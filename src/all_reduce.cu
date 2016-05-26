/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 ************************************************************************/

#include <assert.h>

#include "core.h"
#include "enqueue.h"
#include "crc32.h"

#include "primitives.h"


__device__ inline void getSliceSizeAndOffset(int *size, int *offset, int slice,
    int numSlices, int numBigSlices, int numSmallSlices, int bigSliceN,
    int smallSliceN, int lastSliceN) {
  if (slice < numBigSlices) {
    *size = bigSliceN;
    *offset = slice * bigSliceN;
  } else {
    *size = (slice < numBigSlices + numSmallSlices) ? smallSliceN
        : ((slice == numSlices - 1) ? lastSliceN : 0);
    *offset = numBigSlices * bigSliceN + (slice - numBigSlices) * smallSliceN;
  }
}


template<typename T>
struct AllReduceRingArgs {
  int ThisId;

  T ** ThisPtrToNextOutput;
  T ** PrevPtrToThisOutput;
  volatile int* __restrict__ NextOpCounter;
  volatile int* __restrict__ PrevOpCounter;

  volatile T * __restrict__ ThisBuffer;
  volatile T * __restrict__ NextBuffer;

  // local and remote flags
  volatile int * __restrict__ ThisNewDataAvailableFlag;
  volatile int * __restrict__ NextNewDataAvailableFlag;
  volatile int * __restrict__ ThisChunkDoneFlag;
  volatile int * __restrict__ PrevChunkDoneFlag;
};

template<typename T>
struct AllReduceKernelArgs {
  // general parameters
  int NumGPUs;
  int N;
  int opIndex;
  volatile int * __restrict__ opCounter;
  int * __restrict__ doneCount;

  // some pre-computed sizes
  int SliceSize;
  int SliceOffset;
  int ChunkSize;
  int NumChunks;

  // local and remote input, output, and buffer
  const T * __restrict__ ThisInput;
  volatile T * __restrict__ ThisOutput;

  AllReduceRingArgs<T> rings[MAXRINGS];
};

template<int THREADS, int UNROLL, class FUNC, bool PUSHRECV, typename T>
__launch_bounds__(THREADS+WARP_SIZE, 1)
__global__ void AllReduceKernel(const AllReduceKernelArgs<T> args) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  __shared__ volatile T* nextOutput;
  __shared__ AllReduceRingArgs<T> ring;

  if (tid == 0) {
    ring = args.rings[bid];

    if (PUSHRECV) {
      WaitFlag prevCommOp(ring.PrevOpCounter);
      prevCommOp.wait(args.opIndex);

      *((T * volatile *)ring.PrevPtrToThisOutput) = (T*)args.ThisOutput;
      Wait([=] {
        return *((T * volatile *)ring.ThisPtrToNextOutput) != nullptr;
      });
      nextOutput =
        *((volatile T * volatile *)ring.ThisPtrToNextOutput);
      *ring.ThisPtrToNextOutput = nullptr;
    } else {
      WaitFlag nextCommOp(ring.NextOpCounter);
      nextCommOp.wait(args.opIndex);
    }
  }
  __syncthreads();

  WaitFlag thisChunkDone(ring.ThisChunkDoneFlag);
  WaitFlag thisDataReady(ring.ThisNewDataAvailableFlag);
  PostFlag prevChunkDone(ring.PrevChunkDoneFlag);
  PostFlag nextDataReady(ring.NextNewDataAvailableFlag);

  typedef Primitives<THREADS, UNROLL, 2, T, FUNC> Prims;

  int step = 0;
  for (int chunk=bid; chunk<args.NumChunks; chunk+=gridDim.x) {
    // calculate slice size.  for all chunks except (possibly) the last one,
    // this will just be args.SliceSize. For the last one, it may be smaller
    int bigSliceN   = args.SliceSize;
    int smallSliceN = 0;
    int lastSliceN  = 0;
    int numSlices   = args.NumGPUs;
    int numBigSlices   = numSlices;
    int numSmallSlices = 0;

    // last chunk
    if ((chunk + 1 == args.NumChunks) && (args.N % args.ChunkSize > 0)) {
      if (!PUSHRECV) {
        thisChunkDone.wait(2*step); // TODO handle slice resize more elegantly.
      }
      CalcLastChunk<THREADS, UNROLL, T>(&bigSliceN, &smallSliceN, &lastSliceN,
          &numSlices, &numBigSlices, &numSmallSlices, args.N, args.NumChunks,
          args.ChunkSize);
    }

    // this offset is only applied to Data pointers, not to Buffer pointers,
    // since we only have one buffer per chunk
    int chunkOffset = chunk * args.ChunkSize;

    /////////////// begin AllReduce steps ///////////////

    // step 0: push data to next GPU
    int slice = ring.ThisId;
    int offset;
    int sliceSize;
    getSliceSizeAndOffset(&sliceSize, &offset, slice, numSlices,
        numBigSlices, numSmallSlices, bigSliceN, smallSliceN, lastSliceN);

    if (PUSHRECV) {
      Prims::Copy(
          args.ThisInput  + chunkOffset + offset,
          ring.NextBuffer               + offset,
          sliceSize,
          step++, nextDataReady);
    } else {
      Prims::Copy(
          args.ThisInput  + chunkOffset + offset,
          ring.NextBuffer               + offset,
          sliceSize,
          step++, thisChunkDone, nextDataReady);
    }

    // steps j with 1 <= j < k - 1, where k = number of GPUs:
    // reduce and copy to next GPU
    for (int j=1; j<args.NumGPUs-1; ++j) {
      slice = (args.NumGPUs + slice - 1) % args.NumGPUs;
      getSliceSizeAndOffset(&sliceSize, &offset, slice, numSlices,
          numBigSlices, numSmallSlices, bigSliceN, smallSliceN, lastSliceN);

      Prims::Reduce(
          ring.ThisBuffer               + offset,
          args.ThisInput  + chunkOffset + offset,
          ring.NextBuffer               + offset,
          sliceSize,
          step++, thisDataReady, nextDataReady);
    }

    // step k - 1: reduce this buffer and data, which will produce the final
    // result that we store in this data and push to the next GPU
    slice = (args.NumGPUs + slice - 1) % args.NumGPUs;
    getSliceSizeAndOffset(&sliceSize, &offset, slice, numSlices,
        numBigSlices, numSmallSlices, bigSliceN, smallSliceN, lastSliceN);

    if (PUSHRECV) {
      Prims::ReduceCopy(
          ring.ThisBuffer               + offset,
          args.ThisInput  + chunkOffset + offset,
          nextOutput      + chunkOffset + offset,
          args.ThisOutput + chunkOffset + offset,
          sliceSize,
          step++, thisDataReady, nextDataReady);
    } else {
      Prims::ReduceCopy(
          ring.ThisBuffer               + offset,
          args.ThisInput  + chunkOffset + offset,
          ring.NextBuffer               + offset,
          args.ThisOutput + chunkOffset + offset,
          sliceSize,
          step++, thisDataReady, nextDataReady);
    }

    // steps j with k <= j < 2*k-2: copy result to next GPU
    for (int j=1; j<args.NumGPUs-1; ++j) {
      slice = (args.NumGPUs + slice - 1) % args.NumGPUs;
      getSliceSizeAndOffset(&sliceSize, &offset, slice, numSlices,
          numBigSlices, numSmallSlices, bigSliceN, smallSliceN, lastSliceN);

      if (PUSHRECV) {
        Prims::Copy(
            args.ThisOutput + chunkOffset + offset,
            nextOutput      + chunkOffset + offset,
            sliceSize,
            step++, thisDataReady, nextDataReady);
      } else {
        Prims::DoubleCopy(
            ring.ThisBuffer               + offset,
            ring.NextBuffer               + offset,
            args.ThisOutput + chunkOffset + offset,
            sliceSize,
            step++, thisDataReady, nextDataReady);
      }
    }

    if (!PUSHRECV) {
      // Make final copy from buffer to dest.
      slice = (args.NumGPUs + slice - 1) % args.NumGPUs;
      getSliceSizeAndOffset(&sliceSize, &offset, slice, numSlices,
          numBigSlices, numSmallSlices, bigSliceN, smallSliceN, lastSliceN);

      // Here we need to copy from buffer to this output.
      Prims::Copy(
          ring.ThisBuffer               + offset,
          args.ThisOutput + chunkOffset + offset,
          sliceSize,
          step++, thisDataReady, prevChunkDone);
    }
  }

  // wait for the last data to be pushed to us
  if (tid == 0) {
    if (PUSHRECV) {
      thisDataReady.wait(2*step); // wait to receive last data
    } else {
      thisChunkDone.wait(2*step); // wait for last flag update
      *ring.ThisChunkDoneFlag = 0;
    }

    // Each CTA resets its own flags
    *ring.ThisNewDataAvailableFlag = 0;

    // Last CTA increments comm's operation counts
    if (atomicAdd(args.doneCount, 1) == gridDim.x-1) {
      *args.doneCount = 0;
      __threadfence_system(); // Technically need to ensure that cleared flags
                              // are visible before incrementing op counter.
      *args.opCounter = args.opIndex+1;
    }
  }
}

template<class FUNC, typename T>
ncclResult_t ncclAllReduceWithTypeAndFunc(const void* sendbuff, void* recvbuff,
    const int count, ncclComm* comm, cudaStream_t stream) {
  if (count == 0)
    return ncclSuccess;

  enum {THREADS = 256};
  enum {UNROLL = 8};
  enum {UNROLL_SIZE = THREADS*UNROLL};

  AllReduceKernelArgs<T> args;
  args.NumGPUs = comm->nDev;
  args.N = count;
  args.opIndex = comm->opSched;
  args.opCounter = comm->opCounter;
  args.doneCount = comm->devMem->flags + MAXFLAGS-1;

  // START
  //const int minChunkSize = comm->nDev * 2 * UNROLL_SIZE * sizeof(PackType) / sizeof(T);
  //const int maxNumChunks = count / minChunkSize;
  //const int nRings = std::min(comm->nRings, maxNumChunks);
  //const int 
  //N <= numGpus * chunksPerGpu * chunkSize <= N+numGpus*minChunkSize-1
  // STOP

  const int minSlice = 2 * UNROLL_SIZE * sizeof(PackType) / sizeof(T);
  const int atomSize = minSlice * comm->nDev;
  const int numAtoms = (count + atomSize-1) / atomSize;
  const int nRings = min(numAtoms, comm->nRings);
  const int maxAtomsPerChunk = (comm->buffSize / (nRings * sizeof(T) * atomSize));
  assert (maxAtomsPerChunk > 1);
  const int bufferOffset = maxAtomsPerChunk * atomSize;

  if (numAtoms == nRings) {
    args.SliceSize = minSlice;
    args.ChunkSize = atomSize;
    args.NumChunks = numAtoms;
  } else { // numAtoms > nRings
    int minNumChunks = (numAtoms + maxAtomsPerChunk-1) / maxAtomsPerChunk;
    int targetChunks = ((minNumChunks + nRings-1) / nRings) * nRings;
    int atomsPerChunk = numAtoms / targetChunks;
    if (numAtoms % targetChunks > 1) {
      atomsPerChunk += 1;
      args.NumChunks = (numAtoms+atomsPerChunk-1) / atomsPerChunk;
    } else {
      args.NumChunks = targetChunks;
    }

    args.SliceSize = minSlice * atomsPerChunk;
    args.ChunkSize = atomSize * atomsPerChunk;
  }

  args.ThisInput = (const T*)sendbuff;
  args.ThisOutput = (volatile T*)recvbuff;

  for(int r=0; r<nRings; ++r) {
    AllReduceRingArgs<T>& ring = args.rings[r];
    int index = comm->ringIdx[r];
    int nextId = comm->ncclFromRing[r][(index + 1) % comm->nDev];
    int prevId = comm->ncclFromRing[r][(index + comm->nDev - 1) % comm->nDev];

    ring.ThisId = index;
    ring.ThisPtrToNextOutput = (T**)&(comm->ptrs[nextId].local->recvPtrs[r]);
    ring.PrevPtrToThisOutput = (T**)&(comm->ptrs[prevId].remote->recvPtrs[r]);
    ring.NextOpCounter = comm->ptrs[nextId].opCounter;
    ring.PrevOpCounter = comm->ptrs[prevId].opCounter;
    ring.ThisBuffer = (volatile T*)comm->ptrs[prevId].local->buff + r*bufferOffset;
    ring.NextBuffer = (volatile T*)comm->ptrs[nextId].remote->buff + r*bufferOffset;
    ring.ThisNewDataAvailableFlag = comm->ptrs[prevId].local->flags + r;
    ring.NextNewDataAvailableFlag = comm->ptrs[nextId].remote->flags + r;
    ring.ThisChunkDoneFlag = comm->ptrs[nextId].local->flags + nRings + r;
    ring.PrevChunkDoneFlag = comm->ptrs[prevId].remote->flags + nRings + r;
  }

  // print CRC checksum of input
  int myRank;
  if (ncclPrintCRCs) {
    myRank = comm->userFromRing[0][comm->ringIdx[0]];
    printCRCDev((unsigned char*)sendbuff, count*sizeof(T), myRank, stream);
  }

  dim3 grid(nRings, 1, 1);
  dim3 block(THREADS+1, 1, 1);
  void* argptrs[] = {&args};
  if( comm->useRemoteRecv ) {
    CUDACHECK(cudaLaunchKernel(
        (void*)AllReduceKernel<THREADS, UNROLL, FUNC, true, T>,
        grid, block, argptrs, 0, stream));
  } else {
    CUDACHECK(cudaLaunchKernel(
        (void*)AllReduceKernel<THREADS, UNROLL, FUNC, false, T>,
        grid, block, argptrs, 0, stream));
  }

  // print CRC checksum of output
  if (ncclPrintCRCs) {
    printCRCDev((unsigned char*)recvbuff, count*sizeof(T), myRank, stream);
  }

  return ncclSuccess;
}


template<typename T>
ncclResult_t ncclAllReduceWithType(const void* sendbuff,
    void* recvbuff, int count, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  switch (op) {
  case ncclSum:
    return ncclAllReduceWithTypeAndFunc<FuncSum<T>, T>(
        sendbuff, recvbuff, count, comm, stream);
  case ncclProd:
    return ncclAllReduceWithTypeAndFunc<FuncProd<T>, T>(
        sendbuff, recvbuff, count, comm, stream);
  case ncclMax:
    return ncclAllReduceWithTypeAndFunc<FuncMax<T>, T>(
        sendbuff, recvbuff, count, comm, stream);
  case ncclMin:
    return ncclAllReduceWithTypeAndFunc<FuncMin<T>, T>(
        sendbuff, recvbuff, count, comm, stream);
  }
  return ncclInvalidOperation;
}

class AllReduceFunctor {
public:
  ncclResult_t operator()(const void* sendbuff, void* recvbuff,
      int count, ncclDataType_t datatype, ncclRedOp_t op, int /*root*/,
      ncclComm* comm, cudaStream_t stream) {

    switch (datatype) {
    case ncclChar:
      return ncclAllReduceWithType<char>(sendbuff, recvbuff, count, op,
          comm, stream);
    case ncclInt:
      return ncclAllReduceWithType<int>(sendbuff, recvbuff, count, op,
          comm, stream);
#ifdef CUDA_HAS_HALF
    case ncclHalf:
      return ncclAllReduceWithType<half>(sendbuff, recvbuff, count, op,
          comm, stream);
#endif
    case ncclFloat:
      return ncclAllReduceWithType<float>(sendbuff, recvbuff, count, op,
          comm, stream);
    case ncclDouble:
      return ncclAllReduceWithType<double>(sendbuff, recvbuff, count, op,
          comm, stream);
    case ncclInt64:
      return ncclAllReduceWithType<long long>(sendbuff, recvbuff, count, op,
          comm, stream);
    case ncclUint64:
      return ncclAllReduceWithType<unsigned long long int>(sendbuff, recvbuff, count, op,
          comm, stream);
    }

    return ncclInvalidType;
  }
};

NCCL_API(ncclResult_t, ncclAllReduce, const void* sendbuff, void* recvbuff, int count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, int count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream) {
  return enqueue(AllReduceFunctor(), sendbuff, recvbuff, count, datatype, op, 0,
      comm, stream);
}

