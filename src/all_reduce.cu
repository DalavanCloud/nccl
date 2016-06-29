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

template <int THREADS, typename T> __device__ __forceinline__
void LoadRing(const DevRing<char>* src, DevRing<T>* dst) {
  enum { NUM_WORDS = sizeof(DevRing<char>) / sizeof(long long) };
  static_assert(sizeof(DevRing<char>) % sizeof(long long) == 0, "Bad alignment");
  static_assert(THREADS >= NUM_WORDS, "Not enough threads to load DevRing");
  static_assert(sizeof(DevRing<char>) == sizeof(DevRing<T>), "DevRing size mismatch");
  long long* lldst = reinterpret_cast<long long*>(dst);
  const long long* llsrc = reinterpret_cast<const long long*>(src);
  if (threadIdx.x < NUM_WORDS) {
    lldst[threadIdx.x] = llsrc[threadIdx.x];
  }
}

template<typename T>
struct AllReduceKernelArgs {
  // general parameters
  int NumGPUs;
  int N;
  int opIndex;
  volatile int * __restrict__ opCounter;
  int * __restrict__ doneCount;
  bool pushrecv;

  // some pre-computed sizes
  int SliceSize;
  int SliceOffset;
  int ChunkSize;
  int NumChunks;

  // local and remote input, output, and buffer
  const T * __restrict__ ThisInput;
  T * __restrict__ ThisOutput;

  DevRing<char>* rings;
};


template<int THREADS, int UNROLL, class FUNC, typename T>
__launch_bounds__(THREADS+WARP_SIZE, 1)
__global__ void AllReduceKernel(const AllReduceKernelArgs<T> args) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  __shared__ T* sharedNextOutput;
  __shared__ DevRing<T> ring;
  bool pushrecv = args.pushrecv;

  LoadRing<THREADS>(args.rings+bid, &ring);
  __syncthreads();

  if (tid == 0) {
    if (pushrecv) {
      WaitFlag prevCommOp(ring.prevOpCounter);
      prevCommOp.wait(args.opIndex);

      *ring.sendPtrToPrev = (T*)args.ThisOutput;
      Wait([=] {
        return *ring.recvPtrFromNext != nullptr;
      });
      sharedNextOutput = *ring.recvPtrFromNext;
      *ring.recvPtrFromNext = nullptr;
    } else {
      WaitFlag nextCommOp(ring.nextOpCounter);
      nextCommOp.wait(args.opIndex);
    }
  }
  __syncthreads();

  WaitFlag thisChunkDone(ring.recvFlagFromNext);
  WaitFlag thisDataReady(ring.recvFlagFromPrev);
  PostFlag prevChunkDone(ring.sendFlagToPrev);
  PostFlag nextDataReady(ring.sendFlagToNext);

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
      if (!pushrecv) {
        thisChunkDone.wait(2*step); // TODO handle slice resize more elegantly.
      }
      CalcLastChunk<THREADS, UNROLL, T>(&bigSliceN, &smallSliceN, &lastSliceN,
          &numSlices, &numBigSlices, &numSmallSlices, args.N, args.NumChunks,
          args.ChunkSize);
    }

    // this offset is only applied to Data pointers, not to Buffer pointers,
    // since we only have one buffer per chunk
    int chunkOffset = chunk * args.ChunkSize;

    // Compute pointers
    const T * __restrict__ thisInput = args.ThisInput  + chunkOffset;
    T * __restrict__ thisOutput =  args.ThisOutput + chunkOffset;
    T * __restrict__ prevInput = ring.recvBuffer;
    T * __restrict__ nextOutput =  ring.sendBuffer;

    /////////////// begin AllReduce steps ///////////////

    // step 0: push data to next GPU
    int slice = ring.userRank[args.NumGPUs-1];
    int offset;
    int sliceSize;
    getSliceSizeAndOffset(&sliceSize, &offset, slice, numSlices,
        numBigSlices, numSmallSlices, bigSliceN, smallSliceN, lastSliceN);

    if (pushrecv) {
      Prims::Copy(
          thisInput  + offset,
          nextOutput + offset,
          sliceSize,
          step++, nextDataReady);
    } else {
      Prims::Copy(
          thisInput  + offset,
          nextOutput + offset,
          sliceSize,
          step++, thisChunkDone, nextDataReady);
    }

    // k-2 steps: reduce and copy to next GPU
    for (int j=2; j<args.NumGPUs; ++j) {
      slice = ring.userRank[args.NumGPUs-j];
      getSliceSizeAndOffset(&sliceSize, &offset, slice, numSlices,
          numBigSlices, numSmallSlices, bigSliceN, smallSliceN, lastSliceN);

      Prims::Reduce(
          prevInput  + offset,
          thisInput  + offset,
          nextOutput + offset,
          sliceSize,
          step++, thisDataReady, nextDataReady);
    }

    if (pushrecv) nextOutput = sharedNextOutput + chunkOffset;

    // step k - 1: reduce this buffer and data, which will produce the final
    // result that we store in this data and push to the next GPU
    slice = ring.userRank[0];
    getSliceSizeAndOffset(&sliceSize, &offset, slice, numSlices,
        numBigSlices, numSmallSlices, bigSliceN, smallSliceN, lastSliceN);

    Prims::ReduceCopy(
        prevInput  + offset,
        thisInput  + offset,
        nextOutput + offset,
        thisOutput + offset,
        sliceSize,
        step++, thisDataReady, nextDataReady);

    // k-2 steps: copy result to next GPU
    if (pushrecv) {
      for (int j=1; j<args.NumGPUs-1; ++j) {
	slice = ring.userRank[args.NumGPUs - j];
	getSliceSizeAndOffset(&sliceSize, &offset, slice, numSlices,
	    numBigSlices, numSmallSlices, bigSliceN, smallSliceN, lastSliceN);

	Prims::Copy(
	    thisOutput + offset,
	    nextOutput + offset,
	    sliceSize,
	    step++, thisDataReady, nextDataReady);
      }
    } else {
      for (int j=1; j<args.NumGPUs-1; ++j) {
	slice = ring.userRank[args.NumGPUs - j];
	getSliceSizeAndOffset(&sliceSize, &offset, slice, numSlices,
	    numBigSlices, numSmallSlices, bigSliceN, smallSliceN, lastSliceN);

	Prims::DoubleCopy(
	    prevInput  + offset,
	    nextOutput + offset,
	    thisOutput + offset,
	    sliceSize,
	    step++, thisDataReady, nextDataReady);
      }

      // Make final copy from buffer to dest.
      slice = ring.userRank[1];
      getSliceSizeAndOffset(&sliceSize, &offset, slice, numSlices,
          numBigSlices, numSmallSlices, bigSliceN, smallSliceN, lastSliceN);

      // Here we need to copy from buffer to this output.
      Prims::Copy(
          prevInput  + offset,
          thisOutput + offset,
          sliceSize,
          step++, thisDataReady, prevChunkDone);
    }
  }

  // wait for the last data to be pushed to us
  if (tid == 0) {
    if (pushrecv) {
      thisDataReady.wait(2*step); // wait to receive last data
    } else {
      thisChunkDone.wait(2*step); // wait for last flag update
      *ring.recvFlagFromNext = 0;
    }

    // Each CTA resets its own flags
    *ring.recvFlagFromPrev = 0;

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
ncclResult_t RingAllReduce(const void* sendbuff, void* recvbuff,
    const int count, ncclComm* comm, cudaStream_t stream) {
  if (count == 0)
    return ncclSuccess;

  AllReduceKernelArgs<T> args;
  args.NumGPUs = comm->nRanks;
  args.N = count;
  args.opIndex = comm->opSched;
  args.opCounter = comm->opCounter;
  args.doneCount = comm->devMem->flags + MAXFLAGS-1;

  const int minSlice = 2 * NCCL_UNROLL_SIZE * sizeof(PackType) / sizeof(T);
  const int atomSize = minSlice * comm->nRanks;
  const int numAtoms = (count + atomSize-1) / atomSize;
  const int nRings = min(numAtoms, comm->nRings);
  const int maxAtomsPerChunk = comm->buffSizePerRing / (sizeof(T) * atomSize);
  assert (maxAtomsPerChunk > 1);

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
  args.ThisOutput = (T*)recvbuff;
  args.rings = comm->devRing;
  args.pushrecv = comm->globalMemSpace;

  if (comm->nRanks == 1) {
    if (sendbuff != recvbuff)
      CUDACHECK(cudaMemcpyAsync(recvbuff, sendbuff, count*sizeof(T), cudaMemcpyDeviceToDevice, stream));
  } else {
    LAUNCH_KERNEL(AllReduceKernel, args, stream, nRings, (comm->p2ptype == ncclComm::NVLINK));
  }

  return ncclSuccess;
}

template<typename T, template <typename> class RedOp>
class AllReduce {
  public:
  static ncclResult_t entry(const void* sendbuff, void* recvbuff,
      int count, int /*root*/, ncclComm* comm, cudaStream_t stream) {
    return RingAllReduce<RedOp<T>, T>(sendbuff, recvbuff, count, comm, stream);
  }
};

NCCL_API(ncclResult_t, ncclAllReduce, const void* sendbuff, void* recvbuff, int count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, int count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream) {
  return enqueue<AllReduce>(sendbuff, recvbuff, count, datatype, op, 0, comm, stream);
}

