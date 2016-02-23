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

#include <nvToolsExt.h>
#include <assert.h>

#include "core.h"
#include "common_kernel.h"
#include "copy_kernel.h"
#include "enqueue.h"
#include "reduce_kernel.h"

/* HIERARCHY
 *
 * The data is split into CHUNKS, and each CHUNK is split into NUM_SUBCHUNKS
 * SUBCHUNKS, where each SUBCHUNK is an independent, complete reduction. Each
 * GPU has a buffer that can fit an entire CHUNK, so that all SUBCHUNKS can be
 * processed without checking that the buffer on the receiving GPU is empty. A
 * SUBCHUNK is split into NUM_GPUS SLICES and each GPU works on a different
 * SLICE at the same time. Before moving on the the next SLICE in the reduction
 * algorithm, the GPU has to check whether it has received the data from the
 * previous GPU it needs for this SLICE. To hide the latency of this
 * communication, each GPU processes all the SLICES of all the SUBCHUNKS in
 * sequence before moving on to the next SLICE. Each SLICE is split into a
 * certain number of UNROLLS (determined by the buffer size) and each thread
 * performs UNROLL_COUNT single-data-element operations inside an UNROLL. As the
 * name suggests, the UNROLL_COUNT operations within an UNROLL are unrolled.
*/

// Number of threads used to perform copies, etc. Must be multiple of 32.
// An additional thread is used to handle threadfences, so the CUDA blocks
// have dimension NUM_THREADS+1.
#define NUM_THREADS     256

// Each thread unrolls the innermost loop of the copy or reduction operations
// to this many single-data-element instructions
#define UNROLL_COUNT    8

#define UNROLL_SIZE     (UNROLL_COUNT * NUM_THREADS)

// To hide the latency associated with the synchronization between different
// subchunks, we interleave the independent subchunks so that more data can be
// transferred while the sync is in progress. This is the number of subchunks
// that are active at the same time
#define NUM_SUBCHUNKS   2


// If this is called with STEP, it means that we just finished processing the
// data for step STEP on this GPU, which is the data required on the next GPU
// for step STEP + 1, so we signal the next GPU that its data for step STEP + 1
// is available. This is called by one particular consumer warp and so we select
// the first thread in the warp to set the flag.
#define SIGNAL_NEW_DATA_AVAILABLE(chunk, subchunk, step)                      \
    do {                                                                      \
      __threadfence_system();                                                 \
      *ring.NextNewDataAvailableFlag =                                        \
          NUM_SUBCHUNKS*((chunk) * (2*args.NumGPUs-2) + (step)) + subchunk+1; \
    } while (0)

// This is called by all producer threads, but only thread 0 spins on the flag,
#define WAIT_FOR_NEW_DATA(chunk, subchunk, step)                              \
    do {                                                                      \
      if (tid == 0) {                                                         \
        int val = NUM_SUBCHUNKS*((int)(chunk) * (2*args.NumGPUs-2) + (step))  \
            + subchunk + 1;                                                   \
        Wait([=] { return *ring.ThisNewDataAvailableFlag >= val; });          \
      }                                                                       \
      BAR(sync, 1, NUM_THREADS);                                              \
    } while (0)

#define SIGNAL_CHUNK_DONE(chunk, subchunk)                                    \
    do {                                                                      \
      *ring.PrevChunkDoneFlag = NUM_SUBCHUNKS*(chunk) + subchunk + 1;         \
    } while (0)

#define WAIT_FOR_CHUNK(chunk, subchunk)                                       \
    do {                                                                      \
      if (tid == 0) {                                                         \
        int val = NUM_SUBCHUNKS * (chunk) + subchunk + 1;                     \
        Wait([=] { return *ring.ThisChunkDoneFlag >= val; });                 \
      }                                                                       \
      BAR(sync, 1, NUM_THREADS);                                              \
    } while (0)


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

  // some pre-computed sizes
  int SliceSize;
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
  __shared__ volatile void * nextOutput;
  __shared__ AllReduceRingArgs<T> ring;
  ring = args.rings[bid];

  // First wait for args.PrevPtrToThisOutput to become nullptr to ensure that
  // the previous GPU is done with a previous collective operation.
  if (tid == 0) {
    Wait([=] {
      return *((T * volatile *)ring.PrevPtrToThisOutput) == nullptr;
    });

    *((T * volatile *)ring.PrevPtrToThisOutput) = (T*)args.ThisOutput;

    Wait([=] {
      return *((T * volatile *)ring.ThisPtrToNextOutput) != nullptr;
    });

    if (PUSHRECV)
      nextOutput =
        *((volatile void * volatile *)ring.ThisPtrToNextOutput);
  }
  __syncthreads();


  int chunk;
  for (chunk = bid; chunk < args.NumChunks; chunk+=gridDim.x) {
    // calculate slice size.  for all chunks except (possibly) the last one,
    // this will just be args.SliceSize. For the last one, it may be smaller
    int bigSliceN   = args.SliceSize;
    int smallSliceN = 0;
    int lastSliceN  = 0;
    int numSlices   = args.NumGPUs * NUM_SUBCHUNKS;
    int numBigSlices   = numSlices;
    int numSmallSlices = 0;

    // last chunk
    if ((chunk + 1 == args.NumChunks) && (args.N % args.ChunkSize > 0))
      CalcLastChunk<THREADS, UNROLL, T>(&bigSliceN, &smallSliceN, &lastSliceN,
          &numSlices, &numBigSlices, &numSmallSlices, args.N, args.NumChunks,
          args.ChunkSize);

    // this offset is only applied to Data pointers, not to Buffer pointers,
    // since we only have one buffer per chunk
    int chunkOffset = chunk * args.ChunkSize;

    /////////////// begin AllReduce steps ///////////////

    // step 0: push data to next GPU
    int step = 0;
    int slice = ring.ThisId;
    int offset;
    int sliceSize;

    if (tid < THREADS) {
      for(int s=0; s<NUM_SUBCHUNKS; ++s) {
        if (s > 0) { slice += args.NumGPUs; }
        getSliceSizeAndOffset(&sliceSize, &offset, slice, numSlices,
            numBigSlices, numSmallSlices, bigSliceN, smallSliceN, lastSliceN);

        if (!PUSHRECV && chunk > 0) {
          WAIT_FOR_CHUNK(chunk-gridDim.x, s);
        }

        Copy<UNROLL, THREADS>(
            ring.NextBuffer + offset,
            args.ThisInput + chunkOffset + offset,
            sliceSize);

        __syncthreads();
      }
    } else { // is consumer thread
      for(int s=0; s<NUM_SUBCHUNKS; ++s) {
        __syncthreads();
        SIGNAL_NEW_DATA_AVAILABLE(chunk, s, step);
      }
    }

    // steps j with 1 <= j < k - 1, where k = number of GPUs:
    // reduce and copy to next GPU
    for (step = 1; step < args.NumGPUs - 1; ++step) {
      if (tid < THREADS) {
        slice = (args.NumGPUs + slice - 1) % args.NumGPUs;
        for(int s=0; s<NUM_SUBCHUNKS; ++s) {
          if (s > 0) { slice += args.NumGPUs; }
          getSliceSizeAndOffset(&sliceSize, &offset, slice, numSlices,
              numBigSlices, numSmallSlices, bigSliceN, smallSliceN, lastSliceN);

          WAIT_FOR_NEW_DATA(chunk, s, step-1);

          Reduce<UNROLL, THREADS, FUNC>(
              ring.NextBuffer + offset,
              ring.ThisBuffer + offset,
              args.ThisInput + chunkOffset + offset,
              sliceSize);

          __syncthreads();
        }
      } else {
        for(int s=0; s<NUM_SUBCHUNKS; ++s) {
          __syncthreads();
          SIGNAL_NEW_DATA_AVAILABLE(chunk, s, step);
        }
      }
    }

    // step k - 1: reduce this buffer and data, which will produce the final
    // result that we store in this data and push to the next GPU
    step = args.NumGPUs - 1;

    if (tid < THREADS) {
      slice = (args.NumGPUs + slice - 1) % args.NumGPUs;
      for(int s=0; s<NUM_SUBCHUNKS; ++s) {
        if (s > 0) { slice += args.NumGPUs; }
        getSliceSizeAndOffset(&sliceSize, &offset, slice, numSlices,
            numBigSlices, numSmallSlices, bigSliceN, smallSliceN, lastSliceN);

        WAIT_FOR_NEW_DATA(chunk, s, step-1);

        if (PUSHRECV) {
          ReduceAndCopy<UNROLL, THREADS, FUNC>(
              (volatile T *)nextOutput + chunkOffset + offset,
              args.ThisOutput + chunkOffset + offset,
              ring.ThisBuffer + offset,
              args.ThisInput + chunkOffset + offset,
              sliceSize);
        } else {
          ReduceAndCopy<UNROLL, THREADS, FUNC>(
              ring.NextBuffer + offset,
              args.ThisOutput + chunkOffset + offset,
              ring.ThisBuffer + offset,
              args.ThisInput + chunkOffset + offset,
              sliceSize);
        }

        __syncthreads();
      }
    } else {
      for(int s=0; s<NUM_SUBCHUNKS; ++s) {
        __syncthreads();
        SIGNAL_NEW_DATA_AVAILABLE(chunk, s, step);
      }
    }

    // steps j with k <= j < 2*k-2: copy result to next GPU
    for (step = args.NumGPUs; step < 2 * args.NumGPUs - 2; ++step) {
      if (tid < THREADS) {
        slice = (args.NumGPUs + slice - 1) % args.NumGPUs;
        for(int s=0; s<NUM_SUBCHUNKS; ++s) {
          if (s > 0) { slice += args.NumGPUs; }
          getSliceSizeAndOffset(&sliceSize, &offset, slice, numSlices,
              numBigSlices, numSmallSlices, bigSliceN, smallSliceN, lastSliceN);

          WAIT_FOR_NEW_DATA(chunk, s, step-1);

          if( PUSHRECV ) {
            Copy<UNROLL, THREADS>(
                (volatile T *)nextOutput + chunkOffset + offset,
                args.ThisOutput + chunkOffset + offset,
                sliceSize);
          } else {
            DoubleCopy<UNROLL, THREADS>(
                ring.NextBuffer + offset,
                args.ThisOutput + chunkOffset + offset,
                ring.ThisBuffer + offset,
                sliceSize);
          }

          __syncthreads();
        }
      } else {
        for(int s=0; s<NUM_SUBCHUNKS; ++s) {
          __syncthreads();
          SIGNAL_NEW_DATA_AVAILABLE(chunk, s, step);
        }
      }
    }

    if (!PUSHRECV) {
      // Make final copy from buffer to dest.
      if (tid < THREADS) {
        slice = (args.NumGPUs + slice - 1) % args.NumGPUs;
        for(int s=0; s<NUM_SUBCHUNKS; ++s) {
          if (s > 0) { slice += args.NumGPUs; }
          getSliceSizeAndOffset(&sliceSize, &offset, slice, numSlices,
              numBigSlices, numSmallSlices, bigSliceN, smallSliceN, lastSliceN);

          WAIT_FOR_NEW_DATA(chunk, s, step-1);

          // Here we need to copy from buffer to this output.
          Copy<UNROLL, THREADS>(
              args.ThisOutput + chunkOffset + offset,
              ring.ThisBuffer + offset,
              sliceSize);

          __syncthreads();
        }
      } else {
        for(int s=0; s<NUM_SUBCHUNKS; ++s) {
          __syncthreads();
          if(chunk+gridDim.x < args.NumChunks) {
            SIGNAL_CHUNK_DONE(chunk, s);
          }
        }
      }
    }
  }

  // wait for the last data to be pushed to us
  if (PUSHRECV && tid < THREADS) {
    WAIT_FOR_NEW_DATA(chunk-gridDim.x, NUM_SUBCHUNKS-1, 2*args.NumGPUs-3);
  }

  if (tid == 0) {
    *ring.ThisNewDataAvailableFlag = 0;
    if(!PUSHRECV) {
      *ring.ThisChunkDoneFlag = 0;
    }
    *ring.ThisPtrToNextOutput = nullptr;
  }
}

template<class FUNC, typename T>
ncclResult_t ncclAllReduceWithTypeAndFunc(const void* sendbuff, void* recvbuff,
    const int count, ncclComm* comm, cudaStream_t stream) {
  if (count == 0)
    return ncclSuccess;

  AllReduceKernelArgs<T> args;
  args.NumGPUs = comm->nDev;
  args.N = count;

  const int minSlice = UNROLL_SIZE * sizeof(PackType) / sizeof(T);
  const int atomSize = minSlice * NUM_SUBCHUNKS * comm->nDev;
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
    ring.ThisBuffer = (volatile T*)comm->ptrs[prevId].local->buff + r*bufferOffset;
    ring.NextBuffer = (volatile T*)comm->ptrs[nextId].remote->buff + r*bufferOffset;
    ring.ThisNewDataAvailableFlag = comm->ptrs[prevId].local->flags + r;
    ring.NextNewDataAvailableFlag = comm->ptrs[nextId].remote->flags + r;
    ring.ThisChunkDoneFlag = comm->ptrs[nextId].local->flags + nRings + r;
    ring.PrevChunkDoneFlag = comm->ptrs[prevId].remote->flags + nRings + r;
  }

  if( comm->useRemoteRecv ) {
    AllReduceKernel<NUM_THREADS, UNROLL_COUNT, FUNC, true, T>
        <<<nRings, NUM_THREADS + 1, 0, stream>>>(args);
  } else {
    AllReduceKernel<NUM_THREADS, UNROLL_COUNT, FUNC, false, T>
        <<<nRings, NUM_THREADS + 1, 0, stream>>>(args);
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

extern "C" DSOGLOBAL
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, int count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream) {
  return enqueue(AllReduceFunctor(), sendbuff, recvbuff, count, datatype, op, 0,
      comm, stream);
}

