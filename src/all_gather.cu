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

#include <algorithm>
#include <cassert>

#include "core.h"
#include "common_kernel.h"
#include "copy_kernel.h"
#include "enqueue.h"
#include "crc32.h"

/* HIERARCHY
 *
 * The data is split into CHUNKS, and each CHUNK is split into NUM_SUBCHUNKS
 * SUBCHUNKS, where each SUBCHUNK is processed independently. A SUBCHUNK is
 * split into numUnroll UNROLLS and each thread performs UNROLL_COUNT
 * single-data-element operations inside an UNROLL. As the name suggests, the
 * UNROLL_COUNT operations within an UNROLL are unrolled.
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
#define SIGNAL_NEW_DATA_AVAILABLE(chunk, subchunk, step)         \
    do {                                                         \
      __threadfence_system();                                    \
      *ring.NextNewDataAvailableFlag = subchunk + 1              \
          + NUM_SUBCHUNKS*((chunk) * (args.NumGPUs-1) + (step)); \
    } while (0)

// This is called by all producer threads, but only thread 0 spins on the flag,
#define WAIT_FOR_NEW_DATA(chunk, subchunk, step)                        \
    do {                                                                \
      if (tid == 0) {                                                   \
        int val = subchunk + 1                                          \
            + NUM_SUBCHUNKS*((int)(chunk) * (args.NumGPUs-1) + (step)); \
        Wait([=] { return *ring.ThisNewDataAvailableFlag >= val; });   \
      }                                                                 \
      BAR(sync, 1, NUM_THREADS);                                        \
    } while (0)

#define SIGNAL_CHUNK_DONE(chunk, subchunk)                              \
    do {                                                                \
      __threadfence_system();                                           \
      *ring.PrevChunkDoneFlag = NUM_SUBCHUNKS*(chunk) + (subchunk) + 1; \
    } while (0)

#define WAIT_FOR_PREV_CHUNK(chunk, subchunk)                  \
    do {                                                      \
      if (tid == 0) {                                         \
        int val = NUM_SUBCHUNKS*(chunk) + subchunk + 1;       \
        Wait([=] { return *ring.ThisChunkDoneFlag >= val; }); \
      }                                                       \
      BAR(sync, 1, NUM_THREADS);                              \
    } while (0)

__device__ inline void getSliceSizeAndChunkSize(int *sliceSize, int slice,
    int numSlices, int numBigSlices, int numSmallSlices, int bigSliceN,
    int smallSliceN, int lastSliceN) {
  if (slice < numBigSlices) {
    *sliceSize = bigSliceN;
  } else {
    *sliceSize = (slice < numBigSlices + numSmallSlices) ? smallSliceN
        : ((slice == numSlices - 1) ? lastSliceN : 0);
  }
}

template<typename T>
struct AllGatherRingArgs {
  int * UserFromRing;

  T ** ThisPtrToNextOutput;
  T ** PrevPtrToThisOutput;
  volatile int * __restrict__ NextOpCounter;
  volatile int * __restrict__ PrevOpCounter;

  volatile T * __restrict__ ThisBuffer;
  volatile T * __restrict__ NextBuffer;

  // local and remote flags
  volatile int * __restrict__ ThisNewDataAvailableFlag;
  volatile int * __restrict__ NextNewDataAvailableFlag;
  volatile int * __restrict__ ThisChunkDoneFlag;
  volatile int * __restrict__ PrevChunkDoneFlag;
};

template<typename T>
struct AllGatherKernelArgs {
  // general parameters
  int NumGPUs;
  int N;
  int opIndex;
  volatile int* __restrict__ opCounter;
  int * __restrict__ doneCount;

  // some pre-computed sizes
  int SliceSize;
  int ChunkSize;
  int NumChunks;
  int BufferSliceStride;

  // local input and output
  const T * __restrict__ ThisInput;
  volatile T * __restrict__ ThisOutput;

  AllGatherRingArgs<T> rings[MAXRINGS];
};

__device__ inline int GetBlock(const int step,
    const int * const userFromRing, const int numGPUs) {
  return userFromRing[(numGPUs - step) % numGPUs];
}

template<int THREADS, int UNROLL, bool PUSHRECV, typename T>
__global__ void AllGatherKernel(const AllGatherKernelArgs<T> args) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  __shared__ volatile void * nextOutput;
  __shared__ AllGatherRingArgs<T> ring;
  ring = args.rings[bid];

  if (tid == 0) {
    if (PUSHRECV) {
      Wait([=] { return *ring.PrevOpCounter == args.opIndex; });
      *((T* volatile*)ring.PrevPtrToThisOutput) = (T*)args.ThisOutput;
      Wait([=] { return *(T* volatile*)ring.ThisPtrToNextOutput != nullptr; });
      nextOutput = *((volatile void * volatile *)ring.ThisPtrToNextOutput);
      *ring.ThisPtrToNextOutput = nullptr;
    } else { // !PUSHRECV
      Wait([=] { return *ring.NextOpCounter == args.opIndex; });
    }
  }
  __syncthreads();

  int chunk;
  for (chunk = bid; chunk < args.NumChunks; chunk+=gridDim.x) {
    // calculate slice size.  for all chunks except (possibly) the last one,
    // this will just be args.SliceSize. For the last one, it may be smaller
    int bigSliceN   = args.SliceSize;
    int smallSliceN = 0;
    int lastSliceN  = 0;
    int numSlices   = NUM_SUBCHUNKS;
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

    // step 0: copy the resident block from the ThisInput to ThisOutput and also
    // to NextOutput
    int step = 0;
    int block = GetBlock(step, ring.UserFromRing, args.NumGPUs);
    int outputOffset = chunkOffset + block * args.N;
    int inputOffset = chunkOffset;
    int bufferOffset;
    int sliceSize;

    if (!PUSHRECV) {
      bufferOffset = block * args.BufferSliceStride;
    }

    // Copy from ThisInput
    if (tid < THREADS) {
      for(int s=0; s<NUM_SUBCHUNKS; ++s) {
        getSliceSizeAndChunkSize(&sliceSize, s, numSlices, numBigSlices,
            numSmallSlices, bigSliceN, smallSliceN, lastSliceN);

        if (PUSHRECV) {
          DoubleCopy<UNROLL, THREADS>(
              args.ThisOutput + outputOffset,
              (volatile T *)nextOutput + outputOffset,
              args.ThisInput + inputOffset,
              sliceSize);
        } else {
          WAIT_FOR_PREV_CHUNK(chunk-gridDim.x, s);
          DoubleCopy<UNROLL, THREADS>(
              args.ThisOutput + outputOffset,
              ring.NextBuffer + bufferOffset,
              args.ThisInput + inputOffset,
              sliceSize);
        }
        __syncthreads();

        outputOffset += sliceSize;
        inputOffset += sliceSize;
        if (!PUSHRECV)
          bufferOffset += sliceSize;
      }
    } else {
      for(int s=0; s<NUM_SUBCHUNKS; ++s) {
        __syncthreads();
        SIGNAL_NEW_DATA_AVAILABLE(chunk, s, step);
      }
    }

    // steps j with 0 < j < k - 1:
    // copy a block that was pushed to this GPU to the next GPU
    for (step = 1; step < args.NumGPUs - 1; ++step) {
      block = GetBlock(step, ring.UserFromRing, args.NumGPUs);
      outputOffset = chunkOffset + block * args.N;
      if (!PUSHRECV) {
        bufferOffset = block * args.BufferSliceStride;
      }

      if (tid < THREADS) {
        for(int s=0; s<NUM_SUBCHUNKS; ++s) {
          getSliceSizeAndChunkSize(&sliceSize, s, numSlices, numBigSlices,
              numSmallSlices, bigSliceN, smallSliceN, lastSliceN);
          WAIT_FOR_NEW_DATA(chunk, s, step-1);

          if (PUSHRECV) {
            Copy<UNROLL, THREADS>(
                (volatile T *)nextOutput + outputOffset,
                args.ThisOutput + outputOffset,
                sliceSize);
          } else {
            DoubleCopy<UNROLL, THREADS>(
                ring.NextBuffer + bufferOffset,
                args.ThisOutput + outputOffset,
                ring.ThisBuffer + bufferOffset,
                sliceSize);
          }
          __syncthreads();

          outputOffset += sliceSize;
          if (!PUSHRECV)
            bufferOffset += sliceSize;
        }
      } else {
        for(int s=0; s<NUM_SUBCHUNKS; ++s) {
          __syncthreads();
          SIGNAL_NEW_DATA_AVAILABLE(chunk, s, step);
        }
      }
    }

    if (!PUSHRECV) {
      step = args.NumGPUs - 1;
      block = GetBlock(step, ring.UserFromRing, args.NumGPUs);
      outputOffset = chunkOffset + block * args.N;
      bufferOffset = block * args.BufferSliceStride;

      // Make final copy from buffer to dest.
      if (tid < THREADS) {
        for(int s=0; s<NUM_SUBCHUNKS; ++s) {
          getSliceSizeAndChunkSize(&sliceSize, s, numSlices, numBigSlices,
              numSmallSlices, bigSliceN, smallSliceN, lastSliceN);
          WAIT_FOR_NEW_DATA(chunk, s, step-1);

          Copy<UNROLL, THREADS>(
              args.ThisOutput + outputOffset,
              ring.ThisBuffer + bufferOffset,
              sliceSize);

          __syncthreads();

          outputOffset += sliceSize;
          bufferOffset += sliceSize;
        }
      } else {
        for(int s=0; s<NUM_SUBCHUNKS; ++s) {
          __syncthreads();
          SIGNAL_CHUNK_DONE(chunk, s);
        }
      }
    }
  }

  // wait for the last data to be pushed to us
  if (tid < THREADS) {
    if (PUSHRECV)
      WAIT_FOR_NEW_DATA(chunk-gridDim.x, NUM_SUBCHUNKS-1, args.NumGPUs-2);
    else
      WAIT_FOR_PREV_CHUNK(chunk-gridDim.x, NUM_SUBCHUNKS-1);

    if (tid == 0) {
      *ring.ThisNewDataAvailableFlag = 0;
      *ring.ThisChunkDoneFlag = 0;

      if (atomicAdd(args.doneCount, 1) == gridDim.x-1) {
        *args.doneCount = 0;
        __threadfence_system();

        *args.opCounter = args.opIndex+1;
      }
    }
  }
}

template<typename T>
ncclResult_t ncclAllGatherWithType(const void* sendbuff, void* recvbuff,
    int count, ncclComm* comm, int numUnroll, cudaStream_t stream) {
  if (count == 0)
      return ncclSuccess;

  AllGatherKernelArgs<T> args;
  args.NumGPUs = comm->nRanks;
  args.N = count;
  args.opIndex = comm->opSched;
  args.opCounter = comm->opCounter;
  args.doneCount = comm->devMem->flags + MAXFLAGS-1;

  const int minSlice = UNROLL_SIZE * sizeof(PackType) / sizeof(T);
  const int minChunk = NUM_SUBCHUNKS * minSlice;
  const int atomSize = minChunk * comm->nRanks;
  const int numAtoms = (count + minChunk-1) / minChunk;
  const int nRings = min(numAtoms, comm->nRings);

  const int bufferVPerRing = comm->buffSize / (sizeof(PackType) * nRings);
  const int bufferNPerRing = bufferVPerRing * sizeof(PackType) / sizeof(T);
  const int misalignedN = count % (sizeof(PackType) / sizeof(T));
  const int maxAtomsPerChunk = (bufferNPerRing - misalignedN*comm->nRanks) / atomSize;
  assert(maxAtomsPerChunk>1);

  if (numAtoms == nRings) {
    args.SliceSize = minSlice;
    args.NumChunks = numAtoms;
  } else {
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
  }

  args.ChunkSize = args.SliceSize * NUM_SUBCHUNKS;
  args.BufferSliceStride = minChunk * maxAtomsPerChunk + misalignedN;

  args.ThisInput = (const T*)sendbuff;
  args.ThisOutput = (volatile T*)recvbuff;

  for(int r=0; r<nRings; ++r) {
    AllGatherRingArgs<T>& ring = args.rings[r];
    NodeRef* next = comm->ptrs + comm->ncclFromRing[r][(comm->nRanks > 0) ? 1 : 0];
    NodeRef* prev = comm->ptrs + comm->ncclFromRing[r][comm->nRanks - 1];

    /* Block j is coming from sendbuff[j], which lives on device with logical
     * index comm->ringFromUser[j]. But the block ordering does not necessarily
     * follow the ring ordering. Hence the order in which a particular GPU
     * processes the different blocks (the correspondence between the step in
     * the reduction algorithm and the block on which a GPU operates in that
     * particular step) is not the same as the ring order.
     *
     * Say we have 4 GPUs and comm->userFromRing = { 1, 2, 0, 3 }. Then there are 3
     * step in the all-gather algorithm and block 0 comes from device 2, block 1
     * from 0, block 2 from device 1, and block 3 comes from device 3. In the
     * first step of the algorithm, each GPU must copy its own block from its
     * sendbuff to the appropriate location in its recvbuff. The blocks that a
     * GPU has to process in the next steps is determined by the previous step
     * because each GPU only hands off data to the next GPU in the ring.
     *
     * In the above example, we get the following table of which block is
     * processed by each GPU in a given step. The columns correspond to the
     * different GPUs while the rows are the steps in the algorithm.
     *
     *      GPU 0   1   2   3
     * step
     *    0     1   2   0   3
     *    1     3   1   2   0
     *    2     0   3   1   2
     *
     * We note the the rows in the above table are just comm->userFromRing in the
     * first step and the list is cyclicly permuted to the right for each next
     * step. The columns, which are what the individual GPUs need to know, are
     * comm->userFromRing traversed backwards and starting at index k for GPU k.
     * These columns are what we put into args.BlockVsStep to tell the GPU which
     * block it needs to be processing at a particular step. */
    ring.UserFromRing = comm->devUserFromRing[r];

    ring.ThisPtrToNextOutput = (T**)&(next->local->recvPtrs[r]);
    ring.PrevPtrToThisOutput = (T**)&(prev->remote->recvPtrs[r]);
    ring.NextOpCounter = next->opCounter;
    ring.PrevOpCounter = prev->opCounter;
    ring.ThisBuffer = (volatile T*)prev->local->buff + r*bufferNPerRing;
    ring.NextBuffer = (volatile T*)next->remote->buff + r*bufferNPerRing;
    ring.ThisNewDataAvailableFlag = prev->local->flags + r;
    ring.NextNewDataAvailableFlag = next->remote->flags + r;
    ring.ThisChunkDoneFlag = next->local->flags + nRings + r;
    ring.PrevChunkDoneFlag = prev->remote->flags + nRings + r;
  }

  // print CRC checksum of input
  int myRank;
  if (ncclPrintCRCs) {
    myRank = comm->userFromRing[0][0];
    printCRCDev((unsigned char*)sendbuff, count*sizeof(T), myRank, stream);
  }

  if (comm->nRanks == 1) {
    if (sendbuff != recvbuff)
      CUDACHECK(cudaMemcpyAsync(recvbuff, sendbuff, count*sizeof(T), cudaMemcpyDeviceToDevice, stream));
  } else {
    dim3 grid(nRings, 1, 1);
    dim3 block(NUM_THREADS+1, 1, 1);
    void* argptrs[] = {&args};
    if( comm->globalMemSpace ) {
      CUDACHECK(cudaLaunchKernel(
	    (void*)AllGatherKernel<NUM_THREADS, UNROLL_COUNT, true, T>,
	    grid, block, argptrs, 0, stream));
    } else {
      CUDACHECK(cudaLaunchKernel(
	    (void*)AllGatherKernel<NUM_THREADS, UNROLL_COUNT, false, T>,
	    grid, block, argptrs, 0, stream));
    }
  }

  // print CRC checksum of output
  if (ncclPrintCRCs) {
    printCRCDev((unsigned char*)recvbuff, comm->nRanks*count*sizeof(T), myRank, stream);
  }

  return ncclSuccess;
}

class AllGatherFunctor {
public:
  ncclResult_t operator()(const void* sendbuff, void* recvbuff,
      int count, ncclDataType_t datatype, ncclRedOp_t /*dummy operation*/,
      int /*dummy root*/, ncclComm* comm, cudaStream_t stream) {
    int numUnroll = 16; // this is optimal on dt07 with 4 GPUs

    switch (datatype) {
    case ncclChar:
      return ncclAllGatherWithType<char>(sendbuff, recvbuff, count, comm,
          numUnroll, stream);
    case ncclInt:
      return ncclAllGatherWithType<int>(sendbuff, recvbuff, count, comm,
          numUnroll, stream);
#ifdef CUDA_HAS_HALF
    case ncclHalf:
      return ncclAllGatherWithType<half>(sendbuff, recvbuff, count, comm,
          numUnroll, stream);
#endif
    case ncclFloat:
      return ncclAllGatherWithType<float>(sendbuff, recvbuff, count, comm,
          numUnroll, stream);
    case ncclDouble:
      return ncclAllGatherWithType<double>(sendbuff, recvbuff, count, comm,
          numUnroll, stream);
    case ncclInt64:
      return ncclAllGatherWithType<long long>(sendbuff, recvbuff, count, comm,
          numUnroll, stream);
    case ncclUint64:
      return ncclAllGatherWithType<unsigned long long>(sendbuff, recvbuff, count, comm,
          numUnroll, stream);
    }
    return ncclInvalidType;
  }
};

NCCL_API(ncclResult_t, ncclAllGather, const void* sendbuff, int count, ncclDataType_t datatype,
    void* recvbuff, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclAllGather(const void* sendbuff, int count, ncclDataType_t datatype,
    void* recvbuff, ncclComm_t comm, cudaStream_t stream) {
  return enqueue(AllGatherFunctor(), sendbuff, recvbuff, count, datatype,
      ncclSum, 0, comm, stream);
}
