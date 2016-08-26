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

#define NUM_SUBSTEPS 2
#define NUM_BUFCHUNKS 2

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
struct ReduceScatterKernelArgs {
  // general parameters
  int nRanks;
  int buffSize;
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

// Increase Step and poffset/noffset for buffer sync
#define NEXT_STEP \
  step++; \
  poffset = noffset; \
  noffset += sliceSize; \
  if (noffset == buffSize) noffset = 0;

#define ALIGN_SIZE(size, align) \
  size = ((size + (align) - 1) / (align)) * (align);

template<int THREADS, int UNROLL, class FUNC, typename T>
__launch_bounds__(THREADS+WARP_SIZE, 1)
__global__ void ReduceScatterKernel(const ReduceScatterKernelArgs<T> args) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  __shared__ DevRing<T> ring;

  LoadRing<THREADS>(args.rings+bid, &ring);
  __syncthreads();

  if (tid == 0) {
    WaitFlag prevCommOp(ring.prevOpCounter, 0);
    WaitFlag nextCommOp(ring.nextOpCounter, 0);
    prevCommOp.wait(args.opIndex);
    nextCommOp.wait(args.opIndex);
  }
  __syncthreads();

  WaitFlag waitDoneFromNext(ring.recvFlagFromNext, -NUM_BUFCHUNKS*NUM_SUBSTEPS);
  WaitFlag waitReadyFromPrev(ring.recvFlagFromPrev, -1*NUM_SUBSTEPS);
  PostFlag postDoneToPrev(ring.sendFlagToPrev, -1*NUM_SUBSTEPS);
  PostFlag postReadyToNext(ring.sendFlagToNext, 0);

  typedef Primitives<THREADS, UNROLL, NUM_SUBSTEPS, T, FUNC> Prims;

  const int size = args.N;
  const int nranks = args.nRanks;
  const int buffSize = args.buffSize / sizeof(T);
  const int sliceSize = buffSize / NUM_BUFCHUNKS;
  
  int step = 0;
  int poffset, noffset = 0;

  // Compute pointers
  const T * __restrict__ thisInput = args.ThisInput;
  T * __restrict__ thisOutput =  args.ThisOutput;
  T * __restrict__ prevInput = ring.recvBuffer;
  T * __restrict__ nextOutput =  ring.sendBuffer;

  for (int chunkOffset = bid*sliceSize; chunkOffset < size; chunkOffset += gridDim.x*sliceSize) {
    /////////////// begin ReduceScatter steps ///////////////
    int offset;
    int maxOffset = size-chunkOffset;
    int rankDest;

    // step 0: push data to next GPU
    rankDest = ring.userRank[nranks-1];
    offset = chunkOffset + rankDest * size;

    Prims::Copy(
        thisInput  + offset,
        nextOutput + noffset,
        sliceSize, maxOffset,
        step,
        waitDoneFromNext, waitReadyFromPrev,
        postReadyToNext, postDoneToPrev);

    NEXT_STEP; // Increases step, poffset, noffset

    // k-2 steps: reduce and copy to next GPU
    for (int j=2; j<nranks; ++j) {
      rankDest = ring.userRank[nranks-j];
      offset = chunkOffset + rankDest * size;

      Prims::Reduce(
          prevInput  + poffset,
          thisInput  + offset,
          nextOutput + noffset,
          sliceSize, maxOffset,
          step,
          waitDoneFromNext, waitReadyFromPrev,
          postReadyToNext, postDoneToPrev);

      NEXT_STEP;
    }

    // step k - 1: reduce this buffer and data, which will produce the final
    // result that we store in this data and push to the next GPU
    rankDest = ring.userRank[0];
    offset = chunkOffset + rankDest * size;

    Prims::Reduce(
        prevInput  + poffset,
        thisInput  + offset,
        thisOutput + chunkOffset,
        sliceSize, maxOffset,
        step,
        waitDoneFromNext, waitReadyFromPrev,
        postReadyToNext, postDoneToPrev);

    NEXT_STEP;
  }

  // wait for the last data to be pushed to us
  if (tid == 0) {
    // Wait for last update from next then reset the flag
    waitDoneFromNext.wait(NUM_SUBSTEPS*(step+NUM_BUFCHUNKS-1));
    *ring.recvFlagFromNext = 0;

    // Wait for last update from prev then reset the flag
    waitReadyFromPrev.wait(NUM_SUBSTEPS*(step+1));
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

#define KERNEL(K, THREADS) \
  CUDACHECK(cudaLaunchKernel( \
            (void*)K<THREADS, UNROLL, FUNC, T>, \
            grid, block, argptrs, 0, stream))

#define LAUNCH_KERNEL(K, args, stream, nblocks, nvlink) do { \
  enum {PCIE_THREADS = 512, NVLINK_THREADS = 128}; \
  enum {UNROLL = 8}; \
  int nthreads = nvlink ? NVLINK_THREADS : PCIE_THREADS; \
  dim3 grid(nblocks, 1, 1); \
  dim3 block(nthreads+1, 1, 1); \
  void* argptrs[] = {&args}; \
  if (nvlink) KERNEL(K, NVLINK_THREADS); else KERNEL(K, PCIE_THREADS); \
}while (false)

template<class FUNC, typename T>
ncclResult_t RingReduceScatter(const void* sendbuff, void* recvbuff,
    const int count, ncclComm* comm, cudaStream_t stream) {
  if (count == 0)
    return ncclSuccess;

  ReduceScatterKernelArgs<T> args;
  args.nRanks = comm->nRanks;
  args.buffSize = comm->buffSizePerRing;
  args.N = count;
  args.opIndex = comm->opSched;
  args.opCounter = comm->opCounter;
  args.doneCount = &comm->devMem->doneCount;

  args.ThisInput = (const T*)sendbuff;
  args.ThisOutput = (T*)recvbuff;
  args.rings = comm->devRing;
  args.pushrecv = comm->globalMemSpace;

  if (comm->nRanks == 1) {
    if (sendbuff != recvbuff)
      CUDACHECK(cudaMemcpyAsync(recvbuff, sendbuff, count*sizeof(T), cudaMemcpyDeviceToDevice, stream));
  } else {
    LAUNCH_KERNEL(ReduceScatterKernel, args, stream, comm->nRings, (comm->p2ptype == ncclComm::NVLINK));
  }

  return ncclSuccess;
}

template<typename T, template <typename> class RedOp>
class ReduceScatter {
  public:
  static ncclResult_t entry(const void* sendbuff, void* recvbuff,
      int count, int /*root*/, ncclComm* comm, cudaStream_t stream) {
    return RingReduceScatter<RedOp<T>, T>(sendbuff, recvbuff, count, comm, stream);
  }
};

NCCL_API(ncclResult_t, ncclReduceScatter, const void* sendbuff, void* recvbuff, int recvcount,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff, int recvcount,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  return enqueue<ReduceScatter>(sendbuff, recvbuff, recvcount, datatype, op, 0, comm, stream);
}

