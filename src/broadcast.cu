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
struct BroadcastKernelArgs {
  // general parameters
  int nRanks;
  int root;
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

// Increase Step and boffset for buffer sync
#define NEXT_STEP \
  step++; \
  boffset += sliceSize; \
  if (boffset == buffSize) boffset = 0;

#define ALIGN_SIZE(size, align) \
  size = ((size + (align) - 1) / (align)) * (align);

template<int THREADS, int UNROLL, typename T>
__launch_bounds__(THREADS+WARP_SIZE, 1)
__global__ void BroadcastKernel(const BroadcastKernelArgs<T> args) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  __shared__ T* sharedNextOutput;
  __shared__ DevRing<T> ring;
  bool pushrecv = args.pushrecv;

  LoadRing<THREADS>(args.rings+bid, &ring);
  __syncthreads();

  if (tid == 0) {
    WaitFlag prevCommOp(ring.prevOpCounter, 0);
    WaitFlag nextCommOp(ring.nextOpCounter, 0);
    prevCommOp.wait(args.opIndex);
    nextCommOp.wait(args.opIndex);
    if (pushrecv) {
      *ring.sendPtrToPrev = (T*)args.ThisOutput;
      Wait([=] {
        return *ring.recvPtrFromNext != nullptr;
      });
      sharedNextOutput = *ring.recvPtrFromNext;
      *ring.recvPtrFromNext = nullptr;
    }
  }
  __syncthreads();

  WaitFlag waitDoneFromNext(ring.recvFlagFromNext, (1-NUM_BUFCHUNKS)*NUM_SUBSTEPS);
  WaitFlag waitReadyFromPrev(ring.recvFlagFromPrev, 0);
  PostFlag postDoneToPrev(ring.sendFlagToPrev, 0);
  PostFlag postReadyToNext(ring.sendFlagToNext, 0);

  typedef Primitives<THREADS, UNROLL, NUM_SUBSTEPS, T> Prims;

  const int size = args.N;
  const int rank = ring.userRank[0];
  const int nextRank = ring.userRank[1];
  const int root = args.root;
  const int buffSize = args.buffSize / sizeof(T);
  const int sliceSize = buffSize / NUM_BUFCHUNKS;
  
  int step = 0;
  int boffset = 0;

  // Compute pointers
  const T * __restrict__ thisInput = args.ThisInput;
  T * __restrict__ thisOutput =  args.ThisOutput;
  T * __restrict__ prevInput = ring.recvBuffer;
  T * __restrict__ nextOutput =  ring.sendBuffer;

  for (int offset = bid*sliceSize; offset < size; offset += gridDim.x*sliceSize) {
    int maxOffset = size-offset;
    if (rank == root) {
      Prims::Copy(
          thisInput + offset,
          pushrecv ? sharedNextOutput + offset : nextOutput + boffset,
          sliceSize, maxOffset,
          step,
          waitDoneFromNext,
          postReadyToNext);
    } else if (nextRank == root) {
      if (pushrecv) maxOffset = 0; // Only wait for signals
      Prims::Copy(
          prevInput  + boffset,
          thisOutput + offset,
          sliceSize, maxOffset,
          step,
          waitReadyFromPrev,
          postDoneToPrev);
    } else {
      if (pushrecv) {
        Prims::Copy(
            thisOutput + offset,
            sharedNextOutput + offset,
            sliceSize, maxOffset,
            step,
            waitDoneFromNext, waitReadyFromPrev,
            postReadyToNext, postDoneToPrev);
      } else {
        Prims::DoubleCopy(
            prevInput + boffset,
            thisOutput + offset,
            nextOutput + boffset,
	    sliceSize, maxOffset,
            step,
            waitDoneFromNext, waitReadyFromPrev,
            postReadyToNext, postDoneToPrev);
      }
    }
    NEXT_STEP; // Increases step, boffset
  }

  // wait for the last data to be pushed to us
  if (tid == 0) {
    if (nextRank != root) {
      // Wait for last update from next then reset the flag
      waitDoneFromNext.wait(NUM_SUBSTEPS*(step+NUM_BUFCHUNKS-1));
      *ring.recvFlagFromNext = 0;
    }

    if (rank != root) {
      // reset the flag
      *ring.recvFlagFromPrev = 0;
    }

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
            (void*)K<THREADS, UNROLL, T>, \
            grid, block, argptrs, 0, stream))

#define LAUNCH_KERNEL(K, args, stream, nblocks, nvlink) do { \
  enum {PCIE_THREADS = 256, NVLINK_THREADS = 128}; \
  enum {UNROLL = 8}; \
  int nthreads = nvlink ? NVLINK_THREADS : PCIE_THREADS; \
  dim3 grid(nblocks, 1, 1); \
  dim3 block(nthreads+1, 1, 1); \
  void* argptrs[] = {&args}; \
  if (nvlink) KERNEL(K, NVLINK_THREADS); else KERNEL(K, PCIE_THREADS); \
}while (false)

template<typename T>
ncclResult_t RingBroadcast(void* buff, const int count, const int root,
    ncclComm* comm, cudaStream_t stream) {
  if (count == 0)
    return ncclSuccess;

  BroadcastKernelArgs<T> args;
  args.nRanks = comm->nRanks;
  args.root = root;
  args.buffSize = comm->buffSizePerRing;
  args.N = count;
  args.opIndex = comm->opSched;
  args.opCounter = comm->opCounter;
  args.doneCount = &comm->devMem->doneCount;

  args.ThisInput = (const T*)buff;
  args.ThisOutput = (T*)buff;
  args.rings = comm->devRing;
  args.pushrecv = comm->globalMemSpace;

  if (comm->nRanks != 1) {
    LAUNCH_KERNEL(BroadcastKernel, args, stream, comm->nRings, (comm->p2ptype == ncclComm::NVLINK));
  }

  return ncclSuccess;
}

template<typename T, template <typename> class DummyOp>
class Broadcast {
  public:
  static ncclResult_t entry(const void* sendbuff, void* recvbuff,
      int count, int root, ncclComm* comm, cudaStream_t stream) {
    return RingBroadcast<T>(recvbuff, count, root, comm, stream);
  }
};

NCCL_API(ncclResult_t, ncclBcast, void* buff, int count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclBcast(void* buff, int count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream) {
  return enqueue<Broadcast, FuncNull>(nullptr, buff, count, datatype, root, comm, stream);
}

