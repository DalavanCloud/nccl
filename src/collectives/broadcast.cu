/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "common_coll.h"
#include "enqueue.h"
#include "primitives.h"

#define NUM_SUBSTEPS 4
#define NUM_BUFCHUNKS 2

// Increase Step and boffset for buffer sync
#define NEXT_STEP \
  step++; \
  boffset += sliceSize; \
  if (boffset == buffSize) boffset = 0;

#define ALIGN_SIZE(size, align) \
  size = ((size + (align) - 1) / (align)) * (align);

template<int THREADS, int UNROLL, class FUNC, typename T>
__launch_bounds__(THREADS+WARP_SIZE, 1)
__global__ void BroadcastKernel(const KernelArgs<T> args) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  __shared__ T* sharedNextOutput;
  struct ncclComm* comm = args.comm;
  struct ncclRing* ring = comm->rings+bid;
  int prevdirect = ring->recv.conn.direct;
  int nextdirect = ring->send.conn.direct;

  WaitFlag waitDoneFromNext(ring->send.conn.head, (NUM_BUFCHUNKS-1)*NUM_SUBSTEPS);
  WaitFlag waitReadyFromPrev(ring->recv.conn.tail, 0);
  PostFlag postDoneToPrev(ring->recv.conn.head, 0, NULL, 0);
  PostFlag postReadyToNext(ring->send.conn.tail, 0, ring->send.conn.fifo, NUM_BUFCHUNKS*NUM_SUBSTEPS);

  typedef Primitives<THREADS, UNROLL, NUM_SUBSTEPS, T> Prims;

  const ssize_t size = args.N;
  const int buffSize = ring->buffSize / sizeof(T);
  const int sliceSize = buffSize / NUM_BUFCHUNKS;
  const int rank = ring->devUserRanks[0];
  const int nextRank = ring->devUserRanks[1];
  const int root = args.root;

  if (tid == 0) {
    // Update in case we skipped some collectives
    *ring->recv.conn.opCount = args.opCount;
    if (nextRank != root) {
      // Wait for next to be ready
      WaitFlag waitOpCountNext(ring->send.conn.opCount, 0);
      waitOpCountNext.wait(args.opCount);
    }
    if (rank != root && prevdirect) {
      *ring->recv.conn.ptrExchange = args.ThisOutput;
    }
    if (nextRank != root && nextdirect) {
      void* volatile* ptr = &(ring->devMem->ptrExchange);
      while (*ptr == nullptr);
      sharedNextOutput = (T*)*ptr;
      *ptr = nullptr;
    }
  }
  __syncthreads();
  
  uint64_t step = 0ULL;
  int boffset = 0;

  // Compute pointers
  const T * __restrict__ thisInput = args.ThisInput;
  T * __restrict__ thisOutput = args.ThisOutput;
  T * __restrict__ prevInput = (T*)ring->recv.conn.buff;
  T * __restrict__ nextOutput = (T*)ring->send.conn.buff;

  for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += gridDim.x*sliceSize) {
    int chunkSize = min(sliceSize, DIVUP(size-gridOffset,gridDim.x));
    ALIGN_SIZE(chunkSize, THREADS*sizeof(uint64_t)/sizeof(T));
    ssize_t offset = gridOffset + bid*chunkSize;
    int maxOffset = min(chunkSize, size-offset);

    if (rank == root) {
      if (thisInput == thisOutput) {
        Prims::Copy(
            thisInput  + offset,
            nextdirect ? (sharedNextOutput + offset) : (nextOutput + boffset),
            sliceSize, maxOffset,
            step,
            waitDoneFromNext,
            postReadyToNext);
      } else {
        Prims::DoubleCopy(
            thisInput  + offset,
            thisOutput + offset,
            nextdirect ? (sharedNextOutput + offset) : (nextOutput + boffset),
            sliceSize, maxOffset,
            step,
            waitDoneFromNext,
            postReadyToNext);
      }
    } else if (nextRank == root) {
      if (prevdirect) maxOffset = 0; // Only wait for signals
      Prims::Copy(
          prevInput  + boffset,
          thisOutput + offset,
          sliceSize, maxOffset,
          step,
          waitReadyFromPrev,
          postDoneToPrev);
    } else {
      if (prevdirect) {
        Prims::Copy(
            thisOutput + offset,
            nextdirect ? (sharedNextOutput + offset) : (nextOutput + boffset),
            sliceSize, maxOffset,
            step,
            waitDoneFromNext, waitReadyFromPrev,
            postReadyToNext, postDoneToPrev);
      } else {
        Prims::DoubleCopy(
            prevInput + boffset,
            thisOutput + offset,
	    nextdirect ? (sharedNextOutput + offset) : (nextOutput + boffset),
            sliceSize, maxOffset,
            step,
            waitDoneFromNext, waitReadyFromPrev,
            postReadyToNext, postDoneToPrev);
      }
    }
    NEXT_STEP; // Increases step, boffset
  }

  if (tid == 0) {
    if (nextRank != root) { 
      // Wait for next to have consumed data before resetting the flag
      waitDoneFromNext.wait(NUM_SUBSTEPS*(step + NUM_BUFCHUNKS - 1));
      *ring->send.conn.head = 0ULL;
    }
    *ring->recv.conn.tail = 0ULL;
    __threadfence_system();
    *ring->recv.conn.opCount = args.opCount+1;
  }
}

#include "ll_kernel.h"

#define NEXT_STEP_LL \
  boffset += llSliceSize; \
  if (boffset == llBuffSize) boffset = 0; \
  flag++; \
  step++;

template<int THREADS, class FUNC, typename T>
__global__ void BroadcastKernelSmall(const KernelArgs<T> args) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  struct ncclComm* comm = args.comm;
  struct ncclRing* ring = comm->rings+bid;
  volatile uint64_t * recvHeadPtr = ring->recv.conn.llHead;
  volatile uint64_t * sendHeadPtr = ring->send.conn.llHead;
  volatile int * sizesFifo = ring->send.conn.llFifo;
  uint64_t sendHead = sendHeadPtr[0];
  const int rank = comm->rank;
  const int nextRank = ring->devUserRanks[1];
  const int root = args.root;

  typedef LLPrimitives<THREADS, T, FUNC> LL;

  const ssize_t size = args.N;
  const int llBuffSize = LL_BUFF_SIZE / (2*sizeof(uint64_t));
  const int llSliceSize = llBuffSize / NUM_LL_CHUNKS;
  const int sliceSize = llSliceSize * sizeof(uint64_t) / sizeof(T);

  uint64_t step = ring->send.conn.llStep;
  uint32_t flag = step + 1;
  int boffset = llSliceSize * STEP_TO_SLOT(step);

  // Compute pointers
  const T * __restrict__ thisInput = args.ThisInput;
  T * __restrict__ thisOutput = args.ThisOutput;
  union ncclLLFifoLine * prevInput = (union ncclLLFifoLine *)ring->recv.conn.llBuff;
  union ncclLLFifoLine * nextOutput = (union ncclLLFifoLine *)ring->send.conn.llBuff;

  for (ssize_t offset = 0; offset < size; offset += sliceSize) {
    int chunkSize = min(sliceSize, size-offset);
    ALIGN_SIZE(chunkSize, THREADS*sizeof(uint64_t)/sizeof(T));
    int maxOffset = min(chunkSize, size-offset);
    if (rank == root) {
      WAIT_NEXT;
      LL::ReduceCopy(
          thisInput + offset,
          nextOutput + boffset,
          maxOffset, flag);
      POST_SIZE;
      NEXT_STEP_LL;
    } else if (nextRank == root) {
      LL::ReduceCopy(
          prevInput + boffset,
          thisOutput + offset,
          maxOffset, flag);
      NEXT_STEP_LL;
      ACK_PREV;
    } else {
      WAIT_NEXT;
      LL::ReduceCopy(
          prevInput + boffset,
          thisOutput + offset,
          nextOutput + boffset,
          maxOffset, flag, flag);
      POST_SIZE;
      NEXT_STEP_LL;
      ACK_PREV;
    }
  }

  // We need everyone to acknowledge data even if they didn't receive anything
  // so that the next collective can start right away.
  ACK_PREV;

  FIFO_CLEANING_AND_SAVE_STEP(flag);
}

#define UNROLL 8

template<class FUNC, typename T>
ncclResult_t RingBroadcast(const void* sendbuff, void* recvbuff, const size_t count, const int root,
    ncclComm* comm, cudaStream_t stream) {
  if (comm->nRanks == 1) {
    if (sendbuff != recvbuff)
      CUDACHECK(cudaMemcpyAsync(recvbuff, sendbuff, count*sizeof(T), cudaMemcpyDeviceToDevice, stream));
  } else {
    ArgsSetup(sendbuff, recvbuff, root, count, comm);
    if (count*sizeof(T) <= comm->llThreshold) {
      NCCLCHECK(transportSaveProxies(1, NUM_LL_CHUNKS, 1, 1, 2*count*sizeof(T), proxyPatternFrom(root), comm, 1, 1));
      SAVE_KERNEL_SMALL(BroadcastKernelSmall, comm, FUNC, T, stream);
    } else {
      NCCLCHECK(transportSaveProxies(NUM_SUBSTEPS, NUM_BUFCHUNKS, 1, 1, count*sizeof(T), proxyPatternFrom(root), comm, comm->nRings, 0));
      SAVE_KERNEL(BroadcastKernel, comm, UNROLL, FUNC, T, stream);
      comm->opCount++;
    }
  }

  return ncclSuccess;
}

template<typename T, template<typename> class RedOp>
class Broadcast {
  public:
  static ncclResult_t entry(const void* sendbuff, void* recvbuff,
      size_t count, int root, ncclComm* comm, cudaStream_t stream) {
    return RingBroadcast<RedOp<T>, T>(sendbuff, recvbuff, count, root, comm, stream);
  }
};

ncclResult_t ncclBroadcastFunc(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  return enqueue<Broadcast>(sendbuff, recvbuff, count, datatype, op, root, comm, stream);
}

NCCL_API(ncclResult_t, ncclBcast, void* buff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclBcast(void* buff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream) {
  return ncclEnqueueCheck(ncclBroadcastFunc, "Bcast", buff, buff, count, datatype,
     ncclSum, root, comm, stream);
}

