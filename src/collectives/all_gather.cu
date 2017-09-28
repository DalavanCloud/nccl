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
__global__ void AllGatherKernel(const KernelArgs<T> args) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  __shared__ T* sharedNextOutput;
  struct ncclComm* comm = args.comm;
  struct ncclRing* ring = comm->rings+bid;
  int prevdirect = ring->recv.conn.direct;
  int nextdirect = ring->send.conn.direct;

  WaitFlag waitDoneFromNext(ring->send.conn.head, NUM_BUFCHUNKS*NUM_SUBSTEPS);
  WaitFlag waitReadyFromPrev(ring->recv.conn.tail, NUM_SUBSTEPS);
  PostFlag postDoneToPrev(ring->recv.conn.head, NUM_SUBSTEPS, NULL, 0);
  PostFlag postReadyToNext(ring->send.conn.tail, 0, ring->send.conn.fifo, NUM_BUFCHUNKS*NUM_SUBSTEPS);

  typedef Primitives<THREADS, UNROLL, NUM_SUBSTEPS, T> Prims;

  const ssize_t size = args.N;
  const int nranks = comm->nRanks;
  const int buffSize = ring->buffSize / sizeof(T);
  const int sliceSize = buffSize / NUM_BUFCHUNKS;

  if (tid == 0) {
    // Update in case we skipped some collectives
    *ring->recv.conn.opCount = args.opCount;
    // Wait for next to be ready
    WaitFlag waitOpCountNext(ring->send.conn.opCount, 0);
    waitOpCountNext.wait(args.opCount);
    if (prevdirect) {
      *ring->recv.conn.ptrExchange = args.ThisOutput;
    }
    if (nextdirect) {
      void* volatile* ptr = &(ring->devMem->ptrExchange);
      while (*ptr == nullptr);
      sharedNextOutput = (T*)*ptr;
      *ptr = nullptr;
    }
  }
  __syncthreads();

  uint64_t step = 0ULL;
  int poffset, noffset = 0;

  // Compute pointers
  const T * __restrict__ thisInput = args.ThisInput;
  T * __restrict__ thisOutput = args.ThisOutput;
  T * __restrict__ prevInput = (T*)ring->recv.conn.buff;
  T * __restrict__ nextOutput = (T*)ring->send.conn.buff;

  for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += gridDim.x*sliceSize) {
    int chunkSize = min(sliceSize, DIVUP(size-gridOffset,gridDim.x));
    ALIGN_SIZE(chunkSize, THREADS*sizeof(uint64_t)/sizeof(T));
    ssize_t chunkOffset = gridOffset + bid*chunkSize;

    /////////////// begin AllGather steps ///////////////
    ssize_t offset;
    int maxOffset = min(chunkSize, size-chunkOffset);
    int rankDest;

    // step 0: push data to next GPU
    rankDest = ring->devUserRanks[0];
    offset = chunkOffset + rankDest * size;

    if (thisInput + chunkOffset == thisOutput + offset) { // In place
      Prims::Copy(
          thisInput  + chunkOffset,
          nextdirect ? (sharedNextOutput + offset) : (nextOutput + noffset),
          sliceSize, maxOffset,
          step,
          waitDoneFromNext,
          postReadyToNext);
    } else {
      Prims::DoubleCopy(
          thisInput  + chunkOffset,
          thisOutput + offset,
	  nextdirect ? (sharedNextOutput + offset) : (nextOutput + noffset),
          sliceSize, maxOffset,
          step,
          waitDoneFromNext,
          postReadyToNext);
    }

    NEXT_STEP; // Increases step, poffset, noffset

    // k-2 steps: copy to next GPU
    if (prevdirect) {
      for (int j=1; j<nranks-1; ++j) {
        rankDest = ring->devUserRanks[nranks-j];
        offset = chunkOffset + rankDest * size;

        Prims::Copy(
            thisOutput + offset,
	    nextdirect ? (sharedNextOutput + offset) : (nextOutput + noffset),
            sliceSize, maxOffset,
            step,
            waitDoneFromNext, waitReadyFromPrev,
            postReadyToNext, postDoneToPrev);

        NEXT_STEP;
      }
      Prims::Copy(
          NULL,
          NULL,
          0, 0,
          step,
          waitReadyFromPrev,
          postDoneToPrev);
    } else {
      for (int j=1; j<nranks-1; ++j) {
        rankDest = ring->devUserRanks[nranks-j];
        offset = chunkOffset + rankDest * size;

        Prims::DoubleCopy(
            prevInput + poffset,
            thisOutput + offset,
	    nextdirect ? (sharedNextOutput + offset) : (nextOutput + noffset),
            sliceSize, maxOffset,
            step,
            waitDoneFromNext, waitReadyFromPrev,
            postReadyToNext, postDoneToPrev);

        NEXT_STEP;
      }

      // Make final copy from buffer to dest.
      rankDest = ring->devUserRanks[1];
      offset = chunkOffset + rankDest * size;

      // Here we need to copy from buffer to this output.
      Prims::Copy(
          prevInput + poffset,
          thisOutput + offset,
          sliceSize, maxOffset,
          step,
          waitReadyFromPrev,
          postDoneToPrev);
    }
  }

  if (tid == 0) {
    waitDoneFromNext.wait(NUM_SUBSTEPS*(step + NUM_BUFCHUNKS));
    *ring->send.conn.head = 0ULL;
    *ring->recv.conn.tail = 0ULL;
    __threadfence_system();
    *ring->recv.conn.opCount = args.opCount+1;
  }
}

#include "ll_kernel.h"

#define NEXT_STEP_LL \
  poffset = noffset; \
  pflag = nflag; \
  noffset += llSliceSize; \
  if (noffset == llBuffSize) { noffset = 0; } \
  nflag++; \
  step++;

template<int THREADS, class FUNC, typename T>
__global__ void AllGatherKernelSmall(const KernelArgs<T> args) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  struct ncclComm* comm = args.comm;
  struct ncclRing* ring = comm->rings+bid;
  volatile uint64_t * recvHeadPtr = ring->recv.conn.llHead;
  volatile uint64_t * sendHeadPtr = ring->send.conn.llHead;
  volatile int * sizesFifo = ring->send.conn.llFifo;
  uint64_t sendHead = sendHeadPtr[0];

  typedef LLPrimitives<THREADS, T, FUNC> LL;

  const ssize_t size = args.N;
  //const int rank = comm->rank;
  const int nranks = comm->nRanks;
  const int llBuffSize = LL_BUFF_SIZE / (2*sizeof(uint64_t));
  const int llSliceSize = llBuffSize / NUM_LL_CHUNKS;
  const int sliceSize = llSliceSize * sizeof(uint64_t) / sizeof(T);

  uint64_t step = ring->send.conn.llStep;
  uint32_t pflag, nflag = step + 1;
  int poffset, noffset = llSliceSize * STEP_TO_SLOT(step);

  // Compute pointers
  const T * __restrict__ thisInput = args.ThisInput;
  T * __restrict__ thisOutput = args.ThisOutput;
  union ncclLLFifoLine * prevInput = (union ncclLLFifoLine *)ring->recv.conn.llBuff;
  union ncclLLFifoLine * nextOutput = (union ncclLLFifoLine *)ring->send.conn.llBuff;

  for (ssize_t chunkOffset = 0; chunkOffset < size; chunkOffset += sliceSize) {
    /////////////// begin AllGather steps ///////////////
    ssize_t offset;
    int maxOffset = min(sliceSize, size-chunkOffset);
    int rankDest;

    // step 0: push data to next GPU
    rankDest = ring->devUserRanks[0];
    offset = chunkOffset + rankDest * size;

    WAIT_NEXT;
    if (thisInput + chunkOffset == thisOutput + offset) { // In place
      LL::ReduceCopy(
          thisInput  + chunkOffset,
          nextOutput + noffset,
          maxOffset, nflag);
    } else {
      LL::ReduceCopy(
          thisInput  + chunkOffset,
          thisOutput + offset,
          nextOutput + noffset,
          maxOffset, nflag);
    }
    POST_SIZE;

    NEXT_STEP_LL;

    // k-2 steps: copy to next GPU
    for (int j=1; j<nranks-1; ++j) {
      rankDest = ring->devUserRanks[nranks-j];
      offset = chunkOffset + rankDest * size;

      WAIT_NEXT;
      LL::ReduceCopy(
          prevInput  + poffset,
          thisOutput + offset,
          nextOutput + noffset,
          maxOffset, pflag, nflag);
      POST_SIZE;
      ACK_PREV;

      NEXT_STEP_LL;
    }

    // step k-1: final store
    rankDest = ring->devUserRanks[1];
    offset = chunkOffset + rankDest * size;

    LL::ReduceCopy(
        prevInput  + poffset,
        thisOutput + offset,
        maxOffset, pflag);
    ACK_PREV;
  }

  FIFO_CLEANING_AND_SAVE_STEP(nflag);
}

#define UNROLL 8

template<class FUNC, typename T>
ncclResult_t RingAllGather(const void* sendbuff, void* recvbuff,
    const size_t count, ncclComm* comm, cudaStream_t stream) {
  if (comm->nRanks == 1) {
    if (sendbuff != recvbuff)
      CUDACHECK(cudaMemcpyAsync(recvbuff, sendbuff, count*sizeof(T), cudaMemcpyDeviceToDevice, stream));
  } else {
    ArgsSetup(sendbuff, recvbuff, 0, count, comm);
    if (count*sizeof(T)*comm->nRanks <= comm->llThreshold) {
      NCCLCHECK(transportSaveProxies(1, NUM_LL_CHUNKS, comm->nRanks-1, 1, 2*count*sizeof(T), proxyPatternRing, comm, 1, 1));
      SAVE_KERNEL_SMALL(AllGatherKernelSmall, comm, FUNC, T, stream);
    } else {
      NCCLCHECK(transportSaveProxies(NUM_SUBSTEPS, NUM_BUFCHUNKS, comm->nRanks-1, 1, count*sizeof(T), proxyPatternRing, comm, comm->nRings, 0));
      SAVE_KERNEL(AllGatherKernel, comm, UNROLL, FUNC, T, stream);
      comm->opCount++;
    }
  }

  return ncclSuccess;
}

template<typename T, template<typename> class RedOp>
class AllGather {
  public:
  static ncclResult_t entry(const void* sendbuff, void* recvbuff,
      size_t count, int /*root*/, ncclComm* comm, cudaStream_t stream) {
    return RingAllGather<RedOp<T>, T>(sendbuff, recvbuff, count, comm, stream);
  }
};

ncclResult_t ncclAllGatherFunc(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  return enqueue<AllGather>(sendbuff, recvbuff, count, datatype, op, root, comm, stream);
}

NCCL_API(ncclResult_t, ncclAllGather, const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  return ncclEnqueueCheck(ncclAllGatherFunc, "AllGather", sendbuff, recvbuff, sendcount, datatype, 
      ncclSum, 0, comm, stream);
}

