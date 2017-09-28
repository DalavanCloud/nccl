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
__global__ void ReduceKernel(const KernelArgs<T> args) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  struct ncclComm* comm = args.comm;
  struct ncclRing* ring = comm->rings+bid;

  WaitFlag waitDoneFromNext(ring->send.conn.head, (NUM_BUFCHUNKS-1)*NUM_SUBSTEPS);
  WaitFlag waitReadyFromPrev(ring->recv.conn.tail, 0);
  PostFlag postDoneToPrev(ring->recv.conn.head, 0, NULL, 0);
  PostFlag postReadyToNext(ring->send.conn.tail, 0, ring->send.conn.fifo, NUM_BUFCHUNKS*NUM_SUBSTEPS);

  typedef Primitives<THREADS, UNROLL, NUM_SUBSTEPS, T, FUNC> Prims;

  const ssize_t size = args.N;
  const int nranks = comm->nRanks;
  const int buffSize = ring->buffSize / sizeof(T);
  const int sliceSize = buffSize / NUM_BUFCHUNKS;
  const int rank = ring->devUserRanks[0];
  const int prevRank = ring->devUserRanks[nranks-1];
  const int root = args.root;

  if (tid == 0) {
    // Update in case we skipped some collectives
    *ring->recv.conn.opCount = args.opCount;

    if (rank != root) {
      // Wait for next to be ready
      WaitFlag waitOpCountNext(ring->send.conn.opCount, 0);
      waitOpCountNext.wait(args.opCount);
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
    if (prevRank == root) {
      Prims::Copy(
          thisInput + offset,
          nextOutput + boffset,
          sliceSize, maxOffset,
          step,
          waitDoneFromNext,
          postReadyToNext);
    } else if (rank == root) {
      Prims::Reduce(
          prevInput  + boffset,
          thisInput + offset,
          thisOutput + offset,
          sliceSize, maxOffset,
          step,
          waitReadyFromPrev,
          postDoneToPrev);
    } else {
      Prims::Reduce(
          prevInput + boffset,
          thisInput + offset,
          nextOutput + boffset,
          sliceSize, maxOffset,
          step,
          waitDoneFromNext, waitReadyFromPrev,
          postReadyToNext, postDoneToPrev);
    }
    NEXT_STEP; // Increases step, boffset
  }

  if (tid == 0) {
    if (rank != root) { 
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
__global__ void ReduceKernelSmall(const KernelArgs<T> args) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  struct ncclComm* comm = args.comm;
  struct ncclRing* ring = comm->rings+bid;
  volatile uint64_t * recvHeadPtr = ring->recv.conn.llHead;
  volatile uint64_t * sendHeadPtr = ring->send.conn.llHead;
  volatile int * sizesFifo = ring->send.conn.llFifo;
  uint64_t sendHead = sendHeadPtr[0];
  const int nranks = comm->nRanks;
  const int rank = comm->rank;
  const int prevRank = ring->devUserRanks[nranks-1];
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
    if (prevRank == root) {
      WAIT_NEXT;
      LL::ReduceCopy(
          thisInput + offset,
          nextOutput + boffset,
          maxOffset, flag);
      POST_SIZE;
      NEXT_STEP_LL;
    } else if (rank == root) {
      LL::ReduceCopy(
          thisInput + offset,
          prevInput  + boffset,
          thisOutput + offset,
          maxOffset, flag);
      NEXT_STEP_LL;
      ACK_PREV;
    } else {
      WAIT_NEXT;
      LL::ReduceCopy(
          thisInput + offset,
          prevInput + boffset,
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
ncclResult_t RingReduce(const void* sendbuff, void* recvbuff, const size_t count, const int root,
    ncclComm* comm, cudaStream_t stream) {
  if (comm->nRanks == 1) {
    if (sendbuff != recvbuff)
      CUDACHECK(cudaMemcpyAsync(recvbuff, sendbuff, count*sizeof(T), cudaMemcpyDeviceToDevice, stream));
  } else {
    ArgsSetup(sendbuff, recvbuff, root, count, comm);
    if (count*sizeof(T) <= comm->llThreshold) {
      NCCLCHECK(transportSaveProxies(1, NUM_LL_CHUNKS, 1, 1, 2*count*sizeof(T), proxyPatternTo(root), comm, 1, 1));
      SAVE_KERNEL_SMALL(ReduceKernelSmall, comm, FUNC, T, stream);
    } else {
      NCCLCHECK(transportSaveProxies(NUM_SUBSTEPS, NUM_BUFCHUNKS, 1, 1, count*sizeof(T), proxyPatternTo(root), comm, comm->nRings, 0));
      SAVE_KERNEL(ReduceKernel, comm, UNROLL, FUNC, T, stream);
      comm->opCount++;
    }
  }

  return ncclSuccess;
}

template<typename T, template<typename> class RedOp>
class ReduceFunctor {
  public:
  static ncclResult_t entry(const void* sendbuff, void* recvbuff,
      size_t count, int root, ncclComm* comm, cudaStream_t stream) {
    return RingReduce<RedOp<T>, T>(sendbuff, recvbuff, count, root, comm, stream);
  }
};

ncclResult_t ncclReduceFunc(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  return enqueue<ReduceFunctor>(sendbuff, recvbuff, count, datatype, op, root, comm, stream);
}

NCCL_API(ncclResult_t, ncclReduce, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  return ncclEnqueueCheck(ncclReduceFunc, "Reduce", sendbuff, recvbuff, count, datatype,
      op, root, comm, stream);
}

