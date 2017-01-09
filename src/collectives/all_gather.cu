/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
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

  WaitFlag waitDoneFromNext(ring->send.conn.head, -NUM_BUFCHUNKS*NUM_SUBSTEPS);
  WaitFlag waitReadyFromPrev(ring->recv.conn.tail, -1*NUM_SUBSTEPS);
  PostFlag postDoneToPrev(ring->recv.conn.head, -1*NUM_SUBSTEPS, NULL, 0);
  PostFlag postReadyToNext(ring->send.conn.tail, 0, NULL, 0);

  typedef Primitives<THREADS, UNROLL, NUM_SUBSTEPS, T> Prims;

  const int size = args.N;
  const int nranks = comm->nRanks;
  const int buffSize = ring->buffSize / sizeof(T);
  const int sliceSize = buffSize / NUM_BUFCHUNKS;

  if (tid == 0) {
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
  
  int step = 0;
  int poffset, noffset = 0;

  // Compute pointers
  const T * __restrict__ thisInput = args.ThisInput;
  T * __restrict__ thisOutput = args.ThisOutput;
  T * __restrict__ prevInput = (T*)ring->recv.conn.buff;
  T * __restrict__ nextOutput = (T*)ring->send.conn.buff;

  for (int chunkOffset = bid*sliceSize; chunkOffset < size; chunkOffset += gridDim.x*sliceSize) {
    /////////////// begin AllGather steps ///////////////
    int offset;
    int maxOffset = size-chunkOffset;
    int rankDest;

    // step 0: push data to next GPU
    rankDest = ring->userRanks[0];
    offset = chunkOffset + rankDest * size;

    if (thisInput == thisOutput) {
      Prims::Copy(
          thisInput  + offset,
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
        rankDest = ring->userRanks[nranks-j];
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
        rankDest = ring->userRanks[nranks-j];
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
      rankDest = ring->userRanks[1];
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
    *ring->send.conn.head = 0;
    *ring->recv.conn.tail = 0;
    __threadfence_system();
    *ring->recv.conn.opCount = args.opCount+1;
  }
}

#define PCIE_THREADS 512
#define NVLINK_THREADS 128
#define UNROLL 8

template<class FUNC, typename T>
ncclResult_t RingAllGather(const void* sendbuff, void* recvbuff,
    const int count, ncclComm* comm, cudaStream_t stream) {
  if (count == 0)
    return ncclSuccess;

  if (comm->nRanks == 1) {
    if (sendbuff != recvbuff)
      CUDACHECK(cudaMemcpyAsync(recvbuff, sendbuff, count*sizeof(T), cudaMemcpyDeviceToDevice, stream));
  } else {
    NCCLCHECK(transportStartProxies(NUM_SUBSTEPS, NUM_BUFCHUNKS, comm->nRanks-1, 1, count*sizeof(T), count*sizeof(T), proxyPatternRing, comm));
    KernelArgs<T> args;
    ArgsSetup(&args, sendbuff, recvbuff, 0, count, comm);
    if (comm->nRings > 1) {
      LAUNCH_KERNEL(AllGatherKernel, NVLINK_THREADS, UNROLL, FUNC, T, args, stream);
    } else {
      LAUNCH_KERNEL(AllGatherKernel, PCIE_THREADS, UNROLL, FUNC, T, args, stream);
    }
  }

  return ncclSuccess;
}

template<typename T, template<typename> class RedOp>
class AllGather {
  public:
  static ncclResult_t entry(const void* sendbuff, void* recvbuff,
      int count, int /*root*/, ncclComm* comm, cudaStream_t stream) {
    return RingAllGather<RedOp<T>, T>(sendbuff, recvbuff, count, comm, stream);
  }
};

NCCL_API(ncclResult_t, ncclAllGather, const void* sendbuff, int count, ncclDataType_t datatype,
    void* recvbuff, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclAllGather(const void* sendbuff, int count, ncclDataType_t datatype,
    void* recvbuff, ncclComm_t comm, cudaStream_t stream) {
  return enqueue<AllGather, FuncNull>(sendbuff, recvbuff, count, datatype, 0, comm, stream);
}

