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

  WaitFlag waitDoneFromNext(ring->send.conn.head, (1-NUM_BUFCHUNKS)*NUM_SUBSTEPS);
  WaitFlag waitReadyFromPrev(ring->recv.conn.tail, 0);
  PostFlag postDoneToPrev(ring->recv.conn.head, 0, NULL, 0);
  PostFlag postReadyToNext(ring->send.conn.tail, 0, NULL, 0);

  typedef Primitives<THREADS, UNROLL, NUM_SUBSTEPS, T> Prims;

  const int size = args.N;
  const int buffSize = ring->buffSize / sizeof(T);
  const int sliceSize = buffSize / NUM_BUFCHUNKS;
  const int rank = ring->userRanks[0];
  const int nextRank = ring->userRanks[1];
  const int root = args.root;

  if (tid == 0) {
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
  
  int step = 0;
  int boffset = 0;

  // Compute pointers
  const T * __restrict__ thisInput = args.ThisInput;
  T * __restrict__ thisOutput = args.ThisOutput;
  T * __restrict__ prevInput = (T*)ring->recv.conn.buff;
  T * __restrict__ nextOutput = (T*)ring->send.conn.buff;

  for (int gridOffset = 0; gridOffset < size; gridOffset += gridDim.x*sliceSize) {
    int chunkSize = min(sliceSize, DIVUP(size-gridOffset,gridDim.x));
    ALIGN_SIZE(chunkSize, THREADS*UNROLL);
    int offset = gridOffset + bid*chunkSize;
    int maxOffset = size-offset;

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
      *ring->send.conn.head = 0;
    }
    *ring->recv.conn.tail = 0;
    __threadfence_system();
    *ring->recv.conn.opCount = args.opCount+1;
  }
}

#define PCIE_THREADS 256
#define NVLINK_THREADS 128
#define UNROLL 8

template<class FUNC, typename T>
ncclResult_t RingBroadcast(const void* sendbuff, void* recvbuff, const int count, const int root,
    ncclComm* comm, cudaStream_t stream) {
  if (comm->nRanks == 1) {
    if (sendbuff != recvbuff)
      CUDACHECK(cudaMemcpyAsync(recvbuff, sendbuff, count*sizeof(T), cudaMemcpyDeviceToDevice, stream));
  } else {
    NCCLCHECK(transportStartProxies(NUM_SUBSTEPS, NUM_BUFCHUNKS, 1, 1, count*sizeof(T), count*sizeof(T), proxyPatternFrom(root), comm));
    KernelArgs<T> args;
    ArgsSetup(&args, sendbuff, recvbuff, root, count, comm);
    if (comm->nRings > 1) {
      LAUNCH_KERNEL(BroadcastKernel, NVLINK_THREADS, UNROLL, FUNC, T, args, stream);
    } else {
      LAUNCH_KERNEL(BroadcastKernel, PCIE_THREADS, UNROLL, FUNC, T, args, stream);
    }
  }

  return ncclSuccess;
}

template<typename T, template<typename> class RedOp>
class Broadcast {
  public:
  static ncclResult_t entry(const void* sendbuff, void* recvbuff,
      int count, int root, ncclComm* comm, cudaStream_t stream) {
    return RingBroadcast<RedOp<T>, T>(sendbuff, recvbuff, count, root, comm, stream);
  }
};

NCCL_API(ncclResult_t, ncclBcast, void* buff, int count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclBcast(void* buff, int count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream) {
  NCCLCHECK(ArgsCheck(buff, buff, count, datatype, ncclSum, root, comm, "Bcast"));
  return enqueue<Broadcast, FuncNull>(buff, buff, count, datatype, root, comm, stream);
}

