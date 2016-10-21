/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "enqueue.h"
#include "primitives.h"

#define NUM_SUBSTEPS 2
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

  if (tid == 0) {
    // Wait for prev and next to be ready
    Wait([=] {
        return *ring->recv.conn.head == 0;
    });
    Wait([=] {
        return *ring->send.conn.tail == 0;
    });
    
    if (prevdirect) {
      *ring->recv.conn.ptrExchange = (T*)args.ThisOutput;
    }
    if (nextdirect) {
      Wait([=] {
        return *(ring->send.conn.ptrExchange) != nullptr;
      });
      sharedNextOutput = (T*)*ring->send.conn.ptrExchange;
      *ring->send.conn.ptrExchange = nullptr;
    }
  }
  __syncthreads();

  WaitFlag waitDoneFromNext(ring->send.conn.head, (1-NUM_BUFCHUNKS)*NUM_SUBSTEPS);
  WaitFlag waitReadyFromPrev(ring->recv.conn.tail, 0);
  PostFlag postDoneToPrev(ring->recv.conn.head, 0);
  PostFlag postReadyToNext(ring->send.conn.tail, 0);

  typedef Primitives<THREADS, UNROLL, NUM_SUBSTEPS, T> Prims;

  const int size = args.N;
  const int buffSize = ring->buffSize / sizeof(T);
  const int sliceSize = buffSize / NUM_BUFCHUNKS;
  const int rank = ring->userRanks[0];
  const int nextRank = ring->userRanks[1];
  const int root = args.root;
  
  int step = 0;
  int boffset = 0;

  // Compute pointers
  const T * __restrict__ thisInput = args.ThisInput;
  T * __restrict__ thisOutput = args.ThisOutput;
  T * __restrict__ prevInput = (T*)ring->recv.conn.buff;
  T * __restrict__ nextOutput = (T*)ring->send.conn.buff;

  for (int offset = bid*sliceSize; offset < size; offset += gridDim.x*sliceSize) {
    int maxOffset = size-offset;
    if (rank == root) {
      Prims::Copy(
          thisInput + offset,
	  nextdirect ? (sharedNextOutput + offset) : (nextOutput + boffset),
          sliceSize, maxOffset,
          step,
          waitDoneFromNext,
          postReadyToNext);
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

  // wait for the last data to be pushed to us
  if (tid == 0) {
    if (nextRank != root) {
      // Wait for last update from next then reset the flag
      waitDoneFromNext.wait(NUM_SUBSTEPS*(step+NUM_BUFCHUNKS-1));
      *ring->send.conn.head = 0;
    }

    if (rank != root) {
      // reset the flag
      *ring->recv.conn.tail = 0;
    }
  }
}

#define PCIE_THREADS 256
#define NVLINK_THREADS 128
#define UNROLL 8

template<class FUNC, typename T>
ncclResult_t RingBroadcast(void* buff, const int count, const int root,
    ncclComm* comm, cudaStream_t stream) {
  if (count == 0)
    return ncclSuccess;

  if (comm->nRanks != 1) {
    KernelArgs<T> args;
    ArgsSetup(&args, buff, buff, root, count, comm);
    if (comm->p2ptype == ncclComm::NVLINK) {
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
    return RingBroadcast<RedOp<T>, T>(recvbuff, count, root, comm, stream);
  }
};

NCCL_API(ncclResult_t, ncclBcast, void* buff, int count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclBcast(void* buff, int count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream) {
  return enqueue<Broadcast, FuncNull>(nullptr, buff, count, datatype, root, comm, stream);
}

