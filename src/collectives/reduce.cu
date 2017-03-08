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

  WaitFlag waitDoneFromNext(ring->send.conn.head, (1-NUM_BUFCHUNKS)*NUM_SUBSTEPS);
  WaitFlag waitReadyFromPrev(ring->recv.conn.tail, 0);
  PostFlag postDoneToPrev(ring->recv.conn.head, 0, NULL, 0);
  PostFlag postReadyToNext(ring->send.conn.tail, 0, ring->send.conn.fifo, NUM_BUFCHUNKS*NUM_SUBSTEPS);

  typedef Primitives<THREADS, UNROLL, NUM_SUBSTEPS, T, FUNC> Prims;

  const int size = args.N;
  const int nranks = comm->nRanks;
  const int buffSize = ring->buffSize / sizeof(T);
  const int sliceSize = buffSize / NUM_BUFCHUNKS;
  const int rank = ring->devUserRanks[0];
  const int prevRank = ring->devUserRanks[nranks-1];
  const int root = args.root;

  if (rank != root && tid == 0) {
    // Wait for next to be ready
    WaitFlag waitOpCountNext(ring->send.conn.opCount, 0);
    waitOpCountNext.wait(args.opCount);
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
    ALIGN_SIZE(chunkSize, THREADS*UNROLL*sizeof(uint64_t)/sizeof(T));
    int offset = gridOffset + bid*chunkSize;
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
      *ring->send.conn.head = 0;
    }
    *ring->recv.conn.tail = 0;
    __threadfence_system();
    *ring->recv.conn.opCount = args.opCount+1;
  }
}

#define UNROLL 8

template<class FUNC, typename T>
ncclResult_t RingReduce(const void* sendbuff, void* recvbuff, const size_t count, const int root,
    ncclComm* comm, cudaStream_t stream) {
  if (comm->nRanks == 1) {
    if (sendbuff != recvbuff)
      CUDACHECK(cudaMemcpyAsync(recvbuff, sendbuff, count*sizeof(T), cudaMemcpyDeviceToDevice, stream));
  } else {
    NCCLCHECK(transportStartProxies(NUM_SUBSTEPS, NUM_BUFCHUNKS, 1, 1, count*sizeof(T), proxyPatternTo(root), comm));
    KernelArgs<T> args;
    ArgsSetup(&args, sendbuff, recvbuff, root, count, comm);
    LAUNCH_KERNEL(ReduceKernel, comm->nThreads, UNROLL, FUNC, T, args, stream);
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
  return enqueueCheck(ncclReduceFunc, "Reduce", sendbuff, recvbuff, count, datatype,
      op, root, comm, stream);
}

