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
__global__ void ReduceScatterKernel(const KernelArgs<T> args) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  struct ncclComm* comm = args.comm;
  struct ncclRing* ring = comm->rings+bid;

  WaitFlag waitDoneFromNext(ring->send.conn.head, -NUM_BUFCHUNKS*NUM_SUBSTEPS);
  WaitFlag waitReadyFromPrev(ring->recv.conn.tail, -1*NUM_SUBSTEPS);
  PostFlag postDoneToPrev(ring->recv.conn.head, -1*NUM_SUBSTEPS, NULL, 0);
  PostFlag postReadyToNext(ring->send.conn.tail, 0, ring->send.conn.fifo, NUM_BUFCHUNKS*NUM_SUBSTEPS);

  typedef Primitives<THREADS, UNROLL, NUM_SUBSTEPS, T, FUNC> Prims;

  const int size = args.N;
  const int nranks = comm->nRanks;
  const int buffSize = ring->buffSize / sizeof(T);
  const int sliceSize = buffSize / NUM_BUFCHUNKS;

  if (tid == 0) {
    // Wait for next to be ready
    WaitFlag waitOpCountNext(ring->send.conn.opCount, 0);
    waitOpCountNext.wait(args.opCount);
  }
  __syncthreads();
  
  int step = 0;
  int poffset, noffset = 0;

  // Compute pointers
  const T * __restrict__ thisInput = args.ThisInput;
  T * __restrict__ thisOutput = args.ThisOutput;
  T * __restrict__ prevInput = (T*)ring->recv.conn.buff;
  T * __restrict__ nextOutput = (T*)ring->send.conn.buff;

  for (int gridOffset = 0; gridOffset < size; gridOffset += gridDim.x*sliceSize) {
    int chunkSize = min(sliceSize, DIVUP(size-gridOffset,gridDim.x));
    ALIGN_SIZE(chunkSize, THREADS*sizeof(uint64_t)/sizeof(T));
    int chunkOffset = gridOffset + bid*chunkSize;

    /////////////// begin ReduceScatter steps ///////////////
    int offset;
    int maxOffset;
    int rankDest;

    // step 0: push data to next GPU
    rankDest = ring->devUserRanks[nranks-1];
    offset = chunkOffset + rankDest * size;
    maxOffset = min(chunkSize, size-chunkOffset);

    Prims::Copy(
        thisInput  + offset,
        nextOutput + noffset,
        sliceSize, maxOffset,
        step,
        waitDoneFromNext,
        postReadyToNext);

    NEXT_STEP; // Increases step, poffset, noffset

    // k-2 steps: reduce and copy to next GPU
    for (int j=2; j<nranks; ++j) {
      rankDest = ring->devUserRanks[nranks-j];
      offset = chunkOffset + rankDest * size;
      maxOffset = min(chunkSize, size-chunkOffset);

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

    // step k-1: reduce this buffer and data, which will produce the final
    // result that we store in this data and push to the next GPU
    rankDest = ring->devUserRanks[0];
    offset = chunkOffset + rankDest * size;
    maxOffset = min(chunkSize, size-chunkOffset);

    Prims::Reduce(
        prevInput  + poffset,
        thisInput  + offset,
        thisOutput + chunkOffset,
        sliceSize, maxOffset,
        step,
        waitReadyFromPrev,
        postDoneToPrev);
  }

  if (tid == 0) {
    waitDoneFromNext.wait(NUM_SUBSTEPS*(step + NUM_BUFCHUNKS));
    *ring->send.conn.head = 0;
    *ring->recv.conn.tail = 0;
    __threadfence_system();
    *ring->recv.conn.opCount = args.opCount+1;
  }
}

#define UNROLL 8

template<class FUNC, typename T>
ncclResult_t RingReduceScatter(const void* sendbuff, void* recvbuff,
    const size_t count, ncclComm* comm, cudaStream_t stream) {
  if (comm->nRanks == 1) {
    if (sendbuff != recvbuff)
      CUDACHECK(cudaMemcpyAsync(recvbuff, sendbuff, count*sizeof(T), cudaMemcpyDeviceToDevice, stream));
  } else {
    NCCLCHECK(transportStartProxies(NUM_SUBSTEPS, NUM_BUFCHUNKS, comm->nRanks-1, 1, count*sizeof(T), proxyPatternRing, comm));
    KernelArgs<T> args;
    ArgsSetup(&args, sendbuff, recvbuff, 0, count, comm);
    LAUNCH_KERNEL(ReduceScatterKernel, comm->nThreads, UNROLL, FUNC, T, args, stream);
  }

  return ncclSuccess;
}

template<typename T, template <typename> class RedOp>
class ReduceScatter {
  public:
  static ncclResult_t entry(const void* sendbuff, void* recvbuff,
      size_t count, int /*root*/, ncclComm* comm, cudaStream_t stream) {
    return RingReduceScatter<RedOp<T>, T>(sendbuff, recvbuff, count, comm, stream);
  }
};

ncclResult_t ncclReduceScatterFunc(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  return enqueue<ReduceScatter>(sendbuff, recvbuff, count, datatype, op, root, comm, stream);
}

NCCL_API(ncclResult_t, ncclReduceScatter, const void* sendbuff, void* recvbuff, size_t recvcount,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  return ncclEnqueueCheck(ncclReduceScatterFunc, "ReduceScatter", sendbuff, recvbuff, recvcount, datatype,
      op, 0, comm, stream);
}

