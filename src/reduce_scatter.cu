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

  if (tid == 0) {
    // Wait for prev and next to be ready
    Wait([=] {
        return *ring->recv.conn.head == 0;
    });
    Wait([=] {
        return *ring->send.conn.tail == 0;
    });
  }
  __syncthreads();

  WaitFlag waitDoneFromNext(ring->send.conn.head, -NUM_BUFCHUNKS*NUM_SUBSTEPS);
  WaitFlag waitReadyFromPrev(ring->recv.conn.tail, -1*NUM_SUBSTEPS);
  PostFlag postDoneToPrev(ring->recv.conn.head, -1*NUM_SUBSTEPS);
  PostFlag postReadyToNext(ring->send.conn.tail, 0);

  typedef Primitives<THREADS, UNROLL, NUM_SUBSTEPS, T, FUNC> Prims;

  const int size = args.N;
  const int nranks = comm->nRanks;
  const int buffSize = ring->buffSize / sizeof(T);
  const int sliceSize = buffSize / NUM_BUFCHUNKS;
  
  int step = 0;
  int poffset, noffset = 0;

  // Compute pointers
  const T * __restrict__ thisInput = args.ThisInput;
  T * __restrict__ thisOutput = args.ThisOutput;
  T * __restrict__ prevInput = (T*)ring->recv.conn.buff;
  T * __restrict__ nextOutput = (T*)ring->send.conn.buff;

  for (int chunkOffset = bid*sliceSize; chunkOffset < size; chunkOffset += gridDim.x*sliceSize) {
    /////////////// begin ReduceScatter steps ///////////////
    int offset;
    int maxOffset = size-chunkOffset;
    int rankDest;

    // step 0: push data to next GPU
    rankDest = ring->userRanks[nranks-1];
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
      rankDest = ring->userRanks[nranks-j];
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

    // step k-1: reduce this buffer and data, which will produce the final
    // result that we store in this data and push to the next GPU
    rankDest = ring->userRanks[0];
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
    *ring->send.conn.head = 0;

    // Wait for last update from prev then reset the flag
    waitReadyFromPrev.wait(NUM_SUBSTEPS*(step+1));
    *ring->recv.conn.tail = 0;
  }
}

#define PCIE_THREADS 512
#define NVLINK_THREADS 128
#define UNROLL 8

template<class FUNC, typename T>
ncclResult_t RingReduceScatter(const void* sendbuff, void* recvbuff,
    const int count, ncclComm* comm, cudaStream_t stream) {
  if (count == 0)
    return ncclSuccess;

  if (comm->nRanks == 1) {
    if (sendbuff != recvbuff)
      CUDACHECK(cudaMemcpyAsync(recvbuff, sendbuff, count*sizeof(T), cudaMemcpyDeviceToDevice, stream));
  } else {
    KernelArgs<T> args;
    ArgsSetup(&args, sendbuff, recvbuff, 0, count, comm);
    if (comm->p2ptype == ncclComm::NVLINK) {
      LAUNCH_KERNEL(ReduceScatterKernel, NVLINK_THREADS, UNROLL, FUNC, T, args, stream);
    } else {
      LAUNCH_KERNEL(ReduceScatterKernel, PCIE_THREADS, UNROLL, FUNC, T, args, stream);
    }
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

