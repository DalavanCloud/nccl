/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_ENQUEUE_H_
#define NCCL_ENQUEUE_H_

#include "core.h"
#include "reduce_kernel.h"
#include "crc32.h"
#include "group.h"

/* Syncronize previous collective (if in different stream) and enqueue
 * collective. Work is performed asynchronously with the host thread.
 * The ColFunc class should be templated on the datatype and reduction
 * operator (if applicable) and define a static entry() method as
 * follows.
 *   template <typename T, template <typename> class RedOp>
 *   class CollectiveFunctor {
 *     public:
 *     static ncclResult_t entry(const void* sendbuff, void* recvbuff, size_t count,
 *         int root, ncclComm* comm, cudaStream_t stream);
 *   };
 * The entry() method can assume that the appropriate cuda device has been set. */
template< template<typename, template<typename> class> class ColFunc,
          typename T,
          template<typename> class Op >
ncclResult_t enqueue(const void* sendbuff,
                     void* recvbuff,
                     size_t count,
                     int root,
                     ncclComm_t comm,
                     cudaStream_t stream)
{
  if (stream != comm->prevStream) { // sync required for calls in different streams
    comm->prevStream = stream;
    CUDACHECK(cudaStreamWaitEvent(stream, comm->doneEvent, 0));
  }

  // print CRC checksum of input
  if (ncclPrintCRCs) {
    printCRCDev((unsigned char*)sendbuff, count*sizeof(T), comm->rank, stream);
  }

  ncclResult_t ret;
  ret = ColFunc<T, Op>::entry(sendbuff, recvbuff, count, root, comm, stream);

  // print CRC checksum of output
  if (ncclPrintCRCs) {
    printCRCDev((unsigned char*)recvbuff, count*sizeof(T), comm->rank, stream);
  }
  
  // Always have to record done event because we don't know what stream next
  // collective will be in.
  CUDACHECK(cudaEventRecord(comm->doneEvent, stream));
  return ret;
}


// This version decodes type
template< template<typename, template<typename> class> class ColFunc,
          template<typename> class Op >
ncclResult_t enqueue(const void* sendbuff,
                     void* recvbuff,
                     size_t count,
                     ncclDataType_t type,
                     int root,
                     ncclComm_t comm,
                     cudaStream_t stream)
{
  switch(type) {
  case ncclInt8:
    return enqueue<ColFunc, int8_t, Op>(sendbuff, recvbuff, count, root, comm, stream);
  case ncclUint8:
    return enqueue<ColFunc, uint8_t, Op>(sendbuff, recvbuff, count, root, comm, stream);
  case ncclInt32:
    return enqueue<ColFunc, int32_t, Op>(sendbuff, recvbuff, count, root, comm, stream);
  case ncclUint32:
    return enqueue<ColFunc, uint32_t, Op>(sendbuff, recvbuff, count, root, comm, stream);
  case ncclFloat16:
    return enqueue<ColFunc, half, Op>(sendbuff, recvbuff, count, root, comm, stream);
  case ncclFloat32:
    return enqueue<ColFunc, float, Op>(sendbuff, recvbuff, count, root, comm, stream);
  case ncclFloat64:
    return enqueue<ColFunc, double, Op>(sendbuff, recvbuff, count, root, comm, stream);
  case ncclInt64:
    return enqueue<ColFunc, int64_t, Op>(sendbuff, recvbuff, count, root, comm, stream);
  case ncclUint64:
    return enqueue<ColFunc, uint64_t, Op>(sendbuff, recvbuff, count, root, comm, stream);
  default:
    WARN("Invalid ncclType %d", type);
    return ncclInvalidArgument;
  }
}

// This version decodes both type and reduction op
template< template<typename, template<typename> class> class ColFunc>
ncclResult_t enqueue(const void* sendbuff,
                     void* recvbuff,
                     size_t count,
                     ncclDataType_t type,
                     ncclRedOp_t op,
                     int root,
                     ncclComm_t comm,
                     cudaStream_t stream)
{
  switch(op) {
  case ncclSum:
    return enqueue<ColFunc, FuncSum>(sendbuff, recvbuff, count, type, root, comm, stream);
  case ncclProd:
    return enqueue<ColFunc, FuncProd>(sendbuff, recvbuff, count, type, root, comm, stream);
  case ncclMax:
    return enqueue<ColFunc, FuncMax>(sendbuff, recvbuff, count, type, root, comm, stream);
  case ncclMin:
    return enqueue<ColFunc, FuncMin>(sendbuff, recvbuff, count, type, root, comm, stream);
  default:
    WARN("Invalid ncclRedOp: %d", op);
    return ncclInvalidArgument;
  }
}

typedef ncclResult_t(*ncclFunc_t)(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream);

struct asyncThreadArgs {
  // Ret must be the first argument
  ncclResult_t ret;
  ncclFunc_t func;
  const void* sendbuff;
  void* recvbuff;
  size_t count;
  ncclDataType_t type;
  ncclRedOp_t op;
  int root;
  ncclComm_t comm;
  cudaStream_t stream;
};

ncclResult_t ncclEnqueueCheck(ncclFunc_t func, const char* primName, const void* sendbuff, 
    void* recvbuff, size_t count, ncclDataType_t type, ncclRedOp_t op, int root,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclCpuBarrierCheckin(ncclComm_t comm);
ncclResult_t ncclCpuBarrierWait(ncclComm_t comm);

#endif // End include guard

