/*************************************************************************
 * Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 ************************************************************************/

#ifndef enqueue_h_
#define enqueue_h_

#include "core.h"
#include "reduce_kernel.h"
#include "crc32.h"

/* Syncronize previous collective (if in different stream) and enqueue
 * collective. Work is performed asynchronously with the host thread.
 * The ColFunc class should be templated on the datatype and reduction
 * operator (if applicable) and define a static entry() method as
 * follows.
 *   template <typename T, template <typename> class RedOp>
 *   class CollectiveFunctor {
 *     public:
 *     static ncclResult_t entry(const void* sendbuff, void* recvbuff, int count,
 *         int root, ncclComm* comm, cudaStream_t stream);
 *   };
 * The entry() method can assume that the appropriate cuda device has been set. */
template< template<typename, template<typename> class> class ColFunc,
          typename T,
          template<typename> class Op >
ncclResult_t enqueue(const void* sendbuff,
                     void* recvbuff,
                     int count,
                     int root,
                     ncclComm_t comm,
                     cudaStream_t stream)
{
  if (stream != comm->prevStream) { // sync required for calls in different streams
    comm->prevStream = stream;
    CUDACHECK( cudaStreamWaitEvent(stream, comm->doneEvent, 0) );
  }

  // print CRC checksum of input
  if (ncclPrintCRCs) {
    printCRCDev((unsigned char*)sendbuff, count*sizeof(T), comm->userFromRing[0][0], stream);
  }

  ncclResult_t ret;
  ret = ColFunc<T, Op>::entry(sendbuff, recvbuff, count, root, comm, stream);

  // print CRC checksum of output
  if (ncclPrintCRCs) {
    printCRCDev((unsigned char*)recvbuff, count*sizeof(T), comm->userFromRing[0][0], stream);
  }
  
  // Always have to record done event because we don't know what stream next
  // collective will be in.
  CUDACHECK( cudaEventRecord(comm->doneEvent, stream) );
  comm->opSched += 1;
  return ret;
}


// This version decodes type
template< template<typename, template<typename> class> class ColFunc,
          template<typename> class Op >
ncclResult_t enqueue(const void* sendbuff,
                     void* recvbuff,
                     int count,
                     ncclDataType_t type,
                     int root,
                     ncclComm_t comm,
                     cudaStream_t stream)
{
  switch(type) {
  case ncclChar:
    return enqueue<ColFunc, char, Op>(sendbuff, recvbuff, count, root, comm, stream);
  case ncclInt:
    return enqueue<ColFunc, int, Op>(sendbuff, recvbuff, count, root, comm, stream);
#ifdef CUDA_HAS_HALF
  case ncclHalf:
    return enqueue<ColFunc, half, Op>(sendbuff, recvbuff, count, root, comm, stream);
#endif
  case ncclFloat:
    return enqueue<ColFunc, float, Op>(sendbuff, recvbuff, count, root, comm, stream);
  case ncclDouble:
    return enqueue<ColFunc, double, Op>(sendbuff, recvbuff, count, root, comm, stream);
  case ncclInt64:
    return enqueue<ColFunc, long long, Op>(sendbuff, recvbuff, count, root, comm, stream);
  case ncclUint64:
    return enqueue<ColFunc, unsigned long long, Op>(sendbuff, recvbuff, count, root, comm, stream);
  default:
    WARN("Invalid ncclType %d", type);
    return ncclInvalidType;
  }
}

// This version decodes both type and reduction op
template< template<typename, template<typename> class> class ColFunc>
ncclResult_t enqueue(const void* sendbuff,
                     void* recvbuff,
                     int count,
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
    return ncclInvalidOperation;
  }
}

#endif // End include guard

