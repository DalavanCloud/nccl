/*************************************************************************
 * Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENCE.txt for license information
 ************************************************************************/

#ifndef enqueue_h_
#define enqueue_h_

#include "core.h"

/* Syncronize previous collective (if in different stream) and enqueue
 * collective. Work is performed asynchronously with the host thread.
 * The actual collective should be a functor with the
 * following signature.
 * ncclResult_t collective(void* sendbuff, void* recvbuff,
 *                         int count, ncclDataType_t type, ncclRedOp_t op,
 *                         int root, ncclComm_t comm);
 * The collective can assume that the appropriate cuda device has been set. */
template<typename ColFunc>
ncclResult_t enqueue(ColFunc colfunc,
                     const void* sendbuff,
                     void* recvbuff,
                     int count,
                     ncclDataType_t type,
                     ncclRedOp_t op,
                     int root,
                     ncclComm_t comm,
                     cudaStream_t stream)
{
  // No need for a mutex here because we assume that all enqueue operations happen in a fixed
  // order on all devices. Thus, thread race conditions SHOULD be impossible.

  if (stream != comm->prevStream) { // sync required for calls in different streams
    comm->prevStream = stream;
    CUDACHECK( cudaStreamWaitEvent(stream, comm->doneEvent, 0) );
  }

  // Launch the collective here
  ncclResult_t ret = colfunc(sendbuff, recvbuff, count, type, op, root, comm, stream);

  // Always have to record done event because we don't know what stream next
  // collective will be in.
  CUDACHECK( cudaEventRecord(comm->doneEvent, stream) );
  comm->opSched += 1;
  return ret;
}

#endif // End include guard

