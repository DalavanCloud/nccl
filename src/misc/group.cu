/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "group.h"
#include "debug.h"
#include "enqueue.h"
#include <assert.h>

#define MAX_ASYNC_OPS 128
thread_local pthread_t ncclGroupThreads[MAX_ASYNC_OPS];
thread_local int ncclGroupIndex = 0;
thread_local bool ncclGroupMode = false;
thread_local ncclResult_t ncclGroupError = ncclSuccess;

bool ncclAsyncMode() {
  return ncclGroupMode;
}

ncclResult_t ncclAsyncErrCheck(ncclResult_t ret) {
  if (ncclGroupError == ncclSuccess) ncclGroupError = ret;
  return ret;
}

struct ncclInitArgs {
  ncclInitFunc_t func;
  int cudaDev;
  ncclComm_t* newcomm;
  int ndev;
  ncclUniqueId commId;
  int myrank; 
};
struct ncclCollArgs {
  ncclComm_t comm;
};

enum ncclAsyncFuncType {
  ASYNC_FUNC_INVALID = 0,
  ASYNC_FUNC_INIT = 1,
  ASYNC_FUNC_COLL = 2,
};
struct ncclAsyncArgs {
  ncclResult_t ret;
  enum ncclAsyncFuncType funcType;
  union {
    ncclCollArgs coll;
    ncclInitArgs init;
  };
};

thread_local struct ncclAsyncArgs ncclGroupArgs[MAX_ASYNC_OPS];

ncclResult_t ncclSetDevice(int cudaDev) {
  CUDACHECK(cudaSetDevice(cudaDev));
  return ncclSuccess;
}

#define CHECK(a) do { \
  if ((args->ret = (a)) != ncclSuccess) { \
    INFO("%s:%d -> %d [Async thread]", __FILE__, __LINE__, args->ret); \
    return args; \
  } \
} while(0)

void* ncclAsyncThreadMain(void* args_) {
  struct ncclAsyncArgs* args = (struct ncclAsyncArgs*)args_;
  CHECK(ncclSetDevice(args->init.cudaDev));
  CHECK(args->init.func(args->init.newcomm, args->init.ndev, args->init.commId, args->init.myrank));
  return args;
}

ncclResult_t ncclAsyncInit(ncclInitFunc_t func, int cudaDev, ncclComm_t* newcomm, int ndev, ncclUniqueId commId, int myrank) {
  if (ncclGroupIndex == MAX_ASYNC_OPS) {
    WARN("Too many async operations in progress, max is %d", MAX_ASYNC_OPS);
    return ncclInternalError;
  }
  int index = ncclGroupIndex++;
  struct ncclAsyncArgs* args = ncclGroupArgs+index;
  args->funcType = ASYNC_FUNC_INIT;
  args->init.func = func;
  args->init.cudaDev = cudaDev;
  args->init.newcomm = newcomm;
  args->init.ndev = ndev;
  memcpy(&args->init.commId, &commId, sizeof(commId));
  args->init.myrank = myrank;
  // We need to use threads for Init
  pthread_create(ncclGroupThreads+index, NULL, ncclAsyncThreadMain, args);
  return ncclSuccess;
}

ncclResult_t ncclAsyncColl(ncclComm_t comm) {
  int index = ncclGroupIndex++;
  struct ncclAsyncArgs* args = ncclGroupArgs+index;
  args->funcType = ASYNC_FUNC_COLL;
  args->coll.comm = comm;
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclGroupStart);
ncclResult_t ncclGroupStart() {
  ncclGroupMode = true;
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclGroupEnd);
ncclResult_t ncclGroupEnd() {
  int savedDev;
  CUDACHECK(cudaGetDevice(&savedDev));
  int done = ncclGroupIndex;
  int doneArray[ncclGroupIndex];

  ncclResult_t ret = ncclGroupError;
  if (ret != ncclSuccess) goto end;

  for (int i=0; i<ncclGroupIndex; i++) {
    struct ncclAsyncArgs* args = ncclGroupArgs+i;
    if (args->funcType == ASYNC_FUNC_COLL) {
      NCCLCHECK(ncclCpuBarrierCheckin(args->coll.comm));
    }
  }

  for (int i=0; i<ncclGroupIndex; i++) doneArray[i] = 0;
  while (done) {
    for (int i=0; i<ncclGroupIndex; i++) {
      struct ncclAsyncArgs* args = ncclGroupArgs+i;
      if (doneArray[i] == 1) continue;
      if (args->funcType == ASYNC_FUNC_INIT) {
        int err = pthread_tryjoin_np(ncclGroupThreads[i], NULL);
        if (err == EBUSY) continue;
        if (err != 0) return ncclSystemError;
      } else { // ASYNC_FUNC_COLL
        CUDACHECK(cudaSetDevice(args->coll.comm->cudaDev));
        NCCLCHECK(ncclCpuBarrierWait(args->coll.comm));
      }
      if (args->ret != ncclSuccess) { ret = args->ret; goto end; }
      doneArray[i] = 1;
      done--;
    }
  }
end:
  CUDACHECK(cudaSetDevice(savedDev));
  ncclGroupError = ncclSuccess;
  ncclGroupIndex = 0;
  ncclGroupMode = false;
  return ret;
}
