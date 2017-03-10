/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "group.h"
#include "debug.h"
#include <assert.h>

#define MAX_ASYNC_OPS 128
thread_local pthread_t ncclGroupThreads[MAX_ASYNC_OPS];
thread_local int ncclGroupIndex = 0;
thread_local bool ncclGroupMode = false;

bool ncclAsyncMode() {
  return ncclGroupMode;
}

struct ncclInitArgs {
  // Ret must be the first argument
  ncclInitFunc_t func;
  int cudaDev;
  ncclComm_t* newcomm;
  int ndev;
  ncclUniqueId commId;
  int myrank; 
};
struct ncclCollArgs {
  // Ret must be the first argument
  ncclCollFunc_t func;
  const void* sendbuff;
  void* recvbuff;
  size_t count;
  ncclDataType_t type;
  ncclRedOp_t op;
  int root;
  ncclComm_t comm;
  cudaStream_t stream;
};

enum ncclAsyncMode {
  ASYNC_MODE_UNKNOWN = 0,
  ASYNC_MODE_SEQ = 1,
  ASYNC_MODE_THREAD = 2,
};
enum ncclAsyncFuncType {
  ASYNC_FUNC_INVALID = 0,
  ASYNC_FUNC_INIT = 1,
  ASYNC_FUNC_COLL = 2,
};
struct ncclAsyncArgs {
  ncclResult_t ret;
  enum ncclAsyncFuncType funcType;
  enum ncclAsyncMode mode;
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
    WARN("< ... > [Async thread]"); \
    return args; \
  } \
} while(0)

void* ncclAsyncThreadMain(void* args_) {
  struct ncclAsyncArgs* args = (struct ncclAsyncArgs*)args_;
  if (args->funcType == ASYNC_FUNC_INIT) {
    CHECK(ncclSetDevice(args->init.cudaDev));
    CHECK(args->init.func(args->init.newcomm, args->init.ndev, args->init.commId, args->init.myrank));
  } else { // Coll
    assert(args->funcType == ASYNC_FUNC_COLL);
    CHECK(ncclSetDevice(args->init.cudaDev));
    CHECK(ncclCpuBarrierWait(args->coll.comm));
    CHECK(args->coll.func(args->coll.sendbuff, args->coll.recvbuff, args->coll.count, args->coll.type, args->coll.op, args->coll.root, args->coll.comm, args->coll.stream));
  }
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
  args->mode = ASYNC_MODE_THREAD;
  pthread_create(ncclGroupThreads+index, NULL, ncclAsyncThreadMain, args);
  return ncclSuccess;
}

ncclResult_t ncclAsyncColl(ncclCollFunc_t func, const void* sendbuff, void* recvbuff, size_t count, 
    ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  int index = ncclGroupIndex++;
  struct ncclAsyncArgs* args = ncclGroupArgs+index;
  args->funcType = ASYNC_FUNC_COLL;
  args->coll.func = func;
  args->coll.sendbuff = sendbuff;
  args->coll.recvbuff = recvbuff;
  args->coll.count = count;
  args->coll.type = type;
  args->coll.op = op;
  args->coll.root = root;
  args->coll.comm = comm;
  args->coll.stream = stream;
  static enum ncclAsyncMode mode = ASYNC_MODE_UNKNOWN;
  if (mode == ASYNC_MODE_UNKNOWN) {
    char* str = getenv("NCCL_ASYNC_MODE");
    if (str && (strcmp(str, "THREAD") == 0)) {
      //INFO("Async mode : Thread");
      mode = ASYNC_MODE_THREAD;
    } else {
      //INFO("Async mode : Sequential");
      mode = ASYNC_MODE_SEQ;
    }
  }
  args->mode = mode;
  if (mode == ASYNC_MODE_THREAD) {
    pthread_create(ncclGroupThreads+index, NULL, ncclAsyncThreadMain, args);
  }
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
  for (int i=0; i<ncclGroupIndex; i++) doneArray[i] = 0;
  while (done) {
    for (int i=0; i<ncclGroupIndex; i++) {
      struct ncclAsyncArgs* args = ncclGroupArgs+i;
      if (doneArray[i] == 1) continue;
      if (args->mode == ASYNC_MODE_THREAD) {
        int err = pthread_tryjoin_np(ncclGroupThreads[i], NULL);
        if (err == EBUSY) continue;
        if (err != 0) return ncclSystemError;
      } else { // ASYNC_MODE_SEQ
        // Only for Collectives
        assert(args->funcType == ASYNC_FUNC_COLL);
        CUDACHECK(cudaSetDevice(args->coll.comm->cudaDev));
        NCCLCHECK(ncclCpuBarrierWait(args->coll.comm));
        args->ret = args->coll.func(args->coll.sendbuff, args->coll.recvbuff, args->coll.count, args->coll.type, args->coll.op, args->coll.root, args->coll.comm, args->coll.stream);
      }
      if (args->ret != ncclSuccess) return args->ret;
      doneArray[i] = 1;
      done--;
    }
  }
  ncclGroupIndex = 0;
  ncclGroupMode = false;
  return ncclSuccess;
}


