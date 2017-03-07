/*************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nccl.h"
#include "core.h"
#include "utils.h"
#include "bootstrap.h"
#include "net.h"
#include <unistd.h>
#include <sys/types.h>

struct extId {
  ncclNetHandle_t extHandle;
  void* extListenComm;
  uint64_t hostHash;
  pid_t pid;
  int* lock;
  int fd;
};

ncclResult_t bootstrapGetUniqueId(ncclUniqueId* out) {
  static_assert(sizeof(extId) < sizeof(ncclUniqueId), "NetId does not fit inside ncclUniqueId");
  extId* id = (extId*)out;

  char hostname[1024];
  getHostName(hostname, 1024);
  NCCLCHECK(ncclNetListen(0, &id->extHandle, &id->extListenComm));
  id->hostHash = getHostHash(hostname);
  id->pid = getpid();
  id->lock = (int*)malloc(sizeof(int));
  *(id->lock) = 0;
  return ncclSuccess;
}

struct extInfo {
  int rank;
  ncclNetHandle_t extHandle;
};

struct extState {
  void** extRecvComm;
  void** extSendComm;
  int root;
  int rank;
  int nranks;
};

ncclResult_t bootstrapInit(ncclUniqueId* commId, int rank, int nranks, void** commState) {
  struct extId* id = (struct extId*)commId;
  struct extState* state = (struct extState*)malloc(sizeof(struct extState));
  state->root = 0;
  state->rank = rank;
  state->nranks = nranks;
  *commState = state;

  char hostname[1024];
  getHostName(hostname, 1024);
  uint64_t hostHash = getHostHash(hostname);
  pid_t pid = getpid();
  if (hostHash == id->hostHash && pid == id->pid) {
    // Try to become root
    if (__sync_bool_compare_and_swap(id->lock, 0, 1))
      state->root = 1;
  }

  if (state->root) {
    state->extSendComm = (void**)malloc(nranks*sizeof(void*));
    state->extRecvComm = (void**)malloc(nranks*sizeof(void*));
    /* Fill my info */
    struct extInfo info;

    /* Receive addresses from all ranks */
    for (int c=0; c<nranks-1; c++) {
      void* tmpRecvComm;
      NCCLCHECK(ncclNetAccept(id->extListenComm, &tmpRecvComm));
      NCCLCHECK(ncclNetRecv(tmpRecvComm, &info, sizeof(info)));
      state->extRecvComm[info.rank] = tmpRecvComm;
      NCCLCHECK(ncclNetConnect(0, info.extHandle, state->extSendComm+info.rank));
    }
    NCCLCHECK(ncclNetCloseListen(id->extListenComm));
    free(id->lock);
  } else {
    state->extSendComm = (void**)malloc(sizeof(void*));
    state->extRecvComm = (void**)malloc(sizeof(void*));
    struct extInfo info;
    info.rank = rank;
    void* tmpListenComm;
    NCCLCHECK(ncclNetListen(0, &info.extHandle, &tmpListenComm));
    NCCLCHECK(ncclNetConnect(0, id->extHandle, state->extSendComm));
    NCCLCHECK(ncclNetSend(state->extSendComm[0], &info, sizeof(info)));
    NCCLCHECK(ncclNetAccept(tmpListenComm, state->extRecvComm));
    NCCLCHECK(ncclNetCloseListen(tmpListenComm));
  }
  return ncclSuccess;
}

ncclResult_t bootstrapAllGather(void* commState, void* allData, int size) {
  struct extState* state = (struct extState*)commState;
  char* data = (char*)allData;
  if (state->root) {
    for (int r=0; r<state->nranks; r++) {
      if (r == state->rank) continue; 
      NCCLCHECK(ncclNetRecv(state->extRecvComm[r], data+r*size, size));
    }
    for (int r=0; r<state->nranks; r++) {
      if (r == state->rank) continue;
      NCCLCHECK(ncclNetSend(state->extSendComm[r], data, size*state->nranks));
    }
  } else {
    NCCLCHECK(ncclNetSend(state->extSendComm[0], data+state->rank*size, size));
    NCCLCHECK(ncclNetRecv(state->extRecvComm[0], data, size*state->nranks));
  }
  return ncclSuccess;
}

ncclResult_t bootstrapRingExchange(void* commState, void* prevNextData, int prev, int next, int size) {
  struct extState* state = (struct extState*)commState;
  char* mydata = (char*)prevNextData;
  int prev_offset = prev*2*size+size, next_offset = next*2*size;
  if (state->root) {
    char* data = (char*)malloc(size*2*state->nranks);
    // Receive from all and build total table
    for (int r=0; r<state->nranks; r++) {
      if (r == state->rank) {
        memcpy(data+r*2*size, mydata, 2*size);
      } else {
	NCCLCHECK(ncclNetRecv(state->extRecvComm[r], data+r*2*size, 2*size));
      }
    }

    // Get prev/next request from everyone and answer.
    for (int r=0; r<state->nranks; r++) {
      if (r == state->rank) {
        memcpy(mydata, data+prev_offset, size);
        memcpy(mydata+size, data+next_offset, size);
      } else {
	int offset;
	NCCLCHECK(ncclNetRecv(state->extRecvComm[r], &offset, sizeof(int)));
	NCCLCHECK(ncclNetSend(state->extSendComm[r], data+offset, size));
	NCCLCHECK(ncclNetRecv(state->extRecvComm[r], &offset, sizeof(int)));
	NCCLCHECK(ncclNetSend(state->extSendComm[r], data+offset, size));
      }
    }
    free(data);
  } else {
    // Send data to root
    NCCLCHECK(ncclNetSend(state->extSendComm[0], mydata, 2*size));

    // Receive prev and next data
    NCCLCHECK(ncclNetSend(state->extSendComm[0], &prev_offset, sizeof(int)));
    NCCLCHECK(ncclNetRecv(state->extRecvComm[0], mydata, size));
    NCCLCHECK(ncclNetSend(state->extSendComm[0], &next_offset, sizeof(int)));
    NCCLCHECK(ncclNetRecv(state->extRecvComm[0], mydata+size, size));
  }
  return ncclSuccess;
}

ncclResult_t bootstrapClose(void* commState) {
  struct extState* state = (struct extState*)commState;
  if (state->root) {
    for (int r=0; r<state->nranks; r++) {
      if (r == state->rank) continue;
      NCCLCHECK(ncclNetCloseSend(state->extSendComm[r]));
      NCCLCHECK(ncclNetCloseRecv(state->extRecvComm[r]));
    }
  } else {
    NCCLCHECK(ncclNetCloseSend(state->extSendComm[0]));
    NCCLCHECK(ncclNetCloseRecv(state->extRecvComm[0]));
  }
  free(state->extSendComm);
  free(state->extRecvComm);
  free(state);
  return ncclSuccess;
}

