/*************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nccl.h"
#include "core.h"
#include "utils.h"
#include "bootstrap.h"
#include "ext.h"
#include <unistd.h>
#include <sys/types.h>

struct extId {
  ncclExtHandle_t extHandle;
  void* extRecvComm;
  uint64_t hostHash;
  pid_t pid;
  int* lock;
  int fd;
};

ncclResult_t bootstrapExtGetUniqueId(ncclUniqueId* out) {
  static_assert(sizeof(extId) < sizeof(ncclUniqueId), "ExtId does not fit inside ncclUniqueId");
  extId* id = (extId*)out;
  NCCLCHECK(ncclExtEnabled());

  char hostname[1024];
  getHostName(hostname, 1024);
  NCCLCHECK(ncclExtGetHandle(&id->extHandle, &id->extRecvComm));
  id->hostHash = getHostHash(hostname);
  id->pid = getpid();
  id->lock = (int*)malloc(sizeof(int));
  *(id->lock) = 0;
  return ncclSuccess;
}

struct extInfo {
  int rank;
  ncclExtHandle_t extHandle;
};

struct extState {
  void* extRecvComm;
  void** extSendComm;
  int root;
  int rank;
  int nranks;
};

ncclResult_t bootstrapExtInit(ncclUniqueId* commId, int rank, int nranks, void** commState) {
  if (ncclExtEnabled() != ncclSuccess) {
    *commState = NULL;
    return ncclSuccess;
  }
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
    state->extRecvComm = id->extRecvComm;
    /* Fill my info */
    struct extInfo info;

    /* Receive addresses from all ranks */
    for (int c=0; c<nranks-1; c++) {
      NCCLCHECK(ncclExtRecv(state->extRecvComm, &info, sizeof(info)));
      NCCLCHECK(ncclExtConnectHandle(info.extHandle, state->extSendComm+info.rank));
    }
    for (int r=0; r<nranks; r++) {
      if (r == rank) continue;
      // Just for sync
      NCCLCHECK(ncclExtSend(state->extSendComm[r], &rank, sizeof(int)));
    }
    free(id->lock);
  } else {
    state->extSendComm = (void**)malloc(sizeof(void*));
    struct extInfo info;
    info.rank = rank;
    NCCLCHECK(ncclExtGetHandle(&info.extHandle, &state->extRecvComm));
    NCCLCHECK(ncclExtConnectHandle(id->extHandle, state->extSendComm));
    NCCLCHECK(ncclExtSend(state->extSendComm[0], &info, sizeof(info)));
    int dummy;
    NCCLCHECK(ncclExtRecv(state->extRecvComm, &dummy, sizeof(int)));
  }
  return ncclSuccess;
}

ncclResult_t bootstrapExtAllGather(void* commState, void* allData, int size) {
  struct extState* state = (struct extState*)commState;
  char* data = (char*)allData;
  if (state->root) {
    for (int r=0; r<state->nranks; r++) {
      if (r == state->rank) { 
        memcpy(data+r*size, data+state->rank*size, size);
      } else {
        int go = 1;
        NCCLCHECK(ncclExtSend(state->extSendComm[r], &go, sizeof(int)));
	NCCLCHECK(ncclExtRecv(state->extRecvComm, data+r*size, size));
      }
    }
    for (int r=0; r<state->nranks; r++) {
      if (r == state->rank) continue;
      NCCLCHECK(ncclExtSend(state->extSendComm[r], data, size*state->nranks));
    }
  } else {
    int go;
    NCCLCHECK(ncclExtRecv(state->extRecvComm, &go, sizeof(int)));
    NCCLCHECK(ncclExtSend(state->extSendComm[0], data+state->rank*size, size));
    NCCLCHECK(ncclExtRecv(state->extRecvComm, data, size*state->nranks));
  }
  return ncclSuccess;
}

ncclResult_t bootstrapExtRingExchange(void* commState, void* prevNextData, int prev, int next, int size) {
  struct extState* state = (struct extState*)commState;
  char* mydata = (char*)prevNextData;
  int prev_offset = prev*2*size+size, next_offset = next*2*size;
  if (state->root) {
    int go = 1;
    char* data = (char*)malloc(size*2*state->nranks);
    // Receive from all and build total table
    for (int r=0; r<state->nranks; r++) {
      if (r == state->rank) {
        memcpy(data+r*2*size, mydata, 2*size);
      } else {
        NCCLCHECK(ncclExtSend(state->extSendComm[r], &go, sizeof(int)));
	NCCLCHECK(ncclExtRecv(state->extRecvComm, data+r*2*size, 2*size));
      }
    }

    // Get prev/next request from everyone and answer.
    for (int r=0; r<state->nranks; r++) {
      if (r == state->rank) {
        memcpy(mydata, data+prev_offset, size);
        memcpy(mydata+size, data+next_offset, size);
      } else {
	int offset;
        NCCLCHECK(ncclExtSend(state->extSendComm[r], &go, sizeof(int)));
	NCCLCHECK(ncclExtRecv(state->extRecvComm, &offset, sizeof(int)));
	NCCLCHECK(ncclExtSend(state->extSendComm[r], data+offset, size));
	NCCLCHECK(ncclExtRecv(state->extRecvComm, &offset, sizeof(int)));
	NCCLCHECK(ncclExtSend(state->extSendComm[r], data+offset, size));
      }
    }
    free(data);
  } else {
    int go;
    NCCLCHECK(ncclExtRecv(state->extRecvComm, &go, sizeof(int)));
    // Send data to root
    NCCLCHECK(ncclExtSend(state->extSendComm[0], mydata, 2*size));

    // Receive prev and next data
    NCCLCHECK(ncclExtRecv(state->extRecvComm, &go, sizeof(int)));
    NCCLCHECK(ncclExtSend(state->extSendComm[0], &prev_offset, sizeof(int)));
    NCCLCHECK(ncclExtRecv(state->extRecvComm, mydata, size));
    NCCLCHECK(ncclExtSend(state->extSendComm[0], &next_offset, sizeof(int)));
    NCCLCHECK(ncclExtRecv(state->extRecvComm, mydata+size, size));
  }
  return ncclSuccess;
}

struct ncclBootstrap bootstrapExt = {
  bootstrapExtGetUniqueId,
  bootstrapExtInit,
  bootstrapExtAllGather,
  bootstrapExtRingExchange
};
