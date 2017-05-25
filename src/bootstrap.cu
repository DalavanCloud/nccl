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

#define NCCLCHECKJUMP(call, out) do { \
  ncclResult_t res = call; \
  if (res != ncclSuccess) { \
    /* Print the back trace*/ \
    INFO("%s:%d -> %d [bthread]", __FILE__, __LINE__, res); \
    goto out; \
  } \
} while (0);

struct extId {
  ncclNetHandle_t extHandle;
  void* extListenComm;
  uint64_t hostHash;
  pid_t pid;
  int fd;
  pthread_t boostrapThread;
};

struct bootstrapOp {
  int op;
  int size;
};

struct extInfo {
  int rank;
  int nranks;
  ncclNetHandle_t extHandle;
};

enum { 
  BOOTSTRAP_ALLGATHER = 1,
  BOOTSTRAP_RINGEXCHANGE,
};

static void *bootstrapRoot(void* commId) {
  struct extInfo info;
  struct extId* id = (struct extId*)commId;
  struct bootstrapOp bop;
  void **extSendComm = NULL;
  void **extRecvComm = NULL;
  int size, alloc_size = 0; 
  char* data = NULL;

  /* Receive addresses from all ranks */
  int nranks = 0, c = 0;
  do {
      void* tmpRecvComm;
      NCCLCHECKJUMP(ncclNetAccept(id->extListenComm, &tmpRecvComm), out);
      NCCLCHECKJUMP(ncclNetRecv(tmpRecvComm, &info, sizeof(info)), out);
      if (!c) { 
          extSendComm = (void**)calloc(info.nranks, sizeof(void*));
          extRecvComm = (void**)calloc(info.nranks, sizeof(void*));
          nranks = info.nranks;
      }

      if (nranks != info.nranks) { 
	  WARN("Bootstrap Root : mismatch in rank count from procs %d : %d\n", nranks, info.nranks);
	  goto out;
      }

      extRecvComm[info.rank] = tmpRecvComm;
      NCCLCHECKJUMP(ncclNetConnect(0, info.extHandle, extSendComm+info.rank), out);
      c++;
  } while (c < nranks);

  do {
      NCCLCHECKJUMP(ncclNetRecv(extRecvComm[0], &bop, sizeof(struct bootstrapOp)), out);
      if (bop.size == -1) { 
          break;
      } else { 
	  size = bop.size;
	  if (size*nranks*2 > alloc_size) { 
	      if (data) free(data);
	      data = (char *)malloc(size*nranks*2);
	      alloc_size = size*nranks*2;
	  }
      }

      if (bop.op == BOOTSTRAP_ALLGATHER) {  
          for (int r=0; r<nranks; r++) {
              NCCLCHECKJUMP(ncclNetRecv(extRecvComm[r], data+size*r, size), out);
          }

          for (int r=0; r<nranks; r++) {
              NCCLCHECKJUMP(ncclNetSend(extSendComm[r], data, size*nranks), out);
          }
      } else if (bop.op == BOOTSTRAP_RINGEXCHANGE) {
	  // Receive from all and build total table
          for (int r=0; r<nranks; r++) {
              NCCLCHECKJUMP(ncclNetRecv(extRecvComm[r], data+r*2*size, 2*size), out);
          }
        
          // Get prev/next request from everyone and answer.
          for (int r=0; r<nranks; r++) {
              int offset;
              NCCLCHECKJUMP(ncclNetRecv(extRecvComm[r], &offset, sizeof(int)), out);
              NCCLCHECKJUMP(ncclNetSend(extSendComm[r], data+offset, size), out);
              NCCLCHECKJUMP(ncclNetRecv(extRecvComm[r], &offset, sizeof(int)), out);
              NCCLCHECKJUMP(ncclNetSend(extSendComm[r], data+offset, size), out);
          }
      } else {
	  WARN("Bootstrap Root : invalid op type received %d\n", bop.op);
	  break;
      }
  } while (1);

out:
  ncclNetCloseListen(id->extListenComm);
  for (int r=0; r<nranks; r++) {
      if (extSendComm[r]) ncclNetCloseSend(extSendComm[r]);
      if (extRecvComm[r]) ncclNetCloseRecv(extRecvComm[r]);
  }
  free(commId);
  if (extSendComm) free(extSendComm);
  if (extRecvComm) free(extRecvComm);
  return NULL;
}

ncclResult_t bootstrapGetUniqueId(ncclUniqueId* out) {
  static_assert(sizeof(extId) < sizeof(ncclUniqueId), "NetId does not fit inside ncclUniqueId");
  extId* id = (extId*)out;

  char hostname[1024];
  getHostName(hostname, 1024);
  NCCLCHECK(ncclNetListen(0, &id->extHandle, &id->extListenComm));
  id->hostHash = getHostHash(hostname);
  id->pid = getpid();

  ncclUniqueId* threadIdCopy = (ncclUniqueId*)malloc(sizeof(ncclUniqueId));
  memcpy(threadIdCopy, id, sizeof(ncclUniqueId));
  pthread_create(&id->boostrapThread, NULL, bootstrapRoot, (void *)threadIdCopy);
  return ncclSuccess;
}

struct extState {
  void* extRecvComm;
  void* extSendComm;
  int rank;
  int nranks;
};

ncclResult_t bootstrapInit(ncclUniqueId* commId, int rank, int nranks, void** commState) {
  struct extId* id = (struct extId*)commId;
  struct extState* state = (struct extState*)malloc(sizeof(struct extState));
  state->rank = rank;
  state->nranks = nranks;
  *commState = state;

  struct extInfo info;
  info.rank = rank;
  info.nranks = nranks;
  void* tmpListenComm;
  NCCLCHECK(ncclNetListen(0, &info.extHandle, &tmpListenComm));
  NCCLCHECK(ncclNetConnect(0, id->extHandle, &state->extSendComm));
  NCCLCHECK(ncclNetSend(state->extSendComm, &info, sizeof(info)));
  NCCLCHECK(ncclNetAccept(tmpListenComm, &state->extRecvComm));
  NCCLCHECK(ncclNetCloseListen(tmpListenComm));

  return ncclSuccess;
}

ncclResult_t bootstrapAllGather(void* commState, void* allData, int size) {
  struct extState* state = (struct extState*)commState;
  char* data = (char*)allData;
  struct bootstrapOp bop;

  bop.op = BOOTSTRAP_ALLGATHER;
  bop.size = size;

  if (!state->rank) { 
      NCCLCHECK(ncclNetSend(state->extSendComm, &bop, sizeof(struct bootstrapOp)));
  } 

  NCCLCHECK(ncclNetSend(state->extSendComm, data+state->rank*size, size));
  NCCLCHECK(ncclNetRecv(state->extRecvComm, data, size*state->nranks));

  return ncclSuccess;
}

ncclResult_t bootstrapRingExchange(void* commState, void* prevNextData, int prev, int next, int size) {
  struct extState* state = (struct extState*)commState;
  char* mydata = (char*)prevNextData;
  int prev_offset = prev*2*size+size, next_offset = next*2*size;

  struct bootstrapOp bop;
  bop.op = BOOTSTRAP_RINGEXCHANGE;
  bop.size = size;

  if (!state->rank) {
      NCCLCHECK(ncclNetSend(state->extSendComm, &bop, sizeof(struct bootstrapOp))); 
  }

  // Send data to root
  NCCLCHECK(ncclNetSend(state->extSendComm, mydata, 2*size));

  // Receive prev and next data
  NCCLCHECK(ncclNetSend(state->extSendComm, &prev_offset, sizeof(int)));
  NCCLCHECK(ncclNetRecv(state->extRecvComm, mydata, size));
  NCCLCHECK(ncclNetSend(state->extSendComm, &next_offset, sizeof(int)));
  NCCLCHECK(ncclNetRecv(state->extRecvComm, mydata+size, size));


  return ncclSuccess;
}

ncclResult_t bootstrapClose(void* commState) {
  struct extState* state = (struct extState*)commState;
  struct bootstrapOp bop;
  bop.size = -1;

  if (!state->rank) { 
      NCCLCHECK(ncclNetSend(state->extSendComm, &bop, sizeof(struct bootstrapOp)));
  } 

  NCCLCHECK(ncclNetCloseSend(state->extSendComm));
  NCCLCHECK(ncclNetCloseRecv(state->extRecvComm));

  free(state);

  return ncclSuccess;
}
