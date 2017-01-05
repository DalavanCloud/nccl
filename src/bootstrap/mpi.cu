/*************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nccl.h"
#include "core.h"
#include "utils.h"
#include "bootstrap.h"
#include "mpi.h"
#include <unistd.h>
#include <sys/types.h>

struct mpiId {
  int mpiRank;
  int mpiTag;
  uint64_t hostHash;
  pid_t pid;
  int* lock;
  int fd;
};

ncclResult_t bootstrapMpiGetUniqueId(ncclUniqueId* out) {
  static_assert(sizeof(mpiId) < sizeof(ncclUniqueId), "MpiId does not fit inside ncclUniqueId");
  mpiId* id = (mpiId*)out;
  if (ncclMpiEnabled() == 0)
    return ncclInternalError;

  char hostname[1024];
  getHostName(hostname, 1024);
  ncclMpiCommRank(&id->mpiRank);
  ncclMpiGetTag(&id->mpiTag);
  id->hostHash = getHostHash(hostname);
  id->pid = getpid();
  id->lock = (int*)malloc(sizeof(int));
  *(id->lock) = 0;
  return ncclSuccess;
}

struct mpiInfo {
  int rank;
  int mpiRank;
  int mpiTag;
};

struct mpiState {
  struct mpiInfo* rankInfo;
  int root;
  int rank;
  int nranks;
};

ncclResult_t bootstrapMpiInit(ncclUniqueId* commId, int rank, int nranks, void** commState) {
  if (ncclMpiEnabled() == 0) {
    *commState = NULL;
    return ncclSuccess;
  }
  struct mpiId* id = (struct mpiId*)commId;
  int root = 0;
  struct mpiInfo* rankInfo;

  char hostname[1024];
  getHostName(hostname, 1024);
  uint64_t hostHash = getHostHash(hostname);
  pid_t pid = getpid();
  if (hostHash == id->hostHash && pid == id->pid) {
    // Try to become root
    if (__sync_bool_compare_and_swap(id->lock, 0, 1))
      root = 1;
  }

  if (root) {
    rankInfo = (struct mpiInfo*)malloc(nranks*sizeof(struct mpiInfo));
    /* Fill my info */
    struct mpiInfo info;
    rankInfo[rank].mpiRank = id->mpiRank;
    rankInfo[rank].mpiTag = id->mpiTag;
    rankInfo[rank].rank = rank;

    /* Receive addresses from all ranks */
    for (int c=0; c<nranks-1; c++) {
      ncclMpiRecv(-1, &info, sizeof(info), id->mpiTag);
      memcpy(rankInfo+info.rank, &info, sizeof(info));
    }
    for (int r=0; r<nranks; r++) {
      if (r == rank) continue;
      // Just for sync
      ncclMpiSend(rankInfo[r].mpiRank, &rank, sizeof(int), rankInfo[r].mpiTag);
    }
    free(id->lock);
  } else {
    rankInfo = (struct mpiInfo*)malloc(sizeof(struct mpiInfo));
    rankInfo->mpiRank = id->mpiRank;
    rankInfo->mpiTag = id->mpiTag;
    struct mpiInfo info;
    info.rank = rank;
    ncclMpiCommRank(&info.mpiRank);
    ncclMpiGetTag(&info.mpiTag);
    ncclMpiSend(id->mpiRank, &info, sizeof(info), id->mpiTag);
    int dummy;
    ncclMpiRecv(id->mpiRank, &dummy, sizeof(int), id->mpiTag);
  }
  
  struct mpiState* state = (struct mpiState*)malloc(sizeof(struct mpiState));
  state->root = root;
  state->rankInfo = rankInfo;
  state->rank = rank;
  state->nranks = nranks;
  *commState = state;
  return ncclSuccess;
}

ncclResult_t bootstrapMpiAllGather(void* commState, void* allData, int size) {
  struct mpiState* state = (struct mpiState*)commState;
  char* data = (char*)allData;
  if (state->root) {
    for (int r=0; r<state->nranks; r++) {
      if (r == state->rank) { 
        memcpy(data+r*size, data+state->rank*size, size);
      } else {
	ncclMpiRecv(state->rankInfo[r].mpiRank, data+r*size, size, state->rankInfo[r].mpiTag);
      }
    }
    for (int r=0; r<state->nranks; r++) {
      if (r == state->rank) continue;
      ncclMpiSend(state->rankInfo[r].mpiRank, data, size*state->nranks, state->rankInfo[r].mpiTag);
    }
  } else {
    ncclMpiSend(state->rankInfo[0].mpiRank, data+state->rank*size, size, state->rankInfo[0].mpiTag);
    ncclMpiRecv(state->rankInfo[0].mpiRank, data, size*state->nranks, state->rankInfo[0].mpiTag);
  }
  return ncclSuccess;
}

ncclResult_t bootstrapMpiRingExchange(void* commState, void* prevNextData, int prev, int next, int size) {
  struct mpiState* state = (struct mpiState*)commState;
  char* mydata = (char*)prevNextData;
  int prev_offset = prev*2*size+size, next_offset = next*2*size;
  if (state->root) {
    char* data = (char*)malloc(size*2*state->nranks);
    // Receive from all and build total table
    for (int r=0; r<state->nranks; r++) {
      if (r == state->rank) {
        memcpy(data+r*2*size, mydata, 2*size);
      } else {
	ncclMpiRecv(state->rankInfo[r].mpiRank, data+r*2*size, 2*size, state->rankInfo[r].mpiTag);
      }
    }

    // Get prev/next request from everyone and answer.
    for (int r=0; r<state->nranks; r++) {
      if (r == state->rank) {
        memcpy(mydata, data+prev_offset, size);
        memcpy(mydata+size, data+next_offset, size);
      } else {
	int offset;
	ncclMpiRecv(state->rankInfo[r].mpiRank, &offset, sizeof(int), state->rankInfo[r].mpiTag);
	ncclMpiSend(state->rankInfo[r].mpiRank, data+offset, size, state->rankInfo[r].mpiTag);
	ncclMpiRecv(state->rankInfo[r].mpiRank, &offset, sizeof(int), state->rankInfo[r].mpiTag);
	ncclMpiSend(state->rankInfo[r].mpiRank, data+offset, size, state->rankInfo[r].mpiTag);
      }
    }
    free(data);
  } else {
    // Send data to root
    ncclMpiSend(state->rankInfo[0].mpiRank, mydata, 2*size, state->rankInfo[0].mpiTag);
    // Receive prev and next data
    ncclMpiSend(state->rankInfo[0].mpiRank, &prev_offset, sizeof(int), state->rankInfo[0].mpiTag);
    ncclMpiRecv(state->rankInfo[0].mpiRank, mydata, size, state->rankInfo[0].mpiTag);
    ncclMpiSend(state->rankInfo[0].mpiRank, &next_offset, sizeof(int), state->rankInfo[0].mpiTag);
    ncclMpiRecv(state->rankInfo[0].mpiRank, mydata+size, size, state->rankInfo[0].mpiTag);
  }
  return ncclSuccess;
}

struct ncclBootstrap bootstrapMpi = {
  bootstrapMpiGetUniqueId,
  bootstrapMpiInit,
  bootstrapMpiAllGather,
  bootstrapMpiRingExchange
};
