/*************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_TRANSPORT_H_
#define NCCL_TRANSPORT_H_

#include "nccl.h"
#include <stdint.h>

#define NTRANSPORTS 3

extern struct ncclTransport ncclTransports[];

#define RANK_INFO_SIZE 64
typedef char ncclTinfo_t[RANK_INFO_SIZE];

struct ncclInfo {
  ncclTinfo_t tinfo[NTRANSPORTS];
};

#define CONNECT_SIZE 128
struct ncclConnect {
  char data[CONNECT_SIZE];
};

struct ncclProxyArgs {
  struct ncclRing* ring;
  int substeps;
  int nsteps;
  int opCount;
};

struct ncclTransportComm {
  ncclResult_t (*setup)(ncclTinfo_t*, ncclTinfo_t*, struct ncclConnect*, struct ncclRing*);
  ncclResult_t (*connect)(struct ncclConnect*, struct ncclConnector*);
  ncclResult_t (*free)(void*);
  ncclResult_t (*proxy)(struct ncclProxyArgs*);
};

struct ncclTransport {
  const char name[4];
  ncclResult_t (*fillInfo)(ncclTinfo_t*, int);
  ncclResult_t (*canConnect)(int*, ncclTinfo_t*, ncclTinfo_t*);
  ncclResult_t (*getRings)(int, int, int*, int*, int*, int*, int*, int);
  struct ncclTransportComm send;
  struct ncclTransportComm recv;
};

#include <pthread.h>

typedef ncclResult_t (*threadFunc_t)(struct ncclProxyArgs*);

#define TRANSPORT_PROXY_FIFO_SIZE 16

struct transportProxyInfo {
  struct ncclComm* comm;
  pthread_t thread;
  threadFunc_t func;
  volatile int proxyReady;
  struct ncclProxyArgs argsFifo[TRANSPORT_PROXY_FIFO_SIZE];
  volatile int argsFifoHead;
  volatile int argsFifoTail;
  pthread_cond_t cond;
  pthread_mutex_t mutex;
};

ncclResult_t transportCreateProxy(int type, struct ncclRing* ring, struct ncclComm* comm);
ncclResult_t transportDestroyProxy(struct ncclConnector* connector);

enum proxyMode {
  proxyRing = 0,
  proxyFrom = 1,
  proxyTo = 2
};

static int proxyPatternRing = proxyRing;
static int proxyPatternFrom(int root) { return 1+root; }
static int proxyPatternTo(int root) { return -1-root; }
static enum proxyMode proxyPatternMode(int pattern) { return (pattern == 0) ? proxyRing : ((pattern > 0) ? proxyFrom : proxyTo); }
static int proxyPatternRoot(int pattern) { return (pattern > 0) ? pattern-1 : -pattern-1; }

ncclResult_t transportStartProxies(int substeps, int subchunks, int nsteps_per_round, int nblocks_per_round, int size, int pattern, struct ncclComm* comm);

#include <unistd.h>

// Spin wait until func evaluates to true
template<typename FUNC>
inline void transportProxyWait(const FUNC& func) {
  while (!func()) {
    sched_yield();
  }
}

inline void transportProxyIdle(int idle) {
  sched_yield();
}

static inline int groupFirst(int nranks, int* groups, int group) {
  for (int rank = 0; rank<nranks; rank++) {
    if (groups[rank] == group) return rank;
  }
  return -1;
}

static inline int groupLast(int nranks, int* groups, int group) {
  for (int rank = nranks-1; rank>=0; rank--) {
    if (groups[rank] == group) return rank;
  }
  return -1;
}

static inline int groupPos(int nranks, int* groups, int group, int pos) {
  int skip = pos;
  int rank = 0;
  while (1) {
    if (groups[rank] == group) {
      if (skip == 0) {
       return rank;
      }
      skip--;
    }
    rank++;
    if (rank == nranks) {
      if (skip == pos) {
        // There seems to be no rank of this group. Stop.
        return -1;
      }
      rank = 0;
    }
  }
}

static inline int findConnect(int nranks, int* ranks) {
  for (int i = 0; i<nranks; i++) {
    if (ranks[i] != -1) return i;
  }
  return -1;
}
#endif
