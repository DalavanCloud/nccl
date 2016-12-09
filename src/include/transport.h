#ifndef TRANSPORT_H_
#define TRANSPORT_H_

#include "nccl.h"
#include <stdint.h>

#define NTRANSPORTS 4

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
  int size;
  int nsteps;
  int opCount;
};

struct ncclTransportComm {
  ncclResult_t (*setup)(ncclTinfo_t*, ncclTinfo_t*, struct ncclConnect*, struct ncclRing*);
  ncclResult_t (*connect)(struct ncclConnect*, struct ncclConnector*);
  ncclResult_t (*proxy)(struct ncclProxyArgs*);
};

struct ncclTransport {
  const char name[4];
  ncclResult_t (*fillInfo)(ncclTinfo_t*, int);
  ncclResult_t (*canConnect)(int*, ncclTinfo_t*, ncclTinfo_t*);
  ncclResult_t (*getRings)(int, int, int*, int*, int*, int*, int*);
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
#if 0 // This reduces the CPU load but also impacts performance. It needs some tuning.
  int sleep_time = 2;
  int count = 0;
#endif
  while (!func()) {
#if 0
    if (count > 10) {
      if (sleep_time < 10) sleep_time *= 2;
      usleep(sleep_time);
    }
    count++;
#endif
  }
}

#endif
