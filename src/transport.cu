#include "transport.h"
#include "core.h"
#include "common_kernel.h"

extern struct ncclTransport p2pTransport;
extern struct ncclTransport shmTransport;

struct ncclTransport ncclTransports[NTRANSPORTS] = {
  p2pTransport,
  shmTransport
  // socketTransport,
  // mpiTransport
};

#include <pthread.h>

typedef void* (*threadFunc_t)(void*);

template <int type>
void LaunchProxy(int substeps, int nsteps, struct ncclRing* ring) {
  struct ncclConnector* connector = (type == 0) ? &ring->recv : &ring->send;
  threadFunc_t proxyfunc = (threadFunc_t) ((type == 0) ? connector->transport->recv.proxy : connector->transport->send.proxy);
  if (proxyfunc) {
    if (connector->thread_running == 1)
      pthread_join(connector->thread, NULL);
    struct ncclProxyArgs* args = (struct ncclProxyArgs*)malloc(sizeof(struct ncclProxyArgs));
    args->ring = ring;
    args->substeps = substeps;
    args->nsteps = nsteps;
    connector->thread_running = 1;
    //printf("Launching %s proxy, nsteps = %d\n", type == 0 ? "recv" : "send", nsteps);
    pthread_create(&connector->thread, NULL, proxyfunc, args);
  }
}

ncclResult_t LaunchProxies(int substeps, int subchunks, int nsteps_per_round, int nblocks_per_round, int size, struct ncclComm* comm) {
  for (int r=0; r<comm->nRings; r++) {
    int nrounds = DIVUP(size, comm->nRings * nblocks_per_round * (comm->rings[r].buffSize/subchunks));
    int nsteps = nsteps_per_round * nrounds * substeps;
    LaunchProxy<0>(substeps*subchunks, nsteps, comm->rings+r);
    LaunchProxy<1>(substeps*subchunks, nsteps, comm->rings+r);
  }
  return ncclSuccess;
}

template <int type>
ncclResult_t WaitProxy(struct ncclRing* ring) {
  struct ncclConnector* connector = (type == 0) ? &ring->recv : &ring->send;
  threadFunc_t proxyfunc = (threadFunc_t) ((type == 0) ? connector->transport->recv.proxy : connector->transport->send.proxy);
  if (proxyfunc) {
    ncclResult_t ret;
    pthread_join(connector->thread, (void**)&ret);
    NCCLCHECK(ret);
  }
  return ncclSuccess;
}

ncclResult_t WaitProxies(struct ncclComm* comm) {
  for (int r=0; r<comm->nRings; r++) {
    NCCLCHECK(WaitProxy<0>(comm->rings+r));
    NCCLCHECK(WaitProxy<1>(comm->rings+r));
  }
  return ncclSuccess;
}


