#include "transport.h"
/*************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "common_kernel.h"

extern struct ncclTransport p2pTransport;
extern struct ncclTransport shmTransport;
extern struct ncclTransport netTransport;

struct ncclTransport ncclTransports[NTRANSPORTS] = {
  p2pTransport,
  shmTransport,
  netTransport,
};

static void FifoPullArgs(struct transportProxyInfo* info, struct ncclProxyArgs *args) {
  pthread_mutex_lock(&info->mutex);
  while (info->argsFifoTail == info->argsFifoHead)
    pthread_cond_wait(&info->cond, &info->mutex);
  memcpy(args, info->argsFifo + (info->argsFifoHead % TRANSPORT_PROXY_FIFO_SIZE), sizeof(struct ncclProxyArgs));
  info->argsFifoHead++;
  pthread_cond_signal(&info->cond);
  pthread_mutex_unlock(&info->mutex);
}

static void FifoPushArgs(struct transportProxyInfo* info, struct ncclProxyArgs *args) {
  pthread_mutex_lock(&info->mutex);
  while (info->argsFifoTail == info->argsFifoHead + TRANSPORT_PROXY_FIFO_SIZE)
    pthread_cond_wait(&info->cond, &info->mutex);
  // args may be null in the case of termination
  if (args) memcpy(info->argsFifo + (info->argsFifoTail % TRANSPORT_PROXY_FIFO_SIZE), args, sizeof(struct ncclProxyArgs));
  info->argsFifoTail++;
  pthread_cond_signal(&info->cond);
  pthread_mutex_unlock(&info->mutex);
}

static void WaitProxyReady(struct transportProxyInfo* info) {
  pthread_mutex_lock(&info->mutex);
  while (info->proxyReady == 0)
    pthread_cond_wait(&info->cond, &info->mutex);
  pthread_mutex_unlock(&info->mutex);
}

static void SetProxyReady(struct transportProxyInfo* info) {
  pthread_mutex_lock(&info->mutex);
  info->proxyReady = 1;
  pthread_cond_signal(&info->cond);
  pthread_mutex_unlock(&info->mutex);
}

static void StopProxy(struct transportProxyInfo* info) {
  pthread_mutex_lock(&info->mutex); 
  info->proxyReady = -1;
  // Unblock thread
  info->argsFifoTail++;
  pthread_cond_signal(&info->cond);
  pthread_mutex_unlock(&info->mutex);
}

#define RECV 0
#define SEND 1

static bool NeedProxy(int type, int pattern, struct ncclRing* ring, int nranks) {
  enum proxyMode mode = proxyPatternMode(pattern);
  if (mode == proxyRing) return true;

  /* In chains, one rank does not need a proxy. Let's figure out which one it is */
  int root = proxyPatternRoot(pattern);
  // Which index in the reorganized rings should we compare root against */
  const int myrank = 0, nextrank = 1, prevrank = nranks-1;
  int index = mode == proxyFrom ? 
    /*                            no recv /  no send    if root = */
    /* bcast  */ (type == RECV ?   myrank : nextrank ):
    /* reduce */ (type == RECV ? prevrank :   myrank );
  int rank = ring->userRanks[index];
  return (root != rank);
}

static void StartProxy(int type, int substeps, int nsteps, uint64_t opCount, struct ncclRing* ring, int pattern, int nranks, int llMode) {
  struct ncclConnector* connector = (type == 0) ? &ring->recv : &ring->send;
  struct transportProxyInfo* info = connector->proxyInfo;
  if (info) {
    struct ncclProxyArgs args;
    args.ring = ring;
    args.substeps = substeps;
    args.nsteps = nsteps;
    args.opCount = opCount;
    args.llMode = llMode;
    args.needProxy = NeedProxy(type, pattern, ring, nranks);
    FifoPushArgs(info, &args);
  }
}

ncclResult_t transportSaveProxies(int substeps, int subchunks, int nstepsPerRound, int nblocksPerRound, size_t size, int pattern, struct ncclComm* comm, int nRings, int llMode) {
  struct ncclProxyParams params = { substeps, subchunks, nstepsPerRound, nblocksPerRound, size, pattern, nRings, llMode };
  memcpy(&comm->proxyParams, &params, sizeof(params));
  return ncclSuccess;
}

ncclResult_t transportStartProxies(struct ncclProxyParams* p, struct ncclComm* comm) {
  int nrings = LIMIT_NRINGS(p->size, p->nRings);
  for (int r=0; r<nrings; r++) {
    int buffSize = p->llMode ? LL_BUFF_SIZE : comm->rings[r].buffSize;
    int nrounds = (int)(DIVUP(p->size, nrings * p->nblocksPerRound * (buffSize/p->subchunks)));
    int nsteps = p->nstepsPerRound * nrounds * p->substeps;
    StartProxy(0, p->substeps*p->subchunks, nsteps, comm->opCount, comm->rings+r, p->pattern, comm->nRanks, p->llMode);
    StartProxy(1, p->substeps*p->subchunks, nsteps, comm->opCount, comm->rings+r, p->pattern, comm->nRanks, p->llMode);
  }
  return ncclSuccess;
}

void* persistentThread(void *opaqueInfo) {
  struct transportProxyInfo* info = (struct transportProxyInfo*)opaqueInfo;
  // We need to initialize the context before launching any NCCL cuda kernel,
  // otherwise we would create it during the first cudaMemcpyAsync inside the
  // proxy function and that would cause a deadlock
  cudaSetDevice(info->comm->cudaDev);
  // Signal the main thread the context is created and it can proceed.
  SetProxyReady(info);
  while (1) {
    struct ncclProxyArgs args;
    FifoPullArgs(info, &args);
    if (info->proxyReady == -1) {
      // Main thread asked to stop
      return NULL;
    }
    ncclResult_t res = info->func(&args);
    if (res != ncclSuccess) {
      INFO("%s:%d -> %d [Proxy thread]", __FILE__, __LINE__, res);
    }    
  }
}

ncclResult_t transportCreateProxy(int type, struct ncclRing* ring, struct ncclComm* comm) {
  struct ncclConnector* connector = (type == 0) ? &ring->recv : &ring->send;
  threadFunc_t proxyfunc = (threadFunc_t) ((type == 0) ? connector->transport->recv.proxy : connector->transport->send.proxy);
  if (proxyfunc) {
    struct transportProxyInfo * info = connector->proxyInfo = (struct transportProxyInfo*)malloc(sizeof(struct transportProxyInfo));
    info->comm = comm;
    info->cond = PTHREAD_COND_INITIALIZER;
    info->mutex = PTHREAD_MUTEX_INITIALIZER;
    info->func = proxyfunc;
    info->argsFifoHead = info->argsFifoTail = 0;
    info->proxyReady = 0;
    pthread_create(&connector->proxyInfo->thread, NULL, persistentThread, info);
    // Wait for thread to initialize its CUDA context.
    WaitProxyReady(info);
  }
  return ncclSuccess;
}

ncclResult_t transportDestroyProxy(struct ncclConnector* connector) {
  if (connector->proxyInfo) {
    StopProxy(connector->proxyInfo);
    pthread_join(connector->proxyInfo->thread, NULL);
    free(connector->proxyInfo);
    connector->proxyInfo = NULL;
  }
  return ncclSuccess;
}
