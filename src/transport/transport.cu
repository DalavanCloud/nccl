#include "transport.h"
#include "core.h"
#include "common_kernel.h"

extern struct ncclTransport p2pTransport;
extern struct ncclTransport shmTransport;
extern struct ncclTransport socketTransport;

struct ncclTransport ncclTransports[NTRANSPORTS] = {
  p2pTransport,
  shmTransport,
  socketTransport
  // mpiTransport
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
  memcpy(info->argsFifo + (info->argsFifoTail % TRANSPORT_PROXY_FIFO_SIZE), args, sizeof(struct ncclProxyArgs));
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

static void StartProxy(int type, int substeps, int nsteps, struct ncclRing* ring, int pattern, int nranks) {
  struct ncclConnector* connector = (type == 0) ? &ring->recv : &ring->send;
  struct transportProxyInfo* info = connector->proxyInfo;
  if (info && NeedProxy(type, pattern, ring, nranks)) {
    struct ncclProxyArgs args;
    args.ring = ring;
    args.substeps = substeps;
    args.nsteps = nsteps;
    //printf("Launching %s proxy, nsteps = %d\n", type == 0 ? "recv" : "send", nsteps);
    FifoPushArgs(info, &args);
  }
}

ncclResult_t transportStartProxies(int substeps, int subchunks, int nsteps_per_round, int nblocks_per_round, int size, int pattern, struct ncclComm* comm) {
  for (int r=0; r<comm->nRings; r++) {
    int nrounds = DIVUP(size, comm->nRings * nblocks_per_round * (comm->rings[r].buffSize/subchunks));
    int nsteps = nsteps_per_round * nrounds * substeps;
    StartProxy(0, substeps*subchunks, nsteps, comm->rings+r, pattern, comm->nRanks);
    StartProxy(1, substeps*subchunks, nsteps, comm->rings+r, pattern, comm->nRanks);
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
    ncclResult_t res = info->func(&args);
    if (res != ncclSuccess) {
      WARN("Persistent proxy : proxy function returned with code %d\n", res);
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

