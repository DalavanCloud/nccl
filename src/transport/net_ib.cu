/*************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nccl.h"
#include "core.h"
#include "socket.h"
#include "net.h"

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <poll.h>

#include "infiniband/verbs.h"

int ncclIbPtrSupport(int* supportedTypes) {
  *supportedTypes = NCCL_PTR_HOST;
  return 0;
}

struct ibRequest {
  void* comm;
  struct ibv_mr* mr;
  struct ibv_cq* cq;
  int done;
  int size;
};

static int numRequests = 0;
struct ibRequest* ncclIbRequests = NULL;
int* ncclIbRequestUsed = NULL;
pthread_mutex_t ncclIbLock = PTHREAD_MUTEX_INITIALIZER;

struct ibRequest* ncclIbGetRequest() {
  pthread_mutex_lock(&ncclIbLock);
  for (int i=0; i<numRequests; i++) {
    if (ncclIbRequestUsed[i] == 0) {
      ncclIbRequestUsed[i] = 1; 
      pthread_mutex_unlock(&ncclIbLock);
      return ncclIbRequests + i;
    }
  }
  // No free request found, grow the pool
  int newNumRequests = numRequests + 32;
  struct ibRequest* newRequests = (struct ibRequest*)malloc(newNumRequests*sizeof(struct ibRequest));
  int* newUsed = (int*)malloc(newNumRequests*sizeof(int));
  for (int i=0; i<numRequests; i++) {
    newRequests[i] = ncclIbRequests[i];
    newUsed[i] = ncclIbRequestUsed[i];
  } 
  for (int i=numRequests; i<newNumRequests; i++)
    newUsed[i] = 0;
  free(ncclIbRequests);
  ncclIbRequests = newRequests;
  free(ncclIbRequestUsed);
  ncclIbRequestUsed = newUsed;
  numRequests = newNumRequests;
  pthread_mutex_unlock(&ncclIbLock);
  return ncclIbGetRequest();
}

void ncclIbFreeRequest(struct ibRequest* request) {
  pthread_mutex_lock(&ncclIbLock);
  ncclIbRequestUsed[request-ncclIbRequests] = 0;
  pthread_mutex_unlock(&ncclIbLock);
}

struct ncclIbQpInfo {
  int lid;
  uint8_t ib_port;
  int qpn;
};

struct ncclIbHandle {
  struct socketAddress connectAddr;
  struct ncclIbQpInfo qpInfo;
};

struct ncclIbCommPeer {
  struct ibv_qp* qp;
};

struct ncclIbComm {
  struct ibv_pd* pd;
  struct ibv_comp_channel* cc;
  struct ibv_cq* cq;
  struct ibv_srq* srq;
  int npeers;
  struct ncclIbCommPeer* peers;
};

struct ncclIbRecvComm {
  int listenFd;
  int npeers;
  int npeersActive;
  struct pollfd* fds;
  int dev;
  struct ncclIbComm ibComm;
  int numRecvRequests;
};

struct ncclIbSendComm {
  int fd;
  int ready;
  struct ncclIbComm ibComm;
  int numSendRequests;
};

void ncclIbAddFd(struct ncclIbRecvComm* comm, int fd) {
  if (comm->npeersActive >= comm->npeers) {
    // Grow the number of fds
    comm->npeers += 32;
    comm->fds = (struct pollfd*)realloc(comm->fds, (comm->npeers)*sizeof(struct pollfd));
  }
  comm->fds[comm->npeersActive].fd = fd;
  comm->fds[comm->npeersActive].events = POLLIN;
  comm->npeersActive++;
}

#define MAX_IF_NAME_SIZE 16
static char ncclIbIfName[MAX_IF_NAME_SIZE];
static struct in_addr ncclIbIfAddr;
static int ncclNIbDevs = -1;
struct ncclIbDev {
  int device;
  uint8_t port;
  ibv_context* context;
};

#define MAX_IB_DEVS 16
struct ncclIbDev ncclIbDevs[MAX_IB_DEVS];

int ncclIbTimeout = 14;

static void initDevices() {
  if (ncclNIbDevs == -1) {
    pthread_mutex_lock(&ncclIbLock);
    if (ncclNIbDevs == -1) {
      // Get an IP card for OOB transport
      char* env = getenv("NCCL_SOCKET_IFNAME");
      if (env && strlen(env) > 1) {
        // Specified by user : find or fail
        if (findInterfaces(env, ncclIbIfName, &ncclIbIfAddr, MAX_IF_NAME_SIZE, 1) == 0) {
          WARN("NET/IB : No IP interface found (starting with %s).", env);
          return;
        }
      } else {
        // Try to automatically pick one that will work, but not loopback
        if (findInterfaces("^lo", ncclIbIfName, &ncclIbIfAddr, MAX_IF_NAME_SIZE, 1) == 0) {
          WARN("NET/IB : No IP interface found.");
          return;
        }
      }
      INFO("NET/IB : Using interface %s for sideband communication", ncclIbIfName);
      
      // Detect IB cards
      int nIbDevs;
      ncclNIbDevs = 0;
      struct ibv_device** devices = ibv_get_device_list(&nIbDevs);
      for (int d=0; d<nIbDevs; d++) {
        struct ibv_context * context = ibv_open_device(devices[d]);
        int found = 0;
        if (context) {
          struct ibv_device_attr devAttr;
          if (ibv_query_device(context, &devAttr) == 0) {
            for (int port = 1; port <= devAttr.phys_port_cnt; port++) {
              struct ibv_port_attr portAttr;
              if (ibv_query_port(context, port, &portAttr) != 0) continue;
              if (portAttr.state != IBV_PORT_ACTIVE) continue;
              INFO("Found IB device %d : %s / port %d", d, ibv_get_device_name(devices[d]), port);
              ncclIbDevs[ncclNIbDevs].device = d;
              ncclIbDevs[ncclNIbDevs].port = port;
              ncclIbDevs[ncclNIbDevs].context = context;
              ncclNIbDevs++;
              found++;
            } 
          }
          if (found == 0) ibv_close_device(context);
        }
      }
      ibv_free_device_list(devices);
    }

    char* env = getenv("NCCL_IB_TIMEOUT");
    if (env && strlen(env) > 1) ncclIbTimeout = atoi(env);

    pthread_mutex_unlock(&ncclIbLock);
  }
}

int ncclIbDevices(int* ndev, int** distances) {
  initDevices();
  *ndev = ncclNIbDevs;
  int* dists = (int*)malloc(ncclNIbDevs*sizeof(int));
  for (int i=0; i<ncclNIbDevs; i++) dists[i] = 0;
  *distances = dists;
  return ncclSuccess;
}

static ncclResult_t GetIpAddr(struct in_addr* addr) {
  if (ncclNIbDevs == -1) initDevices();
  memcpy(addr, &ncclIbIfAddr, sizeof(struct in_addr));
  return ncclSuccess;
}

#define MAX_SEND_WR 128
#define MAX_RECV_WR 1024

#define NULLCHECK(cmd) \
  if ((cmd) == NULL) { \
    WARN("IBV call return NULL\n"); \
  }

ncclResult_t ncclIbCreateQp(ibv_context* ctx, uint8_t ib_port, struct ncclIbComm* comm, int* peerIdx) {
  if (comm->npeers == 0) {
    NULLCHECK(comm->pd = ibv_alloc_pd(ctx));
    NULLCHECK(comm->cc = ibv_create_comp_channel(ctx));
    NULLCHECK(comm->cq = ibv_create_cq(ctx, 2048, NULL, comm->cc, 0));
  }

  struct ibv_srq_init_attr srqAttr;
  memset(&srqAttr, 0, sizeof(srqAttr));
  srqAttr.srq_context = ctx;
  srqAttr.attr.max_wr = MAX_RECV_WR;
  srqAttr.attr.max_sge = 1;
  srqAttr.attr.srq_limit = 0;
  if (comm->srq == NULL) NULLCHECK(comm->srq = ibv_create_srq(comm->pd, &srqAttr));

  comm->peers = (struct ncclIbCommPeer*)realloc(comm->peers, (comm->npeers+1)*sizeof(struct ncclIbCommPeer));

  int peer = comm->npeers;
  struct ncclIbCommPeer* commPeer = comm->peers+peer;

  struct ibv_qp_init_attr qpInitAttr;
  memset(&qpInitAttr, 0, sizeof(struct ibv_qp_init_attr));
  qpInitAttr.send_cq = comm->cq;
  qpInitAttr.recv_cq = comm->cq;
  qpInitAttr.srq = comm->srq;
  qpInitAttr.qp_type = IBV_QPT_RC;
  qpInitAttr.cap.max_send_wr = MAX_SEND_WR;
  qpInitAttr.cap.max_recv_wr = 1;
  qpInitAttr.cap.max_send_sge = 1;
  qpInitAttr.cap.max_recv_sge = 1;
  qpInitAttr.cap.max_inline_data = 0;
  qpInitAttr.sq_sig_all = 1;
  NULLCHECK(commPeer->qp = ibv_create_qp(comm->pd, &qpInitAttr));
  struct ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
  qpAttr.qp_state = IBV_QPS_INIT;
  qpAttr.pkey_index = 0;
  qpAttr.port_num = ib_port;
  qpAttr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
  SYSCHECK(ibv_modify_qp(commPeer->qp, &qpAttr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS), "ibv_modify_qp");
  *peerIdx = peer;
  comm->npeers = peer + 1;
  return ncclSuccess;
}

ncclResult_t ncclIbRtrQp(ibv_qp* qp, int qpn, int lid, uint8_t ib_port) {
  struct ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
  qpAttr.qp_state = IBV_QPS_RTR;
  qpAttr.path_mtu = IBV_MTU_2048;
  qpAttr.dest_qp_num = qpn;
  qpAttr.rq_psn = 0;
  qpAttr.max_dest_rd_atomic = 1;
  qpAttr.min_rnr_timer = 12;
  qpAttr.ah_attr.is_global = 0;
  qpAttr.ah_attr.dlid = lid;
  qpAttr.ah_attr.sl = 1;
  qpAttr.ah_attr.src_path_bits = 0;
  qpAttr.ah_attr.port_num = ib_port;
  SYSCHECK(ibv_modify_qp(qp, &qpAttr, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER), "ibv_modify_qp");
  return ncclSuccess;
}

ncclResult_t ncclIbRtsQp(ibv_qp* qp) {
  struct ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
  qpAttr.qp_state = IBV_QPS_RTS;
  qpAttr.timeout = ncclIbTimeout;
  qpAttr.retry_cnt = 7;
  qpAttr.rnr_retry = 7;
  qpAttr.sq_psn = 0;
  qpAttr.max_rd_atomic = 1;
  SYSCHECK(ibv_modify_qp(qp, &qpAttr, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC), "ibv_modify_qp");
  return ncclSuccess;
}

int ncclIbGetHandle(int dev, void* opaqueHandle, void** recvComm) {
  struct ncclIbRecvComm* comm = (struct ncclIbRecvComm*)malloc(sizeof(struct ncclIbRecvComm));
  memset(comm, 0, sizeof(struct ncclIbRecvComm));
  struct ncclIbHandle* handle = (struct ncclIbHandle*) opaqueHandle;
  assert(sizeof(struct ncclIbHandle) < NCCL_NET_HANDLE_MAXSIZE);
  comm->npeers = comm->npeersActive = 0;
  comm->fds = NULL;
  comm->dev = dev;
  NCCLCHECK(GetIpAddr(&(handle->connectAddr.ip_addr)));
  NCCLCHECK(createListenSocket(&comm->listenFd, handle->connectAddr.ip_addr, &handle->connectAddr.port));

  *recvComm = comm;
  return 0;
}

int ncclIbConnectHandle(int dev, void* opaqueHandle, void** sendComm) {
  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)malloc(sizeof(struct ncclIbSendComm));
  memset(comm, 0, sizeof(struct ncclIbSendComm));
  struct ncclIbHandle* handle = (struct ncclIbHandle*) opaqueHandle;
  NCCLCHECK(connectAddress(&handle->connectAddr, ncclIbIfAddr, &comm->fd));
  *sendComm = comm;
  comm->ready = 0;
  
  // IB Setup
  ibv_context* ctx = ncclIbDevs[dev].context;
  uint8_t ib_port = ncclIbDevs[dev].port;
  int peer;
  NCCLCHECK(ncclIbCreateQp(ctx, ib_port, &comm->ibComm, &peer));
  assert(peer == 0);

  // Send my QP Info to receiver through the socket. Hope this won't block.
  struct ibv_port_attr portAttr;
  SYSCHECK(ibv_query_port(ctx, ib_port, &portAttr), "ibv_query_port");
  struct ncclIbQpInfo qpInfo;
  qpInfo.lid = portAttr.lid;
  qpInfo.ib_port = ib_port;
  qpInfo.qpn = comm->ibComm.peers[peer].qp->qp_num;
  NCCLCHECK(socketSend(comm->fd, &qpInfo, sizeof(qpInfo)));
  return 0;
}

int ncclIbAccept(void* recvComm) {
  struct ncclIbRecvComm* comm = (struct ncclIbRecvComm*)recvComm;
  struct sockaddr_in sockaddr;
  socklen_t socklen = sizeof(struct sockaddr_in);
  int peerfd;
  SYSCHECKVAL(accept(comm->listenFd, (struct sockaddr*)&sockaddr, &socklen), "accept", peerfd);
  ncclIbAddFd(comm, peerfd);
  struct ncclIbQpInfo remQpInfo;
  NCCLCHECK(socketReceive(peerfd, &remQpInfo, sizeof(remQpInfo)));

  // IB setup
  ibv_context* ctx = ncclIbDevs[comm->dev].context;
  uint8_t ib_port = ncclIbDevs[comm->dev].port;
  int peer;
  NCCLCHECK(ncclIbCreateQp(ctx, ib_port, &comm->ibComm, &peer));

  struct ibv_qp* qp = comm->ibComm.peers[peer].qp;
  NCCLCHECK(ncclIbRtrQp(qp, remQpInfo.qpn, remQpInfo.lid, remQpInfo.ib_port));
  NCCLCHECK(ncclIbRtsQp(qp));

  // Fill Handle
  struct ibv_port_attr portAttr;
  SYSCHECK(ibv_query_port(ctx, ib_port, &portAttr), "ibv_query_port");
  struct ncclIbQpInfo qpInfo;
  qpInfo.lid = portAttr.lid;
  qpInfo.ib_port = ib_port;
  qpInfo.qpn = qp->qp_num;

  NCCLCHECK(socketSend(peerfd, &qpInfo, sizeof(qpInfo)));
  return 0;
}

ncclResult_t ncclSendCheck(struct ncclIbSendComm* comm) {
  if (comm->ready == 0) {
    struct ncclIbQpInfo remQpInfo;
    struct ibv_qp* qp = comm->ibComm.peers[0].qp;
    NCCLCHECK(socketReceive(comm->fd, &remQpInfo, sizeof(remQpInfo)));
    NCCLCHECK(ncclIbRtrQp(qp, remQpInfo.qpn, remQpInfo.lid, remQpInfo.ib_port));
    NCCLCHECK(ncclIbRtsQp(qp));
    comm->ready = 1;
  }
  return ncclSuccess;
}

int ncclIbTest(void* request, int* done, int* size) {
  struct ibRequest *r = (struct ibRequest*)request;

  for (int wrDone = 1; wrDone;) {
    struct ibv_wc wc;
    SYSCHECKVAL(ibv_poll_cq(r->cq, 1, &wc), "ibv_poll_cq", wrDone);
    if (wrDone == 1) {
      struct ibRequest* doneReq = ncclIbRequests+wc.wr_id;
      //printf("Completion %d, status %d, opcode %d, err %d, len %d, qp_num %d\n", wc.wr_id, wc.status, wc.opcode, wc.vendor_err, wc.byte_len, wc.qp_num); 

      if (wc.opcode == IBV_WC_SEND) {
        struct ncclIbSendComm* sendComm = (struct ncclIbSendComm*)doneReq->comm;
        sendComm->numSendRequests--;
      }
      if (wc.opcode == IBV_WC_RECV) {
        struct ncclIbRecvComm* sendComm = (struct ncclIbRecvComm*)doneReq->comm;
        sendComm->numRecvRequests--;
      }
      if (wc.status != IBV_WC_SUCCESS) return 1;

      if (doneReq->mr != NULL) SYSCHECK(ibv_dereg_mr(doneReq->mr), "ibv_dereg_mr");
      doneReq->size = wc.byte_len;
      doneReq->done = 1;
    }
  }

  *done = 0;
  if (r->done == 1) {
    *done = 1;
    if (size) *size = r->size;
    ncclIbFreeRequest(r);
  }
  return 0;
}


int ncclIbIsend(void* sendComm, void* data, int size, int type, void** request) {
  if (type != NCCL_PTR_HOST) return 1;
  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)sendComm;
  NCCLCHECK(ncclSendCheck(comm));

  struct ibRequest* req = ncclIbGetRequest();
  req->done = 0;
  req->cq = comm->ibComm.cq;
  req->size = size;
  req->comm = sendComm;

  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = req-ncclIbRequests;

  if (size == 0) {
    wr.sg_list = NULL;
    wr.num_sge = 0;
    req->mr = NULL;
  } else {
    struct ibv_mr* mr;
    NULLCHECK(mr = ibv_reg_mr(comm->ibComm.pd, data, size, 0));
    struct ibv_sge sge = { .addr=(uintptr_t)data, .length=(unsigned int)size, .lkey=mr->lkey };
    wr.sg_list = &sge;
    wr.num_sge = 1;
    req->mr = mr;
  }
  wr.opcode = IBV_WR_SEND;

  // Wait for WR to be available in the Send Queue
  while (comm->numSendRequests == MAX_SEND_WR) { 
     int done = 0;
     /* This request is not even posted, but that should make the CQ progress */
     NCCLCHECK((ncclResult_t)ncclIbTest(req, &done, NULL));
     if (comm->numSendRequests == MAX_SEND_WR) sched_yield();
  }

  struct ibv_send_wr* bad_wr;
  SYSCHECK(ibv_post_send(comm->ibComm.peers[0].qp, &wr, &bad_wr), "ibv_post_send");
  comm->numSendRequests++;
  *request = req;
  return 0;
}

int ncclIbIrecv(void* recvComm, void* data, int size, int type, void** request) {
  if (type != NCCL_PTR_HOST) return 1;
  struct ncclIbRecvComm* comm = (struct ncclIbRecvComm*)recvComm;
  struct ibRequest* req = ncclIbGetRequest();
  req->done = 0;
  req->cq = comm->ibComm.cq;
  req->size = size;
  req->comm = recvComm;

  struct ibv_recv_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = req-ncclIbRequests;

  if (size == 0) {
    wr.sg_list = NULL;
    wr.num_sge = 0;
    req->mr = NULL;
  } else {
    struct ibv_mr* mr;
    NULLCHECK(mr = ibv_reg_mr(comm->ibComm.pd, data, size, IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE));
    struct ibv_sge sge = { .addr=(uintptr_t)data, .length=(unsigned int)size, .lkey=mr->lkey };
    wr.sg_list = &sge;
    wr.num_sge = 1;
    req->mr = mr;
  }

  // Wait for WR to be available in the SRQ
  while (comm->numRecvRequests == MAX_RECV_WR) { 
     int done = 0;
     /* This request is not even posted, but that should make the CQ progress */
     NCCLCHECK((ncclResult_t)ncclIbTest(req, &done, NULL));
     if (comm->numRecvRequests == MAX_RECV_WR) sched_yield();
  }

  struct ibv_recv_wr* bad_wr;
  SYSCHECK(ibv_post_srq_recv(comm->ibComm.srq, &wr, &bad_wr), "ibv_post_srq_recv");
  comm->numRecvRequests++;
  *request = req;
  return ncclSuccess;
}

int ncclIbCloseSend(void* sendComm) {
  if (sendComm) {
    struct ncclIbSendComm* comm = (struct ncclIbSendComm*)sendComm;
    close(comm->fd);
    free(comm);
  }
  return 0;
}

int ncclIbCloseRecv(void* recvComm) {
  if (recvComm) {
    struct ncclIbRecvComm* comm = (struct ncclIbRecvComm*)recvComm;
    for (int i=0; i<comm->npeersActive; i++) close(comm->fds[i].fd);
    free(comm->fds);
    free(comm);
  }
  return 0;
}

ncclNet_t ncclNetIb = {
  "IB",
  ncclIbPtrSupport,
  ncclIbDevices,
  ncclIbGetHandle,
  ncclIbConnectHandle,
  ncclIbAccept,
  ncclIbIsend,
  ncclIbIrecv,
  ncclIbTest,
  ncclIbCloseSend,
  ncclIbCloseRecv
};

bool ncclIbSupport() {
  initDevices();
  return ncclNIbDevs > 0;
}

