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
#include <ctype.h>

#include "infiniband/verbs.h"

#define USE_RDMA_WRITE 1
#define MAX_IF_NAME_SIZE 16
#define MAXPATHSIZE 1024
static char ncclIbIfName[MAX_IF_NAME_SIZE];
static struct in_addr ncclIbIfAddr;
static int ncclNIbDevs = -1;
struct ncclIbDev {
  int device;
  uint8_t port;
  ibv_context* context;
  char devPath[MAXPATHSIZE];
};

#define MAX_IB_DEVS 16
struct ncclIbDev ncclIbDevs[MAX_IB_DEVS];
int ncclIbTimeout = 14;
pthread_mutex_t ncclIbLock = PTHREAD_MUTEX_INITIALIZER;

pthread_t ncclIbAsyncThread;
static void* ncclIbAsyncThreadMain(void* args) {
  struct ibv_context* context = (struct ibv_context*)args;
  while (1) {
    struct ibv_async_event event;
    int ret = ibv_get_async_event(context, &event);
    if (ret != 0) break;
    if (event.event_type != IBV_EVENT_COMM_EST)
      WARN("IB Got async event : %s", ibv_event_type_str(event.event_type));
    ibv_ack_async_event(&event);
  }
  return NULL;
}

ncclResult_t getCudaPath(int cudaDev, char** path) {
  char busId[16];
  CUDACHECK(cudaDeviceGetPCIBusId(busId, 16, cudaDev));
  for (int i=0; i<16; i++) busId[i] = tolower(busId[i]);
  char busPath[] =  "/sys/class/pci_bus/0000:00";
  memcpy(busPath+sizeof("/sys/class/pci_bus/")-1, busId, sizeof("0000:00")-1); 

  char pathname[MAXPATHSIZE];
  strcpy(pathname, "/sys/class/pci_bus/");
  int strLen = strlen(pathname);
  int linkLen = readlink(busPath, pathname+strLen, MAXPATHSIZE-strLen);
  if (linkLen == 0) {
    WARN("Could not find link %s", path);
    return ncclSystemError;
  }
  // readlink does not append '\0'. We have to do it.
  pathname[strLen+linkLen] = '\0';
  strncpy(pathname+strlen(pathname), "/device", MAXPATHSIZE-strlen(pathname));
  char* cudaRpath = realpath(pathname, NULL); 
  strncpy(pathname, cudaRpath, MAXPATHSIZE);
  strncpy(pathname+strlen(pathname), "/", MAXPATHSIZE-strlen(pathname));
  strncpy(pathname+strlen(pathname), busId, MAXPATHSIZE-strlen(pathname));
  free(cudaRpath);
  *path = realpath(pathname, NULL); 
  if (*path == NULL) {
    WARN("Could not find real path of %s", pathname);
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t getMlxPath(char* ibdevPath, char** path) {
  char pathname[MAXPATHSIZE];
  strcpy(pathname, "/sys/class/infiniband/");
  int strLen = strlen(pathname);
  int linkLen = readlink(ibdevPath, pathname+strLen, MAXPATHSIZE-strLen);
  if (linkLen == 0) {
    WARN("Could not find link %s", ibdevPath);
    return ncclSystemError;
  }
  // readlink does not append '\0'. We have to do it.
  pathname[strLen+linkLen] = '\0';
  strncpy(pathname+strlen(pathname), "/../..", MAXPATHSIZE-strlen(pathname));
  *path = realpath(pathname, NULL); 
  if (*path == NULL) {
    WARN("Could not find real path of %s", pathname);
    return ncclSystemError;
  }
  return ncclSuccess;
}

enum ncclIbPathDist {
  PATH_PHB = 0,
  PATH_PXB = 1,
  PATH_PIX = 2,
  PATH_SOC = 3
};

static const char* pathDists[] = { "PHB", "PXB", "PIX", "SOC" };

int pciDistance(char* path1, char* path2) {
  int score = 0;
  int depth = 0;
  int same = 1;
  for (int i=0; i<strlen(path1); i++) {
    if (path1[i] != path2[i]) same = 0;
    if (path1[i] == '/') {
      depth++;
      if (same == 1) score++;
    }
  }
  if (score == 3) return PATH_SOC;
  if (score == 4) return PATH_PIX;
  if (score == depth-1)     return PATH_PHB;
  return PATH_PXB;
}

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
              ncclIbDevs[ncclNIbDevs].device = d;
              ncclIbDevs[ncclNIbDevs].port = port;
              ncclIbDevs[ncclNIbDevs].context = context;
              strncpy(ncclIbDevs[ncclNIbDevs].devPath, devices[d]->ibdev_path, MAXPATHSIZE);
              INFO("IB device %d : %s / port %d", d, ibv_get_device_name(devices[d]), port);
              ncclNIbDevs++;
              found++;
              pthread_create(&ncclIbAsyncThread, NULL, ncclIbAsyncThreadMain, context);
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

int ncclIbDevices(int* ndev, int** scores) {
  initDevices();
  *ndev = ncclNIbDevs;
  int cudaDev;
  cudaGetDevice(&cudaDev);
  char* cudaPath;
  getCudaPath(cudaDev, &cudaPath);
  int* sc = (int*)malloc(ncclNIbDevs*sizeof(int));
  for (int d=0; d<ncclNIbDevs; d++) {
    char* mlxPath;
    getMlxPath(ncclIbDevs[d].devPath, &mlxPath);
    int distance = (mlxPath == NULL || cudaPath == NULL) ? PATH_SOC : pciDistance(mlxPath, cudaPath);
    INFO("IB device %d, distance from %d : %s", d, cudaDev, pathDists[distance]);
    sc[d] = 1+PATH_SOC-distance;
    free(mlxPath);
  }
  free(cudaPath);
  *scores = sc;
  return ncclSuccess;
}

int ncclIbPtrSupport(int dev, int* supportedTypes) {
  initDevices();
  *supportedTypes = NCCL_PTR_HOST;
  int ibGdrEnabled = 0;
  char* str = getenv("NCCL_IB_CUDA_SUPPORT");
  if (str && atoi(str) > 0) {
    ibGdrEnabled = 1;
  } else { // auto detect
    int cudaDev;
    cudaGetDevice(&cudaDev);
    char* cudaPath;
    getCudaPath(cudaDev, &cudaPath);
    char* mlxPath;
    getMlxPath(ncclIbDevs[dev].devPath, &mlxPath);
    int distance = (mlxPath == NULL || cudaPath == NULL) ? PATH_SOC : pciDistance(mlxPath, cudaPath);
    free(mlxPath); free(cudaPath);
    if (distance <= PATH_PIX) ibGdrEnabled = 1;
  }
  if (ibGdrEnabled == 1) *supportedTypes |= NCCL_PTR_CUDA;
  return 0;
}

static ncclResult_t GetIpAddr(struct in_addr* addr) {
  if (ncclNIbDevs == -1) initDevices();
  memcpy(addr, &ncclIbIfAddr, sizeof(struct in_addr));
  return ncclSuccess;
}

#define MAX_REQUESTS 64

struct ncclIbQpInfo {
  int lid;
  uint8_t ib_port;
  int qpn;
  uint32_t fifoRkey;
  uint64_t fifoAddr;
};

struct ncclIbHandle {
  struct socketAddress connectAddr;
  struct ncclIbQpInfo qpInfo;
};

struct ncclIbVerbs {
  struct ibv_pd* pd;
  struct ibv_comp_channel* cc;
  struct ibv_cq* cq;
  struct ibv_qp* qp;
  int numRequests;
  struct ibv_mr* mrPool[MAX_REQUESTS];
  int mrRotation;
};

struct ncclIbRequest {
  int used;
  struct ncclIbVerbs* verbs;
  struct ibv_mr* mr;
  int done;
  int size;
};

struct ncclIbListenComm {
  int dev;
  int fd;
};

struct ncclIbReqs {
  int nreqs;
  struct ncclIbRequest* requests;
};

struct ncclIbSendFifo {
  uint64_t addr;
  uint32_t rkey;
  int ready;
};

struct ncclIbSendComm {
  int fd;
  int ready;
  struct ncclIbVerbs verbs;
  struct ncclIbReqs reqs;
  struct ncclIbSendFifo fifo[MAX_REQUESTS];
  struct ibv_mr* fifoMr;
  int fifoHead;
};

struct ncclIbRecvComm {
  int fd;
  int ready;
  struct ncclIbVerbs verbs;
  struct ncclIbReqs reqs;
  uint32_t remFifoRkey;
  uint64_t remFifoAddr;
  int remFifoTail;
  struct ncclIbSendFifo fifoElem;
  struct ibv_mr* fifoElemMr;
  struct ibv_sge fifoSge;
};

#define NULLCHECK(cmd) \
  if ((cmd) == NULL) { \
    WARN("IBV call return NULL\n"); \
  }

ncclResult_t ncclIbCreateQp(ibv_context* ctx, uint8_t ib_port, struct ncclIbVerbs* verbs) {
  NULLCHECK(verbs->pd = ibv_alloc_pd(ctx));
  NULLCHECK(verbs->cc = ibv_create_comp_channel(ctx));
  NULLCHECK(verbs->cq = ibv_create_cq(ctx, MAX_REQUESTS, NULL, verbs->cc, 0));

  struct ibv_qp_init_attr qpInitAttr;
  memset(&qpInitAttr, 0, sizeof(struct ibv_qp_init_attr));
  qpInitAttr.send_cq = verbs->cq;
  qpInitAttr.recv_cq = verbs->cq;
  qpInitAttr.qp_type = IBV_QPT_RC;
  qpInitAttr.cap.max_send_wr = MAX_REQUESTS;
  qpInitAttr.cap.max_recv_wr = MAX_REQUESTS;
  qpInitAttr.cap.max_send_sge = 1;
  qpInitAttr.cap.max_recv_sge = 1;
  qpInitAttr.cap.max_inline_data = 0;
  NULLCHECK(verbs->qp = ibv_create_qp(verbs->pd, &qpInitAttr));
  struct ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
  qpAttr.qp_state = IBV_QPS_INIT;
  qpAttr.pkey_index = 0;
  qpAttr.port_num = ib_port;
  qpAttr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
  SYSCHECK(ibv_modify_qp(verbs->qp, &qpAttr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS), "ibv_modify_qp");
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
  //qpAttr.min_rnr_timer = 12;
  qpAttr.min_rnr_timer = 1;
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
  //qpAttr.rnr_retry = 7;
  qpAttr.rnr_retry = 1;
  qpAttr.sq_psn = 0;
  qpAttr.max_rd_atomic = 1;
  SYSCHECK(ibv_modify_qp(qp, &qpAttr, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC), "ibv_modify_qp");
  return ncclSuccess;
}


int ncclIbListen(int dev, void* opaqueHandle, void** listenComm) {
  struct ncclIbListenComm* comm = (struct ncclIbListenComm*)malloc(sizeof(struct ncclIbListenComm));
  struct ncclIbHandle* handle = (struct ncclIbHandle*) opaqueHandle;
  static_assert(sizeof(struct ncclIbHandle) < NCCL_NET_HANDLE_MAXSIZE, "ncclIbHandle size too large");
  comm->dev = dev;
  NCCLCHECK(GetIpAddr(&(handle->connectAddr.ip_addr)));
  NCCLCHECK(createListenSocket(&comm->fd, handle->connectAddr.ip_addr, &handle->connectAddr.port));
  *listenComm = comm;
  return 0;
}

int ncclIbConnect(int dev, void* opaqueHandle, void** sendComm) {
  //printf("ncclIbConnect\n");
  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)malloc(sizeof(struct ncclIbSendComm));
  memset(comm, 0, sizeof(struct ncclIbSendComm));
  struct ncclIbHandle* handle = (struct ncclIbHandle*) opaqueHandle;
  NCCLCHECK(connectAddress(&handle->connectAddr, ncclIbIfAddr, &comm->fd));
  *sendComm = comm;
  
  // IB Setup
  initDevices(); /*XXX:Anshuman added on 3/2/2017*/
  ibv_context* ctx = ncclIbDevs[dev].context;
  uint8_t ib_port = ncclIbDevs[dev].port;
  //printf("[ncclIbConnect] ncclIbCreateQp\n");
  NCCLCHECK(ncclIbCreateQp(ctx, ib_port, &comm->verbs));

  // Send my QP Info to receiver through the socket. Hope this won't block.
  struct ibv_port_attr portAttr;
  //printf("[ncclIbConnect] ibv_query_port\n");
  SYSCHECK(ibv_query_port(ctx, ib_port, &portAttr), "ibv_query_port");
  struct ncclIbQpInfo qpInfo;
  qpInfo.lid = portAttr.lid;
  qpInfo.ib_port = ib_port;
  qpInfo.qpn = comm->verbs.qp->qp_num;

  // Prepare my fifo
  NULLCHECK(comm->fifoMr = ibv_reg_mr(comm->verbs.pd, comm->fifo, sizeof(struct ncclIbSendFifo)*MAX_REQUESTS, IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ));
  qpInfo.fifoRkey = comm->fifoMr->rkey;
  qpInfo.fifoAddr = (uint64_t)comm->fifo;
   
  NCCLCHECK(socketSend(comm->fd, &qpInfo, sizeof(qpInfo)));
  return 0;
}

int ncclIbAccept(void* listenComm, void** recvComm) {
  //printf("ncclIbAccept\n");
  struct ncclIbListenComm* lComm = (struct ncclIbListenComm*)listenComm;
  struct ncclIbRecvComm* rComm = (struct ncclIbRecvComm*)malloc(sizeof(struct ncclIbRecvComm));
  memset(rComm, 0, sizeof(struct ncclIbRecvComm));
  
  struct sockaddr_in sockaddr;
  socklen_t socklen = sizeof(struct sockaddr_in);
  SYSCHECKVAL(accept(lComm->fd, (struct sockaddr*)&sockaddr, &socklen), "accept", rComm->fd);
  struct ncclIbQpInfo remQpInfo;
  NCCLCHECK(socketReceive(rComm->fd, &remQpInfo, sizeof(remQpInfo)));

  // IB setup
  ibv_context* ctx = ncclIbDevs[lComm->dev].context;
  uint8_t ib_port = ncclIbDevs[lComm->dev].port;
  //printf("[ncclIbAccept] ncclIbCreateQp\n");
  NCCLCHECK(ncclIbCreateQp(ctx, ib_port, &rComm->verbs));

  //printf("[ncclIbAccept] ncclIbRtrQp\n");
  struct ibv_qp* qp = rComm->verbs.qp;
  NCCLCHECK(ncclIbRtrQp(qp, remQpInfo.qpn, remQpInfo.lid, remQpInfo.ib_port));
  NCCLCHECK(ncclIbRtsQp(qp));

  // Retain remote fifo info and prepare my RDMA ops
  rComm->remFifoRkey = remQpInfo.fifoRkey;
  rComm->remFifoAddr = remQpInfo.fifoAddr;
  NULLCHECK(rComm->fifoElemMr = ibv_reg_mr(rComm->verbs.pd, &rComm->fifoElem, sizeof(struct ncclIbSendFifo), IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_READ));
  rComm->fifoSge.addr = (uint64_t)&rComm->fifoElem;
  rComm->fifoSge.length = sizeof(struct ncclIbSendFifo);
  rComm->fifoSge.lkey = rComm->fifoElemMr->lkey;

  // Fill Handle
  struct ibv_port_attr portAttr;
  SYSCHECK(ibv_query_port(ctx, ib_port, &portAttr), "ibv_query_port");
  struct ncclIbQpInfo qpInfo;
  qpInfo.lid = portAttr.lid;
  qpInfo.ib_port = ib_port;
  qpInfo.qpn = qp->qp_num;

  NCCLCHECK(socketSend(rComm->fd, &qpInfo, sizeof(qpInfo)));
  *recvComm = rComm;
  return 0;
}

struct ncclIbRequest* ncclIbGetRequest(struct ncclIbReqs* reqs, struct ncclIbVerbs* verbs) {
  for (int i=0; i<reqs->nreqs; i++) {
    struct ncclIbRequest* req = reqs->requests+i;
    if (req->used == 0) {
      req->used = 1;
      req->mr = NULL;
      req->done = 0;
      req->size = 0;
      req->verbs = verbs;
      return req;
    }
  }
  // No free request found, grow the pool
  int newNumRequests = reqs->nreqs + 32;
  reqs->requests = (struct ncclIbRequest*)realloc(reqs->requests, newNumRequests*sizeof(struct ncclIbRequest));
  for (int i=reqs->nreqs; i<newNumRequests; i++)
    reqs->requests[i].used = 0;
  reqs->nreqs = newNumRequests;
  return ncclIbGetRequest(reqs, verbs);
}

ncclResult_t ncclSendCheck(struct ncclIbSendComm* comm) {
  if (comm->ready == 0) {
    struct ncclIbQpInfo remQpInfo;
    struct ibv_qp* qp = comm->verbs.qp;
    NCCLCHECK(socketReceive(comm->fd, &remQpInfo, sizeof(remQpInfo)));
    NCCLCHECK(ncclIbRtrQp(qp, remQpInfo.qpn, remQpInfo.lid, remQpInfo.ib_port));
    NCCLCHECK(ncclIbRtsQp(qp));
    int go = 1;
    NCCLCHECK(socketSend(comm->fd, &go, sizeof(go)));
    comm->ready = 1;
  }
  return ncclSuccess;
}

ncclResult_t ncclRecvCheck(struct ncclIbRecvComm* comm) {
  if (comm->ready == 0) {
    int go;
    NCCLCHECK(socketReceive(comm->fd, &go, sizeof(go)));
    comm->ready = 1;
  }
  return ncclSuccess;
}

int ncclIbTest(void* request, int* done, int* size) {
  struct ncclIbRequest *r = (struct ncclIbRequest*)request;

  for (int wrDone = 1; wrDone;) {
    struct ibv_wc wc;
    SYSCHECKVAL(ibv_poll_cq(r->verbs->cq, 1, &wc), "ibv_poll_cq", wrDone);
    if (wrDone == 1) {
      //printf("Got completion opcode %d, status %d, wr_id %p, size %d\n", wc.opcode, wc.status, wc.wr_id, wc.byte_len);
      if (wc.status != IBV_WC_SUCCESS) {
        WARN("NET/IB : Got completion with error %d, opcode %d, vendor err %d", wc.status, wc.opcode, wc.vendor_err);
        return 1;
      }
      r->verbs->numRequests--;

      struct ncclIbRequest* doneReq = (struct ncclIbRequest*)wc.wr_id;
      if (doneReq) {
        if (wc.opcode == IBV_WC_RECV) {
          doneReq->size = wc.byte_len;
#ifdef USE_RDMA_WRITE
        } else if (wc.opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
          doneReq->size = wc.imm_data;
#endif
        }
        doneReq->done = 1;
      }  
    }
  }

  *done = 0;
  if (r->done == 1) {
    *done = 1;
    if (size) *size = r->size;
    r->used = 0;
  }
  return 0;
}

// Cache previous MRs to avoid registering/unregistering for each Isend/Irecv
ncclResult_t ncclIbGetMr(struct ncclIbVerbs* verbs, void* data, int size, struct ibv_mr** mrRet) {
  for (int i=0; i<MAX_REQUESTS;i++) {
    if (verbs->mrPool[i] && verbs->mrPool[i]->addr == data && verbs->mrPool[i]->length == size) { 
      *mrRet = verbs->mrPool[i];
      return ncclSuccess;
    }
  }
  int elem = (verbs->mrRotation++)%MAX_REQUESTS;
  if (verbs->mrPool[elem]) SYSCHECK(ibv_dereg_mr(verbs->mrPool[elem]), "ibv_dereg_mr");
  NULLCHECK(verbs->mrPool[elem] = ibv_reg_mr(verbs->pd, data, size, IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE));
  *mrRet = verbs->mrPool[elem];
  return ncclSuccess;
}

int ncclIbIsend(void* sendComm, void* data, int size, int type, void** request) {
  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)sendComm;
  NCCLCHECK(ncclSendCheck(comm));

  struct ncclIbRequest* req = ncclIbGetRequest(&comm->reqs, &comm->verbs);
  req->done = 0;
  req->size = size;
  req->verbs = &comm->verbs;

  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = (uint64_t)req;

  struct ibv_sge sge;
  if (size == 0) {
    wr.sg_list = NULL;
    wr.num_sge = 0;
    req->mr = NULL;
  } else {
    struct ibv_mr* mr;
    NCCLCHECK(ncclIbGetMr(&comm->verbs, data, size, &mr));
    sge.addr=(uintptr_t)data; sge.length=(unsigned int)size; sge.lkey=mr->lkey;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    req->mr = mr;
  }
  wr.opcode = IBV_WR_SEND;
  wr.send_flags = IBV_SEND_SIGNALED;

  // Wait for WR to be available in the Send Queue
  while (comm->verbs.numRequests == MAX_REQUESTS) { 
     int done = 0;
     /* This request is not even posted, but that should make the CQ progress */
     NCCLCHECK((ncclResult_t)ncclIbTest(req, &done, NULL));
     if (comm->verbs.numRequests == MAX_REQUESTS) sched_yield();
  }

  // Wait for receiver to have posted the recv
  volatile struct ncclIbSendFifo* slot = comm->fifo + (comm->fifoHead%MAX_REQUESTS);
  while (slot->ready == 0) sched_yield();
#ifdef USE_RDMA_WRITE
  wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  wr.wr.rdma.remote_addr = slot->addr;
  wr.wr.rdma.rkey = slot->rkey;
  wr.imm_data = size;
#endif
  slot->ready = 0;
  comm->fifoHead++;


  struct ibv_send_wr* bad_wr;
  SYSCHECK(ibv_post_send(comm->verbs.qp, &wr, &bad_wr), "ibv_post_send");
  comm->verbs.numRequests++;
  *request = req;
  return 0;
}

ncclResult_t ncclIbPostFifo(struct ncclIbRecvComm* comm, uint32_t rkey, uint64_t addr) {
  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  struct ncclIbRequest* req = ncclIbGetRequest(&comm->reqs, &comm->verbs);
  wr.wr_id = (uint64_t)req;

  comm->fifoElem.addr = addr;
  comm->fifoElem.rkey = rkey;
  comm->fifoElem.ready = 1;
  wr.wr.rdma.remote_addr = comm->remFifoAddr + (comm->remFifoTail % MAX_REQUESTS) * sizeof(struct ncclIbSendFifo);
  wr.wr.rdma.rkey = comm->remFifoRkey;
  wr.sg_list = &comm->fifoSge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.send_flags = IBV_SEND_SIGNALED;

  // Wait for WR to be available in the RQ
  while (comm->verbs.numRequests == MAX_REQUESTS) { 
     int done = 0;
     /* This request is not even posted, but that should make the CQ progress */
     NCCLCHECK((ncclResult_t)ncclIbTest(req, &done, NULL));
     if (comm->verbs.numRequests == MAX_REQUESTS) sched_yield();
  }

  struct ibv_send_wr* bad_wr;
  SYSCHECK(ibv_post_send(comm->verbs.qp, &wr, &bad_wr), "ibv_post_send");
  comm->verbs.numRequests++;
  comm->remFifoTail++;

  while (req->done == 0) {
    int done;
    NCCLCHECK((ncclResult_t)ncclIbTest(req, &done, NULL));
  }
  req->used = 0;
  
  return ncclSuccess;
}

int ncclIbIrecv(void* recvComm, void* data, int size, int type, void** request) {
  struct ncclIbRecvComm* comm = (struct ncclIbRecvComm*)recvComm;
  struct ncclIbRequest* req = ncclIbGetRequest(&comm->reqs, &comm->verbs);
  NCCLCHECK(ncclRecvCheck(comm));
  req->done = 0;
  req->size = size;
  req->verbs = &comm->verbs;

  struct ibv_recv_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = (uint64_t)req;

  struct ibv_sge sge;
  if (size == 0) {
    wr.sg_list = NULL;
    wr.num_sge = 0;
    req->mr = NULL;
  } else {
    struct ibv_mr* mr;
    NCCLCHECK(ncclIbGetMr(&comm->verbs, data, size, &mr));
    sge.addr=(uintptr_t)data; sge.length=(unsigned int)size; sge.lkey=mr->lkey;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    req->mr = mr;
  }

  // Wait for WR to be available in the RQ
  while (comm->verbs.numRequests == MAX_REQUESTS) { 
     int done = 0;
     /* This request is not even posted, but that should make the CQ progress */
     NCCLCHECK((ncclResult_t)ncclIbTest(req, &done, NULL));
     if (comm->verbs.numRequests == MAX_REQUESTS) sched_yield();
  }

  struct ibv_recv_wr* bad_wr;
  SYSCHECK(ibv_post_recv(comm->verbs.qp, &wr, &bad_wr), "ibv_post_recv");
  comm->verbs.numRequests++;
  *request = req;

  // Post to FIFO to notify sender
  NCCLCHECK(ncclIbPostFifo(comm, req->mr->rkey, (uint64_t)data));
  return ncclSuccess;
}

int ncclIbCloseSend(void* sendComm) {
  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)sendComm;
  if (comm) {
    free(comm->reqs.requests);
    close(comm->fd);
    if (comm->fifoMr != NULL) SYSCHECK(ibv_dereg_mr(comm->fifoMr), "ibv_dereg_mr");
    for (int i=0; i<MAX_REQUESTS; i++) {
      if (comm->verbs.mrPool[i] != NULL) SYSCHECK(ibv_dereg_mr(comm->verbs.mrPool[i]), "ibv_dereg_mr");
    }
    free(comm);
  }
  return 0;
}

int ncclIbCloseRecv(void* recvComm) {
  struct ncclIbRecvComm* comm = (struct ncclIbRecvComm*)recvComm;
  if (comm) {
    free(comm->reqs.requests);
    close(comm->fd);
    if (comm->fifoElemMr != NULL) SYSCHECK(ibv_dereg_mr(comm->fifoElemMr), "ibv_dereg_mr");
    for (int i=0; i<MAX_REQUESTS; i++) {
      if (comm->verbs.mrPool[i] != NULL) SYSCHECK(ibv_dereg_mr(comm->verbs.mrPool[i]), "ibv_dereg_mr");
    }
    free(comm);
  }
  return 0;
}

int ncclIbCloseListen(void* listenComm) {
  struct ncclIbListenComm* comm = (struct ncclIbListenComm*)listenComm;
  if (comm) {
    close(comm->fd);
    free(comm);
  }
  return 0;
}

ncclNet_t ncclNetIb = {
  "IB",
  ncclIbDevices,
  ncclIbPtrSupport,
  ncclIbListen,
  ncclIbConnect,
  ncclIbAccept,
  ncclIbIsend,
  ncclIbIrecv,
  ncclIbTest,
  ncclIbCloseSend,
  ncclIbCloseRecv,
  ncclIbCloseListen
};

bool ncclIbSupport() {
  initDevices();
  return ncclNIbDevs > 0;
}

