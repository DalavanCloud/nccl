#ifndef TRANSPORT_H_
#define TRANSPORT_H_

#include "nccl.h"
#include <stdint.h>

#define NTRANSPORTS 2

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
};

struct ncclTransportComm {
  ncclResult_t (*setup)(ncclTinfo_t*, ncclTinfo_t*, struct ncclConnect*, struct ncclRing*, int*);
  ncclResult_t (*connect)(struct ncclConnect*, struct ncclConnector*);
  ncclResult_t (*proxy)(struct ncclProxyArgs*);
};

struct ncclTransport {
  ncclResult_t (*fillInfo)(ncclTinfo_t*, int rank);
  struct ncclTransportComm send;
  struct ncclTransportComm recv;
};

ncclResult_t LaunchProxies(int substeps, int subchunks, int nsteps_per_round, int nblocks_per_round, int size, struct ncclComm* comm);
ncclResult_t WaitProxies(struct ncclComm* comm);
#endif
