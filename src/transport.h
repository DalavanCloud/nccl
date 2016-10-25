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
  void (*fillInfo)(ncclTinfo_t*);
  struct ncclTransportComm send;
  struct ncclTransportComm recv;
};

#include <unistd.h>

static void getHostName(char* hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i=0; i< maxlen; i++) {
    if (hostname[i] == '.') {
      hostname[i] = '\0';
      return;
    }
  }
}

static uint64_t getHostHash(const char* string) {
  // Based on DJB2, result = result * 33 + char
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++){
    result = ((result << 5) + result) + string[c];
  }
  return result;
}

#include <string.h>

static int getHostNumber(const char* string) {
  int result = 0;
  int len = strlen(string);
  for (int offset = len-1; offset >= 0; offset --) {
   int res = atoi(string+offset);
   if (res <= 0)
     break;
   result = res;
  }
  return result;
}

ncclResult_t LaunchProxies(int substeps, int subchunks, int nsteps_per_round, int nblocks_per_round, int size, struct ncclComm* comm);
ncclResult_t WaitProxies(struct ncclComm* comm);
#endif
