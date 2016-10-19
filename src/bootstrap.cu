#include "bootstrap.h"
#include "bootstrap/socket.h"

#define NBOOTSTRAPS 1
struct ncclBootstrap bootstraps[NBOOTSTRAPS] = {
  bootstrapSocket
};

ncclResult_t bootstrapGetUniqueId(ncclUniqueId* out) {
  for (int b=0; b<NBOOTSTRAPS; b++) {
    if (bootstraps[b].getUniqueId(out) == ncclSuccess)
      return ncclSuccess;
  }
  return ncclInternalError;
}

struct ncclBootstrap* bootstrapInit(ncclUniqueId* id, int rank, int nranks) {
  struct ncclBootstrap* bootstrap = NULL;
  for (int b=0; b<NBOOTSTRAPS; b++) {
    bootstrap = bootstraps[b].init(id, rank, nranks);
    if (bootstrap != NULL)
      break;
  }
  return bootstrap;
}

