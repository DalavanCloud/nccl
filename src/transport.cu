#include "transport.h"

extern struct ncclTransport p2pTransport;
extern struct ncclTransport shmTransport;

struct ncclTransport ncclTransports[NTRANSPORTS] = {
  p2pTransport,
  shmTransport
  // socketTransport,
  // mpiTransport
};


