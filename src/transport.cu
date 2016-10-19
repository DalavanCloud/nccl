#include "transport.h"

extern struct ncclTransport p2pTransport;
//extern struct ncclTransport qpiTransport;

struct ncclTransport ncclTransports[NTRANSPORTS] = {
  p2pTransport
  // qpiTransport,
  // socketTransport,
  // mpiTransport
};


