/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_DEBUG_H_
#define NCCL_DEBUG_H_

#include "core.h"
#include <stdio.h>

typedef enum {NONE=0, VERSION=1, WARN=2, INFO=3, ABORT=4} DebugLevel;
extern DebugLevel ncclDebugLevel;
extern pthread_mutex_t ncclDebugOutputLock;

#define WARN(...) do {                                           \
  if (ncclDebugLevel >= WARN) {                                  \
    pthread_mutex_lock(&ncclDebugOutputLock);                    \
    printf("WARN %s:%d ", __FILE__, __LINE__);                   \
    printf(__VA_ARGS__);                                         \
    printf("\n");                                                \
    fflush(stdout);                                              \
    pthread_mutex_unlock(&ncclDebugOutputLock);                  \
    if (ncclDebugLevel >= ABORT) abort();                        \
  }                                                              \
} while(0)

#define INFO(...) do {                                           \
  if (ncclDebugLevel >= INFO) {                                  \
    pthread_mutex_lock(&ncclDebugOutputLock);                    \
    printf("INFO "); printf(__VA_ARGS__); printf("\n");          \
    fflush(stdout);                                              \
    pthread_mutex_unlock(&ncclDebugOutputLock);                  \
  }                                                              \
} while(0)

extern int ncclPrintCRCs;

static void initDebug() {
  const char* nccl_debug = getenv("NCCL_DEBUG");
  if (nccl_debug == NULL) {
    ncclDebugLevel = NONE;
  } else if (strcmp(nccl_debug, "VERSION") == 0) {
    ncclDebugLevel = VERSION;
  } else if (strcmp(nccl_debug, "WARN") == 0) {
    ncclDebugLevel = WARN;
  } else if (strcmp(nccl_debug, "INFO") == 0) {
    ncclDebugLevel = INFO;
    INFO("NCCL debug level set to INFO");
  } else if (strcmp(nccl_debug, "ABORT") == 0) {
    ncclDebugLevel = ABORT;
    INFO("NCCL debug level set to ABORT");
  }

  const char* nccl_crc = getenv("NCCL_CRC");
  if (nccl_crc != NULL && strcmp(nccl_crc, "PRINT")==0 ) {
    ncclPrintCRCs = 1;
  } else {
    ncclPrintCRCs = 0;
  }

  pthread_mutex_init(&ncclDebugOutputLock, NULL);
}

#endif
