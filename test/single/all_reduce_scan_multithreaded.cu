/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 ************************************************************************/

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cerrno>
#include <pthread.h>
#include <dlfcn.h>
#include <float.h>
#include <sched.h>

#include "nccl.h"
#include "test_utilities.h"
#include <nvToolsExt.h>

void showUsage(const char* bin) {
  printf("\n"
         "Usage: %s <type> <op> <n_min> <n_max> [delta] [gpus] [gpu0 [gpu1 [...]]]\n"
         "Where:\n"
#ifdef CUDA_HAS_HALF
         "    type   =   [char|int|half|float|double|int64|uint64]\n"
#else
         "    type   =   [char|int|float|double|int64|uint64]\n"
#endif
         "    op     =   [sum|prod|max|min]\n"
         "    n_min  >   0\n"
         "    n_max  >=  n_min\n"
         "    delta  >   0\n\n", bin);
  return;
}


static pthread_barrier_t bar;
static pthread_mutex_t mut;
static ncclUniqueId commTag;

struct ParaArgs {
  int rank;
  int device;
  ncclDataType_t type;
  ncclRedOp_t op;
  int n_min;
  int n_max;
  int delta;
  int gpus;
  double* min_ms;
  double* max_ms;
  double* max_err;
  void*  refout;
};

typedef struct nvmlDevice_st* nvmlDevice_t;
class NvmlWrap {
  public:
  NvmlWrap();
  ~NvmlWrap();
  typedef enum {SUCCESS = 0} RetCode;
  RetCode (*DeviceGetHandleByPciBusId)(const char* pciBusId, nvmlDevice_t* device);
  RetCode (*DeviceSetCpuAffinity)(nvmlDevice_t device);
  RetCode (*DeviceClearCpuAffinity)(nvmlDevice_t device);
  const char* (*ErrorString)(RetCode r);

  private:
  void* dlHandle;
  RetCode (*ptrInit)(void);
  RetCode (*ptrShutdown)(void);
};

NvmlWrap* nvml;

#define NVMLCHECK(cmd) {                           \
  NvmlWrap::RetCode e = cmd;                       \
  if( e != NvmlWrap::SUCCESS ) {                   \
    printf("nvml failure %s:%d '%s'\n",            \
        __FILE__,__LINE__,nvml->ErrorString(e));   \
    exit(EXIT_FAILURE);                            \
  }                                                \
} while(0)

NvmlWrap::NvmlWrap() : dlHandle(NULL) {
  dlHandle = dlopen("libnvidia-ml.so.1", RTLD_NOW);
  if (!dlHandle) {
    dlHandle = dlopen("libnvidia-ml.so", RTLD_NOW);
    if (!dlHandle) {
      printf("Failed to open libnvidia-ml.so[.1]");
      exit(EXIT_FAILURE);
    }
  }

  #define LOAD_SYM(handle, symbol, funcptr) do {            \
    void** cast = (void**)&funcptr;                         \
    void* tmp = dlsym(handle, symbol);                      \
    if (tmp == NULL) {                                      \
      printf("dlsym failed on %s - %s", symbol, dlerror()); \
      exit(EXIT_FAILURE);                                   \
    }                                                       \
    *cast = tmp;                                            \
  } while (0)

  LOAD_SYM(dlHandle, "nvmlInit", this->ptrInit);
  LOAD_SYM(dlHandle, "nvmlShutdown", this->ptrShutdown);
  LOAD_SYM(dlHandle, "nvmlDeviceGetHandleByPciBusId", this->DeviceGetHandleByPciBusId);
  LOAD_SYM(dlHandle, "nvmlDeviceSetCpuAffinity", this->DeviceSetCpuAffinity);
  LOAD_SYM(dlHandle, "nvmlDeviceClearCpuAffinity", this->DeviceClearCpuAffinity);
  LOAD_SYM(dlHandle, "nvmlErrorString", this->ErrorString);
  NVMLCHECK(this->ptrInit());
}

NvmlWrap::~NvmlWrap() {
  NVMLCHECK(this->ptrShutdown());
  dlclose(dlHandle);
}

void* ParaFunc(void* vparg) {
  ParaArgs* args = (ParaArgs*)vparg;

  char busid[32] = {0};
  nvmlDevice_t nvmlHandle;
  CUDACHECK(cudaDeviceGetPCIBusId(busid, 32, args->device));
  NVMLCHECK(nvml->DeviceGetHandleByPciBusId(busid, &nvmlHandle));
  NVMLCHECK(nvml->DeviceSetCpuAffinity(nvmlHandle));
  printf("# Rank %d using device %d [%s]\n", args->rank, args->device, busid);

  //cpu_set_t affMask;
  //CPU_ZERO(&affMask);
  //int core = (args->device<4) ? args->device : args->device-4 + 12;
  //CPU_SET(core, &affMask);
  //printf("# Rank %d using core %d and device %d [%s]\n", args->rank, core, args->device, busid);

  CUDACHECK(cudaSetDevice(args->device));
  size_t word = wordSize(args->type);
  size_t maxSize = args->n_max * word;
  void *input, *output;
  double* localError;
  ncclComm_t comm;
  cudaStream_t stream;

  CUDACHECK(cudaMalloc(&input,  maxSize));
  CUDACHECK(cudaMalloc(&output, maxSize));
  CUDACHECK(cudaMallocHost(&localError, sizeof(double)));
  makeRandom(input, args->n_max, args->type, 42+args->rank);

  if (args->rank == 0)
    CUDACHECK(cudaMemcpy(args->refout, input, maxSize, cudaMemcpyDeviceToHost));
  else
    accVec(args->refout, input, args->n_max, args->type, args->op);

  pthread_mutex_unlock(&mut); // Allow other threads to reach InitRank()
  NCCLCHECK(ncclCommInitRank(&comm, args->gpus, commTag, args->rank));
  CUDACHECK(cudaStreamCreate(&stream));

  for(int n=args->n_min; n<=args->n_max; n+=args->delta) {
    size_t bytes = word * n;

    CUDACHECK(cudaMemsetAsync(output, 0, bytes, stream));
    CUDACHECK(cudaStreamSynchronize(stream));
    pthread_barrier_wait(&bar);

    auto start = std::chrono::high_resolution_clock::now();
    NCCLCHECK(ncclAllReduce(input, output, n, args->type, args->op, comm, stream));
    CUDACHECK(cudaStreamSynchronize(stream));
    auto stop = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration_cast<std::chrono::duration<double>>
        (stop - start).count() * 1000.0;

    maxDiff(localError, output, args->refout, n, args->type, stream);
    CUDACHECK(cudaStreamSynchronize(stream));

    pthread_mutex_lock(&mut);
    if (ms < *args->min_ms)
      *args->min_ms = ms;
    if (ms > *args->max_ms)
      *args->max_ms = ms;
    if (*localError > *args->max_err)
      *args->max_err = *localError;
    pthread_mutex_unlock(&mut);
    pthread_barrier_wait(&bar); // wait for all updates

    if (args->rank == 0) { // print rank
      double mb = (double)bytes * 1.e-6;
      double algbw = mb / *args->max_ms;
      double busbw = algbw * (double)(2*args->gpus - 2) / (double)args->gpus;
      printf("%12lu %5.0le %10.3lf %10.3lf %6.2lf %6.2lf\n",
        n*word, *args->max_err, *args->max_ms, *args->min_ms, algbw, busbw);

      *args->max_ms = -1.0;
      *args->min_ms = DBL_MAX;
      *args->max_err = -1.0;
    }
  }

  CUDACHECK(cudaStreamDestroy(stream));
  ncclCommDestroy(comm);
  CUDACHECK(cudaFree(input));
  CUDACHECK(cudaFree(output));
  CUDACHECK(cudaFreeHost(localError));
  pthread_exit(NULL);
}

int main(int argc, char* argv[]) {
  int rc;
  int nvis = 0;
  CUDACHECK(cudaGetDeviceCount(&nvis));
  if (nvis == 0) {
    printf("No GPUs found\n");
    showUsage(argv[0]);
    exit(EXIT_FAILURE);
  }
  nvml = new NvmlWrap();

  ncclDataType_t type;
  ncclRedOp_t op;
  int n_min;
  int n_max;
  int delta;
  int gpus;
  int* list = NULL;

  if (argc < 5) {
    showUsage(argv[0]);
    exit(EXIT_FAILURE);
  }

  type = strToType(argv[1]);
  if (type == nccl_NUM_TYPES) {
    printf("Invalid <type> '%s'\n", argv[1]);
    showUsage(argv[0]);
    exit(EXIT_FAILURE);
  }

  op = strToOp(argv[2]);
  if (op == nccl_NUM_OPS) {
    printf("Invalid <op> '%s'\n", argv[2]);
    showUsage(argv[0]);
    exit(EXIT_FAILURE);
  }

  n_min = strToPosInt(argv[3]);
  if (n_min < 1) {
    printf("Invalid <n_min> '%s'\n", argv[3]);
    showUsage(argv[0]);
    exit(EXIT_FAILURE);
  }

  n_max = strToPosInt(argv[4]);
  if (n_max < n_min) {
    printf("Invalid <n_max> '%s'\n", argv[4]);
    showUsage(argv[0]);
    exit(EXIT_FAILURE);
  }

  if (argc > 5) {
    delta = strToPosInt(argv[5]);
    if (delta < 1) {
      printf("Invalid <delta> '%s'\n", argv[5]);
      showUsage(argv[0]);
      exit(EXIT_FAILURE);
    }
  } else {
    delta = (n_max == n_min) ? 1 : (n_max - n_min+9) / 10;
  }

  if (argc > 6) {
    gpus = strToPosInt(argv[6]);
    if (gpus < 1) {
      printf("Invalid <gpus> '%s'\n", argv[6]);
      showUsage(argv[0]);
      exit(EXIT_FAILURE);
    }
  } else {
    gpus = nvis;
  }

  list = (int*)malloc(gpus*sizeof(int));

  if (argc > 7 && argc != 7+gpus) {
    printf("If given, GPU list must be fully specified.\n");
    showUsage(argv[0]);
    exit(EXIT_FAILURE);
  }

  for(int g=0; g<gpus; ++g) {
    if(argc > 7) {
      list[g] = strToNonNeg(argv[7+g]);
      if (list[g] < 0) {
        printf("Invalid GPU%d '%s'\n", g, argv[7+g]);
        showUsage(argv[0]);
        exit(EXIT_FAILURE);
      } else if (list[g] >= nvis) {
        printf("GPU%d (%d) exceeds visible devices (%d)\n", g, list[g], nvis);
        showUsage(argv[0]);
        exit(EXIT_FAILURE);
      }
    } else {
      list[g] = g % nvis;
    }
  }

  size_t max_size = n_max * wordSize(type);
  void* refout;
  CUDACHECK(cudaMallocHost(&refout, max_size));
  pthread_t* threads = (pthread_t*)malloc(gpus*sizeof(pthread_t));
  ParaArgs* threadArg = (ParaArgs*)malloc(gpus*sizeof(ParaArgs));
  double* minmax = (double*)malloc(3*sizeof(double));
  minmax[0] = DBL_MAX;
  minmax[1] = -1.0;
  minmax[2] = -1.0;

  NCCLCHECK(ncclGetUniqueId(&commTag));
  rc = pthread_barrier_init(&bar, NULL, gpus);
  if (rc) {
    printf("Failed to initialize pthread barrier - %d\n", rc);
    exit(EXIT_FAILURE);
  }
  rc = pthread_mutex_init(&mut, NULL);
  if (rc) {
    printf("Failed to initialize pthread mutex - %d\n", rc);
    exit(EXIT_FAILURE);
  }

  for(int i=0; i<gpus; ++i) {
    threadArg[i].rank = i;
    threadArg[i].device = list[i];
    threadArg[i].type = type;
    threadArg[i].op = op;
    threadArg[i].n_min = n_min;
    threadArg[i].n_max = n_max;
    threadArg[i].delta = delta;
    threadArg[i].gpus = gpus;
    threadArg[i].min_ms = minmax;
    threadArg[i].max_ms = minmax+1;
    threadArg[i].max_err = minmax+2;
    threadArg[i].refout = refout;

    pthread_mutex_lock(&mut); // released in ParaFunc
    rc = pthread_create(threads+i, NULL, ParaFunc, (void*)(threadArg+i));
    if (rc) {
      printf("Error creating thread %d: %d\n", i, rc);
      exit(EXIT_FAILURE);
    }
  }

  for(int i=0; i<gpus; ++i) {
    pthread_join(threads[i], NULL);
  }

  pthread_mutex_destroy(&mut);
  if (rc) {
    printf("Failed to destroy pthread mutex - %d\n", rc);
    exit(EXIT_FAILURE);
  }
  pthread_barrier_destroy(&bar);
  if (rc) {
    printf("Failed to destroy pthread barrier - %d\n", rc);
    exit(EXIT_FAILURE);
  }

  CUDACHECK(cudaFreeHost(refout));
  free(minmax);
  free(threadArg);
  free(threads);
  delete nvml;
  exit(EXIT_SUCCESS);
}

