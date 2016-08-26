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
#include <vector>
#include <sys/time.h>

#include "nccl.h"
#include "test_utilities.h"

#define MAX_POW2 27 // 128 MB
#define NREPS 1

int nDev;
int* devList;
ncclComm_t* comms;
cudaStream_t* streams;
char** sendbuff;
char** recvbuff;
char* reference;
char* results;

int typesSizes[nccl_NUM_TYPES];
double typesDeltas[nccl_NUM_TYPES];

double CheckTypeDelta(int type, char* devmem, char* ref, int count) {
  switch(type) {
    case ncclChar:   return CheckDelta<char>(devmem, ref, count);
    case ncclInt:    return CheckDelta<int>((int*)devmem, (int*)ref, count);
#ifdef CUDA_HAS_HALF
    case ncclHalf:   return CheckDelta<half>((half*)devmem, (half*)ref, count);
#endif
    case ncclFloat:  return CheckDelta<float>((float*)devmem, (float*)ref, count);
    case ncclDouble: return CheckDelta<double>((double*)devmem, (double*)ref, count);
    case ncclInt64:  return CheckDelta<long long>((long long*)devmem, (long long*)ref, count);
    case ncclUint64: return CheckDelta<unsigned long long>((unsigned long long*)devmem, (unsigned long long*)ref, count);
  }
  return 0.0;
}

void AccumulateType(int type, char* ref, char*devmem, int count, ncclRedOp_t op) {
  switch(type) {
    case ncclChar:   return Accumulate<char>(ref, devmem, count, op);
    case ncclInt:    return Accumulate<int>((int*)ref, (int*)devmem, count, op);
#ifdef CUDA_HAS_HALF
    case ncclHalf:   return Accumulate<half>((half*)ref, (half*)devmem, count, op);
#endif
    case ncclFloat:  return Accumulate<float>((float*)ref, (float*)devmem, count, op);
    case ncclDouble: return Accumulate<double>((double*)ref, (double*)devmem, count, op);
    case ncclInt64:  return Accumulate<long long>((long long*)ref, (long long*)devmem, count, op);
    case ncclUint64: return Accumulate<unsigned long long>((unsigned long long*)ref, (unsigned long long*)devmem, count, op);
  }
}

void RandomizeType(int type, char* devmem, int count, int seed) {
  switch(type) {
    case ncclChar:   return Randomize(devmem, count, seed);
    case ncclInt:    return Randomize((int*)devmem, count, seed);
#ifdef CUDA_HAS_HALF
    case ncclHalf:   return Randomize((half*)devmem, count, seed);
#endif
    case ncclFloat:  return Randomize((float*)devmem, count, seed);
    case ncclDouble: return Randomize((double*)devmem, count, seed);
    case ncclInt64:  return Randomize((long long*)devmem, count, seed);
    case ncclUint64: return Randomize((unsigned long long*)devmem, count, seed);
  }
}

typedef int (*test_func_t)(int, int, int, int);

int testBcast(int count, int type, int op, int root) {
  int errors = 0;
  int nbytes = count*typesSizes[type];
  for (int i=0; i<nDev; ++i) {
    CUDACHECK(cudaSetDevice(devList[i]));
    if (i == root) {
      RandomizeType(type, sendbuff[i], count, i);
      CUDACHECK(cudaMemcpy(reference, sendbuff[i], nbytes, cudaMemcpyDeviceToHost));
    } else {
      CUDACHECK(cudaMemset(recvbuff[i], 0, nbytes));
    }
  }

  for (int rep=0; rep<NREPS; ++rep) {
    for (int i=0; i<nDev; ++i) {
      CUDACHECK(cudaSetDevice(devList[i]));
      ncclBcast((i == root) ? sendbuff[i] : recvbuff[i], count, (ncclDataType_t)type, root, comms[i], streams[i]);
    }
  }
  for (int i=0; i<nDev; ++i) {
    CUDACHECK(cudaSetDevice(devList[i]));
    CUDACHECK(cudaStreamSynchronize(streams[i]));
    double delta = CheckDelta<char>((i==root) ? sendbuff[i] : recvbuff[i], reference, nbytes);
    if (delta) errors++;
    if (delta) printf("Bcast size %d, type %d, root %d : delta %g\n", count, type, root, delta);
  }
  return errors;
}

int testAllGather(int count, int type, int op, int root) {
  int errors = 0;
  int sendcount = (count + nDev - 1) / nDev;
  int recvcount = sendcount * nDev;
  int sendnbytes = sendcount*typesSizes[type];
  int recvnbytes = recvcount*typesSizes[type];
  for (int i=0; i<nDev; ++i) {
    CUDACHECK(cudaSetDevice(devList[i]));
    RandomizeType(type, sendbuff[i], sendcount, i);
    CUDACHECK(cudaMemcpy(reference+sendnbytes*i, sendbuff[i], sendnbytes, cudaMemcpyDeviceToHost));
    CUDACHECK(cudaMemset(recvbuff[i], 0, recvnbytes));
  }

  for (int rep=0; rep<NREPS; ++rep) {
    for (int i=0; i<nDev; ++i) {
      CUDACHECK(cudaSetDevice(devList[i]));
      ncclAllGather(sendbuff[i], sendcount, (ncclDataType_t)type, recvbuff[i], comms[i], streams[i]);
    }
  }
  for (int i=0; i<nDev; ++i) {
    CUDACHECK(cudaSetDevice(devList[i]));
    CUDACHECK(cudaStreamSynchronize(streams[i]));
    double delta = CheckTypeDelta(type, recvbuff[i], reference, recvcount);
    if (delta) {
      errors++;
      CUDACHECK(cudaMemcpy(results, recvbuff[i], recvnbytes, cudaMemcpyDeviceToHost));
      printf("Allgather size %d, type %d : delta %g, new %g\n", sendcount, type, delta);
      for (int c=1; c<count; c++) {
	if (type == ncclFloat) {
          float res = *((float*)results+c), ref = *((float*)reference+c);
          if (fabs(res-ref) > typesDeltas[type]*nDev) printf("[%d/%3d] %f != %f (+%f)\n", i, c, res, ref, (ref-res)/ref);
        } else if (type == ncclDouble) {
          double res = *((double*)results+c), ref = *((double*)reference+c);
          if (fabs(ref-res) > typesDeltas[type]*nDev) printf("[%d/%3d] %g != %g (+%g)\n", i, c, res, ref, (ref-res)/ref);
        } else if (c*8 < count*typesSizes[type]) {
          uint64_t res = *((uint64_t*)results+c), ref = *((uint64_t*)reference+c);
          if (res != ref) printf("[%d/%3d] %16lX != %16lX\n", i, c, res, ref);
        }
      }
    }
  }
  return errors;
}

int testAllReduce(int count, int type, int op, int root) {
  int errors = 0;
  int nbytes = count*typesSizes[type];
  for (int i=0; i<nDev; ++i) {
    CUDACHECK(cudaSetDevice(devList[i]));
    RandomizeType(type, sendbuff[i], count, i);
    if(i == 0) {
      CUDACHECK(cudaMemcpy(reference, sendbuff[i], nbytes, cudaMemcpyDeviceToHost));
    } else {
      AccumulateType(type, reference, sendbuff[i], count, (ncclRedOp_t)op);
    }
    CUDACHECK(cudaMemset(recvbuff[i], 0, nbytes));
  }

  for (int rep=0; rep<NREPS; ++rep) {
    for (int i=0; i<nDev; ++i) {
      CUDACHECK(cudaSetDevice(devList[i]));
      ncclAllReduce(sendbuff[i], recvbuff[i], count, (ncclDataType_t)type, (ncclRedOp_t)op, comms[i], streams[i]);
    }
  }
  for (int i=0; i<nDev; ++i) {
    CUDACHECK(cudaSetDevice(devList[i]));
    CUDACHECK(cudaStreamSynchronize(streams[i]));
    double delta = CheckTypeDelta(type, recvbuff[i], reference, count);
    if (delta > typesDeltas[type]*nDev) {
      errors++;
      CUDACHECK(cudaMemcpy(results, recvbuff[i], nbytes, cudaMemcpyDeviceToHost));
      printf("Allreduce size %d, type %d, op %d : delta %g, new %g\n", count, type, op, delta);
#ifdef DEBUG_DETAILS
      for (int c=1; c<count; c++) {
	if (type == ncclFloat) {
          float res = *((float*)results+c), ref = *((float*)reference+c);
          if (fabs(res-ref) > typesDeltas[type]*nDev) printf("[%d/%3d] %f != %f (+%f)\n", i, c, res, ref, (ref-res)/ref);
        } else if (type == ncclDouble) {
          double res = *((double*)results+c), ref = *((double*)reference+c);
          if (fabs(ref-res) > typesDeltas[type]*nDev) printf("[%d/%3d] %g != %g (+%g)\n", i, c, res, ref, (ref-res)/ref);
        } else if (c*8 < count*typesSizes[type]) {
          uint64_t res = *((uint64_t*)results+c), ref = *((uint64_t*)reference+c);
          if (res != ref) printf("[%d/%3d] %16lX != %16lX\n", i, c, res, ref);
        }
      }
#endif
    }
  }
  return errors;
}

#define NCCL_PRIMS 5

test_func_t ncclPrims[NCCL_PRIMS] = {
  testBcast,
  NULL,//testReduce,
  testAllReduce,
  testAllGather,
  NULL,//testReduceScatter
};

int ncclTest() {
  int errors = 0;
  int nccl_prim = rand() % NCCL_PRIMS;
  // Use MAX_POW2-3 because datatypes are up to 8-bytes wide
  int size_pow2 = rand() % (MAX_POW2-3); 
  int size = (1<<size_pow2) + rand() % (1<<size_pow2);
  int type = rand() % nccl_NUM_TYPES;
  int op = rand() % nccl_NUM_OPS;
  int root = rand() % nDev;
#ifndef CUDA_HAS_HALF
  if (type == 2) return 0; // ncclHalf not supported
#endif
  if (ncclPrims[nccl_prim]) {
    printf("Prim %d size %d type %d op %d root %d\n", nccl_prim, size, type, op, root);
    errors += ncclPrims[nccl_prim](size, type, op, root);
  }
  return errors;
}

void usage() {
  printf("Tests all nccl primitives.\n"
      "    Usage: stress_test [time in sec] [number of GPUs]"
      "[GPU 0] [GPU 1] ...\n\n");
}

int main(int argc, char* argv[]) {
  int nVis = 0;
  CUDACHECK(cudaGetDeviceCount(&nVis));

  int T = 0;
  if (argc > 1) {
    int t = sscanf(argv[1], "%d", &T);
    if (t == 0) {
      printf("Error: %s is not an integer!\n\n", argv[1]);
      usage();
      exit(EXIT_FAILURE);
    }
  } else {
    printf("Error: must specify at least time in seconds!\n\n");
    usage();
    exit(EXIT_FAILURE);
  }

  nDev = nVis;
  if (argc > 2) {
    int t = sscanf(argv[2], "%d", &nDev);
    if (t == 0) {
      printf("Error: %s is not an integer!\n\n", argv[1]);
      usage();
      exit(EXIT_FAILURE);
    }
  }
  devList = (int*)malloc(sizeof(int)*nDev);
  for (int i = 0; i < nDev; ++i)
    devList[i] = i % nVis;

  if (argc > 3) {
    if (argc - 3 != nDev) {
      printf("Error: insufficient number of GPUs in list\n\n");
      usage();
      exit(EXIT_FAILURE);
    }

    for (int i = 0; i < nDev; ++i) {
      int t = sscanf(argv[3 + i], "%d", devList + i);
      if (t == 0) {
        printf("Error: %s is not an integer!\n\n", argv[2 + i]);
        usage();
        exit(EXIT_FAILURE);
      }
    }
  }

  typesSizes[ncclChar] = 1; typesSizes[ncclInt] = 4; typesSizes[ncclFloat] = 4;
  typesSizes[ncclDouble] = 8; typesSizes[ncclInt64] = 8; typesSizes[ncclUint64] = 8;
#ifdef CUDA_HAS_HALF
  typesSizes[ncclHalf] = 2;
#endif
  typesDeltas[ncclChar] = 0; typesDeltas[ncclInt] = 0; typesDeltas[ncclFloat] = 1e-6;
  typesDeltas[ncclDouble] = 1e-15; typesDeltas[ncclInt64] = 0; typesDeltas[ncclUint64] = 0;
#ifdef CUDA_HAS_HALF
  typesDeltas[ncclHalf] = 2e-3;
#endif

  comms = (ncclComm_t*)malloc(sizeof(ncclComm_t)*nDev);
  NCCLCHECK(ncclCommInitAll(comms, nDev, devList));

  streams = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);
  sendbuff = (char**)malloc(sizeof(char*)*nDev);
  recvbuff = (char**)malloc(sizeof(char*)*nDev);
  for(int i=0; i<nDev; ++i) {
    CUDACHECK(cudaSetDevice(devList[i]));
    CUDACHECK(cudaStreamCreate(&streams[i]));
    CUDACHECK(cudaMalloc(sendbuff+i, 1<<MAX_POW2));
    CUDACHECK(cudaMalloc(recvbuff+i, 1<<MAX_POW2));
  }

  reference = (char*)malloc(1<<MAX_POW2);
  results = (char*)malloc(1<<MAX_POW2);
 
  struct timeval tv;
  gettimeofday(&tv, NULL);
  int sec_now = tv.tv_sec;
  int sec_start = sec_now;
  srand(sec_start);
  int testcount = 0;
  int errors = 0;

  printf("==== Test starting ====\n");
  while (sec_now <= sec_start + T) {
    errors += ncclTest();
    gettimeofday(&tv, NULL);
    sec_now = tv.tv_sec;
    testcount++;
  }

  printf("==== Test done ====\n");
  printf("%d tests done\n", testcount);
  printf("%d errors\n", errors);
  for(int i=0; i<nDev; ++i)
    ncclCommDestroy(comms[i]);
  free(comms);

  exit(errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
}

