/*************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
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
#include <math.h>
#include <float.h>
#include "mpi.h"

#include "nccl.h"
#include "test_utilities.h"
#include <nvToolsExt.h>

#define MPICHECK(cmd) do {                                  \
  int e = cmd;                                              \
  if( e != MPI_SUCCESS ) {                                  \
    printf("MPI failure %s:%d (%d)\n",__FILE__,__LINE__,e); \
    exit(EXIT_FAILURE);                                     \
  }                                                         \
} while(0)

MPI_Datatype MPITypeFromNCCLType(ncclDataType_t nccl) {
    switch(nccl) {
        case ncclInt8: return MPI_SIGNED_CHAR;
        case ncclUint8: return MPI_UNSIGNED_CHAR;
        case ncclInt32: return MPI_INT;
        case ncclUint32: return MPI_UNSIGNED;
        case ncclInt64: return MPI_LONG_LONG;
        case ncclUint64: return MPI_UNSIGNED_LONG_LONG;
        case ncclFloat16:
            printf("FP16 is not supported by MPI\n");
            exit(EXIT_FAILURE);
        case ncclFloat32: return MPI_FLOAT;
        case ncclFloat64: return MPI_DOUBLE;
        default:
            printf("Unknown type %d\n", (int)nccl);
            exit(EXIT_FAILURE);
    }
}

MPI_Op MPIOpFromNCCLOp(ncclRedOp_t nccl) {
    switch(nccl) {
        case ncclSum: return MPI_SUM;
        case ncclProd: return MPI_PROD;
        case ncclMax: return MPI_MAX;
        case ncclMin: return MPI_MIN;
        default:
            printf("Unknown operation %d\n", (int)nccl);
            exit(EXIT_FAILURE);
    }
}

/*extern "C"
void ncclMpiHook(MPI_Comm comm);*/

void showUsage(const char* bin) {
  printf("\n"
         "Usage: %s <transport> <type> <op> <n_min> <n_max> [delta] [gpu]\n"
         "Where:\n"
         "    transport =   [mpi|socket]\n"
         "    type      =   [char|int|float|double|int64|uint64]\n"
         "    op        =   [sum|prod|max|min]\n"
         "    n_min     >   0\n"
         "    n_max     >=  n_min\n"
         "    delta     >   0\n\n", bin);
  return;
}

int main(int argc, char* argv[]) {
  int nvis = 0;
  CUDACHECK(cudaGetDeviceCount(&nvis));
  if (nvis == 0) {
    printf("No GPUs found\n");
    showUsage(argv[0]);
    exit(EXIT_FAILURE);
  }

  MPI_Init(&argc, &argv);
  int globalRank;
  int localRank;
  char procName[MPI_MAX_PROCESSOR_NAME+1];
  int nameLen;
  int numRanks;
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &globalRank));
  const char* lrString = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
  if (lrString == NULL) { printf("Could not get local rank\n"); exit(EXIT_FAILURE); }
  localRank = strToNonNeg(lrString);
  MPICHECK(MPI_Get_processor_name(procName, &nameLen));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &numRanks));

  ncclDataType_t type;
  ncclRedOp_t op;
  int n_min;
  int n_max;
  int delta;
  int gpu;
  int mpiTransport;

  if (argc < 5) {
    showUsage(argv[0]);
    exit(EXIT_FAILURE);
  }

  if (strcmp("mpi", argv[1]) == 0) {
    mpiTransport = 1;
  } else if (strcmp("socket", argv[1]) == 0) {
    mpiTransport = 0;
  } else {
    printf("Invalid transport '%s'\n", argv[1]);
    exit(EXIT_FAILURE);
  }

  type = strToType(argv[2]);
  if (type == ncclNumTypes) {
    printf("Invalid <type> '%s'\n", argv[2]);
    showUsage(argv[0]);
    exit(EXIT_FAILURE);
  }

  op = strToOp(argv[3]);
  if (op == ncclNumOps) {
    printf("Invalid <op> '%s'\n", argv[3]);
    showUsage(argv[0]);
    exit(EXIT_FAILURE);
  }

  n_min = strToPosInt(argv[4]);
  if (n_min < 1) {
    printf("Invalid <n_min> '%s'\n", argv[4]);
    showUsage(argv[0]);
    exit(EXIT_FAILURE);
  }

  n_max = strToPosInt(argv[5]);
  if (n_max < n_min) {
    printf("Invalid <n_max> '%s'\n", argv[5]);
    showUsage(argv[0]);
    exit(EXIT_FAILURE);
  }

  if (argc > 6) {
    delta = strToPosInt(argv[6]);
    if (delta < 1) {
      printf("Invalid <delta> '%s'\n", argv[6]);
      showUsage(argv[0]);
      exit(EXIT_FAILURE);
    }
  } else {
    delta = (n_max == n_min) ? 1 : (n_max - n_min+9) / 10;
  }

  if (argc > 7) {
    gpu = strToNonNeg(argv[7]);
    if (gpu >= nvis) {
      printf("Invalid <gpu> '%s' must be less than %d\n", argv[7], nvis);
      showUsage(argv[0]);
      exit(EXIT_FAILURE);
    }
  } else if (localRank < nvis) {
    gpu = localRank;
  } else {
    printf("Could not find GPU for rank %d\n", globalRank);
    exit(EXIT_FAILURE);
  }
  CUDACHECK(cudaSetDevice(gpu));
  char busid[32] = {0};
  CUDACHECK(cudaDeviceGetPCIBusId(busid, sizeof(busid), gpu));
  printf("# Rank %d using GPU %d [%s] on host %s\n", globalRank, gpu, busid, procName);

  size_t word = wordSize(type);
  size_t max_size = n_max * word;
  void* refout;
  CUDACHECK(cudaMallocHost(&refout, max_size));

  void *input, *output;
  double* error;
  ncclComm_t comm;
  cudaStream_t stream;

  CUDACHECK(cudaMalloc(&input,  max_size));
  CUDACHECK(cudaMalloc(&output, max_size));
  CUDACHECK(cudaMallocHost(&error, sizeof(double)));
  CUDACHECK(cudaStreamCreate(&stream));
  makeRandom(input, n_max, type, 42+globalRank);

  CUDACHECK(cudaMemcpy(refout, input, max_size, cudaMemcpyDeviceToHost));
  MPICHECK(MPI_Allreduce(MPI_IN_PLACE, refout, n_max,
      MPITypeFromNCCLType(type), MPIOpFromNCCLOp(op), MPI_COMM_WORLD));


/*  if (mpiTransport) {
    ncclMpiHook(MPI_COMM_WORLD);
  }*/

  ncclUniqueId nid;
  if (globalRank == 0) {
    ncclGetUniqueId(&nid);
  }
  MPICHECK(MPI_Bcast(&nid, NCCL_UNIQUE_ID_BYTES, MPI_CHAR, 0, MPI_COMM_WORLD));
  NCCLCHECK(ncclCommInitRank(&comm, numRanks, nid, globalRank));

  if (globalRank == 0) {
    printf("       BYTES ERROR       MSEC  ALGBW  BUSBW\n");
  }

  int failed = 0;
  for(int n=n_min; n<=n_max; n+=delta) {
    size_t bytes = word * n;

    CUDACHECK(cudaMemsetAsync(output, 0, bytes, stream));
    CUDACHECK(cudaStreamSynchronize(stream));

    auto start = std::chrono::high_resolution_clock::now();
    NCCLCHECK(ncclAllReduce(input, output, n, type, op, comm, stream));
    CUDACHECK(cudaStreamSynchronize(stream));
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto stop = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration_cast<std::chrono::duration<double>>
        (stop - start).count() * 1000.0;
    MPICHECK(MPI_Allreduce(MPI_IN_PLACE, &ms, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));

    *error = INFINITY;
    maxDiff(error, output, refout, n, type, stream);
    CUDACHECK(cudaStreamSynchronize(stream));
    MPICHECK(MPI_Allreduce(MPI_IN_PLACE, error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));

    if (globalRank == 0) {
      double mb = (double)bytes * 1.e-6;
      double algbw = mb / ms;
      double busbw = algbw * (double)(2*numRanks - 2) / (double)numRanks;
      printf("%12lu %5.0le %10.3lf %6.2lf %6.2lf\n",
          n*word, *error, ms, algbw, busbw);
    }
    if (*error > deltaMaxValue(type, 1)) failed++;
  }

  MPICHECK(MPI_Allreduce(MPI_IN_PLACE, &failed, 1, MPI_INTEGER, MPI_MAX, MPI_COMM_WORLD));

  if (globalRank == 0) printf("\n Out of bounds values : %d %s\n\n", failed, failed ? "FAILED" : "OK");

  CUDACHECK(cudaStreamDestroy(stream));
  ncclCommDestroy(comm);
  CUDACHECK(cudaFree(input));
  CUDACHECK(cudaFree(output));
  CUDACHECK(cudaFreeHost(error));
  CUDACHECK(cudaFreeHost(refout));
  MPICHECK(MPI_Finalize());
  exit(failed ? EXIT_FAILURE : EXIT_SUCCESS);
}

