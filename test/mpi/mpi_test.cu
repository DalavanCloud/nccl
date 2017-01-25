/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENCE.txt for license information
 ************************************************************************/

#include <sys/types.h>
#include <unistd.h>
#include <stdio.h>

#include "nccl.h"
#include "mpi.h"
#include "test_utilities.h"

#define MAXSIZE (1<<26)
#define NITERS 100

ncclResult_t ncclOp(int collective, int rank, int nranks, int *buff, int size, ncclComm_t comm, cudaStream_t stream) {
  switch (collective) {
    case 0 : // Broadcast
      return ncclBcast(rank == 0 ? (void*)buff : (void*)(buff+MAXSIZE), size, ncclInt, 0, comm, stream);
    case 1 : // Reduce
      return ncclReduce((const void*)buff, (void*)(buff+MAXSIZE), size, ncclInt, ncclSum, 0, comm, stream);
    case 2 : // Allreduce
      return ncclAllReduce((const void*)buff, (void*)(buff+MAXSIZE), size, ncclInt, ncclSum, comm, stream);
    case 3 : // Allgather
      return ncclAllGather((const void*)buff, size/nranks, ncclInt, (void*)(buff+MAXSIZE), comm, stream);
    case 4 : // ReduceScatter
      return ncclReduceScatter((const void*)buff, (void*)(buff+MAXSIZE), size/nranks, ncclInt, ncclSum, comm, stream);
  }
  return ncclSuccess;
}

int checkOp(int collective, int rank, int nranks, int* ptr, int size) {
  int errors = 0;
  int sum = nranks*(nranks+1)/2;
  switch (collective) {
    case 0 : // Broadcast
      if (rank == 0) return 0;
      for (int v=0; v<size; v++) if (ptr[v] != 1) errors++;
      break;
    case 1 : // Reduce
      if (rank != 0) return 0;
    case 2 : // Allreduce
      for (int v=0; v<size; v++) if (ptr[v] != sum) errors++;
      break;
    case 3 : // Allgather
      for (int r = 0; r < nranks; r++) {
        for (int v=0; v<size/nranks; v++) if (ptr[r*(size/nranks)+v] != r+1) errors++;
      }
      break;
    case 4 : // ReduceScatter
      for (int v=0; v<size/nranks; v++) if (ptr[v] != sum) errors++; 
      break;
  }
  return errors;
}

float perfFactorOp(int collective, int nranks) {
  switch (collective) {
    case 2 : // Allreduce
      return 2*(nranks-1.0)/nranks;
    case 3 : // Allgather
      return (nranks-1.0)/nranks;
    case 4 : // ReduceScatter
      return (nranks-1.0)/nranks;
  }
  return 1.0;
}
void printBanner(int collective) {
  printf("\n");
  printf("      **************************************************************************\n");
  switch (collective) {
    case 0 : // Broadcast
      printf("      *                                Broadcast                               *\n");
      break;
    case 1 : // Reduce
      printf("      *                                Reduce                                  *\n");
      break;
    case 2 : // Allreduce
      printf("      *                                Allreduce                               *\n");
      break;
    case 3 : // Allgather
      printf("      *                                Allgather                               *\n");
      break;
    case 4 : // ReduceScatter
      printf("      *                                ReduceScatter                           *\n");
      break;
  }
  printf("      **************************************************************************\n");
}

int benchCollective(int collective, int rank, int nranks, int* ddata, int* hdata, ncclComm_t comm, cudaStream_t stream) {
  // Initialize input values
  for (int v=0; v<MAXSIZE; v++) hdata[v] = rank + 1;
  CUDACHECK(cudaMemcpy(ddata, hdata, MAXSIZE*sizeof(int), cudaMemcpyHostToDevice));
  // Initialize output values
  for (int v=0; v<MAXSIZE; v++) hdata[v] = -1;
  CUDACHECK(cudaMemcpy(ddata+MAXSIZE, hdata, MAXSIZE*sizeof(int), cudaMemcpyHostToDevice));

  // Warm-up
  NCCLCHECK(ncclOp(collective, rank, nranks, ddata, MAXSIZE, comm, stream));
  CUDACHECK(cudaStreamSynchronize(stream));

  int failed = 0;
  if (rank == 0) {
    printBanner(collective);
    printf("        Size (B)       Time (us)   Alg BW (MB/s)   Bus BW (MB/s)          Errors\n");
  }
  for (int size = 1; size <= MAXSIZE; size<<=1) {
    int realSize = size;
    if (collective > 2) {
      if (size > (MAXSIZE/nranks)) continue;
      realSize = size * nranks;
    }
    int nbytes = realSize*sizeof(int);
    int errors = 0;
    CUDACHECK(cudaStreamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    int niters = min(NITERS, 1000000000/realSize);
    for (int i=0; i<niters; i++) {
      NCCLCHECK(ncclOp(collective, rank, nranks, ddata, realSize, comm, stream));
    }
    CUDACHECK(cudaStreamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);
    double delta = MPI_Wtime() - start;
    delta = delta*1e6/niters;

    // Check results
    CUDACHECK(cudaMemcpy(hdata, (ddata+MAXSIZE), nbytes, cudaMemcpyDeviceToHost));
    errors = checkOp(collective, rank, nranks, hdata, realSize);
    MPI_Allreduce(MPI_IN_PLACE, &errors, 1, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) {
      printf(" %15ld %15.2f %15.2f %15.2f %15d\n",
        nbytes,
        delta,
        nbytes/delta,
        nbytes/delta*perfFactorOp(collective, nranks),
        errors);
    }
  }
  if (rank == 0) printf("\n");
  return failed;
}

//extern "C"
//void ncclMpiHook(MPI_Comm comm);

int main(int argc, char *argv[]) {
  ncclUniqueId commId;
  int nranks, rank;
  ncclResult_t ret;

  int threadProvided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &threadProvided);
  printf("provided : %d\n", threadProvided);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  //ncclMpiHook(MPI_COMM_WORLD);

  if (argc < nranks) {
    if (rank == 0)
      printf("Usage : %s <GPU list per rank>\n", argv[0]);
    exit(1);
  }

  int gpu = atoi(argv[rank+1]);

  // We have to set our device before NCCL init
  CUDACHECK(cudaSetDevice(gpu));
  MPI_Barrier(MPI_COMM_WORLD);

  // NCCL Communicator creation
  ncclComm_t comm;
  if (rank == 0) NCCLCHECK(ncclGetUniqueId(&commId));
  MPI_Bcast(&commId, NCCL_UNIQUE_ID_BYTES, MPI_CHAR, 0, MPI_COMM_WORLD);
  ret = ncclCommInitRank(&comm, nranks, commId, rank);
  if (ret != ncclSuccess) {
    printf("NCCL Init failed (%d) '%s'\n", ret, ncclGetErrorString(ret));
    exit(1);
  }

  // CUDA stream creation
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  int *ddata;
  CUDACHECK(cudaMalloc(&ddata, MAXSIZE*2*sizeof(int)));
  int *hdata = (int*) malloc(MAXSIZE*sizeof(int));

  int failed = 0;
  failed += benchCollective(0, rank, nranks, ddata, hdata, comm, stream);
  failed += benchCollective(1, rank, nranks, ddata, hdata, comm, stream);
  failed += benchCollective(2, rank, nranks, ddata, hdata, comm, stream);
  failed += benchCollective(3, rank, nranks, ddata, hdata, comm, stream);
  failed += benchCollective(4, rank, nranks, ddata, hdata, comm, stream);

  CUDACHECK(cudaFree(ddata));
  free(hdata);

  MPI_Finalize();
  ncclCommDestroy(comm);
  return failed;
}
