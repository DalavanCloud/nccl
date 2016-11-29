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

#define MAXSIZE (1<<25)
#define NITERS 100

int main(int argc, char *argv[]) {
  int failed = 0;
  ncclUniqueId commId;
  int nranks, rank;
  ncclResult_t ret;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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

  // Initialize input values
  int *dptr;
  CUDACHECK(cudaMalloc(&dptr, MAXSIZE*2*sizeof(int)));
  int *val = (int*) malloc(MAXSIZE*sizeof(int));
  for (int v=0; v<MAXSIZE; v++) {
    val[v] = rank + 1;
  }
  CUDACHECK(cudaMemcpy(dptr, val, MAXSIZE*sizeof(int), cudaMemcpyHostToDevice));

  // Compute final value
  int ref = nranks*(nranks+1)/2;


  // Warm-up
  NCCLCHECK(ncclAllReduce((const void*)dptr, (void*)(dptr+MAXSIZE), MAXSIZE, ncclInt, ncclSum, comm, stream));
  CUDACHECK(cudaStreamSynchronize(stream));

  // Run allreduce benchmark
  if (rank == 0) printf("\n        Size (B)       Time (us)   Alg BW (MB/s)   Bus BW (MB/s)          Errors\n");
  for (int size = 1; size <= MAXSIZE; size<<=1) {
    int errors = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    for (int i=0; i<NITERS; i++) {
      NCCLCHECK(ncclAllReduce((const void*)dptr, (void*)(dptr+MAXSIZE), size, ncclInt, ncclSum, comm, stream));
    }
    CUDACHECK(cudaStreamSynchronize(stream));
    double delta = MPI_Wtime() - start;

    // Check results
    CUDACHECK(cudaMemcpy(val, (dptr+MAXSIZE), size*sizeof(int), cudaMemcpyDeviceToHost));
    for (int v=0; v<size; v++) {
      if (val[v] != ref) {
        errors++;
        failed = 1;
        //printf("[%d] Error at %d : got %d instead of %d\n", rank, v, val[v], ref);
      }
    }
    MPI_Allreduce(MPI_IN_PLACE, &errors, 1, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) printf(" %15d %15.2f %15.2f %15.2f %15d\n", 
        size*sizeof(int),
        delta*1e6/NITERS,
        size*sizeof(int)*NITERS/(delta*1e6),
        size*sizeof(int)*NITERS/(delta*1e6)*2*(nranks-1)/nranks,
        errors);
  }

  CUDACHECK(cudaFree(dptr));

  MPI_Finalize();
  //ncclCommDestroy(comm);
  return failed;
}
