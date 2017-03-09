/*************************************************************************
 * Copyright (c) 2015-2017, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include "mpi.h"
#include "nccl.h"
#include "debug.h"
#include <sys/types.h>
#include <unistd.h>

extern ncclNet_t ncclNetSocket; 
extern ncclNet_t ncclNetIb; 

#define MAX_REQUESTS 1024
//#define SOCKET 1
#define IB 1
//#define RECV_0_SEND_1 1
//#define RECV_1_SEND_0 1

int tester(ncclNet_t *net, char *data, char *data_d, size_t bytes, int rank, int nranks){
  int failed = 0;
  int *scores;
  int ndev=0;
  char listenHandle[NCCL_NET_HANDLE_MAXSIZE], connectHandle[NCCL_NET_HANDLE_MAXSIZE];
  char *listenComm, *sendComm, *recvComm; 
  int type = 0, cnt = 0;
  //char *request[MAX_REQUESTS];
  MPI_Barrier(MPI_COMM_WORLD);
  if(rank==0){
    if(net->devices(&ndev, &scores)){failed=1; goto out; }
    printf("Rank 0 ndev %d scores : \n", ndev);
    for(int i=0; i<ndev; i++){
      printf("scores[%d] = %d\n", i, scores[i]);
    }
    if(net->listen(0, (void *)listenHandle, (void **)&listenComm)){ failed=1; goto out; }
    if(MPI_Send(listenHandle, NCCL_NET_HANDLE_MAXSIZE, MPI_BYTE, 1/*rank*/, 0, MPI_COMM_WORLD)){ failed=1; goto out; }
    if(net->accept(listenComm, (void **)&recvComm)){ failed=1; goto out; }
    printf("Rank 0 accepted connection from rank 1\n");

    if(MPI_Recv(connectHandle, NCCL_NET_HANDLE_MAXSIZE, MPI_BYTE, 1/*rank*/, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE)){ failed=1; goto out; }
    if(net->connect(0, connectHandle, (void **)&sendComm)){ failed=1; goto out; }
    printf("Rank 0 connected to rank 1 %d\n", getpid());

    char *req;
    type |= NCCL_PTR_HOST;
    //if(net->isend(sendComm, data, bytes, type, (void **)&request[cnt++])){ failed=1; goto out; };
    if(net->isend(sendComm, data, bytes, type, (void **)&req)){ failed=1; goto out; };
    type = 0;
    printf("Rank 0 posted send %d\n", getpid());

    int done=0;
    do {
	    int size = -1;
	    //if(net->test(request[cnt-1], &done, &size)){ failed=1; goto out; }
	    if(net->test(req, &done, &size)){ failed=1; goto out; }
    } while(!done);

    /*type |= NCCL_PTR_HOST;
    if(net->irecv(recvComm, data, bytes, type, (void **)&request[cnt++])){ failed=1; goto out; }
    type = 0;
    printf("Rank 0 posted recv\n");

    type |= NCCL_PTR_HOST;
    if(net->isend(sendComm, data, bytes, type, (void **)&request[cnt++])){ failed=1; goto out; };
    type = 0;
    printf("Rank 0 posted send\n");*/

  }else{
    if(net->devices(&ndev, &scores)){failed=1; goto out; }
    printf("Rank 1 ndev %d scores : \n", ndev);
    for(int i=0; i<ndev; i++){
      printf("scores[%d] = %d\n", i, scores[i]);
    }
    if(MPI_Recv(connectHandle, NCCL_NET_HANDLE_MAXSIZE, MPI_BYTE, 0/*rank*/, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE)){ failed=1; goto out; }
    if(net->connect(0, connectHandle, (void **)&sendComm)){ failed=1; goto out; }
    printf("Rank 1 connected to rank 0\n");

    if(net->listen(0, (void *)listenHandle, (void **)&listenComm)){ failed=1; goto out; }
    if(MPI_Send(listenHandle, NCCL_NET_HANDLE_MAXSIZE, MPI_BYTE, 0/*rank*/, 0, MPI_COMM_WORLD)){ failed=1; goto out; }
    if(net->accept(listenComm, (void **)&recvComm)){ failed=1; goto out; }
    printf("Rank 1 accepted connection from rank 0 %d\n", getpid());

    char *req;
    type |= NCCL_PTR_HOST;
    //if(net->irecv(recvComm, data, bytes, type, (void **)&request[cnt++])){ failed=1; goto out; }
    if(net->irecv(recvComm, data, bytes, type, (void **)&req)){ failed=1; goto out; }
    type = 0;
    printf("Rank 1 posted recv %d\n", getpid());

    int done=0;
    do {
	    int size = -1;
	    //if(net->test(request[cnt-1], &done, &size)){ failed=1; goto out; }
	    if(net->test(req, &done, &size)){ failed=1; goto out; }
    } while(!done);

    /*type |= NCCL_PTR_HOST;
    if(net->irecv(recvComm, data, bytes, type, (void **)&request[cnt++])){ failed=1; goto out; }
    type = 0;
    printf("Rank 1 posted recv\n");

    type |= NCCL_PTR_HOST;
    if(net->isend(sendComm, data, bytes, type, (void **)&request[cnt++])){ failed=1; goto out; };
    type = 0;
    printf("Rank 0 posted send\n");*/
  }
  MPI_Barrier(MPI_COMM_WORLD);
out:
  return failed;
}

#define MAX_SIZE 8//(1024*1024*1024)

int main(int argc, char *argv[]) {
  int nranks, rank;

  int threadProvided;

  initDebug();

  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &threadProvided);
  printf("provided : %d\n", threadProvided);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPI_Barrier(MPI_COMM_WORLD);

  int failed = 0;
  char *data = new char[MAX_SIZE];
  char *data_d;
  cudaMalloc(&data_d, MAX_SIZE);
#if defined(SOCKET) &&  defined(IB)
  ncclNet_t *nets[] = {&ncclNetSocket, &ncclNetIb};
#endif
#if defined(SOCKET) && !defined(IB)
  ncclNet_t *nets[] = {&ncclNetSocket};
#endif
#if defined(IB) && !defined(SOCKET)
  ncclNet_t *nets[] = {&ncclNetIb};
#endif
  for(int i=0; i<sizeof(nets)/sizeof(nets[0]); i++){
    ncclNet_t *net = nets[i]; 
    printf("net %p\n", net);
    printf("net->name %s\n", net->name);
    printf("net->listen %p\n", net->listen);
    //net->listen(0,0,0);
    failed = tester(net, data, data_d, MAX_SIZE, rank, nranks);
    if(failed) { goto out; }
  }
  delete data;
  cudaFree(data_d);
  MPI_Finalize();

out:
  if(failed){
    printf("[%d] Test failed\n", rank);
  }else{
    printf("[%d] : Test successful\n", rank);
  }
  return failed;
}
