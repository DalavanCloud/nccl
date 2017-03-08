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

extern ncclNet_t ncclNetSocket; 
extern ncclNet_t ncclNetIb; 
extern ncclNet_t ncclMpi; 

int Socket_tester(ncclNet_t *net, char *data, size_t bytes, size_t *duration, int dev, int rank, int nranks, MPI_Comm comm){
  if(!rank) printf("Socket tester\n");

  int failed = 0;
  if(rank==0){
    for(int rnk=1; rnk<nranks; rnk++){
      char listenHandle[NCCL_NET_HANDLE_MAXSIZE];
      char *listenComm; 
      //printf("%d listen\n", rank);
      if(net->listen(0, (void *)listenHandle, (void **)&listenComm)){ failed=1; goto out; }
      //printf("%d MPI_Send\n", rank);
      if(MPI_Send(listenHandle, NCCL_NET_HANDLE_MAXSIZE, MPI_BYTE, rnk, 0, comm)){ failed=1; goto out; }

      //printf("%d accept\n", rank);
      char *recvComm;
      if(net->accept(listenComm, (void **)&recvComm)){ failed=1; goto out; }

      //printf("%d closeListen\n", rank);
      if(net->closeListen(listenComm)){ failed=1; goto out; }

      /*ping*/
      int type = 0;
      type |= NCCL_PTR_HOST;
      char *request;
      //printf("%d recv\n", rank);
      struct timeval start, end;
      gettimeofday(&start, NULL);
      if(net->irecv(recvComm, data, bytes, type, (void **)&request)){ failed=1; goto out; }

      int done=0;
      do {
        int size = -1;
        if(net->test(request, &done, &size)){ failed=1; goto out; }
      } while(!done);
      gettimeofday(&end, NULL);
      *duration = (end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec);
      //printf("%d closeRecv\n", rank);
      if(net->closeRecv(recvComm)){ failed=1; goto out; }
    }
  }else{
    char connectHandle[NCCL_NET_HANDLE_MAXSIZE];
    char *sendComm;

    //printf("%d MPI_Recv\n", rank);
    if(MPI_Recv(connectHandle, NCCL_NET_HANDLE_MAXSIZE, MPI_BYTE, 0, 0, comm, MPI_STATUS_IGNORE)){ failed=1; goto out; }

    //printf("%d connect\n", rank);
    if(net->connect(0, connectHandle, (void **)&sendComm)){ failed=1; goto out; }

    /*pong*/  
    int type = 0;
    type |= NCCL_PTR_HOST;
    char *request;
    //printf("%d send\n", rank);
    struct timeval start, end;
    gettimeofday(&start, NULL);
    if(net->isend(sendComm, data, bytes, type, (void **)&request)){ failed=1; goto out; };

    int done=0;
    do {
      int size = -1;
      if(net->test(request, &done, &size)){ failed=1; goto out; }
    } while(!done);
    gettimeofday(&end, NULL);
    *duration = (end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec);
    //printf("%d closeSend\n", rank);
    if(net->closeSend(sendComm)){ failed=1; goto out; }
  } 

out:
  return failed;
}

int IB_tester(ncclNet_t *net, char *data, size_t bytes, size_t *duration, int dev, int rank, int nranks, MPI_Comm comm){
  if(!rank) printf("IB tester\n");

  int failed = 0;
  if(rank==0){
    for(int rnk=1; rnk<nranks; rnk++){
      char listenHandle[NCCL_NET_HANDLE_MAXSIZE];
      char *listenComm; 
      //printf("%d listen\n", rank);
      if(net->listen(0, (void *)listenHandle, (void **)&listenComm)){ failed=1; goto out; }

      //printf("%d MPI_Send\n", rank);
      if(MPI_Send(listenHandle, NCCL_NET_HANDLE_MAXSIZE, MPI_BYTE, rnk, 0, comm)){ failed=1; goto out; }

      //printf("%d accept\n", rank);
      char *recvComm;
      if(net->accept(listenComm, (void **)&recvComm)){ failed=1; goto out; }

      //printf("%d closeListen\n", rank);
      if(net->closeListen(listenComm)){ failed=1; goto out; }

      /*ping*/
      int type = 0;
      type |= NCCL_PTR_HOST;
      char *request;
      //printf("%d recv\n", rank);
      struct timeval start, end;
      gettimeofday(&start, NULL);
      if(net->irecv(recvComm, data, bytes, type, (void **)&request)){ failed=1; goto out; }

      int done=0;
      do {
        int size = -1;
        if(net->test(request, &done, &size)){ failed=1; goto out; }
      } while(!done);
      gettimeofday(&end, NULL);
      *duration = (end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec);
      //printf("%d closeRecv\n", rank);
      if(net->closeRecv(recvComm)){ failed=1; goto out; }
    }
  }else{
    char connectHandle[NCCL_NET_HANDLE_MAXSIZE];
    char *sendComm;

    //printf("%d MPI_Recv\n", rank);
    if(MPI_Recv(connectHandle, NCCL_NET_HANDLE_MAXSIZE, MPI_BYTE, 0, 0, comm, MPI_STATUS_IGNORE)){ failed=1; goto out; }

    //printf("%d connect\n", rank);
    if(net->connect(0, connectHandle, (void **)&sendComm)){ failed=1; goto out; }

    /*pong*/  
    int type = 0;
    type |= NCCL_PTR_HOST;
    char *request;
    //printf("%d send\n", rank);
    struct timeval start, end;
    gettimeofday(&start, NULL);
    if(net->isend(sendComm, data, bytes, type, (void **)&request)){ failed=1; goto out; };

    int done=0;
    do {
      int size = -1;
      if(net->test(request, &done, &size)){ failed=1; goto out; }
    } while(!done);
    gettimeofday(&end, NULL);
    *duration = (end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec);
    //printf("%d closeSend\n", rank);
    if(net->closeSend(sendComm)){ failed=1; goto out; }
  } 

out:
  return failed;
}

extern "C"
void ncclMpiHook(MPI_Comm comm);

int MPI_tester(ncclNet_t *net, char *data, size_t bytes, size_t *duration, int dev, int rank, int nranks, MPI_Comm comm){
  if(!rank) printf("MPI tester\n");

  ncclMpiHook(MPI_COMM_WORLD);
  int failed = 0;
  if(rank==0){
    for(int rnk=1; rnk<nranks; rnk++){
      char listenHandle[NCCL_NET_HANDLE_MAXSIZE];
      char *listenComm; 
      if(net->listen(0, (void *)listenHandle, (void **)&listenComm)){ failed=1; goto out; }
      printf("%d listen\n", rank);

      if(MPI_Send(listenHandle, NCCL_NET_HANDLE_MAXSIZE, MPI_BYTE, rnk, 0, comm)){ failed=1; goto out; }
      printf("%d MPI_Send\n", rank);

      char *recvComm;
      if(net->accept(listenComm, (void **)&recvComm)){ failed=1; goto out; }
      printf("%d accept\n", rank);

      if(net->closeListen(listenComm)){ failed=1; goto out; }
      printf("%d closeListen\n", rank);

      /*ping*/
      int type = 0;
      type |= NCCL_PTR_HOST;
      char *request;
      struct timeval start, end;
      gettimeofday(&start, NULL);
      if(net->irecv(recvComm, data, bytes, type, (void **)&request)){ failed=1; goto out; }
      printf("%d recv\n", rank);

      int done=0;
      do {
        int size = -1;
        if(net->test(request, &done, &size)){ failed=1; goto out; }
      } while(!done);
      gettimeofday(&end, NULL);
      *duration = (end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec);

      if(net->closeRecv(recvComm)){ failed=1; goto out; }
      printf("%d closeRecv\n", rank);
    }
  }else{
    char connectHandle[NCCL_NET_HANDLE_MAXSIZE];
    char *sendComm;
    if(MPI_Recv(connectHandle, NCCL_NET_HANDLE_MAXSIZE, MPI_BYTE, 0, 0, comm, MPI_STATUS_IGNORE)){ failed=1; goto out; }
    printf("%d MPI_Recv\n", rank);

    if(net->connect(0, connectHandle, (void **)&sendComm)){ failed=1; goto out; }
    printf("%d connect\n", rank);

    /*pong*/  
    int type = 0;
    type |= NCCL_PTR_HOST;
    char *request;
    struct timeval start, end;
    gettimeofday(&start, NULL);
    if(net->isend(sendComm, data, bytes, type, (void **)&request)){ failed=1; goto out; };
    printf("%d send\n", rank);

    int done=0;
    do {
      int size = -1;
      if(net->test(request, &done, &size)){ failed=1; goto out; }
    } while(!done);
    gettimeofday(&end, NULL);
    *duration = (end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec);

    if(net->closeSend(sendComm)){ failed=1; goto out; }
    printf("%d closeSend\n", rank);
  } 

out:
  return failed;
}

int (*testers[]) (ncclNet_t *, char *, size_t, size_t *, int, int, int, MPI_Comm) = {&Socket_tester, &IB_tester, &MPI_tester};

#define MAX_SIZE (32*1024*1024)

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
  //ncclNet_t *nets[] = {&ncclNetSocket, &ncclNetIb, &ncclMpi};
  ncclNet_t *nets[] = {&ncclNetSocket, &ncclNetIb};
  for(int i=0; i<sizeof(nets)/sizeof(nets[0]); i++){
    ncclNet_t *net = nets[i]; 
    if(!rank){
      printf("ncclNet implementation %s found \n", net->name);
    }
    for(size_t bytes=1; bytes<=MAX_SIZE; bytes*=32){
      if(!rank) {printf("Send/Recv %lld bytes\n", bytes);}
      size_t duration=0;
      struct timeval start, end;
      gettimeofday(&start, NULL);
      failed = testers[i](net, data, bytes, &duration, 0, rank, nranks, MPI_COMM_WORLD); /*TODO: when NIC is not dev0*/
      gettimeofday(&end, NULL);
      size_t tot_duration = (end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec);
      if(!rank) {printf("Duration (total) %lld us duration (data transfer) %lld us Bandwidth %lld MB/s\n", tot_duration, duration, bytes/duration);}
      if (failed) goto out;
    }
  }
  printf("[%d] : Test successful\n", rank);
  delete data;
  MPI_Finalize();

out:
  if(failed){
    printf("[%d] Test failed\n", rank);
    delete data;
    MPI_Finalize();
  }
  return failed;
}
