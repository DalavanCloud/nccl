/*************************************************************************
 * Copyright (c) 2015-2017, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <stdio.h>
#include <string.h>
#include "mpi.h"
#include "nccl.h"

extern ncclNet_t ncclNetSocket; 
extern ncclNet_t ncclNetIb; 

int Socket_tester(ncclNet_t *net, int dev, int rank, int nranks){
  if(!rank) printf("Socket tester\n");

  int failed = 0;
  if(rank==0){
    for(int rnk=1; rnk<nranks; rnk++){
      char listenHandle[NCCL_NET_HANDLE_MAXSIZE];
      char *listenComm; 
      if(net->listen(0, (void *)listenHandle, (void **)&listenComm)){ failed=1; goto out; }

      char *recvComm;
      if(net->accept(listenComm, (void **)&recvComm)){ failed=1; goto out; }

      if(net->closeListen(listenComm)){ failed=1; goto out; }

      /*ping*/
      char data;
      int type = 0;
      type |= NCCL_PTR_HOST;
      char *request;
      if(net->irecv(recvComm, &data, sizeof(char), type, (void **)&request)){ failed=1; goto out; }

      int done=0;
      do {
        int size = -1;
        if(net->test(request, &done, &size)){ failed=1; goto out; }
      } while(!done);
      if(net->closeRecv(recvComm)){ failed=1; goto out; }
    }
  }else{
    char connectHandle[NCCL_NET_HANDLE_MAXSIZE];
    char *sendComm;
    if(net->connect(0, connectHandle, (void **)&sendComm)){ failed=1; goto out; }

    /*pong*/  
    char data;
    int type = 0;
    type |= NCCL_PTR_HOST;
    char *request;
    if(net->isend(sendComm, &data, sizeof(char), type, (void **)&request)){ failed=1; goto out; };

    int done=0;
    do {
      int size = -1;
      if(net->test(request, &done, &size)){ failed=1; goto out; }
    } while(!done);
    if(net->closeSend(sendComm)){ failed=1; goto out; }
  } 

out:
  return failed;
}

int IB_tester(ncclNet_t *net, int dev, int rank, int nranks){
  int failed = 0;  
  if(!rank) printf("IB tester\n");
  return failed;
}

int (*testers[]) (ncclNet_t *, int, int, int) = {&Socket_tester, &IB_tester};

int main(int argc, char *argv[]) {
  int nranks, rank;

  int threadProvided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &threadProvided);
  printf("provided : %d\n", threadProvided);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPI_Barrier(MPI_COMM_WORLD);

  int failed = 0;
  ncclNet_t *nets[] = {&ncclNetSocket, &ncclNetIb};
  for(int i=0; i<sizeof(nets)/sizeof(nets[0]); i++){
    ncclNet_t *net = nets[i]; 
    if(!rank){
      printf("ncclNet implementation %s found \n", net->name);
    }
    failed = testers[i](net, 0, rank, nranks); /*TODO: when NIC is not dev0*/
    if (failed) goto out;
  }
  MPI_Finalize();

out:
  if(failed){
    printf("[%d] Test failed\n", rank);
  }
  printf("[%d] : Test successful\n", rank);
  return failed;
}
