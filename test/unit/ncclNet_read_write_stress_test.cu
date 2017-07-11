/*************************************************************************
 * Copyright (c) 2015-2017, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <random>
#include "mpi.h"
#include "nccl_net.h"
#include "debug.h"
#include <sys/types.h>
#include <unistd.h>

extern ncclNet_t ncclNetSocket; 
extern ncclNet_t ncclNetIb; 

#define MAX_REQUESTS 128
#define USE_SOCKET 1
#define USE_IB 1
//#define REQUEST_CHECK_EAGER 1
#define REQUEST_CHECK_BATCH 1
//#define REQUEST_CHECK_DELAY_BY_1 1

typedef enum op {
  READ = 0,
  WRITE = 1,
} op_t;

int tester(ncclNet_t *net, char *data, char *data_d, size_t bytes, int type, int rank, int nranks){
  int failed = 0;
  int *scores;
  int ndev=0;
  char listenHandle[NCCL_NET_HANDLE_MAXSIZE], connectHandle[NCCL_NET_HANDLE_MAXSIZE];
  int cnt = 0;
  std::random_device rd;
  std::mt19937 rng(rd());
  std::uniform_int_distribution<int> uni(READ, WRITE);
  MPI_Barrier(MPI_COMM_WORLD);
  if(rank==0){
   char *listenComm, *sendComm[nranks+1], *recvComm[nranks+1]; 
   char *request[2*nranks*MAX_REQUESTS];
   if(net->devices(&ndev, &scores)){failed=1; goto out; }
   INFO("Rank %d ndev %d scores : ", rank, ndev);
   for(int i=0; i<ndev; i++){
      INFO("scores[%d] = %d", i, scores[i]);
    }
    for(int rnk=1; rnk<nranks; rnk++){
	    if(net->listen(0, (void *)listenHandle, (void **)&listenComm)){ failed=1; goto out; }
	    if(MPI_Send(listenHandle, NCCL_NET_HANDLE_MAXSIZE, MPI_BYTE, rnk, 0, MPI_COMM_WORLD)){ failed=1; goto out; }
	    if(net->accept(listenComm, (void **)&recvComm[rnk])){ failed=1; goto out; }
	    INFO("Rank %d accepted connection from rank %d", rank, rnk);

	    if(net->closeListen(listenComm)){ failed=1; goto out; }
	    INFO("Rank %d closed listen comm", rank);

	    if(MPI_Recv(connectHandle, NCCL_NET_HANDLE_MAXSIZE, MPI_BYTE, rnk, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE)){ failed=1; goto out; }
	    if(net->connect(0, connectHandle, (void **)&sendComm[rnk])){ failed=1; goto out; }
	    INFO("Rank %d connected to rank %d", rank, rnk);

            if (type == NCCL_PTR_HOST) {
	      if(net->isend(sendComm[rnk], data, bytes, type, (void **)&request[cnt++])){ failed=1; goto out; }
            } else if (type == NCCL_PTR_CUDA){
	      if(net->isend(sendComm[rnk], data_d, bytes, type, (void **)&request[cnt++])){ failed=1; goto out; }
            }
	    INFO("Rank %d posted first send", rank);

	    int done=0;
	    do {
		    int size = -1;
		    if(net->test(request[cnt-1], &done, &size)){ failed=1; goto out; }
	    } while(!done);
	    INFO("Rank %d completed first send for %s type %d", rank, net->name, type);
    }//for rnk<nranks

    for(int rnk=1; rnk<nranks; rnk++){
            if(!strcmp(net->name, "IB")) {
              for(int i=0; i<MAX_REQUESTS; i++){
                 auto op = i%2;//uni(rng);
                 if(op == READ) {
	            INFO("Rank %d posting %dth op (recv), req %d", rank, i, cnt);
                    if (type == NCCL_PTR_HOST) {
	              if(net->irecv(recvComm[rnk], data, bytes, type, (void **)&request[cnt++])){ failed=1; goto out; }
                    } else if (type == NCCL_PTR_CUDA){
	              if(net->irecv(recvComm[rnk], data_d, bytes, type, (void **)&request[cnt++])){ failed=1; goto out; }
                    }
	            INFO("Rank %d posted %dth op (recv), req %d", rank, i, cnt-1);
#ifdef REQUEST_CHECK_EAGER
		    INFO("Rank %d request %d checking %p", rank, cnt-1, request[cnt-1]);
		    int done=0;
		    do {
			    int size = -1;
			    if(net->test(request[cnt-1], &done, &size)){ failed=1; goto out; }
		    } while(!done);
		    INFO("Rank %d request %d done", rank, cnt-1);
#endif
                 } else if (op == WRITE){
	            INFO("Rank %d posting %dth op (send), req %d", rank, i, cnt);
                    if (type == NCCL_PTR_HOST) {
	              if(net->isend(sendComm[rnk], data, bytes, type, (void **)&request[cnt++])){ failed=1; goto out; }
                    } else if (type == NCCL_PTR_CUDA){
	              if(net->isend(sendComm[rnk], data_d, bytes, type, (void **)&request[cnt++])){ failed=1; goto out; }
                    }
	            INFO("Rank %d posted %dth op (send), req %d", rank, i, cnt - 1);
#ifdef REQUEST_CHECK_EAGER
		    INFO("Rank %d request %d checking %p", rank, cnt-1, request[cnt-1]);
		    int done=0;
		    do {
			    int size = -1;
			    if(net->test(request[cnt-1], &done, &size)){ failed=1; goto out; }
		    } while(!done);
		    INFO("Rank %d request %d done", rank, cnt-1);
#endif
                 } else {
                    WARN("op outside range %d", op);
                 }
#ifdef REQUEST_CHECK_DELAY_BY_1
		 if((i%2 == 0) && (i > 0)){
			 INFO("Rank %d request %d checking", rank, cnt-3);
			 int done=0;
			 do {
				 int size = -1;
				 if(net->test(request[cnt-3], &done, &size)){ failed=1; goto out; }
			 } while(!done);
			 INFO("Rank %d request %d done", rank, cnt-3);
			 INFO("Rank %d request %d checking", rank, cnt-2);
			 done=0;
			 do {
				 int size = -1;
				 if(net->test(request[cnt-2], &done, &size)){ failed=1; goto out; }
			 } while(!done);
			 INFO("Rank %d request %d done", rank, cnt-2);
		 }
#endif
              }
              INFO("Rank %d posted %d ops", rank, MAX_REQUESTS);
#ifdef REQUEST_CHECK_BATCH
	      for(int i=0; i<MAX_REQUESTS; i++) {
                      INFO("Rank %d request %d checking", rank, i);
		      int done=0;
		      do {
			      int size = -1;
			      if(net->test(request[i+1], &done, &size)){ failed=1; goto out; }
		      } while(!done);
                      INFO("Rank %d request %d done", rank, i);
	      }
              INFO("Rank %d completed %d ops", rank, MAX_REQUESTS);
#endif
            }
    }//for rnk<nranks
  }else{
    char *listenComm, *sendComm, *recvComm; 
    char *request[2*MAX_REQUESTS];
    if(net->devices(&ndev, &scores)){failed=1; goto out; }
    INFO("Rank %d ndev %d scores : ", rank, ndev);
    for(int i=0; i<ndev; i++){
      INFO("scores[%d] = %d", i, scores[i]);
    }
    if(MPI_Recv(connectHandle, NCCL_NET_HANDLE_MAXSIZE, MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE)){ failed=1; goto out; }
    if(net->connect(0, connectHandle, (void **)&sendComm)){ failed=1; goto out; }
    INFO("Rank %d connected to rank 0", rank);

    if(net->listen(0, (void *)listenHandle, (void **)&listenComm)){ failed=1; goto out; }
    if(MPI_Send(listenHandle, NCCL_NET_HANDLE_MAXSIZE, MPI_BYTE, 0, 0, MPI_COMM_WORLD)){ failed=1; goto out; }
    if(net->accept(listenComm, (void **)&recvComm)){ failed=1; goto out; }
    INFO("Rank %d accepted connection from rank 0", rank);

    if(net->closeListen(listenComm)){ failed=1; goto out; }
    INFO("Rank %d closed listen comm", rank);

    if(type == NCCL_PTR_HOST) {
      if(net->irecv(recvComm, data, bytes, type, (void **)&request[cnt++])){ failed=1; goto out; }
    } else if (type == NCCL_PTR_CUDA){ 
      if(net->irecv(recvComm, data_d, bytes, type, (void **)&request[cnt++])){ failed=1; goto out; }
    }
    INFO("Rank %d posted first recv", rank);

    int done=0;
    do {
	    int size = -1;
	    if(net->test(request[cnt-1], &done, &size)){ failed=1; goto out; }
    } while(!done);
    INFO("Rank %d completed first recv for %s type %d", rank, net->name, type);

    if(!strcmp(net->name, "IB")){
      for(int i=0; i<MAX_REQUESTS; i++){
        auto op = i%2;//uni(rng);
        if(op == READ) {
#if defined(REQUEST_CHECK_DELAY_BY_1) || defined(REQUEST_CHECK_BATCH)
          INFO("Rank %d posting %dth op (recv), req %d ", rank, i, cnt);
	  if (type == NCCL_PTR_HOST) {
            if(net->irecv(recvComm, data, bytes, type, (void **)&request[cnt++])){ failed=1; goto out; }
          } else if (type == NCCL_PTR_CUDA) {
            if(net->irecv(recvComm, data_d, bytes, type, (void **)&request[cnt++])){ failed=1; goto out; }
          }
          INFO("Rank %d posted %dth op (recv), req %d ", rank, i, cnt - 1);
#endif
#ifdef REQUEST_CHECK_EAGER
	  if (type == NCCL_PTR_HOST) {
            if(net->isend(sendComm, data, bytes, type, (void **)&request[cnt++])){ failed=1; goto out; }
          } else if (type == NCCL_PTR_CUDA) {
            if(net->isend(sendComm, data_d, bytes, type, (void **)&request[cnt++])){ failed=1; goto out; }
          }
          //INFO("Rank %d posted %dth op (send) ", rank, i);
	  INFO("Rank %d request %d checking", rank, i);
	  int done=0;
	  do {
		  int size = -1;
		  if(net->test(request[cnt-1], &done, &size)){ failed=1; goto out; }
	  } while(!done);
	  INFO("Rank %d request %d done", rank, i);
#endif
        } else if (op == WRITE){
#if defined(REQUEST_CHECK_DELAY_BY_1) || defined(REQUEST_CHECK_BATCH)
          INFO("Rank %d posting %dth op (send), req %d ", rank, i, cnt);
	  if (type == NCCL_PTR_HOST) {
            if(net->isend(sendComm, data, bytes, type, (void **)&request[cnt++])){ failed=1; goto out; }
          } else if (type == NCCL_PTR_CUDA) {
            if(net->isend(sendComm, data_d, bytes, type, (void **)&request[cnt++])){ failed=1; goto out; }
          }
          INFO("Rank %d posted %dth op (send), req %d ", rank, i, cnt - 1);
#endif
#ifdef REQUEST_CHECK_EAGER
          if (type == NCCL_PTR_HOST) {
            if(net->irecv(recvComm, data, bytes, type, (void **)&request[cnt++])){ failed=1; goto out; }
          } else if (type == NCCL_PTR_CUDA) {
            if(net->irecv(recvComm, data_d, bytes, type, (void **)&request[cnt++])){ failed=1; goto out; }
          }
          //INFO("Rank %d posted %dth op (recv) ", rank, i);
	  INFO("Rank %d request %d checking", rank, i);
	  int done=0;
	  do {
		  int size = -1;
		  if(net->test(request[cnt-1], &done, &size)){ failed=1; goto out; }
	  } while(!done);
	  INFO("Rank %d request %d done", rank, i);
#endif
        } else {
          WARN("op outside range %d", op);
        }
#if defined(REQUEST_CHECK_DELAY_BY_1)
        if((i%2 == 0) && (i > 0)){
	  INFO("Rank %d request %d checking", rank, cnt-3);
	  int done=0;
	  do {
		  int size = -1;
		  if(net->test(request[cnt-3], &done, &size)){ failed=1; goto out; }
	  } while(!done);
	  INFO("Rank %d request %d done", rank, cnt-3);
	  INFO("Rank %d request %d checking", rank, cnt-2);
	  done=0;
	  do {
		  int size = -1;
		  if(net->test(request[cnt-2], &done, &size)){ failed=1; goto out; }
	  } while(!done);
	  INFO("Rank %d request %d done", rank, cnt-2);
        }
#endif
      }
      INFO("Rank %d posted %d ops", rank, MAX_REQUESTS);
#ifdef REQUEST_CHECK_BATCH
      for(int i=0; i<MAX_REQUESTS; i++) {
              INFO("Rank %d request %d checking", rank, i);
	      int done=0;
	      do {
		      int size = -1;
		      if(net->test(request[i+1], &done, &size)){ failed=1; goto out; }
	      } while(!done);
	      INFO("Rank %d request %d done", rank, i);
      }
      INFO("Rank %d completed %d ops", rank, MAX_REQUESTS);
#endif
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
out:
  return failed;
}

#define MAX_SIZE (1024*1024)

int main(int argc, char *argv[]) {
  int nranks, rank;

  int threadProvided;

  initDebug();

  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &threadProvided);
  INFO("provided : %d", threadProvided);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPI_Barrier(MPI_COMM_WORLD);

  int failed = 0;
  char *data = new char[MAX_SIZE];
  char *data_d;
  cudaMalloc(&data_d, MAX_SIZE);
#if defined(USE_SOCKET) &&  defined(USE_IB)
  ncclNet_t *nets[] = {&ncclNetSocket, &ncclNetIb};
#endif
#if defined(USE_SOCKET) && !defined(USE_IB)
  ncclNet_t *nets[] = {&ncclNetSocket};
#endif
#if !defined(USE_SOCKET) && defined(USE_IB)
  ncclNet_t *nets[] = {&ncclNetIb};
#endif
  for(int i=0; i<sizeof(nets)/sizeof(nets[0]); i++){
    ncclNet_t *net = nets[i]; 
    if(!rank) INFO("net->name %s", net->name);
    if (!strcmp(net->name, "Socket")) {
      int type = NCCL_PTR_HOST;
      failed = tester(net, data, data_d, MAX_SIZE, type, rank, nranks);
      if (failed) goto out;
    } else {
      int type = NCCL_PTR_HOST;
      failed = tester(net, data, data_d, MAX_SIZE, type, rank, nranks);
      if (failed) goto out;
      type = 0; type = NCCL_PTR_CUDA;
      failed = tester(net, data, data_d, MAX_SIZE, type, rank, nranks);
      if (failed) goto out;
    }
  }
  printf("[%d] Test successful", rank);
  MPI_Finalize();
  delete data;
  cudaFree(data_d);

out:
  if(failed){
    printf("[%d] Test failed", rank);
    MPI_Finalize();
    delete data;
    cudaFree(data_d);
  }
  return failed;
}
