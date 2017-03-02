/*************************************************************************
 * Copyright (c) 2015-2017, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <stdio.h>
#include "mpi.h"
#include "nccl.h"

extern ncclNet_t ncclNetSocket; 
extern ncclNet_t ncclNetIb; 

int main(int argc, char *argv[]) {
  int nranks, rank;

  int threadProvided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &threadProvided);
  printf("provided : %d\n", threadProvided);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPI_Barrier(MPI_COMM_WORLD);

  ncclNet_t *nets[] = {&ncclNetSocket, &ncclNetIb};
  for(int i=0; i<sizeof(nets)/sizeof(nets[0]); i++){
    ncclNet_t *net = nets[i]; 
    if(!rank){
      printf("ncclNet implementation %s found \n", net->name);
    }
  }
  int failed = 0;
#if 0
  if(rank==0){
    for(int rnk=1; rnk<nranks; rnk++){
      if(net->listen()){
         goto out;
      }
      if(net->accept()){
         goto out;
      }
      net->closeListen();
      /*ping*/
      net->irecv();
      int done=0;
      do {
        net->test(&done);
      } while(!done);
      net->closeRecv();
    }
  }else{
    if(net->onnect()){
       goto out;
    }
    /*pong*/  
    net->isend();
    int done=0;
    do {
      net->test(&done);
    } while(!done);
    net->closeSend();
  } 
#endif
  MPI_Finalize();

out:
  if(failed){
    printf("[%d] Test failed\n", rank);
  }
  printf("[%d] : Test successful\n", rank);
  return failed;
}
