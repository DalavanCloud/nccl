/*************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_SHM_H_
#define NCCL_SHM_H_

#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

static ncclResult_t shmOpen(const char* shmname, const int shmsize, void** shmPtr, void** devShmPtr, int create) {
  *shmPtr = NULL;
  int fd = shm_open(shmname, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
  if (fd == -1) {
    WARN("shm_open failed to open %s", shmname);
    return ncclSystemError;
  }

  if (create) {
    if (ftruncate(fd, shmsize) == -1) {
      WARN("ftruncate failed to allocate %d bytes", shmsize);
      shm_unlink(shmname);
      close(fd);
      return ncclSystemError;
    }
  }

  void *ptr = (struct ncclSendRecvMem*)mmap(NULL, shmsize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if (ptr == MAP_FAILED) {
    WARN("failure in mmap");
    shm_unlink(shmname);
    return ncclSystemError;
  }

  if (cudaHostRegister(ptr, shmsize, cudaHostRegisterMapped) != cudaSuccess) {
    WARN("failed to register host buffer");
    if (create) shm_unlink(shmname);
    munmap(ptr, shmsize);
    return ncclUnhandledCudaError;
  }   

  if (cudaHostGetDevicePointer(devShmPtr, ptr, 0) != cudaSuccess) {
    WARN("failed to get device pointer for local shmem");
    if (create) shm_unlink(shmname);
    munmap(ptr, shmsize);
    return ncclUnhandledCudaError;
  }
  *shmPtr = ptr;
  return ncclSuccess;
}

static ncclResult_t shmClose(void* shmPtr, void* devShmPtr, const char* shmname, const int shmsize) {
  if (shmname != NULL) shm_unlink(shmname);
  CUDACHECK(cudaHostUnregister(devShmPtr));
  if (munmap(shmPtr, shmsize) != 0) {
    WARN("munmap of shared memory failed");
    return ncclSystemError;
  }
  return ncclSuccess;
}

#endif
