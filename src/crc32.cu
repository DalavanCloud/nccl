/*************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <stdio.h>
#include "core.h" // for CUDACHECK

#define POLYNOMIAL 0x04c11db7L      // Standard CRC-32 ppolynomial
static unsigned int crc_table[256]; // Table of 8-bit remainders
static int tableLoaded = 0;

static void crcInit(void) {
  int i, j;
  unsigned int crc_accum;

  for (i=0;  i<256;  i++) {
    crc_accum = ( i << 24 );
    for ( j = 0;  j < 8;  j++ ) {
      if ( crc_accum & 0x80000000L )
        crc_accum = (crc_accum << 1) ^ POLYNOMIAL;
      else
        crc_accum = (crc_accum << 1);
    }
    crc_table[i] = crc_accum;
  }
}

unsigned calcCRCHost(unsigned char *data_blk_ptr, size_t data_blk_size) {
  if (tableLoaded == 0) {
    crcInit();
    tableLoaded = 1;
  }

  unsigned int crc_accum = 0x11223344; // Initial CRC value used in cuDNN
  int i;
  for (size_t j=0; j<data_blk_size; j++) {
    i = ((int) (crc_accum >> 24) ^ *data_blk_ptr++) & 0xFF;
    crc_accum = (crc_accum << 8) ^ crc_table[i];
  }
  crc_accum = ~crc_accum;
  return crc_accum;
}


static __global__ void CRCKernel(unsigned char* data, int bytes, int rank) {
  __shared__ unsigned crc_table[256];
  __shared__ unsigned char buffer[256];

  // Build table of 8-bit remainders
  int crc_accum = threadIdx.x << 24;
  for (int j=0; j<8; ++j) {
    const int mask = (crc_accum & 0x80000000) ? POLYNOMIAL : 0;
    crc_accum = (crc_accum << 1) ^ mask;
  }
  crc_table[threadIdx.x] = crc_accum;

  unsigned int crc_val = 0x11223344; // Initial CRC value used in cuDNN
  for(int i=threadIdx.x; i<bytes; i+=256) {
    buffer[threadIdx.x] = data[i];
    __syncthreads();

    if (threadIdx.x == 0) {
      const int remaining = bytes - i;
      const int n = (remaining > 256) ? 256 : remaining;
      for(int j=0; j<n; ++j) {
        int t = ((int)(crc_val >> 24) ^ buffer[j]) & 0xFF;
        crc_val = (crc_val << 8) ^ crc_table[t];
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0)
    printf("NCCL Rank %d CRC 0x%.8x\n", rank, ~crc_val);
}

void printCRCDev(unsigned char* data,
                 int bytes,
                 int rank,
                 cudaStream_t stream)
{
  const dim3 grid(1, 1, 1);
  const dim3 block(256, 1, 1);
  void* argptrs[] = {&data, &bytes, &rank};
  CUDACHECK(cudaLaunchKernel((void*)CRCKernel, grid, block, argptrs, 0, stream));
}

