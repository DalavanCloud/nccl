/*************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_CRC32_H_
#define NCCL_CRC32_H_

ncclResult_t printCRCDev(unsigned char * data, int bytes, int rank, cudaStream_t s);
unsigned calcCRCHost(unsigned char * data, size_t bytes);

#endif
