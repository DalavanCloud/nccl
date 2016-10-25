/*************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

ncclResult_t printCRCDev(unsigned char * data, int bytes, int rank, cudaStream_t s);
unsigned calcCRCHost(unsigned char * data, size_t bytes);

