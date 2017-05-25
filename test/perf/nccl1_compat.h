/*************************************************************************
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENCE.txt for license information
 ************************************************************************/

#ifndef NCCL1_COMPAT_H
#define NCCL1_COMPAT_H

#ifndef NCCL_MAJOR // NCCL 1.x
#define ncclNumOps nccl_NUM_OPS
#define ncclNumTypes nccl_NUM_TYPES
//#define ncclInt8 ncclChar
//#define ncclUint8 ncclChar
//#define ncclInt32 ncclInt
//#define ncclUint32 ncclInt
//#define ncclFloat16 ncclHalf
//#define ncclFloat32 ncclFloat
//#define ncclFloat64 ncclDouble
static ncclResult_t ncclGroupStart() { return ncclSuccess; }
static ncclResult_t ncclGroupEnd() { return ncclSuccess; }
#endif

#endif
