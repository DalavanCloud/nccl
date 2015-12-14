/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2015      NVIDIA Corporation. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include <cmath>
#include <nccl.h>

#include "coll_nccl.h"
#include "coll_nccl_misc.h"
#include "ompi/datatype/ompi_datatype.h"
#include "ompi/datatype/ompi_datatype_internal.h"

#include "coll_nccl_kernels.h"

const int blocks = 8;
const int threads = 512;

template <typename T>
__global__ void
mca_coll_nccl_add_kernel(T * a, T * b, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = tid; i < N; i += blockDim.x * gridDim.x)
        a[i] += b[i];
}

template <typename T>
__global__ void
mca_coll_nccl_prod_kernel(T * a, T * b, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = tid; i < N; i += blockDim.x * gridDim.x)
        a[i] *= b[i];
}

template <typename T>
__global__ void
mca_coll_nccl_max_kernel(T * a, T * b, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = tid; i < N; i += blockDim.x * gridDim.x)
        a[i] = max(a[i], b[i]);
}

template <typename T>
__global__ void
mca_coll_nccl_min_kernel(T * a, T * b, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = tid; i < N; i += blockDim.x * gridDim.x)
        a[i] = min(a[i], b[i]);
}

template <typename T>
void
mca_coll_nccl_cuda_op_reduce_withType(T * a, T * b, int N, ncclRedOp_t op,
                                      cudaStream_t stream)
{
    switch(op)
    {
        case ncclSum:
            mca_coll_nccl_add_kernel<T><<<blocks, threads, 0, stream>>>(a,b,N);
            break;
        case ncclProd:
            mca_coll_nccl_prod_kernel<T><<<blocks, threads, 0, stream>>>(a,b,N);
            break;
        case ncclMax:
            mca_coll_nccl_max_kernel<T><<<blocks, threads, 0, stream>>>(a,b,N);
            break;
        case ncclMin:
            mca_coll_nccl_min_kernel<T><<<blocks, threads, 0, stream>>>(a,b,N);
            break;
    }
}

extern "C"
void
mca_coll_nccl_cuda_op_reduce(void * a, void * b, int N,
                        struct ompi_datatype_t *dtype,
                        ncclRedOp_t op, cudaStream_t stream)
{
    switch(dtype->id)
    {
        case OMPI_DATATYPE_MPI_CHAR:
            mca_coll_nccl_cuda_op_reduce_withType<char>((char *)a, (char *)b, N,
                                                        op, stream);
            break;
        case OMPI_DATATYPE_MPI_INT:
            mca_coll_nccl_cuda_op_reduce_withType<int>((int *)a, (int *)b, N,
                                                       op, stream);
            break;
        case OMPI_DATATYPE_MPI_FLOAT:
            mca_coll_nccl_cuda_op_reduce_withType<float>((float *)a, (float *)b,
                                                         N, op, stream);
            break;
        case OMPI_DATATYPE_MPI_DOUBLE:
            mca_coll_nccl_cuda_op_reduce_withType<double>((double *)a,
                                                          (double *)b, N, op, stream);
            break;
    }
    cudaStreamSynchronize(stream);
}
