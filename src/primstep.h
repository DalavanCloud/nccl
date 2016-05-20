/*************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 ************************************************************************/


#ifndef PRIMSTEP_H_
#define PRIMSTEP_H_

#include "copy_kernel.h"
#include "reduce_kernel.h"


// THREADS is number of worker threads per NCCL CTA. An additional thread
// is included in each CTA to handle threadfences. THREADS is used in ASM
// statements, and must be a defined constant (not a template parameter).

#define NUM_THREADS 256


// UNROLL_COUNT controls unroll of innermost loops in Copy, ReduceCopy, etc.

#define UNROLL_COUNT 8


// Generic collective step
template <typename WAITFUNC, typename STEPFUNC, typename POSTFUNC>
__device__ __forceinline__ void GenericStep(int step, int substeps,
    WAITFUNC allready, STEPFUNC operation, int len, POSTFUNC alldone) {
  enum {MAX_SUBSTEPS=4};

  if (threadIdx.x < NUM_THREADS) {
    int sliceSize = len / substeps;
    int sliceOffset = 0;
    #pragma unroll 1
    for (int sub=MAX_SUBSTEPS-substeps; sub<MAX_SUBSTEPS; ++sub) {
      if ( ! std::is_same<WAITFUNC, NOSYNC>::value ) {
        if (threadIdx.x == 0) {
          allready.wait((step-1)*MAX_SUBSTEPS+sub+1);
        }
        BAR(sync, 1, NUM_THREADS);
      }
      operation(sliceOffset, (sub==MAX_SUBSTEPS-1) ? len-sliceOffset : sliceSize);
      sliceOffset += sliceSize;
      __syncthreads();
    }
  } else {
    #pragma unroll 1
    for (int sub=MAX_SUBSTEPS-substeps; sub<MAX_SUBSTEPS; ++sub) {
      __syncthreads();
      if ( ! std::is_same<POSTFUNC, NOSYNC>::value ) {
        __threadfence_system();
        alldone.post(step*MAX_SUBSTEPS+sub+1);
      }
    }
  }
}


template <typename T, typename WAITFUNC, typename POSTFUNC>
__device__ __forceinline__ void CopyStep(int step, int substeps, WAITFUNC allready,
    volatile const T* src, volatile T* dst, int len, POSTFUNC alldone) {

  auto operation = [src, dst] (int offset, int size) {
    Copy<UNROLL_COUNT, NUM_THREADS>(dst+offset, src+offset, size);
  };

  GenericStep(step, substeps, allready, operation, len, alldone);
}


template <typename REDFUNC, typename WAITFUNC, typename POSTFUNC, typename T>
__device__ __forceinline__ void ReduceStep(int step, int substeps, WAITFUNC allready,
    const volatile T* src1, const volatile T* src2, volatile T* dst,
    int len, POSTFUNC alldone) {

  auto operation = [src1, src2, dst] (int offset, int size) {
    Reduce<UNROLL_COUNT, NUM_THREADS, REDFUNC>(dst+offset,
        src1+offset, src2+offset, size);
  };

  GenericStep(step, substeps, allready, operation, len, alldone);
}


template <typename WAITFUNC, typename POSTFUNC, typename T>
__device__ __forceinline__ void DoubleCopyStep(int step, int substeps, WAITFUNC allready,
    volatile const T* src, volatile T* dst1, volatile T* dst2, int len, POSTFUNC alldone) {

  auto operation = [src, dst1, dst2] (int offset, int size) {
    DoubleCopy<UNROLL_COUNT, NUM_THREADS>(dst1+offset,
        dst2+offset, src+offset, size);
  };

  GenericStep(step, substeps, allready, operation, len, alldone);
}


template <typename REDFUNC, typename WAITFUNC, typename POSTFUNC, typename T>
__device__ __forceinline__ void ReduceCopyStep(int step, int substeps, WAITFUNC allready,
    const volatile T* src1, const volatile T* src2, volatile T* dst1, volatile T* dst2, 
    int len, POSTFUNC alldone) {

  auto operation = [src1, src2, dst1, dst2] (int offset, int size) {
    ReduceAndCopy<UNROLL_COUNT, NUM_THREADS, REDFUNC>(dst1+offset,
        dst2+offset, src1+offset, src2+offset, size);
  };

  GenericStep(step, substeps, allready, operation, len, alldone);
}


#endif // end include guard
