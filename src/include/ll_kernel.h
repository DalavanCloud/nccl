/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_LL_KERNEL_H_
#define NCCL_LL_KERNEL_H_


#include "core.h"

static __device__ uint64_t readLL(union ncclLLFifoLine* src, uint32_t flag) {
  volatile uint64_t* valPtr = src->v;
  union ncclLLFifoLine line;
  do {
    line.v[0] = valPtr[0];
    line.v[1] = valPtr[1];
  } while ((line.flag1 != flag) || (line.flag2 != flag));
  uint64_t val = line.data1 + (((uint64_t)line.data2) << 32);
  return val;
}

static __device__ void storeLL(union ncclLLFifoLine* dst, uint64_t val, uint32_t flag) {
  union ncclLLFifoLine line;
  line.data1 = val;
  line.flag1 = flag;
  line.data2 = val >> 32;
  line.flag2 = flag;
  volatile uint64_t* valPtr = dst->v;
  valPtr[0] = line.v[0];
  valPtr[1] = line.v[1];
}

// Using memcpy handles misaligned pointers.
static __device__ uint64_t readAL(uint64_t* src) {
  uint64_t val;
  memcpy((char*)&val, (char*)src, sizeof(uint64_t));
  return val;
}
static __device__ void storeAL(uint64_t* dst, uint64_t val) {
  memcpy((char*)dst, (char*)&val, sizeof(uint64_t));
}

template <int THREADS, typename T, class FUNC>
class LLPrimitives {
  private:
  template <int HAS_SRC1, int HAS_SRC2, int HAS_DST1, int HAS_DST2>
  static __device__ void ReduceCopyGeneric(const T* src1, union ncclLLFifoLine* src2, T* dst1, union ncclLLFifoLine* dst2, int size, uint32_t iflag, uint32_t oflag) {
    if (size <= 0) return;
    size_t size64 = size * sizeof(T) / sizeof(uint64_t);
    uint64_t* src1A = (uint64_t*)src1;
    uint64_t* dst1A = (uint64_t*)dst1;
    int offset = threadIdx.x;
    // Do multiples of 64 bits
    #pragma unroll 1
    for (; offset < size64; offset += THREADS) {
      uint64_t val;
      if (HAS_SRC1) {
        val = readAL(src1A+offset);
        if (HAS_SRC2) val = MULTI<FUNC, T>()(readLL(src2+offset, iflag), val);
      } else if (HAS_SRC2) {
        val = readLL(src2+offset, iflag);
      }
      if (HAS_DST1) storeAL(dst1A+offset, val);
      if (HAS_DST2) storeLL(dst2+offset, val, oflag);
    }
    // Finish last word
    int sizeDone = size64*(sizeof(uint64_t)/sizeof(T));
    int sizeRem = size - sizeDone;
    if (threadIdx.x == 0 && sizeRem) {
      const T* src1B = src1 + sizeDone;
      T* dst1B = dst1 + sizeDone;

      uint64_t lastVal;
      T* vals = (T*)&lastVal;

      if (HAS_SRC2) {
        uint64_t lastVal2 = readLL(src2+size64, iflag);
        T* src2B = (T*)&lastVal2;
        for (int offset = 0; offset < sizeRem; offset++) {
          vals[offset] = HAS_SRC1 ? FUNC()(src2B[offset], src1B[offset]) : src2B[offset];
        }
      } else if (HAS_SRC1) {
        for (int offset = 0; offset < sizeRem; offset++) {
          vals[offset] = src1B[offset];
        }
      }
      if (HAS_DST2) storeLL(dst2+size64, lastVal, oflag);
      if (HAS_DST1) {
        for (int offset = 0; offset < sizeRem; offset++) {
          dst1B[offset] = vals[offset];
        }
      }
    }
  }
  public:
  static __device__ void ReduceCopy(const T* src, union ncclLLFifoLine* dst, int size, uint32_t oflag) {
    return ReduceCopyGeneric<1, 0, 0, 1>(src, NULL, NULL, dst, size, 0, oflag);
  }

  static __device__ void ReduceCopy(union ncclLLFifoLine* src, T* dst, int size, uint32_t iflag) {
    return ReduceCopyGeneric<0, 1, 1, 0>(NULL, src, dst, NULL, size, iflag, 0);
  }

  static __device__ void ReduceCopy(const T* src1, union ncclLLFifoLine* src2, union ncclLLFifoLine* dst, int size, uint32_t iflag, uint32_t oflag) {
    return ReduceCopyGeneric<1, 1, 0, 1>(src1, src2, NULL, dst, size, iflag, oflag);
  }

  static __device__ void ReduceCopy(const T* src1, union ncclLLFifoLine* src2, T* dst, int size, uint32_t iflag) {
    return ReduceCopyGeneric<1, 1, 1, 0>(src1, src2, dst, NULL, size, iflag, 0);
  }

  static __device__ void ReduceCopy(const T* src, T* dst1, union ncclLLFifoLine* dst2, int size, uint32_t oflag) {
    return ReduceCopyGeneric<1, 0, 1, 1>(src, NULL, dst1, dst2, size, 0, oflag);
  }

  static __device__ void ReduceCopy(union ncclLLFifoLine* src, T* dst1, union ncclLLFifoLine* dst2, int size, uint32_t iflag, uint32_t oflag) {
    return ReduceCopyGeneric<0, 1, 1, 1>(NULL, src, dst1, dst2, size, iflag, oflag);
  }

  static __device__ void ReduceCopy(const T* src1, union ncclLLFifoLine* src2, T* dst1, union ncclLLFifoLine* dst2, int size, uint32_t iflag, uint32_t oflag) {
    return ReduceCopyGeneric<1, 1, 1, 1>(src1, src2, dst1, dst2, size, iflag, oflag);
  }
};

// Common macros

#define STEP_TO_SLOT(step) \
  (step % NUM_LL_CHUNKS)

#define WAIT_NEXT \
  if (tid == 0) { \
    while (sendHead + NUM_LL_CHUNKS <= step) { \
      sendHead = sendHeadPtr[0]; \
    } \
  } \
  __syncthreads();

#define POST_SIZE \
  if (tid == 0 && sizesFifo) sizesFifo[step % NUM_LL_CHUNKS] = (maxOffset <= 0) ? -1 : (maxOffset*2*(int)sizeof(T));

#define ACK_PREV \
  __syncthreads();  \
  if (tid == 0) recvHeadPtr[0] = step;

#define FIFO_CLEANING_AND_SAVE_STEP(flag) do { \
  if (step > ring->send.conn.llLastCleaning + LL_CLEAN_FREQ) { \
    /* Reset all flags */ \
    static_assert((LL_BUFF_SIZE % THREADS) == 0, "LL_BUFF_SIZE must be a multiple of THREADS"); \
    static_assert(LL_BUFF_SIZE/(sizeof(union ncclLLFifoLine)*THREADS) > 0, "LL_BUFF_SIZE is less than 16 bytes*THREADS"); \
    const union ncclLLFifoLine resetLine = { 0, flag, 0, flag }; \
    for (int i=0; i<LL_BUFF_SIZE/(sizeof(union ncclLLFifoLine)*THREADS); i++) { \
      prevInput[tid+i*THREADS].i4 = resetLine.i4; \
    } \
    __threadfence_system(); \
    /* Restart from the same slot, only make sure sender waits for data to be reset */ \
    step += NUM_LL_CHUNKS; \
    ACK_PREV; \
    while (sendHeadPtr[0] < step); \
    if (tid == 0) ring->send.conn.llLastCleaning = step; \
  } \
  ring->send.conn.llStep = step; \
} while (0);

#endif
