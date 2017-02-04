/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENCE.txt for license information
 ************************************************************************/


#ifndef SRC_TEST_UTILITIES_H_
#define SRC_TEST_UTILITIES_H_

#include <curand.h>
#include <cerrno>
#include <string>

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Cuda failure %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("NCCL failure %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

template<typename T>
void Randomize(T* const dest, const int N, const int randomSeed);

template<typename T>
void Accumulate(T* dest, const T* contrib, int N, ncclRedOp_t op);

template<typename T>
double CheckDelta(const T* results, const T* expected, int N);

#define CURAND_CHK(cmd)                                                         \
    do {                                                                        \
      curandStatus_t error = (cmd);                                             \
      if (error != CURAND_STATUS_SUCCESS) {                                     \
        printf("CuRAND error %i at %s:%i\n", error, __FILE__ , __LINE__);       \
        exit(EXIT_FAILURE);                                                     \
      }                                                                         \
    } while (false)


template<typename T>
void GenerateRandom(curandGenerator_t generator, T * const dest,
    const int N);

/// TODO: put specialized template functions into another source file.

template<>
void GenerateRandom<int8_t>(curandGenerator_t generator, int8_t * const dest,
    const int N) {
  CURAND_CHK(curandGenerate(generator, (unsigned int*)dest,
      N * sizeof(int8_t) / sizeof(int)));
}
template<>
void GenerateRandom<uint8_t>(curandGenerator_t generator, uint8_t * const dest,
    const int N) {
  CURAND_CHK(curandGenerate(generator, (unsigned int*)dest,
      N * sizeof(uint8_t) / sizeof(int)));
}

template<>
void GenerateRandom<int32_t>(curandGenerator_t generator, int32_t * const dest,
    const int N) {
  CURAND_CHK(curandGenerate(generator, (unsigned int*)dest, N));
}

template<>
void GenerateRandom<uint32_t>(curandGenerator_t generator, uint32_t * const dest,
    const int N) {
  CURAND_CHK(curandGenerate(generator, (unsigned int*)dest, N));
}

template<>
void GenerateRandom<float>(curandGenerator_t generator, float * const dest,
    const int N) {
  CURAND_CHK(curandGenerateUniform(generator, dest, N));
}

template<>
void GenerateRandom<double>(curandGenerator_t generator, double * const dest,
    const int N) {
  CURAND_CHK(curandGenerateUniformDouble(generator, dest, N));
}

template<>
void GenerateRandom<uint64_t>(curandGenerator_t generator, uint64_t * const dest,
    const int N) {
  CURAND_CHK(curandGenerate(generator, (unsigned int *)dest, N*2));
}

template<>
void GenerateRandom<int64_t>(curandGenerator_t generator, int64_t * const dest,
    const int N) {
  CURAND_CHK(curandGenerate(generator, (unsigned int *)dest, N*2));
}

template<typename T>
void Randomize(T* const dest, const int N, const int randomSeed) {
  curandGenerator_t gen;
  CURAND_CHK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32));
  CURAND_CHK(curandSetPseudoRandomGeneratorSeed(gen, randomSeed));
  GenerateRandom<T>(gen, dest, N);
  CURAND_CHK(curandDestroyGenerator(gen));
  CUDACHECK(cudaDeviceSynchronize());
}

__global__ void halve(const float * src, half* dest, int N) {
  for(int tid = threadIdx.x + blockIdx.x*blockDim.x;
      tid < N; tid += blockDim.x * gridDim.x)
    dest[tid] = __float2half(src[tid]);
}

template<>
void Randomize<half>(half* const dest, const int N, const int randomSeed) {
  curandGenerator_t gen;
  CURAND_CHK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32));
  CURAND_CHK(curandSetPseudoRandomGeneratorSeed(gen, randomSeed));

  float* temp;
  CUDACHECK(cudaMalloc(&temp, N*sizeof(float)));
  GenerateRandom<float>(gen, temp, N);
  halve<<<128, 512>>>(temp, dest, N);
  CURAND_CHK(curandDestroyGenerator(gen));
  CUDACHECK(cudaFree(temp));
  CUDACHECK(cudaDeviceSynchronize());
}

void makeRandom(void* ptr, int n, ncclDataType_t type, int seed) {
  if (type == ncclInt8)
    Randomize<int8_t>((int8_t*)ptr, n, seed);
  else if (type == ncclInt8)
    Randomize<uint8_t>((uint8_t*)ptr, n, seed);
  else if (type == ncclInt32)
    Randomize<int32_t>((int32_t*)ptr, n, seed);
  else if (type == ncclHalf)
    Randomize<uint32_t>((uint32_t*)ptr, n, seed);
  else if (type == ncclInt64)
    Randomize<int64_t>((int64_t*)ptr, n, seed);
  else if (type == ncclUint64)
    Randomize<uint64_t>((uint64_t*)ptr, n, seed);
  else if (type == ncclFloat16)
    Randomize<half>((half*)ptr, n, seed);
  else if (type == ncclFloat32)
    Randomize<float>((float*)ptr, n, seed);
  else if (type == ncclFloat64)
    Randomize<double>((double*)ptr, n, seed);

  return;
}

template<typename T, int OP> __global__ static
void accumKern(T* acum, const T* contrib, int N) {
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  int offset = blockDim.x*gridDim.x;
  for(int i=tid; i<N; i+=offset) {
    T c = contrib[i];
    T a = acum[i];
    if(OP == ncclSum) {
      acum[i] = a+c;
    } else if(OP == ncclProd) {
      acum[i] = a*c;
    } else if(OP == ncclMax) {
      acum[i] = (a > c) ? a : c;
    } else if(OP == ncclMin) {
      acum[i] = (a < c) ? a : c;
    }
  }
}

template<> __global__
void accumKern<half, ncclSum>(half* acum, const half* contrib, int N) {
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  int offset = blockDim.x*gridDim.x;
  for(int i=tid; i<N; i+=offset) {
    float c = __half2float(contrib[i]);
    float a = __half2float(acum[i]);
    acum[i] = __float2half( a + c );
  }
}

template<> __global__
void accumKern<half, ncclProd>(half* acum, const half* contrib, int N) {
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  int offset = blockDim.x*gridDim.x;
  for(int i=tid; i<N; i+=offset) {
    float c = __half2float(contrib[i]);
    float a = __half2float(acum[i]);
    acum[i] = __float2half( a * c );
  }
}

template<> __global__
void accumKern<half, ncclMax>(half* acum, const half* contrib, int N) {
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  int offset = blockDim.x*gridDim.x;
  for(int i=tid; i<N; i+=offset) {
    float c = __half2float(contrib[i]);
    float a = __half2float(acum[i]);
    acum[i] = __float2half( (a>c) ? a : c );
  }
}

template<> __global__
void accumKern<half, ncclMin>(half* acum, const half* contrib, int N) {
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  int offset = blockDim.x*gridDim.x;
  for(int i=tid; i<N; i+=offset) {
    float c = __half2float(contrib[i]);
    float a = __half2float(acum[i]);
    acum[i] = __float2half( (a<c) ? a : c );
  }
}

template<typename T>
void accVecType(void* out, void* in, int n, ncclRedOp_t op) {
  switch(op) {
    case ncclSum:  accumKern<T, ncclSum> <<<256,256>>>((T*)out, (T*)in, n); break;
    case ncclProd: accumKern<T, ncclProd><<<256,256>>>((T*)out, (T*)in, n); break;
    case ncclMax:  accumKern<T, ncclMax> <<<256,256>>>((T*)out, (T*)in, n); break;
    case ncclMin:  accumKern<T, ncclMin> <<<256,256>>>((T*)out, (T*)in, n); break;
    default:
      printf("Unknown reduction operation.\n");
      exit(EXIT_FAILURE);
  }
}

template<typename T>
void Accumulate(T* dest, const T* contrib, int N, ncclRedOp_t op) {

  T* devdest;
  CUDACHECK(cudaHostRegister(dest, N*sizeof(T), 0));
  CUDACHECK(cudaHostGetDevicePointer(&devdest, dest, 0));
  accVecType<T>((void*)devdest, (void*)contrib, N, op);
  CUDACHECK(cudaHostUnregister(dest));
}

void accVec(void* out, void* in, int n, ncclDataType_t type, ncclRedOp_t op) {
  switch (type) {
    case ncclInt8:   accVecType<int8_t>   (out, in, n, op); break;
    case ncclUint8:  accVecType<uint8_t>  (out, in, n, op); break;
    case ncclInt32:  accVecType<int32_t>  (out, in, n, op); break;
    case ncclUint32: accVecType<uint32_t> (out, in, n, op); break;
    case ncclHalf:   accVecType<half>     (out, in, n, op); break;
    case ncclFloat:  accVecType<float>    (out, in, n, op); break;
    case ncclDouble: accVecType<double>   (out, in, n, op); break;
    case ncclInt64:  accVecType<int64_t>  (out, in, n, op); break;
    case ncclUint64: accVecType<uint64_t> (out, in, n, op); break;
    default:
      printf("Unknown reduction type.\n");
      exit(EXIT_FAILURE);
  }
}

template<typename T> __device__
double absDiff(T a, T b) {
  return fabs((double)(b - a));
}

template<> __device__
double absDiff<half>(half a, half b) {
  float x = __half2float(a);
  float y = __half2float(b);
  return fabs((double)(y-x));
}

template<typename T, int BSIZE> __global__
void deltaKern(const T* A, const T* B, int N, double* max) {
  __shared__ double temp[BSIZE];
  int tid = threadIdx.x;
  double locmax = 0.0;
  for(int i=tid; i<N; i+=blockDim.x) {

    double delta = absDiff(A[i], B[i]);
    if( delta > locmax )
      locmax = delta;
  }

  temp[tid] = locmax;
  for(int stride = BSIZE/2; stride > 1; stride>>=1) {
    __syncthreads();
    if( tid < stride )
      temp[tid] = temp[tid] > temp[tid+stride] ? temp[tid] : temp[tid+stride];
  }
  __syncthreads();
  if( threadIdx.x == 0)
    *max = temp[0] > temp[1] ? temp[0] : temp[1];
}

template<typename T>
double CheckDelta(const T* results, const T* expected, int N) {
  T* devexp;
  double maxerr;
  double* devmax;
  CUDACHECK(cudaHostRegister((void*)expected, N*sizeof(T), 0));
  CUDACHECK(cudaHostGetDevicePointer((void**)&devexp, (void*)expected, 0));
  CUDACHECK(cudaHostRegister((void*)&maxerr, sizeof(double), 0));
  CUDACHECK(cudaHostGetDevicePointer(&devmax, &maxerr, 0));
  deltaKern<T, 512><<<1, 512>>>(results, devexp, N, devmax);
  CUDACHECK(cudaHostUnregister(&maxerr));
  CUDACHECK(cudaHostUnregister((void*)devexp));
  return maxerr;
}

void maxDiff(double* max, void* first, void* second, int n, ncclDataType_t type, cudaStream_t s) {
  switch (type) {
    case ncclInt8:   deltaKern<int8_t, 512>   <<<1,512,0,s>>>((int8_t*)first, (int8_t*)second, n, max); break;
    case ncclUint8:  deltaKern<uint8_t, 512>  <<<1,512,0,s>>>((uint8_t*)first, (uint8_t*)second, n, max); break;
    case ncclInt32:  deltaKern<int32_t, 512>  <<<1,512,0,s>>>((int32_t*)first, (int32_t*)second, n, max); break;
    case ncclUint32: deltaKern<uint32_t, 512> <<<1,512,0,s>>>((uint32_t*)first, (uint32_t*)second, n, max); break;
    case ncclHalf:   deltaKern<half, 512>     <<<1,512,0,s>>>((half*)first, (half*)second, n, max); break;
    case ncclFloat:  deltaKern<float, 512>    <<<1,512,0,s>>>((float*)first, (float*)second, n, max); break;
    case ncclDouble: deltaKern<double, 512>   <<<1,512,0,s>>>((double*)first, (double*)second, n, max); break;
    case ncclInt64:  deltaKern<int64_t, 512>  <<<1,512,0,s>>>((int64_t*)first, (int64_t*)second, n, max); break;
    case ncclUint64: deltaKern<uint64_t, 512> <<<1,512,0,s>>>((uint64_t*)first, (uint64_t*)second, n, max); break;
    default:
      printf("Unknown reduction type.\n");
      exit(EXIT_FAILURE);
  }
}

std::string TypeName(const ncclDataType_t type) {
  switch (type) {
    case ncclInt8:   return "int8";
    case ncclUint8:  return "uint8";
    case ncclInt32:  return "int32";
    case ncclUint32: return "uint32";
    case ncclHalf:   return "half";
    case ncclFloat:  return "float";
    case ncclDouble: return "double";
    case ncclInt64:  return "int64";
    case ncclUint64: return "uint64";
    default:         return "unknown";
  }
}

std::string OperationName(const ncclRedOp_t op) {
  switch (op) {
    case ncclSum:  return "sum";
    case ncclProd: return "prod";
    case ncclMax:  return "max";
    case ncclMin:  return "min";
    default:       return "unknown";
  }
}

ncclDataType_t strToType(const char* s) {
  if (strcmp(s, "char") == 0)
    return ncclChar;
  if (strcmp(s, "int8") == 0)
    return ncclInt8;
  if (strcmp(s, "uint8") == 0)
    return ncclUint8;
  if (strcmp(s, "int") == 0)
    return ncclInt;
  if (strcmp(s, "int32") == 0)
    return ncclInt32;
  if (strcmp(s, "uint32") == 0)
    return ncclUint32;
  if (strcmp(s, "half") == 0)
    return ncclHalf;
  if (strcmp(s, "float") == 0)
    return ncclFloat;
  if (strcmp(s, "double") == 0)
    return ncclDouble;
  if (strcmp(s, "int64") == 0)
    return ncclInt64;
  if (strcmp(s, "uint64") == 0)
    return ncclUint64;

  return ncclNumTypes;
}

size_t wordSize(ncclDataType_t type) {
  switch(type) {
//  case ncclChar:
    case ncclInt8:
    case ncclUint8: return 1;
//  case ncclHalf:
    case ncclFloat16: return 2;
//  case ncclInt:
    case ncclInt32:
    case ncclUint32:
//  case ncclFloat:
    case ncclFloat32: return 4;
    case ncclInt64:
    case ncclUint64:
//  case ncclDouble:
    case ncclFloat64: return 8;
    default: return 0;
  }
}

double deltaMaxValue(ncclDataType_t type, bool is_reduction) {
  if (is_reduction) {
    switch(type) {
//    case ncclHalf:
      case ncclFloat16: return 5e-2;
//    case ncclFloat:
      case ncclFloat32: return 1e-5;
//    case ncclDouble:
      case ncclFloat64: return 1e-12;

//    case ncclChar:
      case ncclInt8:
      case ncclUint8:
//    case ncclInt:
      case ncclInt32:
      case ncclUint32:
      case ncclInt64:
      case ncclUint64:
      default: return 1e-200;
    }
  }
  return 1e-200;
}

ncclRedOp_t strToOp(const char* s) {
  if (strcmp(s, "sum") == 0)
    return ncclSum;
  if (strcmp(s, "prod") == 0)
    return ncclProd;
  if (strcmp(s, "max") == 0)
    return ncclMax;
  if (strcmp(s, "min") == 0)
    return ncclMin;

  return ncclNumOps;
}

int strToPosInt(const char* s) {
  errno = 0;
  long temp = strtol(s, NULL, 10);
  if (errno != 0 || temp > INT_MAX || temp < 0)
    return 0;
  return (int)temp;
}

int strToNonNeg(const char* s) {
  errno = 0;
  long temp = strtol(s, NULL, 10);
  if (errno != 0 || temp > INT_MAX || temp < 0)
    return -1;
  return (int)temp;
}

#endif // SRC_TEST_UTILITIES_H_
