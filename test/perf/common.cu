/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENCE.txt for license information
 ************************************************************************/

#include "common.h"
#include <pthread.h>
#include <cstdio>
#include <getopt.h>

#ifdef MPI_TRANSPORT
extern "C" {
void ncclMpiHook(MPI_Comm comm);
void ncclMpiLock();
void ncclMpiUnlock();
}

#define MPI_PROTECT(mpicall) do { \
  ncclMpiLock(); \
  mpicall; \
  ncclMpiUnlock(); \
} while(0)
#else
#define MPI_PROTECT(mpicall) mpicall
#endif

ncclDataType_t test_types[9] = {ncclInt8, ncclUint8, ncclInt32, ncclUint32, ncclInt64, ncclUint64, ncclHalf, ncclFloat, ncclDouble};
const char *test_typenames[9] = {"int8", "uint8", "int32", "uint32", "int64", "uint64", "half", "float", "double"};
ncclRedOp_t test_ops[4] = {ncclSum, ncclProd, ncclMax, ncclMin};
const char *test_opnames[4] = {"sum", "prod", "max", "min"};

thread_local int is_main_thread = 0;

static int datacheck = 1;
static int iters = 20;
static int ncclop = ncclSum;
static char ncclopstring[10] = "sum";
static int nccltype = ncclFloat;
static char nccltypestring[10] = "float";
static int ncclroot = 0;

double parsesize(char *value) {
    long long int units;
    double size;

    if (strchr(value, 'G') != NULL) {
        units=1024*1024*1024;
    } else if (strchr(value, 'M') != NULL) {
        units=1024*1024;
    } else if (strchr(value, 'K') != NULL) {
        units=1024;
    } else {
        units=1;
    }

    size = atof(value)*units;
    return size;
}

double DeltaMaxValue(ncclDataType_t type) {
  switch(type) {
    case ncclHalf: return 5e-2;
    case ncclFloat: return 1e-5;
    case ncclDouble: return 1e-12;
    case ncclInt8:
    case ncclUint8:
    case ncclInt32:
    case ncclUint32:
    case ncclInt64:
    case ncclUint64: return 1e-200;
  }
  return 1e-200;
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

template<typename T> __device__
float toFloat(T a) {
  return (float)a;
}
template<> __device__ 
float toFloat(half a) {
  return __half2float(a);
}


template<typename T, int BSIZE> __global__
void deltaKern(void* A_, void* B_, int count, double* max) {
  const T* A = (const T*)A_;
  const T* B = (const T*)B_;
  __shared__ double temp[BSIZE];
  int tid = threadIdx.x;
  double locmax = 0.0;
  for(int i=tid; i<count; i+=blockDim.x) {

    double delta = absDiff(A[i], B[i]);
    if( delta > locmax ) {
      locmax = delta;
#ifdef DEBUG_PRINT
      if (delta > .1) printf("Error at %d/%d : %f != %f\n", i, count, toFloat(A[i]), toFloat(B[i]));
#endif
    }
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


void CheckDelta(void* expected, void* results, int count, ncclDataType_t type, double* devmax) {
  switch (type) {
    case ncclHalf:
      deltaKern<half, 512><<<1, 512>>>(results, expected, count, devmax); break;
    case ncclFloat:
      deltaKern<float, 512><<<1, 512>>>(results, expected, count, devmax); break;
    case ncclDouble:
      deltaKern<double, 512><<<1, 512>>>(results, expected, count, devmax); break;

    case ncclInt8:
    case ncclUint8:
      deltaKern<uint8_t, 512><<<1, 512>>>(results, expected, count, devmax); break;
    case ncclInt32:
    case ncclUint32:
      deltaKern<uint32_t, 512><<<1, 512>>>(results, expected, count, devmax); break;
    case ncclInt64:
    case ncclUint64:
      deltaKern<uint64_t, 512><<<1, 512>>>(results, expected, count, devmax); break;
  }
}

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
void RandomizeType(void* dest, const int N, const int randomSeed) {
  T* ptr = (T*)dest;
  curandGenerator_t gen;
  CURAND_CHK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32));
  CURAND_CHK(curandSetPseudoRandomGeneratorSeed(gen, randomSeed));
  GenerateRandom<T>(gen, ptr, N);
  CURAND_CHK(curandDestroyGenerator(gen));
  CUDACHECK(cudaDeviceSynchronize());
}

__global__ void halve(const float * src, half* dest, int N) {
  for(int tid = threadIdx.x + blockIdx.x*blockDim.x;
      tid < N; tid += blockDim.x * gridDim.x)
    dest[tid] = __float2half(src[tid]);
}

void RandomizeHalf(void* dest, const int N, const int randomSeed) {
  half* ptr = (half*)dest;
  curandGenerator_t gen;
  CURAND_CHK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32));
  CURAND_CHK(curandSetPseudoRandomGeneratorSeed(gen, randomSeed));

  float* temp;
  CUDACHECK(cudaMalloc(&temp, N*sizeof(float)));
  GenerateRandom<float>(gen, temp, N);
  halve<<<128, 512>>>(temp, ptr, N);
  CURAND_CHK(curandDestroyGenerator(gen));
  CUDACHECK(cudaFree(temp));
  CUDACHECK(cudaDeviceSynchronize());
}

void Randomize(void* ptr, const int count, ncclDataType_t type, const int seed) {
  switch (type) {
    case ncclInt8:   RandomizeType<int8_t>  (ptr, count, seed); break;
    case ncclUint8:  RandomizeType<uint8_t> (ptr, count, seed); break;
    case ncclInt32:  RandomizeType<int32_t> (ptr, count, seed); break;
    case ncclUint32: RandomizeType<uint32_t>(ptr, count, seed); break;
    case ncclInt64:  RandomizeType<int64_t> (ptr, count, seed); break;
    case ncclUint64: RandomizeType<uint64_t>(ptr, count, seed); break;
    case ncclHalf:   RandomizeHalf          (ptr, count, seed); break;
    case ncclFloat:  RandomizeType<float>   (ptr, count, seed); break;
    case ncclDouble: RandomizeType<double>  (ptr, count, seed); break;
  }
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

void Accumulate(void* out, void* in, int n, ncclDataType_t type, ncclRedOp_t op) {
  switch (type) {
    case ncclInt8:   accVecType<int8_t>   (out, in, n, op); break;
    case ncclUint8:  accVecType<uint8_t>  (out, in, n, op); break;
    case ncclInt32:  accVecType<int32_t>  (out, in, n, op); break;
    case ncclUint32: accVecType<uint32_t> (out, in, n, op); break;
    case ncclInt64:  accVecType<int64_t>  (out, in, n, op); break;
    case ncclUint64: accVecType<uint64_t> (out, in, n, op); break;
    case ncclHalf:   accVecType<half>     (out, in, n, op); break;
    case ncclFloat:  accVecType<float>    (out, in, n, op); break;
    case ncclDouble: accVecType<double>   (out, in, n, op); break;
    default:
      printf("Unknown reduction type.\n");
      exit(EXIT_FAILURE);
  }
}

#ifdef MPI_TRANSPORT
extern "C"
void ncclMpiHook(MPI_Comm comm);
#endif

void Barrier(struct threadArgs_t* args)
{
  static int i = 0;

  while (args->sync[i] != args->thread) pthread_yield();
  args->sync[i] = args->thread + 1;
  if (args->thread+1 == args->nThreads) {
#ifdef MPI_SUPPORT
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    args->sync[i] = 0;
  } else {
    while (args->sync[i]) pthread_yield();
  }

  i=!i;
}

void RandomizeAccumulate(void* data, void* accum, int count, ncclDataType_t type, ncclRedOp_t op, int seed, int rank) {
  Randomize(data, count, type, seed);
  if (rank == 0) {
    CUDACHECK(cudaMemcpy(accum, data, count*wordSize(type), cudaMemcpyDeviceToHost));
  } else {
    Accumulate(accum, data, count, type, op);
  }
}

double CheckData(struct threadArgs_t* args, ncclDataType_t type, ncclRedOp_t op, int root, int in_place) {
  int count = args->expectedBytes/wordSize(type);
  double maxDelta = 0.0;
  for (int i=0; i<args->nGpus; i++) {
    int device;
    int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i);
    NCCLCHECK(ncclCommCuDevice(args->comms[i], &device));
    CUDACHECK(cudaSetDevice(device));
    void *data = in_place ? ((void *)((uintptr_t)args->recvbuffs[i] + args->recvInplaceOffset*rank)) : args->recvbuffs[i];
    CheckDelta(data , args->expected[i], count, type, args->delta);
    cudaDeviceSynchronize();
    maxDelta = std::max(*(args->deltaHost), maxDelta);

#ifdef DEBUG_PRINT
    if (rank == 0) { 
       int *temp = (int *)malloc(args->expectedBytes);

       printf("\n Expected: ");
       for(int j=0; j<args->expectedBytes/sizeof(int); j++) { 
       	printf("%d:%d ", j, *((int *)args->expectedHost[0] + j));
       }
       printf("\n");

       cudaMemcpy(temp, data, args->expectedBytes, cudaMemcpyDeviceToHost);
       printf("\n Actual: ");
       for (int j=0; j<args->expectedBytes/sizeof(int); j++) { 
       	printf("%d:%d ", j, *((int *)temp + j));
       }
       printf("\n");
       free(temp);
    }
#endif
  }
  if (maxDelta > DeltaMaxValue(type)) args->errors[0]++;
  return maxDelta;
}

void InitSend(struct threadArgs_t* args, ncclDataType_t type, ncclRedOp_t op, int root, int in_place, int is_first) {
  size_t count = args->sendBytes / wordSize(type);
  static int rep = 1;
  for (int i=0; i<args->nGpus; i++) {
    int device;
    int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i);
    NCCLCHECK(ncclCommCuDevice(args->comms[i], &device));
    CUDACHECK(cudaSetDevice(device));
    void* data = in_place ? (void *)((uintptr_t)args->recvbuffs[i] + args->sendInplaceOffset*rank) : args->sendbuffs[i];
    int seed = rank+count+rep+in_place;
    Randomize(data, count, type, seed);

#ifdef DEBUG_PRINT
    if (rank == 2) { 
       int *temp = (int *)malloc(args->sendBytes);
       cudaMemcpy(temp, data, args->sendBytes, cudaMemcpyDeviceToHost);
       printf("\n Send Data at rank %d:", rank);
       for (int i=0; i<args->sendBytes/sizeof(int); i++) { 
       	printf("%d:%d ", i, *((int *)temp + i));
       }
       printf("\n");
    }
#endif

    cudaDeviceSynchronize();
  }
  rep++;
}

#define CHECK 1

void BenchTime(struct threadArgs_t* args, ncclDataType_t type, ncclRedOp_t op, int root, int in_place) {
  size_t count = args->nbytes / wordSize(type);
  
  // Warmup / Sync
  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < args->nGpus; i++) {
    // Intialize data after warmup so that we can overwrite safely
    RunColl((void *)args->sendbuffs[i], 
        (void *)args->recvbuffs[i], count, type, op, root, args->comms[i], args->streams[i]);
  }
  NCCLCHECK(ncclGroupEnd());
  for (int i = 0; i < args->nGpus; ++i) {
    CUDACHECK(cudaStreamSynchronize(args->streams[i]));
  }

  Barrier(args);

  // Performance Benchmark
  auto start = std::chrono::high_resolution_clock::now();
  for (int iter = 0; iter < iters; iter++) { 
      NCCLCHECK(ncclGroupStart());
      for (int i = 0; i < args->nGpus; i++) {
        int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i);
        RunColl((void*)(in_place ? ((void *)((uintptr_t)args->recvbuffs[i] + args->sendInplaceOffset*rank)) : args->sendbuffs[i]), 
            (void*)(in_place ? (void*)((uintptr_t)args->recvbuffs[i] + args->recvInplaceOffset*rank) : args->recvbuffs[i]), 
            count, type, op, root, args->comms[i], args->streams[i]);
      }
      NCCLCHECK(ncclGroupEnd());
  }

  for (int i = 0; i < args->nGpus; ++i) {
    CUDACHECK(cudaStreamSynchronize(args->streams[i]));
  }
  auto delta = std::chrono::high_resolution_clock::now() - start;
  double deltaSec = std::chrono::duration_cast<std::chrono::duration<double>>(delta).count();
  deltaSec = deltaSec/iters;

  double algBw, busBw;
  GetBw(count, wordSize(type), deltaSec, &algBw, &busBw, args->nProcs*args->nThreads*args->nGpus);

  Barrier(args);

  if (datacheck) { 
      InitSend(args, type, op, root, in_place, args->thread == 0 ? 1 : 0);
      InitRecvResult(args, type, op, root, in_place, args->thread == 0 ? 1 : 0);
      cudaDeviceSynchronize();
  }

  //test validation in single itertion, should ideally be included into the multi-iteration run
  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < args->nGpus; i++) {
    int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i);
    RunColl((void*)(in_place ? ((void *)((uintptr_t)args->recvbuffs[i] + args->sendInplaceOffset*rank)) : args->sendbuffs[i]), 
        (void*)(in_place ? (void*)((uintptr_t)args->recvbuffs[i] + args->recvInplaceOffset*rank) : args->recvbuffs[i]), 
        count, type, op, root, args->comms[i], args->streams[i]);
  }
  NCCLCHECK(ncclGroupEnd());

  for (int i = 0; i < args->nGpus; ++i) {
    CUDACHECK(cudaStreamSynchronize(args->streams[i]));
  }

  double maxDelta = 0;
#ifdef CHECK
  if (datacheck) { 
     maxDelta = CheckData(args, type, op, root, in_place);
  } else { 
     maxDelta = -1.0;
  }
#else
     maxDelta = -1.0;
#endif

  //aggregate delta from all threads and procs
  Barrier(args);
  if (args->thread == 0) {
      for (int i=1; i<args->nThreads; i++) { 
          maxDelta += args->deltaThreads[i];
      }
#ifdef MPI_SUPPORT
      MPI_Allreduce(MPI_IN_PLACE, &maxDelta, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
  }
  Barrier(args);

  if (datacheck) { 
     PRINT("  %7.3f  %5.2f  %5.2f  %7.0le", deltaSec * 1.0E3, algBw, busBw,
         maxDelta);
  } else {
     PRINT("  %7.3f  %5.2f  %5.2f  \tN/A", deltaSec * 1.0E3, algBw, busBw);
  }

  args->bw[0] += busBw;
  args->bw_count[0]++;
}

void TimeTest(struct threadArgs_t* args, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName, int root, int inPlace) {
  size_t size;
  int nranks = args->nProcs*args->nGpus*args->nThreads; 
  size_t count, sendCount, recvCount, sendInplaceOffset, recvInplaceOffset, procSharedCount;
  int sameExpected;
  for (size = args->minbytes; size<=args->maxbytes; size = ((args->stepfactor > 1) ? size*args->stepfactor : size+args->stepbytes)) { 
      count = size / wordSize(type);
      getCollByteCount(&sendCount, &recvCount, &sendInplaceOffset, &recvInplaceOffset, &procSharedCount, &sameExpected, (size_t)count, (size_t)nranks);

      args->nbytes = count * wordSize(type); 
      args->sendBytes = sendCount * wordSize(type); 
      args->expectedBytes = recvCount * wordSize(type);
      args->sendInplaceOffset = sendInplaceOffset * wordSize(type);
      args->recvInplaceOffset = recvInplaceOffset * wordSize(type);

      print_line_header(args->nbytes, count, typeName, opName, root);

      BenchTime(args, type, op, root, 0);
      if (inPlace) BenchTime(args, type, op, root, 1);
      PRINT("\n");
  }
}


void* threadRunTests(void* args) {
  RunTest((struct threadArgs_t*)args, ncclroot, (ncclDataType_t)nccltype, nccltypestring, (ncclRedOp_t)ncclop, ncclopstring);

  return NULL;
}

void AllocateBuffs(void **sendbuff, size_t sendBytes, void **recvbuff, size_t recvBytes, void **expected, void **expectedHost, size_t nbytes, int nranks, int sameExpected) {
    static int is_first = 1;
    static void *cached_ptr = NULL;
    static void *cached_hostptr = NULL;

    CUDACHECK(cudaMalloc(sendbuff, sendBytes));
    //work around for inline reduce scatter where recv count is smaller that send count
    CUDACHECK(cudaMalloc(recvbuff, (sendBytes > recvBytes) ? sendBytes : recvBytes));

    if (is_first || !sameExpected) {
        *expectedHost = malloc(recvBytes);
        CUDACHECK(cudaHostRegister(*expectedHost, recvBytes, cudaHostRegisterPortable | cudaHostRegisterMapped));
        CUDACHECK(cudaHostGetDevicePointer(expected, *expectedHost, 0));
        cached_ptr = *expected;
        cached_hostptr = *expectedHost;
        is_first = 0;
    } else {
        *expected = cached_ptr;
        *expectedHost = cached_hostptr;
    }
}
 
int ncclstringtotype (char *str) { 
    int type;

    printf("input %s", str);

    if (!strcmp(str, "int8")) type = (int)ncclInt8; 
    else if (!strcmp(str, "uint8")) type = (int)ncclUint8; 
    else if (!strcmp(str, "int32")) type = (int)ncclInt32; 
    else if (!strcmp(str, "uint32")) type = (int)ncclUint32;
     else if (!strcmp(str, "int64")) type = (int)ncclInt64; 
    else if (!strcmp(str, "uint64")) type = (int)ncclUint64; 
    else if (!strcmp(str, "half")) type = (int)ncclHalf; 
    else if (!strcmp(str, "float")) type = (int)ncclFloat;
    else if (!strcmp(str, "double")) type = (int)ncclDouble;
    else if (!strcmp(str, "all")) type = -1;
    else printf("invalid type, defaulting to uint32... \n"); 

    if(type != (int)ncclUint32) { 
      strcpy(nccltypestring, str);
    }

    return type;
}

int ncclstringtoop (char *str) { 
    int op;

    printf("input %s", str);

    if (!strcmp(str, "sum")) op = (int)ncclSum; 
    else if (!strcmp(str, "prod")) op = (int)ncclProd; 
    else if (!strcmp(str, "min")) op = (int)ncclMin; 
    else if (!strcmp(str, "max")) op = (int)ncclMax; 
    else if (!strcmp(str, "all")) op = -1;
    else printf("invalid op, defaulting to sum... \n"); 

    if(op != (int)ncclSum) { 
      strcpy(ncclopstring, str);
    }

    return op;
}

int main(int argc, char* argv[]) {
 int nThreads = 1, nGpus = 1;
 size_t minBytes = 32*1024*1024, maxBytes = 32*1024*1024, stepBytes = 1*1024*1024, stepFactor = 1;
 int longindex;
 int nProcs = 1, proc = 0;
 int localRank = 0;
 char hostname[1024];
 getHostName(hostname, 1024);
 
 static struct option longopts[] = {
    {"nthreads", required_argument, 0, 't'}, 
    {"ngpus", required_argument, 0, 'g'}, 
    {"minbytes", required_argument, 0, 'b'}, 
    {"maxbytes", required_argument, 0, 'e'}, 
    {"stepbytes", required_argument, 0, 'i'},
    {"stepfactor", required_argument, 0, 'f'},
    {"iters", required_argument, 0, 'n'},
    {"check", required_argument, 0, 'c'},
    {"op", required_argument, 0, 'o'},
    {"datatype", required_argument, 0, 'd'},
    {"root", required_argument, 0, 'r'},
    {"help", no_argument, 0, 'h'}
 };

 while(1) {
      int c;
      c = getopt_long(argc, argv, "t:g:b:e:i:f:n:c:o:d:r:h", longopts, &longindex);

      if (c == -1)
         break;

      switch(c) {
         case 't':
             nThreads = strtol(optarg, NULL, 0);
             break;
         case 'g':
             nGpus = strtol(optarg, NULL, 0);
             break;
         case 'b':
             minBytes = (size_t)parsesize(optarg);
             break;
         case 'e':
             maxBytes = (size_t)parsesize(optarg);
             break;
         case 'i':
             stepBytes = strtol(optarg, NULL, 0);
             break;
         case 'f':
             stepFactor = strtol(optarg, NULL, 0);
             break;
	     case 'n':
	         iters = (int)strtol(optarg, NULL, 0);
	         break;
	     case 'c':
	         datacheck = (int)strtol(optarg, NULL, 0);
	         break;
	     case 'o':
	         ncclop = ncclstringtoop(optarg);
	         break;
	     case 'd':
	         nccltype = ncclstringtotype(optarg);
	         break;
	     case 'r':
	         ncclroot = strtol(optarg, NULL, 0);
	         break;
         case 'h':
	         printf("USAGE: ./test [-t,--nthreads <num threads>] [-g,--ngpus <gpus per thread>] [-b,--minbytes <min size in bytes>] [-e,--maxbytes <max size in bytes>] [-i,--stepbytes <increment size>]"
	         " [-f,--stepfactor <increment factor>] [-n,--iters <iteration count>] [-c,--check <0/1>] [-o,--op <sum/prod/min/max/all>] [-d,--datatype <nccltype/all>] [-r,--root <root>] [-h,--help]\n");
	         return 0;
	     default: 
	         printf("invalid option \n");
	         printf("USAGE: ./test [-t,--nthreads <num threads>] [-g,--ngpus <gpus per thread>] [-b,--minbytes <min size in bytes>] [-e,--maxbytes <max size in bytes>] [-i,--stepbytes <increment size>]"
	         " [-f,--stepfactor <increment factor>] [-n,--iters <iteration count>] [-c, --check <0/1>] [-o,--op <sum/prod/min/max/all>] [-d,--datatype <nccltype/all>] [-r,--root <root>] [-h,--help]\n");
	         return 0;
      }
  }


#ifdef MPI_SUPPORT
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &proc);
  uint64_t hostHashs[nProcs];
  hostHashs[proc] = getHostHash(hostname);
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD);
  for (int p=0; p<nProcs; p++) {
    if (p == proc) break;
    if (hostHashs[p] == hostHashs[proc]) localRank++;
  }
#endif
#ifdef MPI_TRANSPORT
  ncclMpiHook(MPI_COMM_WORLD);
#endif
  is_main_thread = (proc == 0) ? 1 : 0;

  if (proc == 0) { 
      printf("nThread %d nGpus %d minBytes %ld maxBytes %ld step: %ld(%s) iters: %d validation: %d \n", nThreads, nGpus, minBytes, maxBytes, 
      			(stepFactor > 1)?stepFactor:stepBytes, (stepFactor > 1)?"factor":"bytes", iters, datacheck);
  }

  ncclUniqueId ncclId;
  if (proc == 0) {
    NCCLCHECK(ncclGetUniqueId(&ncclId));
  }
#ifdef MPI_SUPPORT
  MPI_Bcast(&ncclId, sizeof(ncclId), MPI_BYTE, 0, MPI_COMM_WORLD);
#endif
  cudaStream_t streams[nGpus*nThreads];
  void* sendbuffs[nGpus*nThreads];
  void* recvbuffs[nGpus*nThreads];
  void* expected[nGpus*nThreads];
  void* expectedHost[nGpus*nThreads];
  void *procSharedHost, *procShared;
  size_t sendBytes, recvBytes, procSharedBytes, sendInplaceOffset, recvInplaceOffset; 
  int sameExpected;

  getCollByteCount(&sendBytes, &recvBytes, &sendInplaceOffset, &recvInplaceOffset, &procSharedBytes, &sameExpected, (size_t)maxBytes, (size_t)nProcs*nGpus*nThreads);

  NCCLCHECK(ncclGroupStart());
  ncclComm_t* comms = (ncclComm_t*)malloc(sizeof(ncclComm_t)*nThreads*nGpus);
  for (int i=0; i<nGpus*nThreads; i++) {
    CUDACHECK(cudaSetDevice(localRank*nThreads*nGpus+i));
    AllocateBuffs(sendbuffs+i, sendBytes, recvbuffs+i, recvBytes, expected+i, expectedHost+i, (size_t)maxBytes, nProcs*nThreads*nGpus, sameExpected);
    CUDACHECK(cudaStreamCreate(streams+i));
    NCCLCHECK(ncclCommInitRank(comms+i, nProcs*nThreads*nGpus, ncclId, proc*nThreads*nGpus+i));
  }
  NCCLCHECK(ncclGroupEnd());

  if (procSharedBytes > 0) { 
      procSharedHost = malloc(procSharedBytes);
      CUDACHECK(cudaHostRegister(procSharedHost, procSharedBytes, cudaHostRegisterPortable | cudaHostRegisterMapped));
      CUDACHECK(cudaHostGetDevicePointer(&procShared, procSharedHost, 0));
  }

  PRINT("# Using devices\n");
  for (int p=0; p<nProcs; p++) {
    if (p == proc) {
      for (int i=0; i<nThreads*nGpus; i++) {
        int cudaDev;
        int rank;
        cudaDeviceProp prop;
        NCCLCHECK(ncclCommCuDevice(comms[i], &cudaDev));
        NCCLCHECK(ncclCommUserRank(comms[i], &rank));
        CUDACHECK(cudaGetDeviceProperties(&prop, cudaDev));
        printf("#   Rank %2d on %10s device %2d [0x%02x] %s\n", rank, hostname, cudaDev,
            prop.pciBusID, prop.name);
        fflush(stdout);
      }
    }
#ifdef MPI_SUPPORT
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    printf("");
    fflush(stdout);
  }

  int errors[nThreads];
  double bw[nThreads];
  double delta[nThreads];
  int bw_count[nThreads];
  for (int t=0; t<nThreads; t++) {
    bw[t] = 0.0;
    errors[t] = bw_count[t] = 0;
  }

  PRINT("\n");
  print_header();

  int* sync = (int*)malloc(sizeof(int)*2);
  sync[0] = 0;
  sync[1] = 0;

  pthread_t threads[nThreads-1];
  struct threadArgs_t args[nThreads];
  for (int t=nThreads-1; t>=0; t--) {
    args[t].minbytes=minBytes;
    args[t].maxbytes=maxBytes;
    args[t].stepbytes=stepBytes;
    args[t].stepfactor=stepFactor;

    args[t].nProcs=nProcs;
    args[t].proc=proc;
    args[t].nThreads=nThreads;
    args[t].thread=t;
    args[t].nGpus=nGpus;
    args[t].sendbuffs = sendbuffs+t*nGpus;
    args[t].recvbuffs = recvbuffs+t*nGpus;
    args[t].comms=comms+t*nGpus;
    args[t].streams=streams+t*nGpus;

    args[t].expectedHost = expectedHost + t*nGpus;
    args[t].expected = expected + t*nGpus;
    args[t].procSharedHost = procSharedHost; 
    args[t].procShared = procShared; 
    args[t].sync = (volatile int*)sync;
    args[t].deltaThreads = delta;
    args[t].deltaHost = (delta + t);
    CUDACHECK(cudaHostRegister(args[t].deltaHost, sizeof(double), 0));
    CUDACHECK(cudaHostGetDevicePointer(&args[t].delta, args[t].deltaHost, 0));
    args[t].errors=errors+t;
    args[t].bw=bw+t;
    args[t].bw_count=bw_count+t;
    if (t)
      pthread_create(threads+t-1, NULL, threadRunTests, args+t);
    else { 
      RunTest((struct threadArgs_t*)args, ncclroot, (ncclDataType_t)nccltype, nccltypestring, (ncclRedOp_t)ncclop, ncclopstring);
    }
  }
  // Wait for other threads
  for (int t=1; t<nThreads; t++) {
    pthread_join(threads[t-1], NULL);
    errors[0] += errors[t];
    bw[0] += bw[t];
    bw_count[0] += bw_count[t];
  }

#ifdef MPI_SUPPORT
    MPI_Allreduce(MPI_IN_PLACE, &errors[0], 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif

  for(int i=0; i<nGpus*nThreads; ++i)
    ncclCommDestroy(comms[i]);
  free(comms);

  char* str = getenv("NCCL_TESTS_MIN_BW");
  double check_avg_bw = str ? atof(str) : -1;
  bw[0] /= bw_count[0];

  PRINT(" Out of bounds values : %d %s\n", errors[0], errors[0] ? "FAILED" : "OK");
  PRINT(" Avg bus bandwidth    : %g %s\n", bw[0], check_avg_bw == -1 ? "" : (bw[0] < check_avg_bw ? "FAILED" : "OK"));
  PRINT("\n");
#ifdef MPI_SUPPORT
  MPI_Finalize();
#endif
  if (errors[0] || bw[0] < check_avg_bw)
    exit(EXIT_FAILURE);
  else 
    exit(EXIT_SUCCESS);
}
