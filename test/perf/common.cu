/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENCE.txt for license information
 ************************************************************************/

#include "common.h"
#include <pthread.h>
#include <cstdio>

thread_local int is_main_thread = 0;

double DeltaMaxValue(ncclDataType_t type) {
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
//    case ncclHalf:
    case ncclFloat16:
      deltaKern<half, 512><<<1, 512>>>(results, expected, count, devmax); break;
//    case ncclFloat:
    case ncclFloat32:
      deltaKern<float, 512><<<1, 512>>>(results, expected, count, devmax); break;
//    case ncclDouble:
    case ncclFloat64:
      deltaKern<double, 512><<<1, 512>>>(results, expected, count, devmax); break;

//    case ncclChar:
    case ncclInt8:
    case ncclUint8:
      deltaKern<uint8_t, 512><<<1, 512>>>(results, expected, count, devmax); break;
//    case ncclInt:
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

void RandomizeAccumulate(void* data, void* accum, int count, ncclDataType_t type, ncclRedOp_t op, int seed, int rank) {
  Randomize(data, count, type, seed);
  if (rank == 0) {
    CUDACHECK(cudaMemcpy(accum, data, count*wordSize(type), cudaMemcpyDeviceToHost));
  } else {
    Accumulate(accum, data, count, type, op);
  }
}

void InitData(struct threadArgs_t* args, ncclDataType_t type, ncclRedOp_t op, int root, int in_place, int is_first) {
  size_t count = args->nbytes / wordSize(type);
  for (int i=0; i<args->nGpus; i++) {
    int device;
    NCCLCHECK(ncclCommCuDevice(args->comms[i], &device));
    CUDACHECK(cudaSetDevice(device));
    void* data = in_place ? args->recvbuffs[i] : args->sendbuffs[i];
    int seed = i+count+in_place;
    Randomize(data, count, type, seed);
    if (is_first && i == 0) {
      CUDACHECK(cudaMemcpy(args->expected, data, count*wordSize(type), cudaMemcpyDeviceToHost));
    } else {
      Accumulate(args->expected, data, count, type, op);
    }
    if (in_place == 0) {
      CUDACHECK(cudaMemset(args->recvbuffs[i], 0, args->nbytes));
    }
    cudaDeviceSynchronize();
  }
}

void BenchTime(struct threadArgs_t* args, ncclDataType_t type, ncclRedOp_t op, int root, int in_place) {
  size_t count = args->nbytes / wordSize(type);
  while (args->sync[0] != args->thread) pthread_yield();
  InitData(args, type, op, root, in_place, args->thread == 0 ? 1 : 0);
  args->sync[0] = args->thread + 1;
  if (args->thread+1 == args->nThreads) {
#if MPI == 1
    // Last thread does the MPI reduction
    void* remote, *remoteHost = malloc(args->nbytes);
    void* myInitialData = malloc(args->nbytes);
    memcpy(myInitialData, args->expectedHost, args->nbytes);
    CUDACHECK(cudaHostRegister(remoteHost, args->nbytes, 0));
    CUDACHECK(cudaHostGetDevicePointer(&remote, remoteHost, 0));
    for (int i=0; i<args->nProcs; i++) {
      if (i == args->proc) {
        MPI_Bcast(myInitialData, args->nbytes, MPI_BYTE, i, MPI_COMM_WORLD);
        free(myInitialData);
      } else {
        MPI_Bcast(remoteHost, args->nbytes, MPI_BYTE, i, MPI_COMM_WORLD);
        Accumulate(args->expected, remote, count, type, op);
        cudaDeviceSynchronize();
      }
    }
    CUDACHECK(cudaHostUnregister(remoteHost));
    free(remoteHost);
#endif
    args->sync[0] = 0;
  } else {
    while (args->sync[0]) pthread_yield();
  }
  

  // Warmup / Sync
  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < args->nGpus; i++) {
    // Use data we can overwrite safely
    void *ptr = in_place ? args->sendbuffs[i] : args->recvbuffs[i];
    NCCLCHECK(ncclAllReduce((const void*)ptr, (void*)ptr, 1, ncclChar, ncclMin, args->comms[i], args->streams[i]));
  }
  NCCLCHECK(ncclGroupEnd());
  for (int i = 0; i < args->nGpus; ++i) {
    CUDACHECK(cudaStreamSynchronize(args->streams[i]));
  }
  
  // Benchmark
  auto start = std::chrono::high_resolution_clock::now();
  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < args->nGpus; i++) {
    RunColl((const void*)(in_place ? args->recvbuffs[i] : args->sendbuffs[i]), 
        (void*)args->recvbuffs[i], count, type, op, root, args->comms[i], args->streams[i]);
  }
  NCCLCHECK(ncclGroupEnd());

  for (int i = 0; i < args->nGpus; ++i) {
    CUDACHECK(cudaStreamSynchronize(args->streams[i]));
  }
  auto delta = std::chrono::high_resolution_clock::now() - start;
  double deltaSec = std::chrono::duration_cast<std::chrono::duration<double>>(delta).count();

  double baseBw = (double)(count * wordSize(type)) / 1.0E9 / deltaSec;
  double algBw, busBw;
  int nranks;
  NCCLCHECK(ncclCommCount(args->comms[0], &nranks));
  GetBw(baseBw, &algBw, &busBw, nranks);

  double maxDelta = CheckData(args, type, op, root);

  PRINT("  %7.3f  %5.2f  %5.2f  %7.0le", deltaSec * 1.0E3, algBw, busBw,
      maxDelta);

  args->bw[0] += busBw;
  args->bw_count[0]++;
}

void TimeTest(struct threadArgs_t* args, ncclDataType_t type, const char* typeName, ncclRedOp_t op, int root, const char* opName) {
  size_t count = args->nbytes / wordSize(type);
  PRINT("%12i  %12i  %6s  %6s", (int)(count*wordSize(type)), (int)(count), typeName, opName);

  BenchTime(args, type, op, root, 0);
  BenchTime(args, type, op, root, 1);
  PRINT("\n");
}

void* threadRunTests(void* args) {
  RunTests((struct threadArgs_t*)args);
  return NULL;
}

int main(int argc, char* argv[]) {
  int nbytes = 0;
  if (argc > 1) {
    int t = sscanf(argv[1], "%d", &nbytes);
    if (t == 0) {
      printf("Error: %s is not an integer!\n\n", argv[1]);
      exit(EXIT_FAILURE);
    }
  } else {
    printf("Error: must specify data size in bytes!\n\n");
    exit(EXIT_FAILURE);
  }

  int nProcs = 1, proc = 0;
  int localRank = 0;
#if MPI == 1
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &proc);
  char hostname[1024];
  getHostName(hostname, 1024);
  uint64_t hostHashs[nProcs];
  hostHashs[proc] = getHostHash(hostname);
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD);
  for (int p=0; p<nProcs; p++) {
    if (p == proc) break;
    if (hostHashs[p] == hostHashs[proc]) localRank++;
  }
#endif
  is_main_thread = (proc == 0) ? 1 : 0;

  int nThreads = 1;
  char* strenv = getenv("NCCL_TESTS_NTHREADS");
  if (strenv) nThreads = atoi(strenv);
  int nGpus = 1;
  strenv = getenv("NCCL_TESTS_GPUSPERTHREAD");
  if (strenv) nGpus = atoi(strenv);

  ncclUniqueId ncclId;
  if (proc == 0) {
    NCCLCHECK(ncclGetUniqueId(&ncclId));
  }
#if MPI == 1
  MPI_Bcast(&ncclId, sizeof(ncclId), MPI_BYTE, 0, MPI_COMM_WORLD);
#endif
  cudaStream_t streams[nGpus*nThreads];
  void* sendbuffs[nGpus*nThreads];
  void* recvbuffs[nGpus*nThreads];

  NCCLCHECK(ncclGroupStart());
  ncclComm_t* comms = (ncclComm_t*)malloc(sizeof(ncclComm_t)*nThreads*nGpus);
  for (int i=0; i<nGpus*nThreads; i++) {
    CUDACHECK(cudaSetDevice(localRank*nThreads*nGpus+i));
    CUDACHECK(cudaMalloc(sendbuffs+i, nbytes));
    CUDACHECK(cudaMalloc(recvbuffs+i, nbytes));
    CUDACHECK(cudaStreamCreate(streams+i));
    NCCLCHECK(ncclCommInitRank(comms+i, nProcs*nThreads*nGpus, ncclId, proc*nThreads*nGpus+i));
  }
  NCCLCHECK(ncclGroupEnd());
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
#if MPI == 1
    MPI_Barrier(MPI_COMM_WORLD);
#endif
  }

  int errors[nThreads];
  double bw[nThreads];
  int bw_count[nThreads];
  for (int t=0; t<nThreads; t++) {
    bw[t] = 0.0;
    errors[t] = bw_count[t] = 0;
  }

  PRINT("\n");
  PRINT("# %10s  %12s  %6s  %6s        out-of-place                    in-place\n", "", "", "", "");
  PRINT("# %10s  %12s  %6s  %6s  %7s  %5s  %5s  %7s  %7s  %5s  %5s  %7s\n", "bytes", "N", "type", "op",
      "time", "algbw", "busbw", "res", "time", "algbw", "busbw", "res");

  void* expected, *expectedHost = malloc(nbytes);
  CUDACHECK(cudaHostRegister(expectedHost, nbytes, 0));
  CUDACHECK(cudaHostGetDevicePointer(&expected, expectedHost, 0));
  int* sync = (int*)malloc(sizeof(int));
  sync[0] = 0;

  pthread_t threads[nThreads-1];
  struct threadArgs_t args[nThreads];
  for (int t=nThreads-1; t>=0; t--) {
    args[t].nbytes=nbytes;

    args[t].nProcs=nProcs;
    args[t].proc=proc;
    args[t].nThreads=nThreads;
    args[t].thread=t;
    args[t].nGpus=nGpus;
    args[t].sendbuffs = sendbuffs+t*nGpus;
    args[t].recvbuffs = recvbuffs+t*nGpus;
    args[t].comms=comms+t*nGpus;
    args[t].streams=streams+t*nGpus;

    args[t].expectedHost = expectedHost;
    args[t].expected = expected;
    args[t].sync = (volatile int*)sync;
    args[t].deltaHost = (double*)malloc(sizeof(double));
    CUDACHECK(cudaHostRegister(args[t].deltaHost, sizeof(double), 0));
    CUDACHECK(cudaHostGetDevicePointer(&args[t].delta, args[t].deltaHost, 0));
    args[t].errors=errors+t;
    args[t].bw=bw+t;
    args[t].bw_count=bw_count+t;
    if (t)
      pthread_create(threads+t-1, NULL, threadRunTests, args+t);
    else
      RunTests(args+t); // Directly execute last thread
  }
  // Wait for other threads
  for (int t=1; t<nThreads; t++) {
    pthread_join(threads[t-1], NULL);
    errors[0] += errors[t];
    bw[0] += bw[t];
    bw_count[0] += bw_count[t];
  }

  for(int i=0; i<nGpus*nThreads; ++i)
    ncclCommDestroy(comms[i]);
  free(comms);

  char* str = getenv("NCCL_TESTS_MIN_BW");
  double check_avg_bw = str ? atof(str) : -1;
  bw[0] /= bw_count[0];

  PRINT(" Out of bounds values : %d %s\n", errors[0], errors[0] ? "FAILED" : "OK");
  PRINT(" Avg bus bandwidth    : %g %s\n", bw[0], check_avg_bw == -1 ? "" : (bw[0] < check_avg_bw ? "FAILED" : "OK"));
  PRINT("\n");
#if MPI == 1
  MPI_Finalize();
#endif
  if (errors[0] || bw[0] < check_avg_bw)
    exit(EXIT_FAILURE);
  else 
    exit(EXIT_SUCCESS);
}
