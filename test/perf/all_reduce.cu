/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENCE.txt for license information
 ************************************************************************/

#include "common.h"

double CheckData(struct threadArgs_t* args, ncclDataType_t type, ncclRedOp_t op, int root) {
  int count = args->nbytes/wordSize(type);
  double maxDelta = 0.0;
  for (int i=0; i<args->nGpus; i++) {
    int device;
    NCCLCHECK(ncclCommCuDevice(args->comms[i], &device));
    CUDACHECK(cudaSetDevice(device));
    CheckDelta(args->recvbuffs[i], args->expected, count, type, args->delta);
    cudaDeviceSynchronize();
    maxDelta = std::max(*(args->deltaHost), maxDelta);
    if (maxDelta > DeltaMaxValue(type)) {
      printf("Error rank%d/thread%d/gpu%d Delta = %g > %g\n", args->proc, args->thread, i, maxDelta, DeltaMaxValue(type));
      args->errors[0]++;
    }
  }
  return maxDelta;
}

void GetBw(double baseBw, double* algBw, double* busBw, int nranks) {
  *algBw = baseBw;
  double factor = 2 * nranks - 2;
  factor /= nranks;
  *busBw = baseBw * factor;
}

void RunColl(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  NCCLCHECK(ncclAllReduce(sendbuff, recvbuff, count, type, op, comm, stream));
}

void RunTestOp(struct threadArgs_t* args, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName) {
  TimeTest(args, type, typeName, op, 0, opName);
}

void RunTestType(struct threadArgs_t* args, ncclDataType_t type, const char* typeName) {
  RunTestOp(args, type, typeName, ncclSum, "sum");
  RunTestOp(args, type, typeName, ncclProd, "prod");
  RunTestOp(args, type, typeName, ncclMax, "max");
  RunTestOp(args, type, typeName, ncclMin, "min");
}

void RunTests(struct threadArgs_t* args) {
  RunTestType(args, ncclInt8, "int8");
  RunTestType(args, ncclUint8, "uint8");
  RunTestType(args, ncclInt32, "int32");
  RunTestType(args, ncclUint32, "uint32");
  RunTestType(args, ncclInt64, "int64");
  RunTestType(args, ncclUint64, "uint64");
  RunTestType(args, ncclHalf, "half");
  RunTestType(args, ncclFloat, "float");
  RunTestType(args, ncclDouble, "double");
}
