/*************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_UTILS_H_
#define NCCL_UTILS_H_

#include <stdint.h>

void getHostName(char* hostname, int maxlen);
uint64_t getHostHash(const char* string);
int getHostNumber(const char* string);
int parseStringList(const char* string, const char* delim, char** tokens, int maxNTokens);

#endif
