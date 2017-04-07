/*************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "utils.h"
#include <unistd.h>

void getHostName(char* hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i=0; i< maxlen; i++) {
    if (hostname[i] == '.') {
      hostname[i] = '\0';
      return;
    }
  }
}

uint64_t getHostHash(const char* string) {
  // Based on DJB2, result = result * 33 + char
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++){
    result = ((result << 5) + result) + string[c];
  }
  return result;
}

#include <string.h>

int getHostNumber(const char* string) {
  int result = 0;
  int len = strlen(string);
  for (int offset = len-1; offset >= 0; offset --) {
   int res = atoi(string+offset);
   if (res <= 0)
     break;
   result = res;
  }
  return result;
}

int parseStringList(const char* string, const char* delim, char** tokens, int maxNTokens) {
  if (!string) return 0;
  int len = strlen(string);
  if (len == 0) return 0;
  char* str = (char *) malloc(sizeof(char) * (len+1));
  strcpy(str, string);
  char *token, *saveptr;
  int j;
  for (j = 0; j < maxNTokens; j++, str = NULL) {
    token = strtok_r(str, delim, &saveptr);
    if (token == NULL) {
      break;
    }
    tokens[j] = token;
  }
  return j;
}
