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

int parseStringList(const char* string, struct netIf* ifList, int maxList) {
  if (!string) return 0;

  const char* ptr = string;
  // Ignore "^" prefix, will be detected outside of this function
  if (ptr[0] == '^') ptr++;

  int ifNum = 0;
  int ifC = 0;
  char c;
  do {
    c = *ptr;
    if (c == ':') {
      if (ifC > 0) {
        ifList[ifNum].prefix[ifC] = '\0';
        ifList[ifNum].port = atoi(ptr+1);
        ifNum++; ifC = 0;
      }
      while (c != ',' && c != '\0') c = *(++ptr);
    } else if (c == ',' || c == '\0') {
      if (ifC > 0) {
        ifList[ifNum].prefix[ifC] = '\0';
        ifList[ifNum].port = -1;
        ifNum++; ifC = 0;
      }
    } else {
      ifList[ifNum].prefix[ifC] = c;
      ifC++;
    }
    ptr++;
  } while (c);
  return ifNum;
}

static bool matchPrefix(const char* string, const char* prefix) {
  return (strncmp(string, prefix, strlen(prefix)) == 0);
}

static bool matchPort(const int port1, const int port2) {
  if (port1 == -1) return true;
  if (port2 == -1) return true;
  if (port1 == port2) return true;
  return false;
}


bool matchIfList(const char* string, int port, struct netIf* ifList, int listSize) {
  // Make an exception for the case where no user list is defined
  if (listSize == 0) return true;

  for (int i=0; i<listSize; i++) {
    if (matchPrefix(string, ifList[i].prefix) 
        && matchPort(port, ifList[i].port)) {
      return true;
    }
  }
  return false;
}
