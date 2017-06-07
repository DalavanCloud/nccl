/*************************************************************************
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_ENV_H_
#define NCCL_ENV_H_

#include <stdlib.h>

#include <unistd.h>
#include <sys/types.h>
#include <pwd.h>

static const char* userHomeDir() {
  struct passwd *pwUser = getpwuid(getuid());
  return pwUser == NULL ? NULL : pwUser->pw_dir;
}

static void setEnvFile(const char* fileName) {
  FILE * file = fopen(fileName, "r");
  if (file == NULL) return;

  char *line = NULL;
  char envVar[1024];
  char envValue[1024];
  size_t n = 0;
  ssize_t read;
  while ((read = getline(&line, &n, file)) != -1) {
    if (line[read-1] == '\n') line[read-1] = '\0';
    int s=0; // Env Var Size
    while (line[s] != '\0' && line[s] != '=') s++;
    if (line[s] == '\0') continue;
    strncpy(envVar, line, min(1024,s));
    envVar[s] = '\0';
    s++;
    strncpy(envValue, line+s, 1024);
    setenv(envVar, envValue, 0);
    char *str = getenv(envVar);
  }
  if (line) free(line);
  fclose(file);
}

static void initEnv() {
  char confFilePath[1024];
  const char * userDir = userHomeDir();
  if (userDir) {
    sprintf(confFilePath, "%s/.nccl.conf", userDir);
    setEnvFile(confFilePath);
  }
  sprintf(confFilePath, "/etc/nccl.conf");
  setEnvFile(confFilePath);
}

#endif
