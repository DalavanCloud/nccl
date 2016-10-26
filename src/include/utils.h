#ifndef UTILS_H_
#define UTILS_H_
#include <stdint.h>
void getHostName(char* hostname, int maxlen);
uint64_t getHostHash(const char* string);
int getHostNumber(const char* string);
#endif
