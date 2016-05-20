/*************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 ************************************************************************/


#ifndef SYNCFUNC_H_
#define SYNCFUNC_H_

#include <type_traits>

template <typename... Args> class SyncFunc;

template<>
class SyncFunc<>
{
  public:
  void __device__ inline set() { }
  void __device__ inline wait(int val) { }
  void __device__ inline post(int val) { }
};

using NOSYNC = SyncFunc<>;

template<typename T>
class SyncFunc<T>
{
  static_assert(std::is_same<T, volatile int*>::value, "SyncFunc requires int flags");
  volatile int* flag;

  public:
  void __device__ inline set(T f) {
    flag = f;
  }
  void __device__ inline wait(int val) {
    while(*flag < val);
  }
  void __device__ inline post(int val) {
    *flag = val;
  }
};

template<typename T, typename... Ts>
class SyncFunc<T, Ts...>
{
  private:
  static_assert(std::is_same<T, int*>::value, "SyncFunc reqires int flags");
  volatile int* flag;
  SyncFunc<Ts...> tail;

  public:
  void __device__ inline set(T first, Ts... others) {
    flag = first;
    tail.set(others...);
  }
  void __device__ inline wait(int val) {
    while(*flag < val);
    tail.wait(val);
  }
  void __device__ inline post(int val) {
    *flag = val;
    tail.post(val);
  }
};

template <typename ... Args_T> __device__ inline
SyncFunc<Args_T...> PackSyncFlags(Args_T... flags) {
  SyncFunc<Args_T...> ret;
  ret.set(flags...);
  return ret;
}

#endif // end include guard

