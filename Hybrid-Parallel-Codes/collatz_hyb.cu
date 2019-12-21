/*
Authors: Jose Garcia Kameron Bush
Collatz code for CS 4380 / CS 5351

Copyright (c) 2019 Texas State University. All rights reserved.

Redistribution in source or binary form, with or without modification,
is *not* permitted. Use in source and binary forms, with or without
modification, is only permitted for academic use in CS 4380 or CS 5351
at Texas State University.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Author: Martin Burtscher
*/

#include <cstdio>
#include <cuda.h>

static const int ThreadsPerBlock = 512;

static int* d_maxlen;

static __global__ void collatz(const long start, const long stop, int* const maxlen)
{
  // todo: process odd values from start (assume start to be odd) to stop (inclusively if stop is odd) with one thread per value (based on code from previous project
  // compute sequence lengths
  //*********has to be only odds*************
  //int maxlen = 0;
  const int i = (threadIdx.x + blockIdx.x * (long)blockDim.x) * 2 + start;
  if (i <= stop) {
    long val = i;
    int len = 1;
    while (val != 1) {
      len++;
      if ((val % 2) == 0) {
        val = val / 2;  // even
      } else {
        val = 3 * val + 1;  // odd
      }
    }
     if (*maxlen < len) {
       atomicMax(maxlen, len);
     }
	  }
  }


void GPU_Init()
{
  int maxlen = 0;
  if (cudaSuccess != cudaMalloc((void **)&d_maxlen, sizeof(int))) {fprintf(stderr, "ERROR: could not allocate memory\n"); exit(-1);}
  if (cudaSuccess != cudaMemcpy(d_maxlen, &maxlen, sizeof(int), cudaMemcpyHostToDevice)) {fprintf(stderr, "ERROR: copying to device failed\n"); exit(-1);}
}

void GPU_Exec(const long start, const long stop)
{
  if (start <= stop) {
    collatz<<<((stop - start + 2) / 2 + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(start, stop, d_maxlen);
  }
}

int GPU_Fini()
{
  int maxlen;

  // todo: copy the result from the device to the host and free the device memory
  if (cudaSuccess != cudaMemcpy(&maxlen, d_maxlen, sizeof(int), cudaMemcpyDeviceToHost)) {
    fprintf(stderr, "ERROR: copying from device failed\n"); exit(-1);
  }
  cudaFree(d_maxlen);

  return maxlen;
}
