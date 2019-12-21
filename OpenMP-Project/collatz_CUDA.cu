

/*
Group Members : Jose Garcia Kameron Bush

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
#include <algorithm>
#include <cuda.h>
#include <sys/time.h>

static const int ThreadsPerBlock = 512;

static __global__ void collatz(const long upper, int const maxlen[])
{
  // compute sequence lengths
  //*********has to be only odds*************
  const int i = (threadIdx.x + blockIdx.x * (long)blockDim.x) * 2 + 1;
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
	int * temp = (int *)maxlen[0];
	if (len > *temp) { //***********check if pointer correct********
      atomicMax(temp, len);
	  int temp2 = *temp;
	  maxlen = &temp2;
	}
}

static void CheckCuda()
{
  cudaError_t e;
  cudaDeviceSynchronize();
  if (cudaSuccess != (e = cudaGetLastError())) {
    fprintf(stderr, "CUDA error %d: %s\n", e, cudaGetErrorString(e));
    exit(-1);
  }
}

int main(int argc, char *argv[])
{
  printf("Collatz v1.2\n");

  // check command line
  if (argc != 2) {fprintf(stderr, "USAGE: %s upper_bound\n", argv[0]); exit(-1);}
  const long upper = atol(argv[1]);
  if (upper < 5) {fprintf(stderr, "ERROR: upper_bound must be at least 5\n"); exit(-1);}
  if ((upper % 2) != 1) {fprintf(stderr, "ERROR: upper_bound must be an odd number\n"); exit(-1);}
  printf("upper bound: %ld\n", upper);
  
  //Initalize
  int* const maxlen = new int [1];
  
  maxlen[0] = 0;
  
  // allocate vectors on GPU
  int* d_maxlen;
  if (cudaSuccess != cudaMalloc((void **)&d_maxlen, 1)) {fprintf(stderr, "ERROR: could not allocate memory\n"); exit(-1);}

  // initialize vectors on GPU
  if (cudaSuccess != cudaMemcpy(d_maxlen, maxlen, 1, cudaMemcpyHostToDevice)) {fprintf(stderr, "ERROR: copying to device failed\n"); exit(-1);}
  
  // start time
  timeval start, end;
  gettimeofday(&start, NULL);
  
  // execute timed code
  //**********************see if number of threads correct
  collatz<<<(upper + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(upper, d_maxlen);
  cudaDeviceSynchronize();

  // end time
  gettimeofday(&end, NULL);
  const double runtime = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
  printf("compute time: %.4f s\n", runtime);

  // get result from GPU
  CheckCuda();
  if (cudaSuccess != cudaMemcpy(maxlen, d_maxlen, 1, cudaMemcpyDeviceToHost)) {fprintf(stderr, "ERROR: copying from device failed\n"); exit(-1);}

  // print result
  printf("longest sequence: %d elements\n", maxlen[0]);

  cudaFree(d_maxlen);
  delete [] maxlen;
  return 0;
}

