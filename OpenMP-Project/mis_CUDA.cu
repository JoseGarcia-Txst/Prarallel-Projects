/*
Group Members : Jose Garcia Kameron Bush

Maximal independent set code for CS 4380 / CS 5351

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

Author: Martin Burtscher;
*/
//2. include cuda.h

#include <cuda.h>
#include <cstdlib>
#include <cstdio>
#include <sys/time.h>
#include "ECLgraph.h"

static const unsigned char in = 2;
static const unsigned char out = 1;
static const unsigned char undecided = 0;
static const int ThreadsPerBlock = 512;

static void CheckCuda()
{
  cudaError_t e;
  cudaDeviceSynchronize();
  if (cudaSuccess != (e = cudaGetLastError())) {
    fprintf(stderr, "CUDA error %d: %s\n", e, cudaGetErrorString(e));
    exit(-1);
  }
}

// source of hash function: https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
//3. Hash fuction to a kernal
static __device__ unsigned int hash(unsigned int val)
{
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  return (val >> 16) ^ val;
}

static __global__ void init(const ECLgraph g, unsigned char* const status, unsigned int* const random)
{
  // initialize arrays
  int v  = threadIdx.x + blockIdx.x * blockDim.x;
  if( v < g.nodes){
    status[v] = undecided;
    random[v] =  hash(v + 1);
  }
}

static __global__ void mis(const ECLgraph g, unsigned char* const status, unsigned int* const random, bool* missing)
{
  // intailized arrays

  //7. Each Thread must exe one iteration of the paralleized for loop. only last thread is blocked
  int v  = threadIdx.x + blockIdx.x * blockDim.x;
  // repeat until all nodes' status has been decided
  // go over all the nodes
  //5. Elinate the three for-v loops entirely as we will lauch one thread for each itteration
  //6. Make sure Excess threads do not do work
  if ( v < g.nodes) {
    if (status[v] == undecided) {
      int i = g.nindex[v];
      // try to find a neighbor whose random number is lower
      while ((i < g.nindex[v + 1]) && ((status[g.nlist[i]] == out) || (random[v] < random[g.nlist[i]]) || ((random[v] == random[g.nlist[i]]) && (v < g.nlist[i])))) {
        i++;
      }

      if (i < g.nindex[v + 1]) {
        // found such a neighbor -> status still unknown
        *missing = true;
      }else{
        // no such neighbor -> status is "in" and all neighbors are "out"
        status[v] = in;
        for (int i = g.nindex[v]; i < g.nindex[v + 1]; i++) {
              status[g.nlist[i]] = out;
        }
      }
    }
  }
}




int main(int argc, char* argv[])
{
  printf("Maximal Independent Set v1.3\n");

  // check command line
  if (argc != 2) {fprintf(stderr, "USAGE: %s input_file\n", argv[0]); exit(-1);}

  // read input
  ECLgraph g = readECLgraph(argv[1]);
  printf("configuration: %d nodes and %d edges (%s)\n", g.nodes, g.edges, argv[1]);

  // allocate arrays
  unsigned char* const status = new unsigned char [g.nodes];
  unsigned int* const random = new unsigned int [g.nodes];
  bool* d_missing;
  unsigned char* d_status;
  unsigned int* d_random;
  if (cudaSuccess != cudaMalloc((void **)&d_missing, sizeof(bool) )) {fprintf(stderr, "ERROR: could not allocate memory\n"); exit(-1);}
  //11.allocate the staus and random array on the GPU
  if (cudaSuccess != cudaMalloc((void **)&d_status, sizeof(unsigned char) * g.nodes )) {fprintf(stderr, "ERROR: could not allocate memory\n"); exit(-1);}
  if (cudaSuccess != cudaMalloc((void **)&d_random, sizeof(unsigned int) * g.nodes )) {fprintf(stderr, "ERROR: could not allocate memory\n"); exit(-1);}

  // start time
  timeval start, end;
  gettimeofday(&start, NULL);
  // execute timed code

  //13. Create ECLgraph that can be passed to Kernal
  ECLgraph d_g = g;
  cudaMalloc((void **)&d_g.nindex, sizeof(int) * (g.nodes + 1));
  cudaMalloc((void **)&d_g.nlist, sizeof(int) * g.edges);
  cudaMemcpy(d_g.nindex, g.nindex, sizeof(int) * (g.nodes + 1), cudaMemcpyHostToDevice);
  cudaMemcpy(d_g.nlist, g.nlist, sizeof(int) * g.edges, cudaMemcpyHostToDevice);

  //8. Move to the kernal launch
  bool missing ;
  init<<<(g.nodes + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(d_g, d_status, d_random);

  do {
    //intailized Missing to False
    missing = false;
    //Copied Missing to the Device
    if (cudaSuccess != cudaMemcpy(d_missing, &missing, sizeof(bool) , cudaMemcpyHostToDevice)) {fprintf(stderr, "ERROR: copying to device failed\n"); exit(-1);}
    //Launch the Kernal
    mis<<<(g.nodes + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(d_g, d_status, d_random, d_missing);
    //Copy Missing back to Host
    if (cudaSuccess != cudaMemcpy(&missing, d_missing, sizeof(bool), cudaMemcpyDeviceToHost)) {fprintf(stderr, "ERROR: copying from device failed\n"); exit(-1);}

  } while (missing);
  //10.call cudaDeviceSynchronize
  cudaDeviceSynchronize();

  // end time
  gettimeofday(&end, NULL);
  const double runtime = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
  printf("compute time: %.4f s\n", runtime);

  CheckCuda();

  // get result from GPU
  //12. Just before determing the set size, copy sthe staus arrary back to HOST
  if (cudaSuccess != cudaMemcpy(status, d_status, sizeof(unsigned char) * g.nodes, cudaMemcpyDeviceToHost)) {fprintf(stderr, "ERROR: copying from device failed\n"); exit(-1);}
  // determine and print set size
  int count = 0;
  for (int v = 0; v < g.nodes; v++) {
    if (status[v] == in) {
      count++;
    }
  }
  printf("elements in set: %d (%.1f%%)\n", count, 100.0 * count / g.nodes);

  // verify result
  for (int v = 0; v < g.nodes; v++) {
    if ((status[v] != in) && (status[v] != out)) {fprintf(stderr, "ERROR: found unprocessed node\n"); exit(-1);}
    if (status[v] == in) {
      for (int i = g.nindex[v]; i < g.nindex[v + 1]; i++) {
        if (status[g.nlist[i]] == in) {fprintf(stderr, "ERROR: found adjacent nodes in MIS\n"); exit(-1);}
      }
    } else {
      bool flag = true;
      for (int i = g.nindex[v]; i < g.nindex[v + 1]; i++) {
        if (status[g.nlist[i]] == in) {
          flag = false;
          break;
        }
      }
      if (flag) {fprintf(stderr, "ERROR: set is not maximal\n"); exit(-1);}
    }
  }

  // clean up
  freeECLgraph(g);
  delete [] status;
  delete [] random;
  ///WHYYYYYYYY
  //cudaFree(d_g);
  cudaFree(d_missing);
  cudaFree(d_status);

  return 0;
}
