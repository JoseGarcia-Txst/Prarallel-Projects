

/*
Group Members : Jose Garcia Kameron Bush

Fractal code for CS 4380 / CS 5351

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
#include <cmath>
#include <algorithm>
#include <sys/time.h>
#include <cuda.h>
#include "BMP43805351.h"
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

static __global__ void fractal(const int width, const int frames, unsigned char* const pic)
{
  const double Delta = 0.002;
  const double xMid = 0.2315059;
  const double yMid = 0.5214880;
  int i  = threadIdx.x + blockIdx.x * blockDim.x;
  // compute pixels of each frame
  double delta = Delta;
  if (i < frames) {
    const int col = i % width;
    const int row = (i / width) % width;
    const int frame = i / (width * width);
    delta = Delta * pow(delta, frame);
    const double xMin = xMid - delta;
    const double yMin = yMid - delta;
    const double dw = 2.0 * delta / width;
    const double cy = yMin + row * dw;
    const double cx = xMin + col * dw;
    double x = cx;
    double y = cy;
    double x2, y2;
    int depth = 256;
    do {
      x2 = x * x;
      y2 = y * y;
      y = 2.0 * x * y + cy;
      x = x2 - y2 + cx;
      depth--;
    } while ((depth > 0) && ((x2 + y2) < 5.0));
    pic[frame * width * width + row * width + col] = (unsigned char) depth;
  }
}

int main(int argc, char *argv[])
{
  printf("Fractal v1.9\n");

  // check command line
  if (argc != 3) {fprintf(stderr, "USAGE: %s frame_width number_of_frames\n", argv[0]); exit(-1);}
  const int width = atoi(argv[1]);
  if (width < 10) {fprintf(stderr, "ERROR: frame_width must be at least 10\n"); exit(-1);}
  const int frames = atoi(argv[2]);
  if (frames < 1) {fprintf(stderr, "ERROR: number_of_frames must be at least 1\n"); exit(-1);}
  printf("width: %d\n", width);
  printf("frames: %d\n", frames);

  // allocate picture array
  unsigned char* pic = new unsigned char [frames * width * width];
  unsigned char* d_pic;
  int size = frames * width * width;

  if (cudaSuccess != cudaMalloc((void **)&d_pic, sizeof(unsigned char ) * size)) {fprintf(stderr, "ERROR: could not allocate memory\n"); exit(-1);}

  //if (cudaSuccess != cudaMemcpy(d_pic, pic, sizeof(unsigned char) * size, cudaMemcpyHostToDevice)) {fprintf(stderr, "ERROR: copying to device failed\n"); exit(-1);}

  // start time
  timeval start, end;
  gettimeofday(&start, NULL);

  // execute timed code
  fractal<<<(size + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(width, frames,  d_pic);
  cudaDeviceSynchronize();

  // end time
  gettimeofday(&end, NULL);
  const double runtime = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
  printf("compute time: %.4f s\n", runtime);

  CheckCuda();
  if (cudaSuccess != cudaMemcpy(pic, d_pic, sizeof(unsigned char) * size, cudaMemcpyDeviceToHost)) {fprintf(stderr, "ERROR: copying from device failed\n"); exit(-1);}

  // write result to BMP files
  if ((width <= 257) && (frames <= 60)) {
    for (int frame = 0; frame < frames; frame++) {
      BMP24 bmp(0, 0, width - 1, width - 1);
      for (int y = 0; y < width - 1; y++) {
        for (int x = 0; x < width - 1; x++) {
          const int p = pic[frame * width * width + y * width + x];
          const int e = pic[frame * width * width + y * width + (x + 1)];
          const int s = pic[frame * width * width + (y + 1) * width + x];
          const int dx = std::min(2 * std::abs(e - p), 255);
          const int dy = std::min(2 * std::abs(s - p), 255);
          bmp.dot(x, y, dx * 0x000100 + dy * 0x000001);
        }
      }
      char name[32];
      sprintf(name, "fractal%d.bmp", frame + 1000);
      bmp.save(name);
    }
  }

  // clean up
  cudaFree(d_pic);
  delete [] pic;
  return 0;
}
