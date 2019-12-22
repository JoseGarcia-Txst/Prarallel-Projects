/*
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
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <sys/time.h>
#include "BMP43805351.h"
#include <cstdio>
#include <sys/time.h>
#include <pthread.h>

static long threads;
static int frames;
static int width;
static unsigned char* pic;



static void* fractal(void* arg)
{
  const long my_rank = (long)arg;
  const double Delta = 0.002;
  const double xMid = 0.2315059;
  const double yMid = 0.5214880;

  // compute pixels of each frame
  double delta = Delta;
  for (long int frame = my_rank; frame < frames; frame += threads) {  // frames
    delta = Delta * pow(0.98,frame);
    const double xMin = xMid - delta;
    const double yMin = yMid - delta;
    const double dw = 2.0 * delta / width;
    for (int row = 0; row < width; row++) {  // rows
      const double cy = yMin + row * dw;
      for (int col = 0; col < width; col++) {  // columns
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
        pic[frame * width * width + row * width + col] = (unsigned char)depth;
      }
    }
  }
  return NULL;
}

int main(int argc, char *argv[])
{
  printf("Fractal v1.9\n");

  // check command line
  if (argc != 4) {fprintf(stderr, "USAGE: %s frame_width number_of_frames\n", argv[0]); exit(-1);}
  width = atoi(argv[1]);
  if (width < 10) {fprintf(stderr, "ERROR: frame_width must be at least 10\n"); exit(-1);}
  frames = atoi(argv[2]);
  if (frames < 1) {fprintf(stderr, "ERROR: number_of_frames must be at least 1\n"); exit(-1);}
  printf("width: %d\n", width);
  printf("frames: %d\n", frames);
  threads = atol(argv[3]);
  if (threads < 1) {fprintf(stderr, "ERROR: threads must be at least 1\n"); exit(-1);}
  printf("threads: %ld\n", threads);

  // initialize pthread variables
  pthread_t* const handle = new pthread_t [threads - 1];

  pic = new unsigned char [frames * width * width];

  // start time
  timeval start, end;
  gettimeofday(&start, NULL);

  // Launch Threads
  for (long thread = 0; thread < threads - 1; thread++) {
    pthread_create(&handle[thread], NULL, fractal, (void *)thread);
  }
  // work for master
  fractal((void*)(threads - 1));

  // join threads
    for (long thread = 0; thread < threads - 1; thread++) {
    pthread_join(handle[thread], NULL);
  }
  // end time
  gettimeofday(&end, NULL);
  const double runtime = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
  // malloc: Incorrect checksum for freed object 0x7faa80402a98: probably modified after being freed.Corrupt value: 0xa5a6a400ae
  printf("compute time: %.4f s\n", runtime);
  // In the Runtime

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
  // malloc: Incorrect checksum for freed object 0x7faa80402a98: probably modified after being freed.Corrupt value: 0xa5a6a400ae
  delete [] handle;
  delete [] pic;
  // Either in these two
  return 0;
}

