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

Author: (Professor) Dr. Martin Burtscher & Jose Garcia
*/

#include <cstdio>
#include <cmath>
#include <algorithm>
#include <sys/time.h>
#include "BMP43805351.h"

int threads;
int frames;

static void fractal(const int width, const int frames, unsigned char* const pic)
{
  const double Delta = 0.002;
  const double xMid = 0.2315059;
  const double yMid = 0.5214880;

  // compute pixels of each frame
  double delta = Delta;
    for (int frame = 0; frame < frames; frame++) {  // frames
      delta = Delta * pow(0.98, frame);
    const double xMin = xMid - delta;
    const double yMin = yMid - delta;
    const double dw = 2.0 * delta / width;
        #pragma omp parallel for default(none) num_threads(threads) shared(width, pic,frames,frame, delta, xMin, yMin, dw)
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
    delta *= 0.98;
  }
}

int main(int argc, char *argv[])
{
  printf("Fractal v1.9\n");

  // check command line
  if (argc != 4) {fprintf(stderr, "USAGE: %s frame_width number_of_frames\n", argv[0]); exit(-1);}
  const int width = atoi(argv[1]);
  if (width < 10) {fprintf(stderr, "ERROR: frame_width must be at least 10\n"); exit(-1);}
  frames = atoi(argv[2]);
  if (frames < 1) {fprintf(stderr, "ERROR: number_of_frames must be at least 1\n"); exit(-1);}
  printf("width: %d\n", width);
  printf("frames: %d\n", frames);
    threads = atoi(argv[3]);
    if (threads < 1) {fprintf(stderr, "ERROR: thread_count must be at least 1\n"); exit(-1);}
    printf("thread count: %d\n", threads);

  // allocate picture array
  unsigned char* pic = new unsigned char [frames * width * width];

  // start time
  timeval start, end;
  gettimeofday(&start, NULL);

  // execute timed code
  fractal(width, frames, pic);

  // end time
  gettimeofday(&end, NULL);
  const double runtime = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
  printf("compute time: %.4f s\n", runtime);

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
  delete [] pic;
  return 0;
}

