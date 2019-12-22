/*
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

13. Initialize and destroy the mutex outside of the timed code section.
14. Make sure your code runs correctly for different numbers of threads, i.e., it produces the
same results as the serial code.
15. Free all dynamically allocated memory before normal program termination.
16. Do not allocate or free any dynamic memory in the timed code section.

*/

#include <cstdio>
#include <algorithm>
#include <sys/time.h>
#include <cstdlib>
#include <pthread.h>

//shared variables
static long threads;
static int long upper;
static long globalMax = 0;
pthread_mutex_t mutex;

static void* collatz(void* arg)
{
  const long my_rank = (long)arg;
  // compute sequence lengths
  int maxlen = 0;
  for (long i = 2 * my_rank + 1; i <= upper; i += 2* (threads)) {
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
    maxlen = std::max(maxlen, len);
  }

  pthread_mutex_lock(&mutex);
  if(maxlen > globalMax)
    globalMax = maxlen;
  pthread_mutex_unlock(&mutex);
  return NULL;
}

int main(int argc, char *argv[])
{
  printf("Collatz v1.2\n");

  // check command line
  if (argc != 3) {fprintf(stderr, "USAGE: %s upper_bound\n", argv[0]); exit(-1);}
  upper = atol(argv[1]);
  if (upper < 5) {fprintf(stderr, "ERROR: upper_bound must be at least 5\n"); exit(-1);}
  if ((upper % 2) != 1) {fprintf(stderr, "ERROR: upper_bound must be an odd number\n"); exit(-1);}
  printf("upper bound: %ld\n", upper);
  threads = atol(argv[2]);
  if (threads < 1) {fprintf(stderr, "ERROR: threads must be at least 1\n"); exit(-1);}
  printf("threads: %ld\n", threads);

  // initialize pthread variables
  pthread_t* const handle = new pthread_t [threads - 1];

  // start time
  timeval start, end;
  gettimeofday(&start, NULL);

  // Launch Threads
  for (long thread = 0; thread < threads - 1; thread++) {
    pthread_create(&handle[thread], NULL, collatz, (void *)thread);
  }

  // work for master
  collatz((void*)(threads - 1));

  // join threads
  for (long thread = 0; thread < threads - 1; thread++) {
    pthread_join(handle[thread], NULL);
  }

  // execute timed code
  //const int maxlen = collatz(upper);

  // end time
  gettimeofday(&end, NULL);
  const double runtime = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
  printf("compute time: %.4f s\n", runtime);

  // print result
  printf("longest sequence: %d elements\n", globalMax);

  pthread_mutex_destroy(&mutex);
  return 0;
}

