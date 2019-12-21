# Hybrid Code ( CUDA, Open MP, MPI )

## Background 
Collatz (3n+1) is an algorthim where given a number if even divide by 2, else if odd multiply number by 3 and add 1, stop when number becomes zero.

## collatz_hyb.cpp
Includes all 3 (CUDA, OpenMP, MPI)
We ran multiple test to find an optimal soultion were the HOST and DEVICE can both produce and output together. The performance was used when the HOST took only 10% of the work.
