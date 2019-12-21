# Hybrid Code ( CUDA, Open MP, MPI )

## Background 
Collatz (3n+1) is an algorthim where given a number if even divide by 2, else if odd multiply number by 3 and add 1, stop when number goes to 1.

## collatz_hyb.cpp
Includes all 3 (CUDA, OpenMP, MPI)
We ran multiple test to find an optimal soultion were the HOST and DEVICE can both produce and output together. The performance was used when the HOST took only 10% of the work.

## collatz_hyb.cu
Code for the Device to take the data and parse it through the GPU using CUDA

## collatz_hyb_noMPI.cpp
Steping stone to make sure we had CUDA, and OpenMP working together, once this was complete we can move foward and adding MPI support
