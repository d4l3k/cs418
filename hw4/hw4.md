# CPSC 418 - Homework 4
Tristan Rice, q7w9a, 25886145

## 3

Using: lin13.ugrad.cs.ubc.ca

### b

```
q7w9a@lin13  ~/cs418/hw4   master ●  ./a.out f 10000000 10000
f(n, ...): t_elapsed =  5.960e-01, throughput =  2.040e+09
 q7w9a@lin13  ~/cs418/hw4   master ●  ./a.out f 1000000 10000
f(n, ...): t_elapsed =  1.400e-01, throughput =  1.007e+10
 q7w9a@lin13  ~/cs418/hw4   master ●  ./a.out f 256 10000
f(n, ...): t_elapsed =  8.000e-03, throughput =  3.200e+08
 q7w9a@lin13  ~/cs418/hw4   master ●  ./a.out f 256 1000000
f(n, ...): t_elapsed =  5.200e-02, throughput =  4.923e+09
 q7w9a@lin13  ~/cs418/hw4   master ●  ./a.out f 256 10000000
f(n, ...): t_elapsed =  2.360e-01, throughput =  1.085e+10
 q7w9a@lin13  ~/cs418/hw4   master ●  ./a.out f 256 100000000
f(n, ...): t_elapsed =  2.948e+00, throughput =  1.399e+09
 q7w9a@lin13  ~/cs418/hw4   master ●  ./a.out f 512 1000000
f(n, ...): t_elapsed =  5.600e-02, throughput =  9.143e+09
 q7w9a@lin13  ~/cs418/hw4   master ●  ./a.out f 512 5000000
f(n, ...): t_elapsed =  1.480e-01, throughput =  1.730e+10
 q7w9a@lin13  ~/cs418/hw4   master ●  ./a.out f 1024 25000000
f(n, ...): t_elapsed =  6.240e-01, throughput =  6.611e+09
 q7w9a@lin13  ~/cs418/hw4   master ●  ./a.out f 1024 2500000
f(n, ...): t_elapsed =  7.200e-02, throughput =  3.556e+10
 q7w9a@lin13  ~/cs418/hw4   master ●  ./a.out f 65536 2500000
f(n, ...): t_elapsed =  2.364e+00, throughput =  2.670e+08
 q7w9a@lin13  ~/cs418/hw4   master ●  ./a.out f 1024 2500000
f(n, ...): t_elapsed =  8.400e-02, throughput =  3.048e+10
 q7w9a@lin13  ~/cs418/hw4   master ●  ./a.out f 1024 3000000
f(n, ...): t_elapsed =  1.000e-01, throughput =  3.072e+10
 q7w9a@lin13  ~/cs418/hw4   master ●  ./a.out f 2048 1500000
f(n, ...): t_elapsed =  4.800e-02, throughput =  6.400e+10
 q7w9a@lin13  ~/cs418/hw4   master ●  ./a.out f 2048 3000000
f(n, ...): t_elapsed =  8.000e-02, throughput =  2.311e+10
 q7w9a@lin13  ~/cs418/hw4   master ●  ./a.out f 2048 3000000
f(n, ...): t_elapsed =  1.040e-01, throughput =  1.778e+10
 q7w9a@lin13  ~/cs418/hw4   master ●  ./a.out f 2048 3000000
f(n, ...): t_elapsed =  8.400e-02, throughput =  2.201e+10
 q7w9a@lin13  ~/cs418/hw4   master ●  ./a.out f 2048 1500000
f(n, ...): t_elapsed =  4.400e-02, throughput =  6.982e+10
 q7w9a@lin13  ~/cs418/hw4   master ●  ./a.out f 2048 1000000
f(n, ...): t_elapsed =  2.800e-02, throughput =  7.314e+10
 q7w9a@lin13  ~/cs418/hw4   master ●  ./a.out f 2048 100000
f(n, ...): t_elapsed =  1.200e-02, throughput =  1.707e+10
 q7w9a@lin13  ~/cs418/hw4   master ●  ./a.out f 2048 1000000
f(n, ...): t_elapsed =  4.000e-02, throughput =  5.120e+10
 q7w9a@lin13  ~/cs418/hw4   master ●  ./a.out f 2048 1000000
f(n, ...): t_elapsed =  3.600e-02, throughput =  5.689e+10
 q7w9a@lin13  ~/cs418/hw4   master ●  ./a.out f 2048 1000000
f(n, ...): t_elapsed =  2.400e-02, throughput =  8.533e+10
 q7w9a@lin13  ~/cs418/hw4   master ●  ./a.out f 2048 1000000
f(n, ...): t_elapsed =  3.200e-02, throughput =  6.400e+10
 q7w9a@lin13  ~/cs418/hw4   master ●  ./a.out f 2048 1500000
f(n, ...): t_elapsed =  5.200e-02, throughput =  5.908e+10
 q7w9a@lin13  ~/cs418/hw4   master ●  ./a.out f 3048 1500000
f(n, ...): t_elapsed =  5.600e-02, throughput =  4.947e+09
 q7w9a@lin13  ~/cs418/hw4   master ●  ./a.out f 4096 1500000
f(n, ...): t_elapsed =  8.800e-02, throughput =  2.101e+10
 q7w9a@lin13  ~/cs418/hw4   master ●  ./a.out f 4096 15000000
f(n, ...): t_elapsed =  8.840e-01, throughput =  1.482e+09
 q7w9a@lin13  ~/cs418/hw4   master ●  ./a.out f 4096 1500000
f(n, ...): t_elapsed =  1.000e-01, throughput =  1.849e+10
```

Through experimentation, I found that $n=2048$ and $m=1000000$ had the highest
throughput at 8.533e10/s iterations of the inner loop of the kernel. Each of
those iterations requires 4 multiplies and 1 add which means 4 floating point
operations since there's a combined multiply/add operation. Thus, we achieve a
performance of 341.32GFlops.


### c

```
q7w9a@lin13  ~/cs418/hw4   master ●  ./a.out f_cpu 2048 1000000
f_cpu(n, ...): t_elapsed =  1.336e+01, throughput =  1.533e+08
```

Speed up of: 13.36/2.400e-02=566.67

### d

When writing cuda operations you want to maximize utilization of the cores as
well as have a high enough ratio of floating point operations to memory
operations to not be blocked by memory bandwidth. With a large M value, there's
a lot of floating point operations for every memory access. It's also good to
have the number of kernels be a multiple of the number of threads so you don't
have any wasted computations. The number of threads should be sufficiently large
as to make full use of all the cores.

## 4


