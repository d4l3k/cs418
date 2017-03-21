#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>

#define NewArray(Type, N) ((Type *)(malloc(N*sizeof(Type))))

#define cudaTry(cudaStatus) _cudaTry(cudaStatus, __FILE__, __LINE__)

void _cudaTry(cudaError_t cudaStatus, const char *fileName, int lineNumber) {
  if(cudaStatus != cudaSuccess) {
    fprintf(stderr, "%s in %s line %d\n",
        cudaGetErrorString(cudaStatus), fileName, lineNumber);
    exit(1);
  }
}


void f(float *x, float *xm, uint n, uint m) {
  /* you need to write this */
}

void f_cpu(float *x, float *xm, uint n, uint m) {
  float a = 5.0/2.0;
  for(uint i = 0; i < n; i++) {
    float y = x[i];
    for(uint j = 0; j < m; j++)
      y = a*(y*y*y - y);
    x[i] = y;
  }
}

void f_test(uint n, uint m, int gpu_flag) {
  float *x, *xm;
  struct rusage r0, r1;

  x = NewArray(float, n);
  xm = NewArray(float, n);
  for(uint i = 0; i < n; i++)
    x[i] = (1.0+i)/(n+1);
  getrusage(RUSAGE_SELF, &r0);
  if(gpu_flag) f(x, xm, n, m);
  else f_cpu(x, xm, n, m);
  getrusage(RUSAGE_SELF, &r1);
  double t_elapsed =   (r1.ru_utime.tv_sec - r0.ru_utime.tv_sec)
                     + 1e-6*(r1.ru_utime.tv_usec - r0.ru_utime.tv_usec);
  printf("f%s(n, ...): t_elapsed = %10.3e\n",
         gpu_flag ? "" : "_cpu", t_elapsed);
}

__global__ void saxpy_kernel(uint n, float a, float *x, float *y) {
  uint i = blockIdx.x*blockDim.x + threadIdx.x; // nvcc built-ins
  if(i < n)
    y[i] = a*x[i] + y[i];
  }

void saxpy(uint n, float a, float *x, float *y) {
  int size = n*sizeof(float);
  float *dev_x, *dev_y; // will be allocated on the GPU.

  cudaTry(cudaMalloc((void**)(&dev_x), size));
  cudaTry(cudaMalloc((void**)(&dev_y), size));
  cudaTry(cudaMemcpy(dev_x, x, size, cudaMemcpyHostToDevice));
  cudaTry(cudaMemcpy(dev_y, y, size, cudaMemcpyHostToDevice));

  saxpy_kernel<<<ceil(n/256.0),256>>>(n, a, dev_x, dev_y);
  
  cudaTry(cudaMemcpy(y, dev_y, size, cudaMemcpyDeviceToHost));
  cudaTry(cudaFree(dev_x));
  cudaTry(cudaFree(dev_y));
}

void saxpy_cpu(uint n, float a, float *x, float *y) {
  /* you need to write this */
}

void saxpy_test(uint n, float a, int gpu_flag) {
  float *x, *y;
  struct rusage r0, r1;

  x = NewArray(float, n);
  y = NewArray(float, n);
  for(uint i = 0; i < n; i++) {
    x[i] = i;
    y[i] = i*i;
  }
  getrusage(RUSAGE_SELF, &r0);
  if(gpu_flag) saxpy(n, a, x, y);
  else saxpy_cpu(n, a, x, y);
  getrusage(RUSAGE_SELF, &r1);
  double t_elapsed =   (r1.ru_utime.tv_sec - r0.ru_utime.tv_sec)
                     + 1e-6*(r1.ru_utime.tv_usec - r0.ru_utime.tv_usec);
  printf("saxpy%s(n, ...): t_elapsed = %10.3e\n",
         gpu_flag ? "" : "_cpu", t_elapsed);
}


// f_main: for command lines of the form
//   hw4 f n m
// or
//   hw4 f_cpu n m
void f_main(int argc, char **argv, int gpu_flag) {
  uint n, m;
  if(argc != 4) {
    fprintf(stderr, "usage: hw4 %s n m\n", argv[1]);
    exit(1);
  }
  n = strtoul(argv[2], NULL, 10);
  m = strtoul(argv[3], NULL, 10);
  f_test(n, m, gpu_flag);
}
  
// saxpy_main: for command lines of the form
//   hw4 saxpy n [a]
// or
//   hw4 saxpy n [a]
// The parameter 'a' is optional.  If omitted, we set a=3.0.
void saxpy_main(int argc, char **argv, int gpu_flag) {
  int n;
  float a;
  if((argc < 3) || (4 < argc)) {
    fprintf(stderr, "usage: hw4 saxpy n [a]\n");
    exit(1);
  }
  n = strtoul(argv[2], NULL, 10);
  if(argc >= 4) a = atof(argv[3]);
  else a = 3.0;
  saxpy_test(n, a, gpu_flag);
}
  
int main(int argc, char **argv) {
  if(argc < 2) {
    fprintf(stderr, "usage: hw4 testName testArgs\n");
    exit(1);
  }
  if(strcmp(argv[1], "f") == 0) f_main(argc, argv, true);
  else if(strcmp(argv[1], "f_cpu") == 0) f_main(argc, argv, false);
  else if(strcmp(argv[1], "saxpy") == 0) saxpy_main(argc, argv, true);
  else if(strcmp(argv[1], "saxpy_cpu") == 0) saxpy_main(argc, argv, false);
  else {
    fprintf(stderr, "hw4, unrecognized test case: %s\n", argv[1]);
    exit(1);
  }
  exit(0);
}
