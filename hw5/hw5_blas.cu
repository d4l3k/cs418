/* hw5_blas.cu: Apply the power method to find the largest eigenvalue
   / eigenvector pair for a matrix using the cuBLAS library on a GPU.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <assert.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

//--------------------------------------------------------------------------
// Fortran column major indexing.  We don't actually need the number
// of columns (n), but we'll include it as an argument anyway for when
// we do C-style (row major) indexing.
#define IDX2C(i,j,m,n) (((j)*(m))+(i))

//--------------------------------------------------------------------------
// Just in case there is a cuBLAS error.
const char *cublasGetErrorString(cublasStatus_t status) {
  switch(status) {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";
  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";
  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";
  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";
  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "CUBLAS_STATUS_NOT_SUPPORTED";
  case CUBLAS_STATUS_LICENSE_ERROR:
    return "CUBLAS_STATUS_LICENSE_ERROR";
  default:
    return "UNKNOWN!!!";
  }
}

//--------------------------------------------------------------------------
// Mark's code to handle CUDA errors.

#define cudaTry(cudaStatus) _cudaTry(cudaStatus, __FILE__, __LINE__)

void _cudaTry(cudaError_t cudaStatus, const char *fileName, int lineNumber) {
  if(cudaStatus != cudaSuccess) {
    fprintf(stderr, "%s in %s line %d\n",
        cudaGetErrorString(cudaStatus), fileName, lineNumber);
    exit(1);
  }
}

// Ian's code to handle cuBLAS errors.
#define cublasTry(cublasStatus) _cublasTry(cublasStatus, __FILE__, __LINE__)

void _cublasTry(cublasStatus_t cublasStatus, 
	       const char *fileName, int lineNumber) {
  if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "%s in %s line %d\n",
	    cublasGetErrorString(cublasStatus), fileName, lineNumber);
    exit(EXIT_FAILURE);
  }
}


//--------------------------------------------------------------------------
// Load data array from a file.  File is assumed to be ASCII and
// contain rows * columns floats (in column major order).  Each entry
// is separated from the next by a newline.  It is assumed that input
// array data has already been allocated sufficient memory.

void load_data(const char *filename, const uint rows, 
	       const uint cols, float *data) {

  FILE *fh = fopen(filename, "r");
  if(fh == NULL) {
    fprintf(stderr, "Unable to open file %s for input", filename);
    exit(1);
  }

  // Load the data.
  for(uint i = 0; i < rows; i++)
    for(uint j = 0; j < cols; j++) {
      if(fscanf(fh, "%e", &data[IDX2C(i, j, rows, cols)]) != 1) {
	fprintf(stderr, "Unable to read element (%d,%d) of file %s",
		i, j, filename);
	exit(1);
      }
    }

  fclose(fh);
}

//--------------------------------------------------------------------------
// Save data array into a file.  File is written in ASCII format and
// contains rows * cols of floats (in column major order).  Each entry
// is separated from the next by a newline.
void save_data(const char *filename, const uint rows, 
	       const uint cols, float *data) {

  FILE *fh = fopen(filename, "w");
  if(fh == NULL) {
    fprintf(stderr, "Unable to open file %s for output", filename);
    exit(1);
  }

  // Save the data.
  for(uint i = 0; i < rows; i++)
    for(uint j = 0; j < cols; j++)
      if(fprintf(fh, "%e\n", data[IDX2C(i, j, rows, cols)]) == 0) {
	fprintf(stderr, "Unable to write element (%d,%d) of file %s",
		i, j, filename);
	exit(1);
      }

  fclose(fh);
}


//--------------------------------------------------------------------------
// Run the power iteration on matrix A of size n x n, starting with
// vector b0 for k iterations.  Assuming that convergence has been
// reached, return the magnitude of the largest eignenvalue (a scalar)
// in lambda_mag and the corresponding eigenvector in bk.  Caller
// should allocate space for a, b0, lambda_mag and bk.
void power_iteration_cpu(const uint n, const float *a, const float *b0, 
			 const uint k, float *lambda_mag, float *bk) {

  cublasHandle_t handle;

  float lambda, lambda_inv;
  // Constants for the BLAS calls.
  float alpha = 1.0;
  float beta = 0.0;

  // Set up cuBLAS.
  cublasTry(cublasCreate(&handle));

  // Allocate space on the GPU.
  float *dev_a, *dev_bk;
  cudaTry(cudaMalloc((void **)(&(dev_a)), n * n * sizeof(float)));
  cudaTry(cudaMalloc((void **)(&(dev_bk)), n * sizeof(float)));

  // Copy over the input data.
  cublasTry(cublasSetMatrix(n, n, sizeof(*a), a, n, dev_a, n));
  cublasTry(cublasSetVector(n, sizeof(*b0), b0, 1, dev_bk, 1));

  // Remaining iterations can use bk as the input vector.
  for(uint i = 0; i < k; i++) {
    // bk <- A * bk.
    cublasTry(cublasSgemv(handle, CUBLAS_OP_N, n, n, 
			  &alpha, dev_a, n, dev_bk, 1, &beta, dev_bk, 1));
    // lambda <- ||bk||.
    cublasTry(cublasSnrm2(handle, n, dev_bk, 1, &lambda));

    // bk <- bk / lambda.
    lambda_inv = 1.0f / lambda;
    cublasTry(cublasSscal(handle, n, &lambda_inv, dev_bk, 1));
  }

  // Get the final value of bk as the approximation of the eigenvector.
  cublasTry(cublasGetVector(n, sizeof(*dev_bk), dev_bk, 1, bk, 1));

  // Return the estimate of the magnitude of the largest eigenvalue.
  // The estimated eigenvector is already stored in bk.
  *lambda_mag = lambda;

  // Free the device memory.
  cudaTry(cudaFree((void *)(dev_a)));
  cudaTry(cudaFree((void *)(dev_bk)));
}

int main(int argc, char **argv) {

  if((argc < 5) || (argc > 6)) {
    fprintf(stderr, "\nusage: hw5_blas <matrix_size> <filename_A> %s\n",
	    "<filename_b0> <iterations> [filename_bk]");
    exit(1);
  }

  // Figure out the problem.
  int n = atoi(argv[1]);
  char *file_A = argv[2];
  char *file_b0 = argv[3];
  int k = atoi(argv[4]);

  printf("\nPower iteration parameters:\n  n = %d", n);
  printf("\n  filename_A = %s\n  filename_b0 = %s\n  iterations = %d\n", 
	 file_A, file_b0, k);

  // Allocate some memory.
  float *a = (float *)malloc(n * n * sizeof(float));
  assert(a != NULL);
  float *b0 = (float *)malloc(n * sizeof(float));
  assert(b0 != NULL);
  float *bk = (float *)malloc(n * sizeof(float));
  assert(bk != NULL);
  float lambda_mag;

  // Load the input data.
  load_data(file_A, n, n, a);
  load_data(file_b0, n, 1, b0);

  // Run the power iteration.
  power_iteration_cpu(n, a, b0, k, &lambda_mag, bk);

  // If necessary, save the eigenvector in the output file.
  if(argc == 6) {
    char *file_bk = argv[6];
    save_data(file_bk, n, 1, bk);
  }

  // Print the estimated eigenvector magnitude.
  printf("\nAfter %d iterations, magnitude of largest eigenvalue: %5.3f\n", 
	 k, lambda_mag);

  // Free up all the memory.
  free(a);
  free(b0);
  free(bk);

  exit(0);
}
