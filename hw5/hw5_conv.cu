#include "cudaConvolution.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

// Size of the blocks on the GPU.  This is the smallest possible
// square block size that is an integer multiple of a warp.  You may
// modify these values if you want.
#define BLOCK_SIZE_X 8
#define BLOCK_SIZE_Y 8

// Size of the stencils.  Do not modify.
#define STENCIL_WIDTH_X 21
#define STENCIL_WIDTH_Y 11

// Global variables to store the convolution stencils.
float *hos_stencil_1dx = NULL;
float *hos_stencil_1dy = NULL;

////////////////////////////////////////////////////////////////
///////////////////////// CUDA kernels /////////////////////////
////////////////////////////////////////////////////////////////
// TO DO: Modify the code in the kernels below to answer the homework
// questions.

__global__ void conv1h_basic_kernel(int width, int height,
  float *dev_input, float *dev_output) {

  // TODO: This is only an example kernel: it reverses the greyscale
  // value of the input image but does not otherwise modify it.

  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  if((x < width) && (y < height)) {
    int image_offset = y * width + x;
    dev_output[image_offset] = 1.0f - dev_input[image_offset];
  }
}

__global__ void conv1v_basic_kernel(int width, int height,
  float *dev_input, float *dev_output) {
  // TODO: Complete this kernel.
}

__global__ void conv1h_tiled_kernel(int width, int height,
  float *dev_input, float *dev_output) {
  // TODO: Complete this kernel.
}

__global__ void conv1v_tiled_kernel(int width, int height,
  float *dev_input, float *dev_output) {
  // TODO: Complete this kernel.
}

//////////////////////////////////////////////////////////////////
///////////////////////// Host functions /////////////////////////
//////////////////////////////////////////////////////////////////
// TO DO: Modify the code in the kernels below to answer the homework
// questions.
//
// Notes:
//
// float *hos_stencil_1dx is a host global pointer containing a 1D
// array of length STENCIL_SIZE_X with the stencil data to be used for the
// horizontal convolution.
//
// float *hos_stencil_1dy is a host global pointer containing a 1D
// array of length STENCIL_SIZE_Y with the stencil data to be used for the
// vertical convolution.

void conv1h_basic(int width, int height, float *hos_data_in, float *hos_data_out) {

  // TODO: This host function is mostly complete, but you will need to
  // add some code to set up the constant memory on the device to
  // store the stencil and you may want to modify the grid and block
  // structure for the kernel.

  float *dev_image_in_buffer;
  float *dev_image_out_buffer;

  // Allocate space on the device and copy over the input image.
  int image_size = width * height * sizeof(float);

  cudaMalloc(&dev_image_in_buffer, image_size);
  cudaMalloc(&dev_image_out_buffer, image_size);

  cudaMemcpy(dev_image_in_buffer, hos_data_in, image_size,
	     cudaMemcpyHostToDevice);

  // Compute grid and block size
  dim3 blockDim(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
  int grid_size_x = ceil((double) width / BLOCK_SIZE_X);
  int grid_size_y = ceil((double) height / BLOCK_SIZE_Y);
  printf("\ngrid_size (%d, %d)\n", grid_size_x, grid_size_y);
  dim3 gridDim(grid_size_x, grid_size_y, 1);

  // Launch kernel
  conv1h_basic_kernel<<<gridDim, blockDim>>>(width, height,
    dev_image_in_buffer, dev_image_out_buffer);

  // Synchronization
  cudaThreadSynchronize();

  // Retrieve the output image and free the memory on the device.
  cudaMemcpy(hos_data_out, dev_image_out_buffer, image_size,
	     cudaMemcpyDeviceToHost);

  cudaFree(dev_image_in_buffer);
  cudaFree(dev_image_out_buffer);
}

// Q2 (b)
void conv1v_basic(int width, int height, float *hos_data_in, float *hos_data_out) {
  // TO DO: Remove the following two lines and complete this host function.
  fprintf(stderr, "conv1v_basic() not yet implemented\n");
  exit(1);
}

// Q2 (c)
void conv1to2_basic(int width, int height, float *hos_data_in, float *hos_data_out) {
  // TO DO: Remove the following two lines and complete this host function.
  fprintf(stderr, "conv1to2_basic() not yet implemented\n");
  exit(1);
}

// Q3 (a)
void conv1h_tiled(int width, int height, float *hos_data_in, float *hos_data_out) {
  // TO DO: Remove the following two lines and complete this host function.
  fprintf(stderr, "conv1h_tiled() not yet implemented\n");
  exit(1);
}

void conv1v_tiled(int width, int height, float *hos_data_in, float *hos_data_out) {
  // TO DO: Remove the following two lines and complete this host function.
  fprintf(stderr, "conv1v_tiled()() not yet implemented\n");
  exit(1);
}

void conv1to2_tiled(int width, int height, float *hos_data_in, float *hos_data_out) {
  // TO DO: Remove the following two lines and complete this host function.
  fprintf(stderr, "conv2to2_tiled() not yet implemented\n");
  exit(1);
}

/////////////////////////////////////////////////////////////////////////
///////////////// No change to code after this point ////////////////////
/////////////////////////////////////////////////////////////////////////
// DO NOT CHANGE THE CODE BELOW THIS COMMENT (or in any of the other
// files).  Modification of the code below or in the other files may
// cause the autograder to fail, and you may receive a zero for the
// corresponding questions in the homework.

Convolution::~Convolution() {
  free(hos_stencil_1dx);
  free(hos_stencil_1dy);
  
  hos_stencil_1dx = hos_stencil_1dy = NULL;
}

void Convolution::evaluate_gaussian_stencil(float sigma, float *stencil_array,
					    int stencil_width) {
  // Compute the stencil
  float normalization = 0.0f;
  int half_width = stencil_width / 2;
  float interval = 20.0f / stencil_width;
  for (int i = 0; i < stencil_width; i++) {
    float x = (i - half_width) * interval;
    float gaussian = std::exp(-(x * x) / (2 * sigma *sigma));
    stencil_array[i] = gaussian;

    normalization += gaussian;
    //printf("%d: %f - %f\n", i, x, gaussian);
  }

  // Normalize so that stencil sums to 1 and store to stencil_array.
  for (int i = 0; i < stencil_width; i++)
    stencil_array[i] /= normalization;
}


void Convolution::setup_host_gaussian_stencil(float sigma) {
  // Allocate memory, freed in destructor.
  hos_stencil_1dx = (float*)malloc(STENCIL_WIDTH_X * sizeof(float));
  hos_stencil_1dy = (float*)malloc(STENCIL_WIDTH_Y * sizeof(float));

  // Evaluate Gaussian function to create the stencils.
  evaluate_gaussian_stencil(sigma, hos_stencil_1dx, STENCIL_WIDTH_X);
  evaluate_gaussian_stencil(sigma, hos_stencil_1dy, STENCIL_WIDTH_Y);
}

void Convolution::setup_device(int width, int height) {
  cudaDeviceProp prop;

  int ndev;
  cudaGetDeviceCount(&ndev);
  if(ndev < 1) {
    fprintf(stderr, "No CUDA device found\n");
    exit(-1);
  }
  cudaGetDeviceProperties(&prop, 0);

  printf("The GPU is a %s\n", prop.name);
  printf("Cuda capability %d.%d.\n", prop.major, prop.minor);
  printf("Shared memory per block %d bytes.\n", prop.sharedMemPerBlock);
}

void Convolution::load_image_input(const Image &image) {
  if (image.channel != 1) {
    printf("Error: Input image has %d channels (should be 1).\n", image.channel);
  }

  int w = image.width, h = image.height;

  // Allocate host input image buffer
  if (!hos_image_in.pixels) {
    hos_image_in.alloc(image.width, image.height, image.channel);
  }

  memcpy(hos_image_in.pixels, image.pixels, w * h * sizeof(float));
  hos_image_out.alloc(w, h, 1);
}

// Since we've copied input to device in function load_image_input,
// we can just launch kernels here:
void Convolution::run_horizontal_1d() {
  int width = hos_image_in.width, height = hos_image_in.height;

  // Call student's code
  conv1h_basic(width, height, hos_image_in.pixels, hos_image_out.pixels);
}

void Convolution::run_vertical_1d() {
  int width = hos_image_in.width, height = hos_image_in.height;

  // Call student's code
  conv1v_basic(width, height, hos_image_in.pixels, hos_image_out.pixels);
}

void Convolution::run_1to2() {
  int width = hos_image_in.width, height = hos_image_in.height;
  
  conv1to2_basic(width, height, hos_image_in.pixels, hos_image_out.pixels);
}

void Convolution::run_1to2_tiling() {
  int width = hos_image_in.width, height = hos_image_in.height;

  conv1to2_tiled(width, height, hos_image_in.pixels, hos_image_out.pixels);
}

void Convolution::run_2d_tiling() {

}

void Convolution::run_horizontal_1d_tiling() {
  int width = hos_image_in.width, height = hos_image_in.height;

  // Call student's code
  conv1h_tiled(width, height, hos_image_in.pixels, hos_image_out.pixels);
}

void Convolution::run_vertical_1d_tiling() {
  int width = hos_image_in.width, height = hos_image_in.height;

    // Call student's code
  conv1v_tiled(width, height, hos_image_in.pixels, hos_image_out.pixels);
}

const float *Convolution::get_image_buffer() {
  return hos_image_out.pixels;
}

const Image *Convolution::get_image_ptr() {
  return &hos_image_out;
}
