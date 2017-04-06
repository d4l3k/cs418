#pragma once

#include <vector>
#include "image.h"

class Convolution {
 public:
  Convolution() { }
  ~Convolution();

  // host data operations
  void load_image_input(const Image &image);
  void setup_host_gaussian_stencil(float sigma);
  void evaluate_gaussian_stencil(float sigma, float *stencil_array,
				 int stencil_width);
  const float *get_image_buffer();
  const Image *get_image_ptr();

  // device data operations
  void setup_device(int width, int height);

  // Call kernels from host
  void run_horizontal_1d();
  void run_horizontal_1d_tiling();

  void run_vertical_1d();
  void run_vertical_1d_tiling();

  void run_1to2();
  void run_1to2_tiling();

  void run_2d();
  void run_2d_tiling();
  
 private:
  Image hos_image_in, hos_image_out;
};
