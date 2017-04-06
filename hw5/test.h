#pragma once

#include <string>

void verify(int width, int height, const float *pixels, int operation_num, std::string ppm_file, std::string diff_file="");
