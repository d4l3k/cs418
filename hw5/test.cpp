#include "test.h"

#include <limits>
#include <iostream>
#include <cmath>
#include <cassert>
#include <vector>
#include <sstream>

#include "image.h"

void verify(int width, int height, const float *pixels, int operation_num, std::string ppm_file, std::string diff_file) {
  // Replace the .ppm with _ref.ppm
  // This only works with created test cases and will throw error if file not exists
  std::string ref_file = ppm_file;
  ref_file.replace(ref_file.find(".ppm"), std::string::npos, "_" + std::to_string(operation_num) + "_ref.ppm");

  Image reader(ref_file, 1);
  assert(reader.width == width); assert(reader.height == height);

  std::vector<std::string> error_messages;
  for (int i = 0; i < width * height; i++) {
    unsigned char px = static_cast<unsigned char>(255.f * pixels[i]);
    unsigned char ref_px = static_cast<unsigned char>(255.f * reader.pixels[i]);

    unsigned char diff = std::abs(px - ref_px);
    if (diff > 0) {
      int px_ind = i;
      int y = px_ind / width;
      int x = px_ind % width;

      std::ostringstream oss;
      oss << "<(" << x << ", " << height - 1 - y << "), " << (int)px << " (" << pixels[px_ind] << ")" << "=/=" << (int)ref_px << "(ref)" << ">";
      //oss << "<(" << x << ", " << height - 1 - y << "), " << pixels[px_ind] << ">";
      error_messages.emplace_back(oss.str());
    }
  }

  if (error_messages.size() > 0) {
    size_t i = 0;
    for (; i < 10 && i < error_messages.size(); i++)
      std::cerr << "\t" << error_messages[i] << std::endl;
    if (i < error_messages.size()) {
      std::cerr << "\t" << "... Ignore " << error_messages.size() - i << " error pixels ..." << std::endl;
    }

    if (!diff_file.empty()) {
      Image diff(width, height, 1);

      for (int i = 0; i < width * height; i++) {
        diff.pixels[i] = pixels[i];//std::fabs(pixels[i] - reader.pixels[i]);
      }

      diff.save_ppm(diff_file);
      std::cerr << "\tWrote difference image to '" << diff_file << "'." << std::endl;
    }
  }
  else {
    std::cerr << "\tPassed the test '" << ppm_file << "'."  << std::endl;
  }
}
