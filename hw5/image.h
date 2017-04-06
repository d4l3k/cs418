#pragma once

#include <string>

class Image {
 public:
  Image()
    : width(0), height(0), channel(3), pixels(NULL) { }
  Image(int w, int h, int channel = 3);
  Image(const std::string filename, int channel = 3);
  ~Image();

  void alloc(int w, int h, int channel = 3);
  void clear();
  void load_ppm(const std::string filename);
  void save_ppm(const std::string filename) const;

  int width, height;
  int channel;
  float *pixels;
};
