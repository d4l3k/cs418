#include "image.h"

#include <algorithm>
#include <iostream>
#include <fstream>
#include <iterator>
#include <vector>
#include <cassert>

Image::Image(int w, int h, int channel) {
  alloc(w, h, channel);
}

Image::Image(const std::string filename, int ch)
  : channel(ch), pixels(NULL) {
  clear();
  load_ppm(filename);
}

Image::~Image() {
  delete []pixels;
  pixels = NULL;
}

void Image::alloc(int w, int h, int ch) {
  width = w; height = h; channel = ch;
  pixels = new float[channel * width * height];
}

void Image::clear() {
  width = height = 0;
  delete []pixels;
  pixels = NULL;
}

void Image::load_ppm(const std::string filename) {
  std::ifstream in_file(filename.c_str(), std::ios::binary);

  if (!in_file.is_open()) {
    std::cerr << "Error: Could not read from " << filename << std::endl;
    return ;
  }

  std::vector<char> buffer((std::istreambuf_iterator<char>(in_file)),
			   (std::istreambuf_iterator<char>()));

  in_file.close();

  // For simplicity, I won't check comments
  if (buffer[0] != 'P' || buffer[1] != '6') {
    std::cerr << "Error: Parser only supports P6 ppm images. " << std::endl;
    return ;
  }

  char linebreak[] = "\n";
  size_t break_ind = std::find_first_of(buffer.begin(), buffer.end(),
					linebreak, linebreak + 1) - buffer.begin();

  assert(break_ind + 1 < buffer.size());

  // Read size
  if (sscanf(&(buffer[break_ind + 1]), "%d %d\n", &width, &height) != 2) {
    std::cerr << "Error: Could not read size. " << std::endl;
    return ;
  }

  break_ind = std::find_first_of(buffer.begin() + break_ind + 1, buffer.end(),
				 linebreak, linebreak + 1) - buffer.begin();
  assert(break_ind + 1 < buffer.size());

  int range;
  if (sscanf(&(buffer[break_ind + 1]), "%d\n", &range) != 1 ||
      range != 255) {
    std::cerr << "Error: Doesn't support this color range. " << std::endl;
    return ;
  }

  // Read pixels
  break_ind = std::find_first_of(buffer.begin() + break_ind + 1, buffer.end(),
				 linebreak, linebreak + 1) - buffer.begin();
  assert(break_ind + 1 < buffer.size());

  //std::cout << width << ", " << height << " - " << buffer.size() <<
  //  " from " << break_ind + 1 << "; ch: " << channel << std::endl;

  assert((3 * width * height + break_ind + 1) <= buffer.size());
  assert(pixels == NULL);
  pixels = new float[channel * width * height];
  
  size_t px_ind = 0;
  for (int j = height - 1; j >= 0; j--) {
    int row_offset = j * width;
    for (int i = 0; i < width; i++) {
      for (int k = 0; k < channel; k++) {
	pixels[px_ind++] = (unsigned char)buffer[3 * (row_offset + i) + k + break_ind + 1] / 255.f;
      }
    }
  }

  if ((int)px_ind != channel * width * height) {
    std::cerr << "Error: File is incomplete. " << std::endl;
    return ;
  }
}

void write_str_binary(std::ofstream &out_file, std::string str) {
  assert(out_file.is_open());

  out_file.write(str.c_str(), str.size());
}

void Image::save_ppm(std::string filename) const {
  std::ofstream out_file(filename.c_str(), std::ios::binary);

  if (!out_file.is_open()) {
    std::cerr << "Error: Could not write to " << filename << std::endl;
    return ;
  }
  
  write_str_binary(out_file, std::string("P6\n"));
  write_str_binary(out_file, std::to_string(width));
  write_str_binary(out_file, std::string(" "));
  write_str_binary(out_file, std::to_string(height));
  write_str_binary(out_file, std::string("\n"));
  write_str_binary(out_file, std::string("255\n"));

  // todo: clamp
  for (int j = height - 1; j >= 0; j--) {
    for (int i = 0; i < width; i++) {
      const float* ptr = &pixels[channel * (j * width + i)];
      for (int k = 0; k < 3; k++) {
	int ch = std::min(k, channel - 1);
	unsigned char px = static_cast<unsigned char>(255.f * ptr[ch]);
	out_file.put(std::max(std::min(px, (unsigned char)255), (unsigned char)0));
      }
    }
  }

  out_file.close();

  std::cout << "Wrote to file: " << filename << std::endl;
}

