#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include <algorithm>
#include <iostream>
#include <vector>
#include <cassert>
#include <map>
#include <sys/time.h>
#include <sys/resource.h>

#include "image.h"
#include "cudaConvolution.h"
#include "test.h"

std::map<std::string, float> test_convolutions = {
  {"checker20.ppm", 5.f},
  {"checker40.ppm", 5.f},
  {"booby.ppm", 3.f},
  {"stippling.ppm", 3.f}
};
std::vector<std::string> operation_names = {"conv1h_basic", "conv1v_basic", "conv1to2_basic", 
  "conv1h_tiled", "conv1v_tiled", "conv1to2_tiled"};

void usage(const char* progname) {
  printf("Usage: %s [OPTIONS] PPM_FILE\n", progname);
  printf("Test input PPM images in ./images/ are \n\t\t");
  for (auto const &test : test_convolutions) {
    printf("%s; ", test.first.c_str());
  }
  printf("\n");
  printf("Program Options:\n");
  printf("-f  --file   <FILENAME>         Save the convoultion result into a PPM image.\n");
  printf("-s  --sigma  <SIGMA>            Sigma value for the Gaussian kernel (maximum 9; 3 by default).\n");
  printf("-c  --convol <OPERATION_CODE>   Pick the convolution you want to run (0, aka, conv1h_basic by default).\n"
         "                                Here is a list of operations:\n");
  for (size_t i = 0; i < operation_names.size(); i++) {
    printf("                                %zu: %s;\n", i, operation_names[i].c_str());
  }

  printf("-n  --iter   <ITERATIONS>       Run for some iterations (1 by default). When the test case is small, to make the timing more noticable, you may want to use this option.\n");
  printf("-t  --test   <FILENAME>         Test correctness with pre-defined cases and results.\n"
	 "                                (If passing in a PPM image file name, the program would save difference image to the file)\n");
  printf("-?  --help                      Help message.\n");
}

int main(int argc, char** argv) {
  if (0)
  {
    Image checker(400, 400, 1);
    int checker_size = 40;
    for (int j = 0; j < checker.height; j++) {
      for (int i = 0; i < checker.width; i++) {
	int offset = j * checker.width + i;
	if ((j / checker_size) % 2 == (i / checker_size) % 2) {
	  checker.pixels[offset] = 1.f;
	}
	else
	  checker.pixels[offset] = 0.f;
      }
    }

    checker.save_ppm("./checker.ppm");
    exit(0);
  }

  std::string input_filename;
  std::string ppm_filename = "";
  std::string weight_filename = "";
  int num_rounds = 1;
  bool to_test = false;
  float sigma = 3.f;
  int operation_num = 0;

  // Parse commandline options
  int opt;
  static struct option long_options[] = {
    {"help",0, 0,'?'},
    {"file",1, 0,'f'},
    {"iter", 1, 0,'n'},
    {"convol", 1, 0,'c'},
    {"sigma", 1, 0,'s'},
    {"test",0, 0,'t'},
    {0 ,0, 0, 0}
  };

  while ((opt = getopt_long(argc, argv, "f:n:s:c:dt?", long_options, NULL)) != EOF) {
    switch (opt) {
    case 'f':
      ppm_filename = optarg;
      break;
    case 'n':
      if (optarg[0] != ':')
	    num_rounds = std::stoi(std::string(optarg));
      break;
    case 'c':
      if (optarg[0] != ':')
      operation_num = std::stoi(std::string(optarg));
      break;
    case 's':
      if (optarg[0] != ':')
      sigma = std::stof(std::string(optarg));
      break;
    case 't':
      to_test = true;
      break;
    case '?':
    default:
      usage(argv[0]);
      return 1;
    }
  }
  // Parse commandline options

  if (optind + 1 > argc) {
    fprintf(stderr, "Error: Missing cities file name.\n");
    usage(argv[0]);
    return 1;
  }

  if (operation_num >= (int)operation_names.size()) {
    fprintf(stderr, "Error: Unknown operation: %d.\n", operation_num);
    usage(argv[0]);
    return 1;
  }

  input_filename = argv[optind];

  // If going to test result, then use predefined values for sigma
  if (to_test) {
    // Since I'm bad at c++ regex, I'm going to iterate on map...
    for (auto const entry : test_convolutions) {
      if (input_filename.find(entry.first) != std::string::npos) {
	sigma = entry.second;
	std::cerr << "For testing '" << operation_names[operation_num] << "' sigma is set to " << sigma << "..." << std::endl;
	break;
      }
    }
  }

  Convolution convolution;

  if (input_filename.find(".ppm") != std::string::npos) {
    Image reader(input_filename, 1);

    convolution.setup_host_gaussian_stencil(sigma);
    convolution.setup_device(reader.width, reader.height);
    convolution.load_image_input(reader);
  }
  else {
    std::cerr << "Error: Invalid input PPM image '" << input_filename << "'." << std::endl;
    usage(argv[0]);
    return 1;
  }

  struct rusage r0, r1;
  std::vector<std::string> descriptions;
  std::vector<double> timings;

  //Run
  if (descriptions.size() == 0) {
    descriptions.emplace_back(operation_names[operation_num] + "\t\t");
  }

  timings.resize(descriptions.size(), 0.);

  for (int i = 0; i < num_rounds; i++) {
    getrusage(RUSAGE_SELF, &r0);

    switch (operation_num) {
      case 0:
        convolution.run_horizontal_1d();
        break;
      case 1:
        convolution.run_vertical_1d();
        break;
      case 2:
        convolution.run_1to2();
        break;
      case 3:
        convolution.run_horizontal_1d_tiling();
        break;
      case 4:
        convolution.run_vertical_1d_tiling();
        break;
      case 5:
        convolution.run_1to2_tiling();
        break;
      default: break;
    }

    getrusage(RUSAGE_SELF, &r1);
    timings[0] += (r1.ru_utime.tv_sec - r0.ru_utime.tv_sec)
             + 1e-6*(r1.ru_utime.tv_usec - r0.ru_utime.tv_usec);
  }

  // Save to file
  if (!ppm_filename.empty()) {
    convolution.get_image_ptr()->save_ppm(ppm_filename);
  }

  // Test
  int w = convolution.get_image_ptr()->width;
  int h = convolution.get_image_ptr()->height;

  if (to_test) {
    verify(w, h, convolution.get_image_buffer(), operation_num, input_filename, ppm_filename);
  }

  // Final statistics
  printf("Run %s on a %dx%d image for %d rounds. Sigma is %f.\n", 
	 operation_names[operation_num].c_str(), w, h, num_rounds, sigma);

  for (size_t i = 0; i < timings.size(); i++)
    printf("%sTotal = %10.3e s; Average = %10.3e s.\n", descriptions[i].c_str(), timings[i], timings[i] / num_rounds);

  return 0;
}
