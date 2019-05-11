#ifndef FASHION_H
#define FASHION_H

// C++
#include <fstream>
#include <iostream>
#include <sstream>
// CUDA
#include <thrust/host_vector.h>

using namespace std;
using namespace thrust;

// Data files
#define PATH_TRAIN_DATA "../data/train-images-idx3-ubyte"
#define PATH_TRAIN_LABEL "../data/train-labels-idx1-ubyte"
#define PATH_TEST_DATA "../data/t10k-images-idx3-ubyte"
#define PATH_TEST_LABEL "../data/t10k-labels-idx1-ubyte"
// Tunables
#define MINIBATCH 1000
#define MNIST_SCALE_FACTOR 0.00390625 // 1 / 255 = 0.00390625
#define NUM_TEST 10000
#define NUM_TRAIN 60000
#define RAW_PIXELS_PER_IMG_PADDING 1024

// util.cu
void read_data(const char *data_path, host_vector<host_vector<float> > &data);
void read_label(const char *label_path, host_vector<int> &label);
static int32_t reverse_int32(int32_t val);

#endif
