#ifndef FASHION_H
#define FASHION_H

// C / C++
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
// CUDA Thrust
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
// Utility
#include <time.h>
#include <unistd.h>

// Define namespaces
using namespace std;
using namespace thrust;

// Data files
#define PATH_TRAIN_DATA "../data/train-images-idx3-ubyte"
#define PATH_TRAIN_LABEL "../data/train-labels-idx1-ubyte"
#define PATH_TEST_DATA "../data/t10k-images-idx3-ubyte"
#define PATH_TEST_LABEL "../data/t10k-labels-idx1-ubyte"
// Tunables
#define CONV_KERNEL_SIZE 5
#define LABEL_ONE 1
#define LABEL_ZERO 0
#define LAMDA (3e-1)
#define LEARNIG_RATE 1e-5
#define MAXBYTE 255
#define MNIST_SCALE_FACTOR 0.00390625 // 0.00390625 = 1 / 255
#define MINIBATCH 2
#define NUM_TEST 10000
#define NUM_TRAIN 60000
#define TILE_WIDTH 32
#define RAW_DIM 28
#define RAW_DIM_PADDING 32
#define RAW_PIXELS_PER_IMG 784 // 28 x 28, single-channel image
#define RAW_PIXELS_PER_IMG_PADDING 1024

// Function declarations
// Input functions
void read_data(const char *datapath, host_vector<host_vector<float>> &data);
void read_label(const char *labelPath, host_vector<int> &label);
// ReLU
__global__ void relu_h(float *X, float *Y, int size_in);
void forward_relu(device_vector<float> &input, device_vector<float> &output);
__global__ void backward_relu_h(float *X, float *Y, int size_in);
void backward_relu(device_vector<float> &input, device_vector<float> &output);
// Matrix reduction
__global__ void reduce_to_first_index_h(float *X, int height, int width);
void reduce_to_first_index(device_vector<float> &input, int height, int width);
void reduce_to_first_index(float *input_pointer, int height, int width);
// Add bias to all values
__global__ void forward_bias(float *X, float *b, int N, int ch_in, int h_in,
                             int w_in);
void forward_bias_per_channel(device_vector<float> &input,
                              device_vector<float> &bias, int N, int ch_in,
                              int h_in, int w_in);
// General matrix multiplication
__global__ void gemm_h(float *Md, float *Nd, float *Pd, int M_height_in,
                       int M_width_N_height_in, int N_width_in, int height_out,
                       int width_out);
__global__ void gemm_with_bias_h(float *Md, float *Nd, float *Pd, float *B,
                                 int M_height_in, int M_width_N_height_in,
                                 int N_width_in, int height_out, int width_out);
// Matrix transpose
__global__ void transposeMatrix_h(float *odata, const float *idata,
                                  int height_in, int width_in);
void transposeMatrix(device_vector<float> &XT, device_vector<float> &X,
                     int X_height, int X_width);
void transposeMatrix(float *XT_pointer, float *X_pointer, int input_height,
                     int input_width);
// Backward propagation
__global__ void grad_descent(float *odata, const float *idata, int size);
void backward_bias_per_channel(device_vector<float> &input,
                               device_vector<float> &bias, int N, int h_in,
                               int w_total_in, int w_ch,
                               int w_width_mul_w_height);

// For thrust calculation
struct square {
  __host__ __device__ float operator()(float x) { return x * x; }
};
struct div_h {
  __host__ __device__ float operator()(float x) { return x / MINIBATCH; }
};
struct mul_h {
  __host__ __device__ float operator()(float x) { return x * MINIBATCH; }
};
struct plus_h {
  const float weight_decay_h;

  __host__ __device__ plus_h(const float weight_decay)
      : weight_decay_h(weight_decay) {}
  __host__ __device__ float operator()(float x) {
    return x + LAMDA * weight_decay_h;
  }
};

// Classes
// fully_connect.cu
class FullyConnect {
public:
  void init(int X_h, int X_w_W_h, int W_w);
  void forward();
  void backward();
  host_vector<float> X_c;
  host_vector<float> W_c;
  host_vector<float> b_c;
  host_vector<float> Wgrad_c;
  host_vector<float> bgrad_c;
  host_vector<float> Output_c;
  device_vector<float> X;
  device_vector<float> XT;
  device_vector<float> W;
  device_vector<float> WT;
  device_vector<float> b;
  device_vector<float> Wgrad;
  device_vector<float> bgrad;
  device_vector<float> Output;
  device_vector<float> OutputT;
  int X_width;
  int X_height;
  int XT_width;
  int XT_height;
  int W_width;
  int W_height;
  int WT_width;
  int WT_height;
  int Output_width;
  int Output_height;
  int OutputT_width;
  int OutputT_height;
};

// convolution.cu
class Convolution {
public:
  void init(int minibatch, int in_img_h, int in_img_w, int w_w_h, int w_ch);
  void forward_gpu();
  void backward_gpu();
  device_vector<float> x;
  device_vector<float> w;
  device_vector<float> w_t;
  device_vector<float> b;
  device_vector<float> w_grad;
  device_vector<float> w_grad_tmp;
  device_vector<float> output;
  device_vector<float> bgrad;
  device_vector<float> unroll_x;
  device_vector<float> unroll_x_t;
  int w_width_height;
  int w_ch;
  int unroll_x_width;
  int unroll_x_height;
  int in_img_width;
  int in_img_height;
  int minibatch;
  int out_img_width;
  int out_img_height;
  int out_img_ch;
  int out_width;
  int out_height;
};
__global__ void conv_layer_forward_gpu(float *x, float *w, float *y, int h_in,
                                       int w_in, int w_out, int k, int m);
__global__ void unroll_kernel(int h_in, int w_in, int k, float *x,
                              float *x_unroll);

// pool.cu
class Pool {
public:
  void init(int minib, int X_h, int X_w, int X_ch, int pool_size);
  void forward_GPU_naive(device_vector<float> &input);
  void backward_GPU(device_vector<float> &output);
  host_vector<float> X_c;
  device_vector<float> X;
  host_vector<float> Output_c;
  device_vector<float> Output;
  device_vector<float> b;
  device_vector<float> b_c;
  int X_height;
  int X_width;
  int b_height;
  int b_width;
  int Inputimage_height;
  int Inputimage_width;
  int Inputimage_channel;
  int Outputimage_height;
  int Outputimage_width;
  int Outputimage_channel;
  int Output_height;
  int Output_width;
  int minibatch;
  int pool_size;
};
__global__ void poolingLayer_forward_GPU_naive(float *X, int H_in, int W_in,
                                               float *Y, int M, int pool_size);
__global__ void poolingLayer_backward_GPU(float *X, int H_in, int W_in,
                                          float *Y, int M, int pool_size);

// softmax.cu
class Softmax {
public:
  host_vector<float> delta_c;
  device_vector<float> delta;
  float loss;
  void cross_entropy_loss(int N, host_vector<int> label,
                          host_vector<float> &input, int Width_in, float &loss,
                          int minib);
  void softmax_backward(int N, host_vector<int> label,
                        host_vector<float> &softmax_output,
                        host_vector<float> &delta, int Width_in, int minib);
  void softmax(int N, int Width_in, host_vector<int> &label,
               host_vector<float> &output);
  void accuracy(int N, int Width_in, host_vector<host_vector<float>> &Xtrain,
                host_vector<int> &label, host_vector<float> &output, int minib,
                int &correct_num);
};

// gpu_net.cu
// Main
class GPU_Net {
public:
  GPU_Net();
  ~GPU_Net();
  virtual void train(host_vector<host_vector<float>> &Xtrain,
                     host_vector<int> &Ytrain);
  virtual void test(host_vector<host_vector<float>> &Xtest,
                    host_vector<int> &Ytest);
  Convolution conv1;
  Pool pool1;
  FullyConnect fc1, fc2, fc3, fc4;
  Softmax sm1;
  int correct_num;
};

#endif
