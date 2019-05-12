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

/***** Function declarations ***************************/
void read_data(const char *datapath, host_vector<host_vector<float>> &data);
void read_label(const char *labelPath, host_vector<int> &label);
void flatten(host_vector<host_vector<float>> &input,
             host_vector<float> &output);
// size_in -> entire width of
// vector(MINIBATCH*Outputimage_channel*Outputimage_height*Outputimage_width)
void forward_relu(device_vector<float> &input, device_vector<float> &output);
// size_in -> entire width of
// vector(MINIBATCH*Outputimage_channel*Outputimage_height*Outputimage_width)
void backward_relu(device_vector<float> &input, device_vector<float> &output);
void reduceTofirstindex(device_vector<float> &input, int H_in, int W_in);
void reduceTofirstindex(float *input_pointer, int H_in, int W_in);
// option 1 -> forward
// option 2 -> backward
void relu_h_gpu_test(host_vector<float> &input, device_vector<float> &comp,
                     int size_in, int test_number, int option);
// size_in -> entire width of
// vector(MINIBATCH*Outputimage_channel*Outputimage_height*Outputimage_width)
void forward_bias_per_channel(device_vector<float> &input,
                              device_vector<float> &bias, int N, int ch_in,
                              int h_in, int w_in);
void backward_bias_per_channel(device_vector<float> &input,
                               device_vector<float> &bias, int N, int h_in,
                               int w_total_in, int w_ch,
                               int w_width_mul_w_height);
void backward_bias(device_vector<float> &input, device_vector<float> &bias,
                   int N, int ch_in, int h_in, int w_in);
void forward_bias_gpu_test(host_vector<float> &input,
                           device_vector<float> &bias,
                           device_vector<float> &comp, int N, int ch_in,
                           int h_in, int w_in, int test_number);
void transposeMatrix(device_vector<float> &XT, device_vector<float> &X,
                     int X_height, int X_width);
void transposeMatrix(float *XT_pointer, float *X_pointer, int input_height,
                     int input_width);
void transposeMatrix_gpu_test(host_vector<float> &Output_c,
                              host_vector<float> &input_c, int height_in,
                              int width_in, int test_number);
__global__ void forward_bias(float *X, float *b, int N, int ch_in, int h_in,
                             int w_in);
// bx = output_WIDTH, by = output_HEIGH
__global__ void gemm_h(float *Md, float *Nd, float *Pd, int M_height_in,
                       int M_width_N_height_in, int N_width_in, int height_out,
                       int width_out);
__global__ void gemm_with_bias_h(float *Md, float *Nd, float *Pd, float *B,
                                 int M_height_in, int M_width_N_height_in,
                                 int N_width_in, int height_out, int width_out);
// bx = input_WIDTH, by = input_HEIGHT
__global__ void transposeMatrix_h(float *odata, const float *idata,
                                  int height_in, int width_in);
// bx*tx = idata_width*idata*height
__global__ void grad_descent(float *odata, const float *idata, int size);
// blocknumber -> size_in/1024
__global__ void relu_h(float *X, float *Y, int size_in);
// blocknumber -> size_in/1024
__global__ void backward_relu_h(float *X, float *Y, int size_in);
// blocknumber -> Input_width/1024
__global__ void reduceTofirstindex_h(float *X, int H_in, int W_in);
__global__ void backward_bias(float *X, float *b, int N, int ch_in, int h_in,
                              int w_in);

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

/***** CLASS declarations ***************************/

class FullyConnect {

public:
  void init(int X_h, int X_w_W_h, int W_w);
  void forward();
  void backward();

  // M*(C*H*W)
  host_vector<float> X_c; // when back, dE_dX
  host_vector<float> W_c;
  host_vector<float> b_c;
  host_vector<float> Wgrad_c;
  host_vector<float> bgrad_c;
  host_vector<float> Output_c;
  device_vector<float> X; // when back, dE_dX
  device_vector<float> XT;
  device_vector<float> W;
  device_vector<float> WT;
  device_vector<float> b;
  device_vector<float> Wgrad;
  device_vector<float> bgrad;
  device_vector<float> Output; // when back, dE_dY
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

class Convolution { // M*C*H*W
public:
  void init(int minib, int X_h, int X_w, int W_w_h, int W_ch);
  void forward_gpu();
  void backward_gpu();

  host_vector<float> X_c; // when back, dE_dX
  host_vector<float> W_c;
  host_vector<float> b_c;
  host_vector<float> Wgrad_c;
  host_vector<float> Output_c;
  host_vector<float> bgrad_c;
  host_vector<float> Unroll_X_c;
  device_vector<float> X; // when back, dE_dX
  device_vector<float> W;
  device_vector<float> WT;
  device_vector<float> b;
  device_vector<float> Wgrad;
  device_vector<float> WgradTmp;
  device_vector<float> Output; // when back, dE_dY
  device_vector<float> bgrad;
  device_vector<float> Unroll_X;
  device_vector<float> Unroll_XT;

  int W_width_height;
  int W_channel;
  int X_width;
  int X_height;
  int Unroll_X_width;
  int Unroll_X_height;
  int input_image_width;
  int input_image_height;
  int minibatch;
  int Outputimage_width;
  int Outputimage_height;
  int Outputimage_channel;
  int Output_width;
  int Output_height;
};

__global__ void conv_layer_forward_gpu(float *X, float *W, float *Y, int H_in,
                                       int W_in, int W_out, int K, int M);
__global__ void unroll_kernel(int H_in, int W_in, int K, float *X,
                              float *X_unroll);

class Pool { // M*C*H*W
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

class Softmax { // M*C*H*W
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

#define gpuErrchk(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

#endif
