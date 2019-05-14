#include "fashion.h"

// Input functions: no optimizations
int reverse_int32(int i) {
  unsigned char byte1, byte2, byte3, byte4;
  byte1 = i & MAXBYTE;
  byte2 = (i >> 8) & MAXBYTE;
  byte3 = (i >> 16) & MAXBYTE;
  byte4 = (i >> 24) & MAXBYTE;
  return ((int)byte1 << 24) + ((int)byte2 << 16) + ((int)byte3 << 8) +
         (int)byte4;
}
// Input functions: no optimizations
void read_data(const char *datapath, host_vector<host_vector<float>> &data) {
  ifstream infile(datapath, ios::binary);
  if (!infile.is_open()) {
    printf("FAILED TO OPEN FILE: %s\n", datapath);
    return;
  }
  cout << "== Input test image file: " << datapath << endl;
  // Read the header information
  int magic_number = 0;
  int number_of_images = 0;
  int n_rows = 0;
  int n_cols = 0;
  infile.read((char *)&magic_number, sizeof(magic_number));
  magic_number = reverse_int32(magic_number);
  cout << "magic number: " << magic_number << endl;
  infile.read((char *)&number_of_images, sizeof(number_of_images));
  number_of_images = reverse_int32(number_of_images);
  cout << "number of images: " << number_of_images << endl;
  infile.read((char *)&n_rows, sizeof(n_rows));
  n_rows = reverse_int32(n_rows);
  infile.read((char *)&n_cols, sizeof(n_cols));
  n_cols = reverse_int32(n_cols);
  n_rows += 4;
  n_cols += 4;
  cout << "size of row = " << n_rows << ", size of cols = " << n_cols << endl;
  // Read actual data (uint8 -> float)
  for (int i = 0; i < number_of_images / MINIBATCH; ++i) {
    for (int n = 0; n < MINIBATCH; n++) {
      for (int r = 2; r < n_rows - 2; ++r) {
        for (int c = 2; c < n_cols - 2; ++c) {
          unsigned char temp = 0;
          infile.read((char *)&temp, sizeof(temp));
          data[i][(n_cols * n_rows * n) + (n_rows * r) + c] =
              (float)temp * (float)MNIST_SCALE_FACTOR;
        }
      }
    }
  }
  infile.close();
  cout << "Done. [data: " << datapath << "] [count: " << number_of_images << "]"
       << endl;
}
// Input functions: no optimizations
void read_label(const char *labelPath, host_vector<int> &labels) {
  int number_of_labels = 0;
  ifstream infile(labelPath, ios::binary);
  if (!infile.is_open()) {
    printf("FAILED TO OPEN FILE: %s\n", labelPath);
    return;
  }
  cout << "== Input test label file: " << labelPath << endl;
  int magic_number = 0;
  // read the label information
  infile.read((char *)&magic_number, sizeof(magic_number));
  magic_number = reverse_int32(magic_number);
  cout << "magic number: " << magic_number << endl;
  infile.read((char *)&number_of_labels, sizeof(number_of_labels));
  number_of_labels = reverse_int32(number_of_labels);
  cout << "number of labels: " << number_of_labels << endl;
  for (int i = 0; i < number_of_labels; ++i) {
    unsigned char temp = 0;
    infile.read((char *)&temp, sizeof(temp));
    labels[i] = (int)temp;
  }
  infile.close();
  cout << "Done. [data: " << labelPath << "] [count: " << number_of_labels
       << "] " << endl;
}

// ReLU kernel in forward propagation
__global__ void relu_h(float *X, float *Y, int size_in) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t < size_in) {
    Y[t] = 0.0;
    if (X[t] >= 0)
      Y[t] = X[t];
  }
}

// ReLU in forward propagation
void forward_relu(device_vector<float> &input, device_vector<float> &output) {
  int size_in = input.size();
  float *input_pointer = thrust::raw_pointer_cast(input.data());
  float *output_pointer = thrust::raw_pointer_cast(output.data());
  int block_size = ceil((double)size_in / 1024);
  relu_h<<<block_size, 1024>>>(input_pointer, output_pointer, size_in);
}

// ReLU kernel in backward propagation
__global__ void backward_relu_h(float *X, float *Y, int size_in) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t < size_in) {
    X[t] = 0.0;
    if (X[t] >= 0)
      X[t] = Y[t];
  }
}

// ReLU in backward propagation
void backward_relu(device_vector<float> &input, device_vector<float> &output) {
  int size_in = input.size();
  float *input_pointer = thrust::raw_pointer_cast(input.data());
  float *output_pointer = thrust::raw_pointer_cast(output.data());
  int block_size = ceil((double)size_in / 1024);
  backward_relu_h<<<block_size, 1024>>>(input_pointer, output_pointer, size_in);
}

// Matrix reduction kernel
__global__ void reduce_to_first_index_h(float *X, int height, int width) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  float tmp = 0;
  if (t < width) {
    for (int i = 0; i < height; i++) {
      tmp += X[i * width + t];
    }
    X[t] = tmp;
  }
}

// Matrix reduction: pointer version
void reduce_to_first_index(float *input_pointer, int height, int width) {
  int block_size = ceil((float)width / 1024);
  reduce_to_first_index_h<<<block_size, 1024>>>(input_pointer, height, width);
}

// Matrix reduction: device vector version
void reduce_to_first_index(device_vector<float> &input, int height, int width) {
  float *input_pointer = thrust::raw_pointer_cast(input.data());
  int block_size = ceil((float)width / 1024);
  reduce_to_first_index_h<<<block_size, 1024>>>(input_pointer, height, width);
}

// Add bias to all values kernel
__global__ void forward_bias(float *X, float *b, int N, int ch_in, int h_in,
                             int w_in) {
  int n = blockIdx.x;
  int ch = blockIdx.y;
  int h = threadIdx.x;
  int w = threadIdx.y;
  X[n * ch_in * h_in * w_in + ch * h_in * w_in + h * w_in + w] += b[ch];
}

// Add bias
void forward_bias_per_channel(device_vector<float> &input,
                              device_vector<float> &bias, int N, int ch_in,
                              int h_in, int w_in) {
  dim3 blockDim(N, ch_in);
  dim3 threadDim(h_in, w_in);
  float *input_pointer = thrust::raw_pointer_cast(input.data());
  float *bias_pointer = thrust::raw_pointer_cast(bias.data());
  forward_bias<<<blockDim, threadDim>>>(input_pointer, bias_pointer, N, ch_in,
                                        h_in, w_in);
}

// General matrix multiplication
__global__ void gemm_h(float *Md, float *Nd, float *Pd, int M_height_in,
                       int M_width_N_height_in, int N_width_in, int height_out,
                       int width_out) {
  __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int row = by * TILE_WIDTH + ty;
  int col = bx * TILE_WIDTH + tx;
  float Pvalue = 0;
  for (int m = 0; m < ceilf((float)M_width_N_height_in / TILE_WIDTH); ++m) {
    int mtx = m * TILE_WIDTH + tx;
    int mty = m * TILE_WIDTH + ty;
    if (row < M_height_in && mtx < M_width_N_height_in)
      Mds[ty][tx] = Md[row * M_width_N_height_in + mtx];
    else
      Mds[ty][tx] = 0;
    if (mty < M_width_N_height_in && col < N_width_in)
      Nds[ty][tx] = Nd[mty * N_width_in + col];
    else
      Nds[ty][tx] = 0;
    __syncthreads();
    for (int k = 0; k < TILE_WIDTH; ++k) {
      Pvalue += Mds[ty][k] * Nds[k][tx];
    }
    __syncthreads();
  }
  if (row < height_out && col < width_out)
    Pd[row * width_out + col] = Pvalue;
}
// General matrix multiplication with bias
__global__ void gemm_with_bias_h(float *Md, float *Nd, float *Pd, float *B,
                                 int M_height_in, int M_width_N_height_in,
                                 int N_width_in, int height_out,
                                 int width_out) {
  __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int row = by * TILE_WIDTH + ty;
  int col = bx * TILE_WIDTH + tx;
  float Pvalue = 0;
  for (int m = 0; m < ceilf((float)M_width_N_height_in / TILE_WIDTH); ++m) {
    int mtx = m * TILE_WIDTH + tx;
    int mty = m * TILE_WIDTH + ty;
    if (row < M_height_in && mtx < M_width_N_height_in)
      Mds[ty][tx] = Md[row * M_width_N_height_in + mtx];
    else
      Mds[ty][tx] = 0;
    if (mty < M_width_N_height_in && col < N_width_in)
      Nds[ty][tx] = Nd[mty * N_width_in + col];
    else
      Nds[ty][tx] = 0;
    __syncthreads();
    for (int k = 0; k < TILE_WIDTH; ++k) {
      Pvalue += Mds[ty][k] * Nds[k][tx];
    }
    __syncthreads();
  }
  if (row < height_out && col < width_out)
    Pd[row * width_out + col] = Pvalue + B[col];
}

// Matrix multiplication kernel
__global__ void transposeMatrix_h(float *odata, float *idata, int height,
                                  int width) {
  __shared__ float block[TILE_WIDTH][TILE_WIDTH + 1];
  unsigned int xIndex = blockIdx.x * TILE_WIDTH + threadIdx.x;
  unsigned int yIndex = blockIdx.y * TILE_WIDTH + threadIdx.y;
  if ((xIndex < width) && (yIndex < height)) {
    unsigned int index_in = yIndex * width + xIndex;
    block[threadIdx.y][threadIdx.x] = idata[index_in];
  }
  __syncthreads();
  xIndex = blockIdx.y * TILE_WIDTH + threadIdx.x;
  yIndex = blockIdx.x * TILE_WIDTH + threadIdx.y;
  if ((xIndex < height) && (yIndex < width)) {
    unsigned int index_out = yIndex * height + xIndex;
    odata[index_out] = block[threadIdx.x][threadIdx.y];
  }
}

// Matrix multiplication: device vector version
void transposeMatrix(device_vector<float> &outputT, device_vector<float> &input,
                     int input_height, int input_width) {
  dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
  dim3 numBlocks_transpose_X(ceil((float)input_width / TILE_WIDTH),
                             ceil((float)input_height / TILE_WIDTH));
  float *XT_pointer = thrust::raw_pointer_cast(outputT.data());
  float *X_pointer = thrust::raw_pointer_cast(input.data());
  transposeMatrix_h<<<numBlocks_transpose_X, threadsPerBlock>>>(
      XT_pointer, X_pointer, input_height, input_width);
}

// Matrix multiplication: pointer version
void transposeMatrix(float *XT_pointer, float *X_pointer, int input_height,
                     int input_width) {
  dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
  dim3 numBlocks_transpose_X(ceil((float)input_width / TILE_WIDTH),
                             ceil((float)input_height / TILE_WIDTH));
  transposeMatrix_h<<<numBlocks_transpose_X, threadsPerBlock>>>(
      XT_pointer, X_pointer, input_height, input_width);
}

// Gradient descent
__global__ void grad_descent(float *odata, const float *idata, int size) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t < size) {
    odata[t] -= LEARNIG_RATE * idata[t];
  }
}
// Backward propagation
void backward_bias_per_channel(device_vector<float> &input,
                               device_vector<float> &bias, int N, int h_in,
                               int w_total_in, int w_ch,
                               int w_width_mul_w_height) {
  device_vector<float> input_tmp, input_per_minibatch_T;
  input_tmp.resize(h_in * w_total_in, 0);
  input_per_minibatch_T.resize(h_in * w_total_in, 0);
  input_tmp = input;

  float *input_pointer = thrust::raw_pointer_cast(input_tmp.data());
  float *input_per_minibatch_T_pointer =
      thrust::raw_pointer_cast(input_per_minibatch_T.data());
  float *bias_pointer = thrust::raw_pointer_cast(bias.data());

  for (int i = 0; i < N; i++) {
    transposeMatrix(input_per_minibatch_T_pointer, input_pointer, w_ch,
                    w_width_mul_w_height);
    reduce_to_first_index(input_per_minibatch_T_pointer, w_width_mul_w_height,
                          w_ch);
    input_pointer += w_total_in;
    input_per_minibatch_T_pointer += w_total_in;
  }

  input_per_minibatch_T_pointer =
      thrust::raw_pointer_cast(input_per_minibatch_T.data());
  reduce_to_first_index(input_per_minibatch_T_pointer, h_in, w_total_in);

  int blockDim_b = ceil((float)w_ch / 1024);
  thrust::transform(input_per_minibatch_T.begin(), input_per_minibatch_T.end(),
                    input_per_minibatch_T.begin(), div_h());
  grad_descent<<<blockDim_b, 1024>>>(bias_pointer,
                                     input_per_minibatch_T_pointer, w_ch);
}
