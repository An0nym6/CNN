#include "fashion.h"

// Convolution layer initialization
void Convolution::init(int minibatch, int in_img_h, int in_img_w, int w_w_h,
                       int w_ch) {
  // Define random generator for initializing weights
  std::default_random_engine generator;
  std::normal_distribution<float> distribution(0, 0.1);
  // Initialize the member variables
  this->w_width_height = w_w_h;   // Size of the convolution kernel
  this->w_ch = w_ch;              // Number of convolution kernels (channels)
  this->in_img_width = in_img_w;  // Input image width
  this->in_img_height = in_img_h; // Input image height
  this->minibatch = minibatch;    // Batch size
  this->out_img_width =
      (in_img_width - w_width_height + 1); // Output image width
  this->out_img_height =
      (in_img_height - w_width_height + 1); // Output image height
  this->out_img_ch = w_ch;      // Number of output images (channels)
  this->out_height = minibatch; // Each example per row
  this->out_width = out_img_ch * out_img_height *
                    out_img_width; // Then the size of one row = number of
                                   // channels * image size
  this->unroll_x_width = out_img_width * out_img_height;
  this->unroll_x_height = w_width_height * w_width_height;
  // Use resize to define vector sizes
  this->x.resize(minibatch * in_img_height * in_img_width, 0);
  this->unroll_x.resize(
      w_width_height * w_width_height * out_img_width * out_img_height, 0);
  this->unroll_x_t.resize(
      out_img_width * out_img_height * w_width_height * w_width_height, 0);
  this->w.resize(out_img_ch * w_width_height * w_width_height, 0.5);
  this->w_t.resize(w_width_height * w_width_height * out_img_ch, 0.5);
  // Initialize weights
  for (int i = 0; i < w_ch * w_width_height * w_width_height; i++) {
    w[i] = distribution(generator);
  }
  this->output.resize(minibatch * out_img_ch * out_img_width * out_img_height,
                      0);
  this->w_grad.resize(out_img_ch * w_width_height * w_width_height, 0);
  this->w_grad_tmp.resize(out_img_ch * w_width_height * w_width_height, 0);
}

// Convolution operation for all examples under one batch
// Forward propagation
void Convolution::forward_gpu() {
  // Define GPU kernel configures
  dim3 num_threads(TILE_WIDTH, TILE_WIDTH);
  dim3 num_blocks(minibatch, out_img_ch);
  // Prepare input and output data for GPU kernel
  float *in_pointer = thrust::raw_pointer_cast(x.data());
  float *w_pointer = thrust::raw_pointer_cast(w.data());
  float *out_pointer = thrust::raw_pointer_cast(output.data());
  // Convolution operation for one pixel from one channel (image)
  // under one batch
  conv_layer_forward_gpu<<<num_blocks, num_threads>>>(
      in_pointer, w_pointer, out_pointer, in_img_height, in_img_width,
      out_img_width, w_width_height, out_img_ch);
}

// Backward propagation
void Convolution::backward_gpu() {
  // Define GPU kernel configures for unroll_kernel
  int num_threads = out_img_height * out_img_width;
  int num_blocks = ceil((float)num_threads / 1024);
  // Define GPU kernel configures for gemm_h
  dim3 num_threads_gemm(TILE_WIDTH, TILE_WIDTH);
  dim3 num_blocks_gemm(ceil((float)unroll_x_height / TILE_WIDTH),
                       ceil((float)out_img_ch / TILE_WIDTH));
  // Prepare input and output data for unroll_kernel
  float *x_pointer = thrust::raw_pointer_cast(x.data());
  float *unroll_x_pointer = thrust::raw_pointer_cast(unroll_x.data());
  // Prepare input and output data for gemm_h
  float *out_pointer = thrust::raw_pointer_cast(output.data());
  float *unroll_x_t_pointer = thrust::raw_pointer_cast(unroll_x_t.data());
  float *w_grad_tmp_pointer = thrust::raw_pointer_cast(w_grad_tmp.data());

  // Loop through each example inside one batch
  for (int i = 0; i < minibatch; i++) {
    unroll_kernel<<<num_blocks, 1024>>>(in_img_height, in_img_width,
                                        w_width_height, x_pointer,
                                        unroll_x_pointer);

    // dL/dy * Unroll_x^t  = dy/dW
    transposeMatrix(unroll_x_t, unroll_x, unroll_x_height, unroll_x_width);
    gemm_h<<<num_blocks_gemm, num_threads_gemm>>>(
        out_pointer, unroll_x_t_pointer, w_grad_tmp_pointer, out_img_ch,
        out_img_height * out_img_width, unroll_x_height, out_img_ch,
        unroll_x_height);

    // Sum w_grad
    thrust::transform(w_grad.begin(), w_grad.end(), w_grad_tmp.begin(),
                      w_grad.begin(), thrust::plus<float>());
    // Move to next example
    out_pointer = out_pointer + (out_img_ch * out_img_height * out_img_width);
    x_pointer = x_pointer + (in_img_height * in_img_width);
  }

  // Divide by minibatch
  thrust::transform(w_grad.begin(), w_grad.end(), w_grad.begin(), div_h());

  // Gradient descent
  // Define GPU kernel configures for grad_descent
  int num_blocks_grad = ceil((float)out_img_ch * unroll_x_height / 1024);
  // Prepare input and output data for grad_descent
  float *w_grad_pointer = thrust::raw_pointer_cast(w_grad.data());
  float *w_pointer = thrust::raw_pointer_cast(w.data());
  grad_descent<<<num_blocks_grad, 1024>>>(w_pointer, w_grad_pointer,
                                          out_img_ch * unroll_x_height);
}

// Convolution operation for one pixel from one channel (image) under one batch
__global__ void conv_layer_forward_gpu(float *x, float *w, float *y, int h_in,
                                       int w_in, int w_out, int k, int m) {
  int n, m_, h, w_, p, q;
  n = blockIdx.x;   // Batch index
  m_ = blockIdx.y;  // Channel index
  h = threadIdx.y;  // Pixel (h, w_)
  w_ = threadIdx.x; // Pixel (h, w_)
  float ans = 0;    // Return value
  // Loop over k by k kernel
  if (h < w_out && w_ < w_out) {
    for (p = 0; p < k; p++) {
      for (q = 0; q < k; q++)
        ans = ans + x[n * (h_in * w_in) + (h + p) * w_in + (w_ + q)] *
                        w[m_ * (k * k) + p * k + q];
    }
    // Write out the return value
    y[n * (m * w_out * w_out) + m_ * (w_out * w_out) + h * w_out + w_] = ans;
  }
}

__global__ void unroll_kernel(int h_in, int w_in, int k, float *x,
                              float *x_unroll) {
  int w_out_, h_out_, h_unroll, w_unroll_, p, q;
  int t = blockIdx.x * 1024 + threadIdx.x; // Index of this thread
  int w_out = w_in - k + 1;                // Output image size
  int w_unroll = w_out * w_out;            // Unroll limit

  if (t < w_unroll) {
    h_out_ = t / w_out;                  // Output height
    w_out_ = t % w_out;                  // Output width
    w_unroll_ = h_out_ * w_out + w_out_; // The index of output pixel in image
    for (p = 0; p < k; p++)
      for (q = 0; q < k; q++) {
        h_unroll = p * k + q;
        if ((h_out_ + p) < h_in && (w_out_ + q) < w_in)
          x_unroll[h_unroll * w_unroll + w_unroll_] =
              x[(h_out_ + p) * w_in + w_out_ + q];
      }
  }
}
