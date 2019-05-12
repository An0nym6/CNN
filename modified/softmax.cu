#include "fashion.h"

void Softmax::softmax(int N, int Width_in, host_vector<int> &label,
                      host_vector<float> &output)
{
  float sum;
  float lamda = 1e-8;
  float *output_pointer = thrust::raw_pointer_cast(output.data());

  float tmp;
  for (int i = 0; i < N; i++)
  {
    sum = 0;
    for (int j = 0; j < Width_in; j++)
    {
      tmp = exp(output_pointer[i * Width_in + j]);
      sum += tmp;
    }
    for (int j = 0; j < Width_in; j++)
    {
      tmp = exp(output_pointer[i * Width_in + j]);
      output_pointer[i * Width_in + j] = tmp / (sum + lamda);
    }
  }
}

// N -> current minibatch size
// minib -> minibatch index
void Softmax::accuracy(int N, int Width_in,
                       host_vector<host_vector<float>> &Xtrain,
                       host_vector<int> &label, host_vector<float> &output,
                       int minib, int &correct_num)
{
  int *label_pointer = thrust::raw_pointer_cast(label.data());
  float *output_pointer = thrust::raw_pointer_cast(output.data());
  float estimation_value = -1;
  int estimation_index = -1;
  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < Width_in; j++)
    {
      if (estimation_value < output_pointer[i * Width_in + j])
      {
        estimation_value = output_pointer[i * Width_in + j];
        estimation_index = j;
      }
    }
    if (estimation_index == label_pointer[i + minib * MINIBATCH])
    {
      correct_num++;
    }

    estimation_value = -1;
    estimation_index = -1;
  }
}

// N : MINIBATCH
// minib : current minibatch index
void Softmax::softmax_backward(int N, host_vector<int> label,
                               host_vector<float> &softmax_output,
                               host_vector<float> &delta, int Width_in,
                               int minib)
{
  int *label_pointer = thrust::raw_pointer_cast(label.data());
  float *softmax_output_pointer =
      thrust::raw_pointer_cast(softmax_output.data());
  float *delta_pointer = thrust::raw_pointer_cast(delta.data());
  for (int i = 0; i < N; i++)
  {
    int tmp_label_pointer = label_pointer[i + minib * N];
    for (int j = 0; j < Width_in; j++)
    {
      delta_pointer[i * Width_in + j] =
          (softmax_output_pointer[i * Width_in + j] -
           (float)(tmp_label_pointer ==
                   j)); // 1 is label_pointer value. minibatch sum
    }
  }

  for (int i = 1; i < N; i++)
  {
    for (int j = 0; j < Width_in; j++)
    {
      delta_pointer[0 * Width_in + j] +=
          delta_pointer[i * Width_in +
                        j]; // 1 is label_pointer value. minibatch sum
    }
  }
  for (int j = 0; j < Width_in; j++)
  {
    delta_pointer[0 * Width_in + j] =
        delta_pointer[0 * Width_in + j] / N; // average
  }
  for (int i = 1; i < N; i++)
  {
    for (int j = 0; j < Width_in; j++)
    {
      delta_pointer[i * Width_in + j] =
          delta_pointer[0 * Width_in + j]; // scattering
    }
  }
}

void Softmax::cross_entropy_loss(int N, host_vector<int> label,
                                 host_vector<float> &input, int Width_in,
                                 float &loss, int minib)
{
  loss = 0;
  float hyper_delta = 0.000001;
  float *input_pointer = thrust::raw_pointer_cast(input.data());
  for (int i = 0; i < N; i++)
  {
    int tmp_label = label[i + minib];
    for (int j = 0; j < Width_in; j++)
    {
      float log1;
      float log2;

      log1 = log(input_pointer[i * Width_in + j] + hyper_delta);
      log2 = log(1 - input_pointer[i * Width_in + j] + hyper_delta);

      if (tmp_label == j) // label is scalar
      {
        loss -= (log1 * LABEL_ONE) + log2 * (1 - LABEL_ONE); // minibatch sum
      }
      else
      {
        loss -= (log1 * LABEL_ZERO) + log2 * (1 - LABEL_ZERO); // minibatch sum
      }
    }
  }
  loss = loss / N;
}

/*
 * compute the next power-of-2 of integer n
 */
int nextPowerOfTwo(int n)
{
  n--;
  n = n >> 1 | n;
  n = n >> 2 | n;
  n = n >> 4 | n;
  n = n >> 8 | n;
  n = n >> 16 | n;
  //n = n >> 32 | n; //For 64-bits int
  return ++n;
}

/* 
 *  compute sum of one-dimension array
 *  cnt - length of array
 *  cnt2 - next power-of-2 of cnt
 */
__global__ static void array_sum(int *array, int cnt, int cnt2)
{

  extern __shared__ unsigned int sharedMem[];

  sharedMem[threadIdx.x] = (threadIdx.x < cnt) ? array[threadIdx.x] : 0;

  __syncthreads();
  //cnt2 "must" be a power of two!
  for (unsigned int s = cnt2 / 2; s > 0; s >>= 1)
  {
    if (threadIdx.x < s)
    {
      sharedMem[threadIdx.x] += sharedMem[threadIdx.x + s];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0)
  {
    array[0] = sharedMem[0];
  }
}