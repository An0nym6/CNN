#include "fashion.h"

void Softmax::softmax(int N, int Width_in, host_vector<int> &label,
                      host_vector<float> &output)
{
  float sum;
  float lamda = 1e-8;
  float *output_pointer = thrust::raw_pointer_cast(output.data());

  float tmp;
  // compute the result of each value
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
  // Prepare input and output data for softmax accuracy
  int *label_pointer = thrust::raw_pointer_cast(label.data());
  float *output_pointer = thrust::raw_pointer_cast(output.data());
  float estimation_value = -1;
  int estimation_index = -1;
  // check the accuracy of each sample
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
  // Prepare input and output data for softmax accuracy
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

//softmax entropy loss criterion function
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