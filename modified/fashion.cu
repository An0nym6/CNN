#include "fashion.h"

int main() {
  /* Load data */
  // Features from trainning set
  host_vector<host_vector<float>> x_train(
      NUM_TRAIN / MINIBATCH,
      host_vector<float>(RAW_PIXELS_PER_IMG_PADDING * MINIBATCH, 0));
  // Features from test set
  host_vector<host_vector<float>> x_test(
      NUM_TEST / MINIBATCH,
      host_vector<float>(RAW_PIXELS_PER_IMG_PADDING * MINIBATCH, 0));
  // Labels from training set
  host_vector<int> y_train(NUM_TRAIN, 0);
  // Labels from test set
  host_vector<int> y_test(NUM_TEST, 0);
  // Read features from trainning set
  read_data(PATH_TRAIN_DATA, x_train);
  // Read labels from training set
  read_label(PATH_TRAIN_LABEL, y_train);
  // Read features from test set
  read_data(PATH_TEST_DATA, x_test);
  // Read labels from test set
  read_label(PATH_TEST_LABEL, y_test);

  /* Build model */
  GPU_Net gpu_net;
  gpu_net.train(x_train, y_train);

  /* Predict */
  gpu_net.test(x_test, y_test);
}
