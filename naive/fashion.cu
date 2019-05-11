#include "fashion.h"

int main() {
  /*
   * Load data
   */
  // Features
  host_vector<host_vector<float> > x_train(
      NUM_TRAIN / MINIBATCH,
      host_vector<float>(RAW_PIXELS_PER_IMG_PADDING * MINIBATCH, 0));
  host_vector<host_vector<float> > x_test(
      NUM_TEST / MINIBATCH,
      host_vector<float>(RAW_PIXELS_PER_IMG_PADDING * MINIBATCH, 0));
  // Labels
  host_vector<int> y_train(NUM_TRAIN, 0);
  host_vector<int> y_test(NUM_TEST, 0);
  // Read data from file
  read_data(PATH_TRAIN_DATA, x_train);
  read_label(PATH_TRAIN_LABEL, y_train);
  read_data(PATH_TEST_DATA, x_test);
  read_label(PATH_TEST_LABEL, y_test);

  return 0;
}
