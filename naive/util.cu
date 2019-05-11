#include "fashion.h"

// Read features from files
void read_data(const char *data_path, host_vector<host_vector<float> > &data) {
  ifstream infile(data_path, ios::binary);
  if (!infile.is_open()) {
    printf("FAILED TO OPEN FILE: %s\n", data_path);
    return;
  }
  cout << "== Input test image file: " << data_path << endl;
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
  cout << "Done. [data: " << data_path << "] [count: " << number_of_images
       << "]" << endl;
}

// Read labels from files
void read_label(const char *label_path, host_vector<int> &labels) {
  int number_of_labels = 0;
  ifstream infile(label_path, ios::binary);
  if (!infile.is_open()) {
    printf("FAILED TO OPEN FILE: %s\n", label_path);
    return;
  }
  cout << "== Input test label file: " << label_path << endl;
  int magic_number = 0;
  // Read the label information
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
  cout << "Done. [data: " << label_path << "] [count: " << number_of_labels
       << "] " << endl;
}

// This function should be provided by some headers, but we couldn't find it
// So we copy this code from APPFS documentation
static int32_t reverse_int32(int32_t val) {
  val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
  return (val << 16) | ((val >> 16) & 0xFFFF);
}
