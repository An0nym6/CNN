#include "fashion.h"

// Constructor
GPU_Net::GPU_Net() {
  // conv1 Input = 1x32x32 Output = 8x20x20
  // pool1 Input = 8x20x20 Output = 8x10x10
  // fc1 Input = 8x10x10(800) Output = 800
  // fc2 Input = 800 Output = 400
  // fc3 Input = 400 Output = 200
  // fc3 Input = 200 Output = 10

  conv1.init(MINIBATCH, 32, 32, 1, 5, 8);
  pool1.init(MINIBATCH, 28, 28, 8, 2);
  fc1.init(MINIBATCH, 14 * 14 * 8, 1600);
  fc2.init(MINIBATCH, 1600, 800);
  fc3.init(MINIBATCH, 800, 200);
  fc4.init(MINIBATCH, 200, 10);
  sm1.delta_c.resize(MINIBATCH * 10, 0);
  correct_num = 0;
}

// Desctuctor
GPU_Net::~GPU_Net() {}

// Train on training set
void GPU_Net::train(host_vector<host_vector<float>> &x_train,
                    host_vector<int> &y_train) {
  cout << "Train Start" << endl << fflush;
  // Output variables
  int minibatch_index;
  struct timespec start, finish;
  double elapsed;
  int fine_epoch = 0;
  float max_acc = 0;
  // Run 100 epoches
  for (int epoch = 1; epoch <= 10; epoch++) {
    minibatch_index = 0;
    correct_num = 0;
    elapsed = 0;
    // Run NUM_TRAIN / MINIBATCH batches
    while (minibatch_index < NUM_TRAIN / MINIBATCH) {
      clock_gettime(CLOCK_REALTIME, &start);
      // Convolution layer forward propagation
      conv1.X = x_train[minibatch_index];
      conv1.forward_GPU_naive();
      // Pooling layer forward propagation
      pool1.forward_GPU_naive(conv1.Output);
      forward_bias_per_channel(
          pool1.Output, pool1.b, MINIBATCH, pool1.Outputimage_channel,
          pool1.Outputimage_height, pool1.Outputimage_width);
      forward_relu(pool1.Output, fc1.X);
      // Fully connected layer 1 forward propagation
      fc1.forward();
      forward_relu(fc1.Output, fc2.X);
      // Fully connected layer 2 forward propagation
      fc2.forward();
      forward_relu(fc2.Output, fc3.X);
      // Fully connected layer 3 forward propagation
      fc3.forward();
      forward_relu(fc3.Output, fc4.X);
      // Fully connected layer 4 forward propagation
      fc4.X = fc3.Output;
      fc4.forward();
      fc4.Output_c = fc4.Output;
      // SoftMax forward propagation
      sm1.accuracy(MINIBATCH, 10, x_train, y_train, fc4.Output_c,
                   minibatch_index, correct_num);
      sm1.softmax(MINIBATCH, 10, y_train, fc4.Output_c);
      sm1.cross_entropy_loss(MINIBATCH, y_train, fc4.Output_c, 10, sm1.loss,
                             minibatch_index);
      // Softmax backward propagation
      sm1.softmax_backward(MINIBATCH, y_train, fc4.Output_c, sm1.delta_c, 10,
                           minibatch_index);
      // Fully connected layer 4 backward propagation
      fc4.Output = sm1.delta_c;
      fc4.backward();
      backward_relu(fc3.Output, fc4.X);
      // Fully connected layer 3 backward propagation
      fc3.Output = fc4.X;
      fc3.backward();
      backward_relu(fc2.Output, fc3.X);
      // Fully connected layer 2 backward propagation
      fc2.Output = fc3.X;
      fc2.backward();
      backward_relu(fc1.Output, fc2.X);
      // Fully connected layer 1 backward propagation
      fc1.backward();
      backward_relu(pool1.Output, fc1.X);
      // Pooling layer backward propagation
      backward_bias_per_channel(
          pool1.Output, pool1.b, MINIBATCH, pool1.Output_height,
          pool1.Output_width, pool1.Outputimage_channel,
          pool1.Outputimage_height * pool1.Outputimage_width);
      pool1.backward_GPU(fc1.X);
      // Convolution layer backward propagation
      conv1.Output = pool1.X;
      conv1.backward_GPU_gemm();

      // Calculate output variables
      clock_gettime(CLOCK_REALTIME, &finish);
      elapsed += ((double)finish.tv_sec - start.tv_sec) +
                 ((double)finish.tv_nsec - start.tv_nsec) / 1000000000.0;
      minibatch_index += 1;
      if ((minibatch_index * MINIBATCH) == NUM_TRAIN) {
        float acc = (float)correct_num / NUM_TRAIN;
        if (acc > max_acc) {
          max_acc = acc;
          fine_epoch = epoch;
        }
        correct_num = 0;
        printf("[Epoch %d] minibatch %d (%.7lf images/sec) max_acc %.3f acc "
               "%.3f elapsed time %.3f\n",
               epoch, minibatch_index, minibatch_index * MINIBATCH / elapsed,
               max_acc * 100, acc * 100, elapsed);
        fflush(stdin);
      }
    }
  }
}

// Test on test set
void GPU_Net::test(host_vector<host_vector<float>> &Xtest,
                   host_vector<int> &Ytest) {
  cout << "Test Start" << endl << fflush;
  // Output variables
  int minibatch_index = 0;
  struct timespec start, finish;
  double elapsed = 0;
  correct_num = 0;
  int fine_epoch = 0;
  // Test on NUM_TEST / MINIBATCH batches
  while (minibatch_index < NUM_TEST / MINIBATCH) {
    clock_gettime(CLOCK_REALTIME, &start);
    // Convolution layer forward propagation
    conv1.X = Xtest[minibatch_index];
    conv1.forward_GPU_naive();
    // Pooling layer forward propagation
    pool1.forward_GPU_naive(conv1.Output);
    forward_bias_per_channel(pool1.Output, pool1.b, MINIBATCH,
                             pool1.Outputimage_channel,
                             pool1.Outputimage_height, pool1.Outputimage_width);
    forward_relu(pool1.Output, fc1.X);
    // Fully connected layer 1 forward propagation
    fc1.forward();
    forward_relu(fc1.Output, fc2.X);
    // Fully connected layer 2 forward propagation
    fc2.forward();
    forward_relu(fc2.Output, fc3.X);
    // Fully connected layer 3 forward propagation
    fc3.forward();
    forward_relu(fc3.Output, fc4.X);
    // Fully connected layer 4 forward propagation
    fc4.X = fc3.Output;
    fc4.forward();
    fc4.Output_c = fc4.Output;
    // SoftMax forward propagation
    sm1.accuracy(MINIBATCH, 10, Xtest, Ytest, fc4.Output_c, minibatch_index,
                 correct_num);

    // Calculate output variables
    clock_gettime(CLOCK_REALTIME, &finish);
    elapsed += ((double)finish.tv_sec - start.tv_sec) +
               ((double)finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    minibatch_index += 1;
    if ((minibatch_index * MINIBATCH) == 10000) {
      float acc = (float)correct_num / (10000);
      correct_num = 0;
      printf("[Test](%.7lf images/sec) acc %.3f elapsed time %.3f\n",
             minibatch_index * MINIBATCH / elapsed, acc * 100, elapsed);
      fflush(stdin);
    }
  }
}
