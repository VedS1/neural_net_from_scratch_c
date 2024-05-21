#ifndef PROPAGATION_H
#define PROPAGATION_H

#include "../matrix/matrix.h"

void forward_propagation(Matrix* W1, Matrix* b1, Matrix* W2, Matrix* b2, Matrix* X, Matrix** Z1, Matrix** Z2);
void backward_propagation(Matrix* Z1, Matrix* A1, Matrix* Z2, Matrix* A2, Matrix* W1, Matrix* W2, Matrix* X, Matrix* Y, double m, Matrix** dW1, Matrix** db1, Matrix** dW2, Matrix** db2);
void ReLU_deriv(Matrix* mat);
Matrix* one_hot(Matrix* Y, int max_value);
void update_params(Matrix* W1, Matrix* b1, Matrix* W2, Matrix* b2, Matrix* dW1, Matrix* db1, Matrix* dW2, Matrix* db2, double alpha);
Matrix* get_predictions(Matrix* A2);
double get_accuracy(Matrix* predictions, Matrix* Y);
void gradient_descent(Matrix* X, Matrix* Y, double alpha, int iterations, Matrix** W1, Matrix** b1, Matrix** W2, Matrix** b2);
void init_params(Matrix** W1, Matrix** b1, Matrix** W2, Matrix** b2, int input_size, int hidden_size, int output_size);

#endif // PROPAGATION_H

