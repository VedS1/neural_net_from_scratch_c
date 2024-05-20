#include "../matrix/matrix.h"

void forward_propagation(Matrix W1, Matrix b1, Matrix W2, Matrix b2, Matrix X, Matrix* Z1, Matrix* Z2); //done (I think), Not tested

Matrix* one_hot(Matrix* Y, int max_value); //done, not tested

void ReLU_deriv(Matrix* mat);

void back_propagation(Matrix* Z1, Matrix* A1, Matrix* Z2, Matrix* A2, Matrix* W1, Matrix* W2, Matrix* X, Matrix* Y, double m, Matrix** dW1, Matrix** db1, Matrix** dW2, Matrix** db2);
//TBA backward prop, derivative of relu, update params
