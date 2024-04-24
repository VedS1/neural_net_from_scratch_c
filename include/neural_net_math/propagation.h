#include "../matrix/matrix.h"

void forward_propagation(Matrix W1, Matrix B1, Matrix W2, Matrix B2, Matrix X, Matrix* Z1, Matrix* A1, Matrix* Z2, Matrix* A2); 

Matrix one_hot(Matrix Y, int size, int max_value);


//TBA backward prop, derivative of relu, one-hot, update params,
