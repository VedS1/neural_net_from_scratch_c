#include "propagation.h"

//W1 and W2 are weight matricies
//b1 and b2 are bias vectors
//X input matrix
//Z1 and Z2 linear comb of activation layer?? not rlly sure tbh lol
//A1 and A2 activation matrix result and softmax result
void forward_propagation(Matrix W1, Matrix b1, Matrix W2, Matrix b2, Matrix X, Matrix* Z1, Matrix* Z2) {
    //Z1 = W1 * X + b1
    Z1 = matrix_multi(&W1, &X);
    Z1 = matrix_add(Z1, &b1);
    
    //A1 = swish(Z1)
    Matrix* A1;
    matrix_copy(Z1, A1);
    activationFunctionSwish(A1);

    // Z2 = W2 * A1 + b2
    Z2 = matrix_multi(W2, A1);
    Z2 = matrix_add(Z2, b2);

    // A2 = softmax(Z2)
    A2 = matrix_alloc(Z2->rows, Z2->cols);  // Allocating memory for A2
    matrix_copy(Z2, A2);
    softmax(A2);
}