#include "propagation.h"

//W1 and W2 are weight matricies
//b1 and b2 are bias vectors
//X input matrix
//Z1 and Z2 linear comb of activation layer?? not rlly sure tbh lol
//A1 and A2 activation matrix result and softmax result

void forward_propagation(Matrix W1, Matrix b1, Matrix W2, Matrix b2, Matrix X, Matrix* Z1, Matrix* Z2) {
    //Z1 = W1 * X
    Z1 = matrix_multi(&W1, &X);
    if (!Z1) {
        fprintf(stderr, "Multiplication failed for W1 * X\n");
        return;
    }
    // + b1
    matrix_add(Z1, &b1);

	Matrix* A1 = matrix_alloc((Z1)->rows, (Z1)->cols);
    matrix_copy(Z1, A1);
    activationFunctionSwish(A1); //could b ReLU but ig u used swish, either works idc

    //Z2 = W2 * A1
    Z2 = matrix_multi(&W2, A1);
    if (!Z2) {
        fprintf(stderr, "Multiplication failed for W2 * A1\n");
        matrix_dealloc(A1);
        return;
    }
    // + b2
    matrix_add(Z2, &b2);

    softmax(Z2); 
	//softmax is an activation layer
	//generally, we would put it inside of the function that instantiates the nn;

    matrix_dealloc(A1);  
}

Matrix* one_hot(Matrix* Y, int max_value) {
    // Check if Y is a 1x(?) vector
    if (Y->cols != 1) {
        fprintf(stderr, "Input must be 1x(?) vector.\n");
        return NULL;
    }

    int size = Y->rows;
    Matrix* result = matrix_alloc(size, max_value + 1);
    if (!result) {
        fprintf(stderr, "Failed to allocate memory for one-hot.\n");
        return NULL;
    }

    // Initialize to zero
    for (int i = 0; i < size; i++) {
        for (int j = 0; j <= max_value; j++) {
            result->data[i][j] = 0.0;
        }
    }

    for (int i = 0; i < size; i++) {
        int label = (int)Y->data[i][0];
        if (label < 0 || label > max_value) {
            fprintf(stderr, "Out of range one hot value: %d\n", label);
            matrix_dealloc(result);
            return NULL;
        }
		//change value of vector index to 1 (one hot)
        result->data[i][label] = 1.0;
    }

    return result;
}
