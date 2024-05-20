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

void ReLU_deriv(Matrix* mat){
	if (!mat) {
		fprintf(stderr, "ReLU_deriv error: Input matrix is NULL\n");
		return;
    }
	for (int i = 0; i < mat->rows; i++) {
		for (int j = 0; j < mat->cols; j++) {
			mat->data[i][j] = mat->data[i][j] > 0 ? 1.0 : 0.0; //Relu Derivative is 1 for x >= 0 and 0 for x < 0
		}
	}
}


void backward_propagation(Matrix* Z1, Matrix* A1, Matrix* Z2, Matrix* A2, Matrix* W1, Matrix* W2, Matrix* X, Matrix* Y, double m, Matrix** dW1, Matrix** db1, Matrix** dW2, Matrix** db2) {
    Matrix* one_hot_Y = one_hot(Y, 1); 

    // dZ2 = A2 - one_hot_Y
	Matrix* dZ2 = matrix_alloc(A2->rows, A2->cols);
    matrix_copy(A2, dZ2); 
    matrix_sub(dZ2, one_hot_Y);

    // dW2 = (1/m) * dZ2 * A1.T
	Matrix* A1_trans = matrix_alloc(A1->rows, A1->cols);
	matrix_copy(A1, A1_trans);
    transpose(A1_trans);
    *dW2 = matrix_multi(dZ2, A1_trans);
    matrix_scale(*dW2, 1.0 / m);

    // db2 = sum of dZ2, scaled by 1/m
    *db2 = matrix_sum_cols(dZ2);
    matrix_scale(*db2, 1.0 / m);

    // dZ1 = (W2.T * dZ2) element_multi by ReLU_deriv(Z1)
    Matrix* W2_trans = matrix_alloc(W2->rows, W2->cols);
	matrix_copy(W2, W2_trans);
	transpose(W2_trans);
    Matrix* dZ1 = matrix_multi(W2_trans, dZ2);
    ReLU_deriv(Z1);  
    matrix_element_multi(dZ1, Z1);  

    // dW1 = (1/m) * dZ1 * X.T
    Matrix* X_trans = matrix_alloc(X->rows, X->cols);
	matrix_copy(X, X_trans);
	transpose(X_trans);
    *dW1 = matrix_multi(dZ1, X_trans);
    matrix_scale(*dW1, 1.0 / m);

    // db1 = sum of dZ1 cols, scaled by 1/m
    *db1 = matrix_sum_cols(dZ1);
    matrix_scale(*db1, 1.0 / m);

    // Clean up intermediate matrices
    matrix_dealloc(one_hot_Y);
    matrix_dealloc(dZ2);
    matrix_dealloc(A1_trans);
    matrix_dealloc(W2_trans);
    matrix_dealloc(dZ1);
    matrix_dealloc(X_trans);
}

