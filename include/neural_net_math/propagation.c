#include "propagation.h"

//W1 and W2 are weight matrices
//b1 and b2 are bias vectors
//X input matrix
//Z1 and Z2 linear comb of activation layer?? not rlly sure tbh lol
//A1 and A2 activation matrix result and softmax result

void forward_propagation(Matrix* W1, Matrix* b1, Matrix* W2, Matrix* b2, Matrix* X, Matrix** Z1, Matrix** Z2) {
    // Z1 = W1 * X + b1
    *Z1 = matrix_multi(W1, X);
    if (!*Z1) {
        fprintf(stderr, "Multiplication failed for W1 * X\n");
        return;
    }
    matrix_add(*Z1, b1);

    Matrix* A1 = matrix_alloc((*Z1)->rows, (*Z1)->cols);
    matrix_copy(*Z1, A1);
    activationFunctionSwish(A1);

    // Z2 = W2 * A1 + b2
    *Z2 = matrix_multi(W2, A1);
    if (!*Z2) {
        fprintf(stderr, "Multiplication failed for W2 * A1\n");
        matrix_dealloc(A1);
        return;
    }
    matrix_add(*Z2, b2);

    softmax(*Z2);

    matrix_dealloc(A1);
}

Matrix* one_hot(Matrix* Y, int max_value) {
    if (Y->cols != 1) {
        fprintf(stderr, "Input must be a 1x(?) vector.\n");
        return NULL;
    }

    int size = Y->rows;
    Matrix* result = matrix_alloc(size, max_value + 1);
    if (!result) {
        fprintf(stderr, "Failed to allocate memory for one-hot.\n");
        return NULL;
    }

    for (int i = 0; i < size; i++) {
        int label = (int)Y->data[i];
        if (label < 0 || label > max_value) {
            fprintf(stderr, "Out of range one-hot value: %d\n", label);
            matrix_dealloc(result);
            return NULL;
        }
        result->data[i * (max_value + 1) + label] = 1.0;
    }

    return result;
}

void ReLU_deriv(Matrix* mat) {
    if (!mat) {
        fprintf(stderr, "ReLU_deriv error: Input matrix is NULL\n");
        return;
    }
    for (int i = 0; i < mat->rows * mat->cols; i++) {
        mat->data[i] = mat->data[i] > 0 ? 1.0 : 0.0;
    }
}

// Backward propagation
void backward_propagation(Matrix* Z1, Matrix* A1, Matrix* Z2, Matrix* A2, Matrix* W1, Matrix* W2, Matrix* X, Matrix* Y, double m, Matrix** dW1, Matrix** db1, Matrix** dW2, Matrix** db2) {
    Matrix* one_hot_Y = one_hot(Y, 1);
    if (!one_hot_Y) {
        return;
    }

    // dZ2 = A2 - one_hot_Y
    Matrix* dZ2 = matrix_alloc(A2->rows, A2->cols);
    matrix_copy(A2, dZ2);
    matrix_sub(dZ2, one_hot_Y);

    // dW2 = (1/m) * dZ2 * A1.T
    Matrix* A1_trans = matrix_alloc(A1->cols, A1->rows);
    matrix_copy(A1, A1_trans);
    transpose(A1_trans);
    *dW2 = matrix_multi(dZ2, A1_trans);
    matrix_scale(*dW2, 1.0 / m);

    // db2 = sum of dZ2, scaled by 1/m
    *db2 = matrix_sum_cols(dZ2);
    matrix_scale(*db2, 1.0 / m);

    // dZ1 = (W2.T * dZ2) element-wise multiplied by ReLU_deriv(Z1)
    Matrix* W2_trans = matrix_alloc(W2->cols, W2->rows);
    matrix_copy(W2, W2_trans);
    transpose(W2_trans);
    Matrix* dZ1 = matrix_multi(W2_trans, dZ2);
    ReLU_deriv(Z1);
    matrix_element_multi(dZ1, Z1);

    // dW1 = (1/m) * dZ1 * X.T
    Matrix* X_trans = matrix_alloc(X->cols, X->rows);
    matrix_copy(X, X_trans);
    transpose(X_trans);
    *dW1 = matrix_multi(dZ1, X_trans);
    matrix_scale(*dW1, 1.0 / m);

    // db1 = sum of dZ1 cols, scaled by 1/m
    *db1 = matrix_sum_cols(dZ1);
    matrix_scale(*db1, 1.0 / m);

    matrix_dealloc(one_hot_Y);
    matrix_dealloc(dZ2);
    matrix_dealloc(A1_trans);
    matrix_dealloc(W2_trans);
    matrix_dealloc(dZ1);
    matrix_dealloc(X_trans);
}

//idk this is whatever the python source says so...
void update_params(Matrix* W1, Matrix* b1, Matrix* W2, Matrix* b2, Matrix* dW1, Matrix* db1, Matrix* dW2, Matrix* db2, double alpha) {
    for (int i = 0; i < W1->rows * W1->cols; i++) {
        W1->data[i] -= alpha * dW1->data[i];
    }
    for (int i = 0; i < b1->rows * b1->cols; i++) {
        b1->data[i] -= alpha * db1->data[i];
    }
    for (int i = 0; i < W2->rows * W2->cols; i++) {
        W2->data[i] -= alpha * dW2->data[i];
    }
    for (int i = 0; i < b2->rows * b2->cols; i++) {
        b2->data[i] -= alpha * db2->data[i];
    }
}

//same thing
Matrix* get_predictions(Matrix* A2) {
    Matrix* predictions = matrix_alloc(A2->rows, 1);
    if (!predictions) {
        perror("Failed to allocate predictions matrix");
        return NULL;
    }

    for (int i = 0; i < A2->rows; i++) {
        double max_value = A2->data[i * A2->cols];
        int max_index = 0;
        for (int j = 1; j < A2->cols; j++) {
            if (A2->data[i * A2->cols + j] > max_value) {
                max_value = A2->data[i * A2->cols + j];
                max_index = j;
            }
        }
        predictions->data[i] = max_index;
    }

    return predictions;
}

double get_accuracy(Matrix* predictions, Matrix* Y) {
    int correct_count = 0;
    for (int i = 0; i < Y->rows; i++) {
        if ((int)predictions->data[i] == (int)Y->data[i]) {
            correct_count++;
        }
    }
    return (double)correct_count / Y->rows;
}
//This function was left out for whatever reason, it shows up way earlier in the python but I just put it here.
void init_params(Matrix** W1, Matrix** b1, Matrix** W2, Matrix** b2, int input_size, int hidden_size, int output_size) {
    *W1 = matrix_alloc(hidden_size, input_size);
    *b1 = matrix_alloc(hidden_size, 1);
    *W2 = matrix_alloc(output_size, hidden_size);
    *b2 = matrix_alloc(output_size, 1);

    if (!*W1 || !*b1 || !*W2 || !*b2) {
        fprintf(stderr, "Failed to allocate memory for parameters\n");
        exit(EXIT_FAILURE);
    }

    // Initialize weights with small random values
    for (int i = 0; i < (*W1)->rows * (*W1)->cols; i++) {
        (*W1)->data[i] = ((double)rand() / RAND_MAX) * 0.01;
    }
    for (int i = 0; i < (*W2)->rows * (*W2)->cols; i++) {
        (*W2)->data[i] = ((double)rand() / RAND_MAX) * 0.01;
    }

    // Initialize biases to zero
    for (int i = 0; i < (*b1)->rows * (*b1)->cols; i++) {
        (*b1)->data[i] = 0.0;
    }
    for (int i = 0; i < (*b2)->rows * (*b2)->cols; i++) {
        (*b2)->data[i] = 0.0;
    }
}

//function in order to load the data from the CSV file for training.
void load_data(const char* x_file, const char* y_file, Matrix** X, Matrix** Y, int rows, int cols) {
    *X = matrix_alloc(rows, cols);
    *Y = matrix_alloc(rows, 1);

    FILE* fx = fopen(x_file, "rb");
    FILE* fy = fopen(y_file, "rb");

    if (!fx || !fy) {
        perror("Failed to open data files");
        exit(EXIT_FAILURE);
    }

    fread((*X)->data, sizeof(double), rows * cols, fx);
    fread((*Y)->data, sizeof(double), rows, fy);

    fclose(fx);
    fclose(fy);
}

void gradient_descent(Matrix* X, Matrix* Y, double alpha, int iterations, Matrix** W1, Matrix** b1, Matrix** W2, Matrix** b2) {
    int input_size = X->cols;
    //random values, optimize later
    int hidden_size = 10; 
    int output_size = 2; 
    init_params(W1, b1, W2, b2, input_size, hidden_size, output_size);

    for (int i = 0; i < iterations; i++) {
        Matrix *Z1 = NULL, *Z2 = NULL;
        forward_propagation(*W1, *b1, *W2, *b2, X, &Z1, &Z2);

        Matrix *dW1 = NULL, *db1 = NULL, *dW2 = NULL, *db2 = NULL;
        backward_propagation(Z1, NULL, Z2, NULL, *W1, *W2, X, Y, (double)Y->rows, &dW1, &db1, &dW2, &db2);
        

        update_params(*W1, *b1, *W2, *b2, dW1, db1, dW2, db2, alpha);

        if (i % 10 == 0) {
            printf("Iteration: %d\n", i);
            Matrix* predictions = get_predictions(Z2);
            double accuracy = get_accuracy(predictions, Y);
            printf("Accuracy: %f\n", accuracy);
            matrix_dealloc(predictions);
        }

        matrix_dealloc(Z1);
        matrix_dealloc(Z2);
        matrix_dealloc(dW1);
        matrix_dealloc(db1);
        matrix_dealloc(dW2);
        matrix_dealloc(db2);
    }
}

