#include "matrix.h"

// Allocate a matrix with given rows and columns
Matrix* matrix_alloc(int rows, int cols) {
    Matrix* m = malloc(sizeof(Matrix));
    if (!m) {
        perror("Alloc failed for Matrix struct");
        return NULL;
    }

    m->rows = rows;
    m->cols = cols;
    m->data = malloc(rows * cols * sizeof(double));
    if (!m->data) {
        perror("Alloc failed for Matrix data");
        free(m);
        return NULL;
    }

    return m;
}

// Deallocate a matrix
void matrix_dealloc(Matrix* m) {
    if (m) {
        free(m->data);
        free(m);
    }
}

// Get element from matrix
double get_element(Matrix* matrix, int row, int col) {
    return matrix->data[row * matrix->cols + col];
}

// Set element in matrix
void set_element(Matrix* matrix, int row, int col, double value) {
    matrix->data[row * matrix->cols + col] = value;
}

// Matrix multiplication
Matrix* matrix_multi(Matrix* a, Matrix* b) {
    if (a->cols != b->rows) {
        fprintf(stderr, "Matrix dimension mismatch for multiplication\n");
        return NULL;
    }

    Matrix* result = matrix_alloc(a->rows, b->cols);
    if (!result) return NULL;

    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
            result->data[i * result->cols + j] = 0;
            for (int k = 0; k < a->cols; k++) {
                result->data[i * result->cols + j] += a->data[i * a->cols + k] * b->data[k * b->cols + j];
            }
        }
    }
    return result;
}

// Matrix addition
void matrix_add(Matrix* mat, Matrix* mat2) {
    if (mat2->cols != 1 || mat->rows != mat2->rows) {
        fprintf(stderr, "Matrix 2 must be a 1x(?) vector with the same num. of rows as Matrix1 (add)\n");
        return;
    }
    for (int i = 0; i < mat->rows; i++) {
        for (int j = 0; j < mat->cols; j++) {
            mat->data[i * mat->cols + j] += mat2->data[i];
        }
    }
}

// Matrix subtraction
void matrix_sub(Matrix* mat, Matrix* mat2) {
    if (mat2->cols != 1 || mat->rows != mat2->rows) {
        fprintf(stderr, "Matrix 2 must be a 1x(?) vector with the same num. of rows as Matrix1 (sub)\n");
        return;
    }
    for (int i = 0; i < mat->rows; i++) {
        for (int j = 0; j < mat->cols; j++) {
            mat->data[i * mat->cols + j] -= mat2->data[i];
        }
    }
}

// General Matrix Multiply (GEMM)
void GEMM(Matrix* a, Matrix* b, Matrix* c) {
    if (a->cols != b->rows || a->rows != c->rows || b->cols != c->cols) {
        fprintf(stderr, "Matrix dimension mismatch for GEMM\n");
        return;
    }

    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
            c->data[i * c->cols + j] = 0;  // Initialize C[i][j]
            for (int k = 0; k < a->cols; k++) {
                c->data[i * c->cols + j] += a->data[i * a->cols + k] * b->data[k * b->cols + j];
            }
        }
    }
}

// Transpose matrix
void transpose(Matrix* a) {
    Matrix* newA = matrix_alloc(a->cols, a->rows);
    if (!newA) {
        fprintf(stderr, "Transpose allocation failed!");
        return;
    }
    for (int x = 0; x < a->rows; x++) {
        for (int y = 0; y < a->cols; y++) {
            newA->data[y * newA->cols + x] = a->data[x * a->cols + y];
        }
    }

    free(a->data);
    a->data = newA->data;
    a->rows = newA->cols;
    a->cols = newA->rows;
    free(newA);
}

// Apply softmax function to each row of the matrix
void softmax(Matrix* mat) {
    for (int x = 0; x < mat->rows; x++) {
        double max_value = mat->data[x * mat->cols];
        for (int y = 1; y < mat->cols; y++) {
            if (mat->data[x * mat->cols + y] > max_value) {
                max_value = mat->data[x * mat->cols + y];
            }
        }
        double row_sum = 0;
        for (int y = 0; y < mat->cols; y++) {
            mat->data[x * mat->cols + y] = exp_approx(mat->data[x * mat->cols + y] - max_value);
            row_sum += mat->data[x * mat->cols + y];
        }
        for (int y = 0; y < mat->cols; y++) {
            mat->data[x * mat->cols + y] /= row_sum;
        }
    }
}

// Apply Swish activation function to each element of the matrix
void activationFunctionSwish(Matrix* mat) {
    for (int x = 0; x < mat->rows; x++) {
        for (int y = 0; y < mat->cols; y++) {
            int index = x * mat->cols + y;
            mat->data[index] *= 1.0 / (1.0 + exp_approx(-mat->data[index]));
        }
    }
}

// Apply ReLU activation function to each element of the matrix
void activationFunctionReLU(Matrix* mat) {
    if (!mat) {
        fprintf(stderr, "ReLU error: No input matrix\n");
        return;
    }
    for (int i = 0; i < mat->rows; i++) {
        for (int j = 0; j < mat->cols; j++) {
            int index = i * mat->cols + j;
            mat->data[index] = mat->data[index] > 0 ? mat->data[index] : 0.0;
        }
    }
}

// Copy matrix data from src to dst
void matrix_copy(Matrix* src, Matrix* dst) {
    if (dst == NULL) {
        dst = matrix_alloc(src->rows, src->cols);
    }
    if ((src->rows == dst->rows) && (src->cols == dst->cols)) {
        for (int i = 0; i < src->rows * src->cols; i++) {
            dst->data[i] = src->data[i];
        }
    }
}

// Scale matrix by a scalar value
void matrix_scale(Matrix* mat, double scalar_val) {
    if (!mat) {
        fprintf(stderr, "Input matrix is NULL (scale)");
        return;
    }
    for (int i = 0; i < mat->rows * mat->cols; i++) {
        mat->data[i] *= scalar_val;
    }
}

// Element-wise multiplication of two matrices
void matrix_element_multi(Matrix* mat, Matrix* mat2) {
    if (!mat || !mat2) {
        fprintf(stderr, "Matrix 1 and Matrix 2 are NULL (element multiply)");
        return;
    }
    if (mat->rows != mat2->rows || mat->cols != mat2->cols) {
        fprintf(stderr, "Dimension mismatch (element multiply)\n");
        return;
    }
    for (int i = 0; i < mat->rows * mat->cols; i++) {
        mat->data[i] *= mat2->data[i];
    }
}

// Sum columns of a matrix
Matrix* matrix_sum_cols(Matrix* mat) {
    if (!mat) {
        fprintf(stderr, "Matrix is NULL (sum cols)\n");
        return NULL;
    }

    Matrix* sums = matrix_alloc(1, mat->cols);
    if (!sums) {
        fprintf(stderr, "Failed to allocate memory (sum cols)\n");
        return NULL;
    }

    for (int j = 0; j < mat->cols; j++) {
        sums->data[j] = 0;
        for (int i = 0; i < mat->rows; i++) {
            sums->data[j] += mat->data[i * mat->cols + j];
        }
    }

    return sums;
}

