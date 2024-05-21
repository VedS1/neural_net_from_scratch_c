#ifndef MATRIX_H
#define MATRIX_H

#include <stdio.h>
#include <stdlib.h>
#include "../basic_math_func/our_math.h"


typedef struct {
    double *data;
    int rows;
    int cols;
} Matrix;

// Function prototypes
Matrix* matrix_alloc(int rows, int cols);
void matrix_dealloc(Matrix* m);
double get_element(Matrix* matrix, int row, int col);
void set_element(Matrix* matrix, int row, int col, double value);
Matrix* matrix_multi(Matrix* a, Matrix* b);
void matrix_add(Matrix* mat, Matrix* mat2);
void matrix_sub(Matrix* mat, Matrix* mat2);
void GEMM(Matrix* a, Matrix* b, Matrix* c);
void transpose(Matrix* a);
void softmax(Matrix* mat);
void activationFunctionSwish(Matrix* mat);
void activationFunctionReLU(Matrix* mat);
void matrix_copy(Matrix* src, Matrix* dst);
void matrix_scale(Matrix* mat, double scalar_val);
void matrix_element_multi(Matrix* mat, Matrix* mat2);
Matrix* matrix_sum_cols(Matrix* mat);

#endif // MATRIX_H

