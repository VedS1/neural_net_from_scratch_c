#include <stdio.h>
#include <stdlib.h>
#include <../our_math.h>

// Basic Matrix Struct 
typedef struct{
	int rows;
	int cols;
	double **data;
} Matrix;

Matrix* matrix_alloc(int rows, int cols);

void matrix_dealloc(Matrix* m);

Matrix* GEMM(Matrix* a, Matrix* b);

Matrix* transpose(Matrix* a);

Matrix* matrix_multi(Matrix* a, Matrix* b);

Matrix* GEMM(Matrix* a, Matrix* b); // C = AB + C

void ReLU(Matrix* mat);

void softmax(Matrix* mat);

void matrix_partial_dealloc(Matrix* m);