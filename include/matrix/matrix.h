#include <stdio.h>
#include <stdlib.h>
#include <../our_math.h>

// Basic Matrix Struct 
typedef struct{
	int rows;
	int cols;
	double **data;
} Matrix;

Matrix* matrix_alloc(int rows, int cols); //DONE | NOT TESTED

void matrix_dealloc(Matrix* m); //DONE | NOT TESTED

Matrix* GEMM(Matrix* a, Matrix* b); //NOT DONE | NOT TESTED

Matrix* transpose(Matrix* a); //DONE | NOT TESTED

Matrix* matrix_multi(Matrix* a, Matrix* b); //DONE | NOT TESTED

Matrix* GEMM(Matrix* a, Matrix* b); //NOT DONE | NOT TESTED

void ReLU(Matrix* mat); //NOT DONE | NOT TESTED

void softmax(Matrix* mat); //NOT DONE | NOT TESTED

void matrix_partial_dealloc(Matrix* m); //DONE | NOT TESTED