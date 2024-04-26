#include <stdio.h>
#include <stdlib.h>
#include "../basic_math_func/our_math.h"

// Basic Matrix Struct 
typedef struct{
	int rows;
	int cols;
	double **data;
} Matrix;

typedef struct {
    void **item;
    int capacity;
	int total;
} DynamArr;

DynamArr* array_alloc(int initial_size);

void array_dealloc(DynamArr* arr);

void array_resize(DynamArr* arr, int new_size);

void array_add(DynamArr* arr, void* item);

void* array_get(DynamArr* arr, int index);

void array_set(DynamArr* arr, int index, void* item);

Matrix* matrix_alloc(int rows, int cols); //DONE | NOT TESTED

void matrix_dealloc(Matrix* m); //DONE | NOT TESTED

void transpose(Matrix* a); //DONE | NOT TESTED

void GEMM(Matrix* a, Matrix* b, Matrix* c); // C = scalarA * AB + scalarB * C

Matrix* matrix_multi(Matrix* a, Matrix* b); //DONE | NOT TESTED

void matrix_add(Matrix* mat, Matrix* mat2);

void matrix_sub(Matrix* mat, Matrix* mat2);

void activationFunctionReLU(Matrix* mat); //NOT DONE | NOT TESTED (U LIED IT DOESNT EXIST)

void softmax(Matrix* mat); //DONE | NOT TESTED

void matrix_partial_dealloc(Matrix* m); //DONE | NOT TESTED

void activationFunctionSwish(Matrix* mat);

void matrix_copy(Matrix* a, Matrix* b);

void matrix_scale(Matrix* mat, double scalar_val); // for the 1/m * (?) in the back prop func, scalar multi

void matrix_element_multi(Matrix* mat, Matrix* mat2); //dot product apparently is not suitable here: https://stats.stackexchange.com/questions/533577/dot-product-vs-element-wise-multiplication

Matrix* matrix_sum_cols(Matrix* mat);

