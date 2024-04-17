#include <stdio.h>
#include <stdlib.h>
#include <../our_math.h>

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

Matrix* GEMM(Matrix* a, Matrix* b); //NOT DONE | NOT TESTED

Matrix* transpose(Matrix* a); //DONE | NOT TESTED

void GEMM(Matrix* a, Matrix* b, Matrix* c); // C = scalarA * AB + scalarB * C

Matrix* matrix_multi(Matrix* a, Matrix* b); //DONE | NOT TESTED

void activationFunctionReLU(Matrix* mat); //NOT DONE | NOT TESTED

void softmax(Matrix* mat); //DONE | NOT TESTED

void matrix_partial_dealloc(Matrix* m); //DONE | NOT TESTED

void activationFunctionSwish(Matrix* mat);
