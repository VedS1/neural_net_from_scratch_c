#include <stdio.h>
#include <stdlib.h>


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

Matrix* matrix_alloc(int rows, int cols);

void matrix_dealloc(Matrix* m);

Matrix* matrix_multi(Matrix* a, Matrix* b);

void GEMM(Matrix* a, Matrix* b, Matrix* c); // C = scalarA * AB + scalarB * C

void ReLU(Matrix* mat);

void softmax(Matrix* mat);
