#include "matrix.h" 

Matrix* matrix_alloc(int rows, int cols) {
    Matrix* m = malloc(sizeof(Matrix));
    if (m == NULL) {
        perror("Alloc failed for Matrix struct");
        return NULL;
    }

    m->rows = rows;
    m->cols = cols;
    m->data = malloc(rows * sizeof(double*));
    if (m->data == NULL) {
        perror("Alloc failed for Matrix data");
        free(m);
        return NULL;
    }

    for (int i = 0; i < rows; i++) {
        m->data[i] = malloc(cols * sizeof(double));
        if (m->data[i] == NULL) {
            for (int j = 0; j < i; j++) {
                free(m->data[j]);
            }
            free(m->data);
            free(m);
            perror("Alloc failed for Matrix rows");
            return NULL;
        }
    }

    return m;
}

void matrix_dealloc(Matrix* m) {
    if (m != NULL) {
        for (int i = 0; i < m->rows; i++) {
            free(m->data[i]); 
        }
        free(m->data); // Free the row pointers
        free(m); // Free the structure
    }
}

Matrix* matrix_multi(Matrix* a, Matrix* b){
	if (a->cols != b->rows) {
    fprintf(stderr, "Not a square Matrix");
    return NULL;
	}
	Matrix* result = matrix_alloc(a->rows, b->cols);
    if (!result) return NULL;
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
            result->data[i][j] = 0;
            for (int k = 0; k < a->cols; k++) {
                result->data[i][j] += a->data[i][k] * b->data[k][j];
            }
        }
    }
	return result;
}

void GEMM(Matrix* a, Matrix* b, Matrix* c){
    if (a->cols != b->rows || a->rows != c->rows || b->cols != c->cols) {
        fprintf(stderr, "Matrix is not Square for GEMM");
        return;
    }

    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
            c->data[i][j] = 0;  // Initialize C[i][j]
            for (int k = 0; k < a->cols; k++) {
                c->data[i][j] += a->data[i][k] * b->data[k][j];
            }
        }
    }
}


