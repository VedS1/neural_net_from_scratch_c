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

void matrix_partial_dealloc(Matrix* m) {
        if (m != NULL) {
        for (int i = 0; i < m->rows; i++) {
            free(m->data[i]); 
        }
        free(m->data); // Free the row pointers
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

void transpose(Matrix* a) {  
    
    Matrix* newA = matrix_alloc(a->cols, a->rows);
    if (!newA) fprintf(stderr, "Transpose allocation failed!");
    for (int x = 0; x < a->rows; x++) {
        for (int y = 0; y < a->cols; y++) {
            newA->data[y][x] = a->data[x][y];
        }
    }

    matrix_partial_dealloc(a);
    
    a->rows = newA->cols;
    a->cols = newA->rows;
    a->data = newA->data;

    free(newA);

}
