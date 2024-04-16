
// Basic Matrix Struct is_sq_matrix corresponds to bool value checking if the matrix is a sqaure matrix
typedef struct{
	int rows;
	int cols;
	double *data;
	int is_sq_matrix;
} Matrix;

Matrix mat_multi(Matrix a, Matrix b);

Matrix mat_add(Matrix a, Matrix b);

void ReLU(Matrix* mat);

void softmax(Matrix* mat);
