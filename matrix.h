typdef Struct{
	int rows;
	int cols;
	double* data;
} Matrix;

Matrix mat_multi(Matrix a, Matrix b);

Matrix mat_add(Matrix a, Matrix b);

void ReLU(Matrix* mat);

void softmax(Matrix* mat);
