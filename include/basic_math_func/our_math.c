#include "our_math.h"
 
// Simple function to compute x^y (or something like that)

double power(double base, int exp){
	double result = 1;
	if (exp == 0) 
		return 1;
	for (int i=0; i < abs(exp); ++i) {
		result *= base; 
	}
	if (exp < 0) { //negative exp
		return 1/result;
	}
	return result;
}

// Exponential function using a series expansion
double exp_approx(double x) {
    double sum = 1.0;  // e^0 is 1
    double term = 1.0;
    for (int n = 1; n < exp_approx_series_length; ++n) {  // Sum the first 20 terms of the series
        term *= x / n;
        sum += term;
    }
    return sum;
}

// Function to approximate ln using Newton-Raphson (simple numerical methods)
double ln_approx(double x) {
    if (x <= 0) {   
        return -1;  // undef for <= 0
    }
    double y = 0;
    for (int i = 0; i < 10; ++i) {  // 10 iter
        y = y - 1 + x / exp_approx(y);
    }
    return y;
}

// Works based on the change of base formula: log_10(x) = ln(x)/ln(10) 
double log10_approx(double x){
	double ln10 = ln_approx(10.0);  // Precompute ln(10)
    return ln_approx(x) / ln10;
}

double sqrt_approx(double x) {
    if (x < 0) {
        printf("Error: Negative input\n");
        return -1; // Return an error for negative inputs
    }
    if (x == 0 || x == 1) {
        return x; 
    }

    double guess = x / 2.0; // Initial guess
    double epsilon = 0.000001; // Define the accuracy 

    // Newton-Raphson Method for approximating a sqrt 
	while (guess - (x / guess) > epsilon || guess - (x / guess) < -epsilon) {
        guess = (guess + x / guess) / 2.0;
    }

    return guess;
}

void swap(double* a, double* b) {
    double placeholder = *a;
    *a = *b;
    *b = placeholder;
}


double sigmoid(double value) {

    return 1.0 / (1.0 + exp_approx(-value));
 
}
double relu(double x) {
    return x > 0 ? x : 0;
}

double relu_derivative(double x) {
    return x > 0 ? 1 : 0;
}

double mean_squared_error(double* y_true, double* y_pred, int length) {
    double sum = 0.0;
    for (int i = 0; i < length; i++) {
        double diff = y_true[i] - y_pred[i];
        sum += diff * diff;
    }
    return sum / length;
}

double cross_entropy_loss(double* y_true, double* y_pred, int length) {
    double sum = 0.0;
    for (int i = 0; i < length; i++) {
        sum += -y_true[i] * log(y_pred[i]) - (1 - y_true[i]) * log(1 - y_pred[i]);
    }
    return sum / length;
}
