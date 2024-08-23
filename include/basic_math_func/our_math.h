#include <stdio.h>
#include <stdlib.h>
#define exp_approx_series_length 20//idk am i supposed to do smthing here???? not rlly sure lmfao

double power(double base, int exp); //DONE | NOT TESTED

double exp_approx(double x); //DONE | NOT TESTED

double ln_approx(double x); //DONE | NOT TESTED

double log10_approx(double x); //DONE | NOT TESTED

double sqrt_approx(double x); //DONE | NOT TESTED

void swap(double* a, double* b); //DONE | NOT TESTED

double sigmoid(double value);

double exp_approx(double x); // Exponential function using a series expansion

double ln_approx(double x); // Function to approximate ln using Newton-Raphson

double log10_approx(double x); // Works based on the change of base formula: log_10(x) = ln(x)/ln(10)

double sqrt_approx(double x); // Approximate square root using Newton-Raphson method
double relu(double x);

double relu_derivative(double x);

double mean_squared_error(double* y_true, double* y_pred, int length);

double cross_entropy_loss(double* y_true, double* y_pred, int length);
