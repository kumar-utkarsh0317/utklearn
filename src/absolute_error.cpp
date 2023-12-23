#include<armadillo>
#include<iostream>
#include "utklearn.hpp"
using namespace std;

double utk::absolute_error(arma::rowvec predictions, arma::rowvec true_value)
{
    if(predictions.n_cols != true_value.n_cols)
    {
        cout<<"predictions and true_values have different elements\n\n";
        return -1;
    }

    double total_sum = 0;
    for(int i = 0; i < predictions.n_cols; i++)
    {
        total_sum += std::abs(predictions(i) - true_value(i));
    }

    return total_sum / predictions.n_cols;
}
