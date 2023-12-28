#pragma once
#include <armadillo>

namespace utk{
    void train_test_split(arma::mat& data,
                        arma::mat& X_train, 
                        arma::mat& X_test, 
                        arma::mat& y_train, 
                        arma::mat& y_test, 
                        float percent
                        );

    void normalize_features(arma::mat& features);

    double mean_absolute_error(arma::rowvec predictions, arma::rowvec true_value);

    double mean_squared_error(arma::rowvec predictions, arma::rowvec true_value);
}