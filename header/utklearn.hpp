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

    // void absolute_error()
}