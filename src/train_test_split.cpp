#include<armadillo>
#include<iostream>
#include "utklearn.hpp"
using namespace std;

void utk::train_test_split(arma::mat& data, 
                    arma::mat& X_train, 
                    arma::mat& X_test, 
                    arma::mat& y_train, 
                    arma::mat& y_test,
                    float percent)
                    {

    cout<<"train_test_split is on process"<<endl;
    const int n_cols = data.n_cols;
    const int n_rows = data.n_rows;

    arma::mat X, y;
    X = data.submat(0, 0, n_rows-2, n_cols-1);
    y = data.submat(n_rows-1, 0, n_rows-1, n_cols-1);

    arma::uvec v = arma::linspace<arma::uvec>(0, n_cols - 1, n_cols);
    v = arma::shuffle(v);

    int n_train_point = n_cols * (percent / 100); 

    //iterating through the vector element
    for (int i = 0; i < n_cols; i++){
        int col_n = v(i);
        if(i < n_train_point){
            X_train = arma::join_horiz(X_train, X.col(col_n));
            y_train = arma::join_horiz(y_train, y.col(col_n));
        }else{
            X_test = arma::join_horiz(X_test, X.col(col_n));
            y_test = arma::join_horiz(y_test, y.col(col_n));
        }
    }
    
    //printing the information
    cout<<"dimension:: (rows, cols)"<<endl;
    cout<<"dimension of data:: ("<<data.n_rows<<" ,"<<data.n_cols<<")"<<endl;
    cout<<"dimension of X_train:: ("<<X_train.n_rows<<" ,"<<X_train.n_cols<<")"<<endl;
    cout<<"dimension of y_train:: ("<<y_train.n_rows<<" ,"<<y_train.n_cols<<")"<<endl;
    cout<<"dimension of X_test:: ("<<X_test.n_rows<<" ,"<<X_test.n_cols<<")"<<endl;
    cout<<"dimension of y_test:: ("<<y_test.n_rows<<" ,"<<y_test.n_cols<<")"<<endl;
}
