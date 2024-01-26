#ifndef FALL_N_MATRIX_WRAPPER 
#define FALL_N_MATRIX_WRAPPER


// Possible matrix packages -------------
// Eigen
#include <Eigen/Dense>
#include <Eigen/Core>

// Blaze

// --------------------------------------

// if Eigen is used, the following aliases are defined.
template<unsigned int N, typename ScalarType = double> 
using SqMatrix = Eigen::Matrix<ScalarType, N, N> ;

template<unsigned int N_rows, unsigned int N_cols, typename ScalarType = double> 
using Matrix = Eigen::Matrix<ScalarType, N_rows, N_cols> ;



#endif