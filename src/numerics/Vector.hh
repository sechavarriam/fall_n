#ifndef FALL_N_VECTOR_WRAPPER 
#define FALL_N_VECTOR_WRAPPER


#include "Matrix.hh"

// Vector defined in terms of the matrix wrapper.
template<unsigned int N, typename ScalarType = double>
using Vector = Matrix<N,1,ScalarType> ;



#endif