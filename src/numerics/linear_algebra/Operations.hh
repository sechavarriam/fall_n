#ifndef FALL_N_MATRIX_VECTOR_OPERATIONS
#define FALL_N_MATRIX_VECTOR_OPERATIONS


#include <concepts>
#include <ranges>

#include <petscvec.h>
#include <petscmat.h>

#include "Matrix.hh"
#include "Vector.hh"


namespace linalg{
    std::floating_point auto dot(const Vector& vec1, const Vector& vec2){
        PetscScalar result;
        VecDot(vec1.vec_, vec2.vec_, &result);
        return result;
    }  
} //namespace linalg


#endif // FALL_N_MATRIX_VECTOR_OPERATIONS