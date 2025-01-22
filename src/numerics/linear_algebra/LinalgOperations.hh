#ifndef FALL_N_MATRIX_VECTOR_OPERATIONS
#define FALL_N_MATRIX_VECTOR_OPERATIONS


#include <concepts>
#include <ranges>

#include <petscvec.h>
#include <petscmat.h>

#include "Matrix.hh"
#include "Vector.hh"


namespace linalg{

    inline std::floating_point auto dot(const Vector& vec1, const Vector& vec2){
        PetscScalar result{0};
        VecTDot(vec1.vec_, vec2.vec_, &result);
        return result;
    }

    inline std::integral auto mat_vec_mult(const DeprecatedDenseMatrix& A, const Vector& x, Vector& y){
        //if (y.vec_ == nullptr){
        //    VecDuplicate(x.vec_, &y.vec_);
        //};
        return MatMult(A.mat_, x.vec_, y.vec_);
    }

    inline Vector mat_vec_mult(const DeprecatedDenseMatrix& A, const Vector& x){
        Vector y;
        MatCreateVecs(A.mat_, nullptr, &y.vec_);
        MatMult(A.mat_, x.vec_, y.vec_);
        return y;
    }

    //enum class ProductPolicy{
    //    AB   = MATPRODUCT_AB,
    //    AtB  = MATPRODUCT_AtB,
    //    ABt  = MATPRODUCT_ABt,
    //    PtAP = MATPRODUCT_PtAP,
    //    RARt = MATPRODUCT_RARt,
    //    ABC  = MATPRODUCT_ABC,
    //};

    inline DeprecatedDenseMatrix mat_mat_PtAP(const DeprecatedDenseMatrix& P, const DeprecatedDenseMatrix& A, const PetscScalar& scalar = 1.0){
        Mat C;
        MatPtAP(A.mat_,P.mat_, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &C);

        MatScale(C, scalar);
        return DeprecatedDenseMatrix{C};
    }



    //template<typename MultiplicationPolicy>
    


    

} //namespace linalg


#endif // FALL_N_MATRIX_VECTOR_OPERATIONS