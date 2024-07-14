#ifndef FALL_N_MATRIX_WRAPPER 
#define FALL_N_MATRIX_WRAPPER


// Possible matrix packages -------------
// Eigen
//#include <Eigen/Dense>
//#include <Eigen/Core>

#include <petscsys.h>
#include <petscvec.h>
#include <petscmat.h>

#include <ranges>
#include <span>

#include "Vector.hh"
class Matrix;
namespace linalg{
    std::floating_point auto dot(const Vector& vec1, const Vector& vec2); 

    std::integral auto mat_vec_mult(const Matrix& A, const Vector& x, Vector& y);
    Vector             mat_vec_mult(const Matrix& A, const Vector& x);

} //namespace linalg
class Matrix // Wrapper Around PETSc DenseMatrix
{
    using PETSc_DENSE_SEQ_Matrix = Mat;
    
    private:
    PETSc_DENSE_SEQ_Matrix mat_;
    
    public:

    bool owns_matrix{true};



    friend std::integral auto linalg::mat_vec_mult(const Matrix& A, const Vector& x, Vector& y);
    friend Vector             linalg::mat_vec_mult(const Matrix& A, const Vector& x);

    void print_content(){MatView(mat_, PETSC_VIEWER_STDOUT_SELF);};
    //Operators
    Matrix& operator+=(const Matrix& other){
        MatAXPY(mat_, 1.0, other.mat_, SAME_NONZERO_PATTERN);
        return *this;
    };
    Matrix operator-= (const Matrix& other){
        MatAXPY(mat_, -1.0, other.mat_, SAME_NONZERO_PATTERN);
        return *this;
    };
    Matrix operator*=(const PetscScalar& scalar){
        MatScale(mat_, scalar);
        return *this;
    };
    Matrix operator/=(const PetscScalar& scalar){
        MatScale(mat_, 1.0/scalar);
        return *this;
    };

    Matrix operator*(const PetscScalar& scalar){
        Mat B;
        MatDuplicate(mat_, MAT_COPY_VALUES, &B);
        MatScale(B, scalar);
        return Matrix(B);
    };

    Vector operator*(const Vector& x){return linalg::mat_vec_mult(*this, x);};

    Matrix operator/(const PetscScalar& scalar){
        Mat B;
        MatDuplicate(mat_, MAT_COPY_VALUES, &B);
        MatScale(B, 1.0/scalar);
        return Matrix(B);
    };

    Matrix operator+(const Matrix& other){
        Mat B;
        MatDuplicate(mat_, MAT_COPY_VALUES, &B);
        MatAXPY(B, 1.0, other.mat_, SAME_NONZERO_PATTERN);  
        return Matrix(B);
    };

    Matrix operator-(const Matrix& other){
        Mat B;
        MatDuplicate(mat_, MAT_COPY_VALUES, &B);
        MatAXPY(B, -1.0, other.mat_, SAME_NONZERO_PATTERN);
        return Matrix(B);
    };

    //Constructors
    Matrix(PETSc_DENSE_SEQ_Matrix mat): mat_(std::move(mat)){};

    Matrix(PetscInt rows, PetscInt cols){
        MatCreateSeqDense(PETSC_COMM_SELF, rows, cols, PETSC_NULLPTR, &mat_);
    };
    
    Matrix(PetscScalar* data, PetscInt rows, PetscInt cols){
        owns_matrix = false;
        MatCreateSeqDense(PETSC_COMM_SELF, rows, cols, data, &mat_);
    };
    
    Matrix(std::ranges::contiguous_range auto const& data, PetscInt extent1, PetscInt extent2){
        owns_matrix = false;
        MatCreateSeqDense(PETSC_COMM_SELF, extent1, extent2, std::to_address(data.begin()), &mat_);
    };
    
    Matrix(std::ranges::contiguous_range auto&& data, PetscInt extent1, PetscInt extent2){
        owns_matrix = false;
        MatCreateSeqDense(PETSC_COMM_SELF, extent1, extent2, std::to_address(data.begin()), &mat_);
    };
    
    //Copy Constructor
    Matrix(const Matrix& other){MatDuplicate(other.mat_, MAT_COPY_VALUES, &mat_);};
    
    //Move Constructor
    Matrix(Matrix&& other){
        mat_ = other.mat_;
        other.mat_ = nullptr;
    };
    
    //Copy Assignment
    Matrix& operator=(const Matrix& other){
        MatDuplicate(other.mat_, MAT_COPY_VALUES, &mat_);
        return *this;
    };
    
    //Move Assignment
    Matrix& operator=(Matrix&& other){
        mat_ = other.mat_;
        other.mat_ = nullptr;
        return *this;
    };

    ~Matrix(){MatDestroy(&mat_);};
};



#endif