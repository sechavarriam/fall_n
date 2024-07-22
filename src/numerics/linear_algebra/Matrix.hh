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
    public:
    
    Mat mat_;
    bool assembly_mode_on{false};

    bool owns_data{true};

    //Wrapper for inserting values.

    void assembly_begin(MatAssemblyType assembly_mode= MAT_FINAL_ASSEMBLY){
        PetscCallVoid(MatAssemblyBegin(mat_, assembly_mode));
        assembly_mode_on = true;
    };

    void assembly_end(MatAssemblyType assembly_mode= MAT_FINAL_ASSEMBLY){
        PetscCallVoid(MatAssemblyEnd(mat_, assembly_mode));
        assembly_mode_on = false;
    };


    void insert_values(PetscInt row, PetscInt col, PetscScalar value, InsertMode insert_mode = INSERT_VALUES){
        PetscCallVoid(MatSetValue     (mat_, row, col, value, insert_mode));
    };

    friend std::integral auto linalg::mat_vec_mult(const Matrix& A, const Vector& x, Vector& y);
    friend Vector             linalg::mat_vec_mult(const Matrix& A, const Vector& x);

    void print_content(){MatView(mat_, PETSC_VIEWER_STDOUT_SELF);};
    
    //Operators
    Matrix& operator+=(const Matrix& other){MatAXPY(mat_, 1.0, other.mat_,SAME_NONZERO_PATTERN); return *this;};
    Matrix& operator-=(const Matrix& other){MatAXPY(mat_,-1.0, other.mat_,SAME_NONZERO_PATTERN);return *this;};
    Matrix& operator*=(const PetscScalar& scalar){MatScale(mat_,     scalar); return *this;};
    Matrix& operator/=(const PetscScalar& scalar){MatScale(mat_, 1.0/scalar); return *this;};

    friend Matrix operator*(const PetscScalar& scalar, const Matrix& mat){
        Mat B;
        MatDuplicate(mat.mat_, MAT_COPY_VALUES, &B);
        MatScale(B, scalar);
        return Matrix(B);
    };

    Matrix operator*(const PetscScalar& scalar){
        Mat B;
        MatDuplicate(mat_, MAT_COPY_VALUES, &B);
        MatScale(B, scalar);
        return Matrix(B);
    };

    Matrix operator*(const Matrix& other){
        Mat C;
        MatMatMult(mat_, other.mat_, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &C);
        return Matrix(C);
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
    //Matrix(Mat mat): mat_(std::move(mat)){};
    Matrix(PETSc_DENSE_SEQ_Matrix mat): mat_(std::move(mat)){};

    Matrix(PetscInt rows, PetscInt cols){
        MatCreateSeqDense(PETSC_COMM_SELF, rows, cols, PETSC_NULLPTR, &mat_);
    };
    
    Matrix(PetscScalar* data, PetscInt rows, PetscInt cols){
        //owns_data = false;
        MatCreateSeqDense(PETSC_COMM_SELF, rows, cols, data, &mat_);
    };
    
    explicit Matrix(std::ranges::contiguous_range auto const& data, PetscInt extent1, PetscInt extent2){
        //std::cout << "PETSc Matrix Constructor - range const&\n";
        MatCreateSeqDense(PETSC_COMM_SELF, extent1, extent2, std::to_address(data.begin()), &mat_);
    };
    
    explicit Matrix(std::ranges::contiguous_range auto&& data, PetscInt extent1, PetscInt extent2){
        //std::cout << "PETSc Matrix Constructor - range&& \n";
        MatCreateSeqDense(PETSC_COMM_SELF, extent1, extent2, std::to_address(data.begin()), &mat_);
    };
    
    //Copy Constructor
    Matrix(const Matrix& other){MatDuplicate(other.mat_, MAT_COPY_VALUES, &mat_);};
    
    //Move Constructor
    Matrix(Matrix&& other){
        //std::cout << "PETSc Matrix Move Constructor\n";
        MatDuplicate(other.mat_, MAT_COPY_VALUES, &mat_);
    };
    
    //Copy Assignment
    Matrix& operator=(const Matrix& other){
        //std::cout << "PETSc Matrix Copy Assignment\n";
        MatDuplicate(other.mat_, MAT_COPY_VALUES, &mat_);
        return *this;
    };
    
    //Move Assignment
    Matrix& operator=(Matrix&& other){
        //std::cout << "PETSc Matrix Move Assignment\n";
        MatDuplicate(other.mat_, MAT_COPY_VALUES, &mat_);
        return *this;
    };

    Matrix(){
        //std::cout << "PETSc Matrix Default Constructor\n";
        MatCreateSeqDense(PETSC_COMM_SELF, 0, 0, PETSC_NULLPTR, &mat_);
        
    }

    ~Matrix(){MatDestroy(&mat_);};
};



#endif