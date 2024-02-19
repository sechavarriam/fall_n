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



class Matrix // Wrapper Around PETSc DenseMatrix
{
    private:
        Mat mat_;
    public:

        void print_content(){
            MatView(mat_, PETSC_VIEWER_STDOUT_SELF);
        };

        Matrix(PetscInt rows, PetscInt cols){
            MatCreateSeqDense(PETSC_COMM_SELF, rows, cols, nullptr, &mat_);
        };

        Matrix(PetscScalar* data, PetscInt rows, PetscInt cols){
            MatCreateSeqDense(PETSC_COMM_SELF, rows, cols, data, &mat_);
        };


        Matrix(std::ranges::contiguous_range auto data, PetscInt extent1, PetscInt extent2){
            MatCreateSeqDense(PETSC_COMM_SELF, extent1, extent2, std::to_address(data.begin()), &mat_);
        };
        

        //auto data(){
        //    PetscInt rows, cols;
        //    MatGetSize(mat_, &rows, &cols);
        //    PetscScalar* p_data;
        //    MatGetArray(mat_, &p_data);
        //    return std::span<PetscScalar>(p_data, rows*cols);
        //}

        //void print_content(){
        //    PetscInt rows, cols;
        //    MatGetSize(mat_, &rows, &cols);
        //    for(auto i:data()) printf("%f ", i);printf("\n");
        //};

        //Operators
        /*
        Matrix& operator+=(const Matrix& other){
            MatAXPY(mat_, 1.0, other.mat_, DIFFERENT_NONZERO_PATTERN);
            return *this;
        };

        Matrix operator-= (const Matrix& other){
            MatAXPY(mat_, -1.0, other.mat_, DIFFERENT_NONZERO_PATTERN);
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
        */
    ~Matrix(){MatDestroy(&mat_);};
};


// Blaze

// --------------------------------------

// if Eigen is used, the following aliases are defined.
//template<unsigned int N, typename ScalarType = double> 
//using SqMatrix = Eigen::Matrix<ScalarType, N, N> ;
//
//template<unsigned int N_rows, unsigned int N_cols, typename ScalarType = double> 
//using Matrix = Eigen::Matrix<ScalarType, N_rows, N_cols> ;



#endif