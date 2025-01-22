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
class DeprecatedDenseMatrix;
namespace linalg{
    std::floating_point auto dot(const Vector& vec1, const Vector& vec2); 

    std::integral auto mat_vec_mult(const DeprecatedDenseMatrix& A, const Vector& x, Vector& y);
    Vector             mat_vec_mult(const DeprecatedDenseMatrix& A, const Vector& x);

} //namespace linalg

class DeprecatedDenseMatrix // Wrapper Around PETSc DenseMatrix
{
    using PETSc_DENSE_SEQ_Matrix = Mat;
    
    private:
    public:
    
    Mat mat_;
    bool assembly_mode_on{false};

    bool owns_data{true};

    // Accessors
    // data() returns a span of the data in the matrix.
    auto data(){
        PetscScalar* data;
        MatDenseGetArray(mat_, std::addressof(data));
        return data;
    };    

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

    friend std::integral auto linalg::mat_vec_mult(const DeprecatedDenseMatrix& A, const Vector& x, Vector& y);
    friend Vector             linalg::mat_vec_mult(const DeprecatedDenseMatrix& A, const Vector& x);

    void print_content(){MatView(mat_, PETSC_VIEWER_STDOUT_SELF);};
    
    //Operators
    DeprecatedDenseMatrix& operator+=(const DeprecatedDenseMatrix& other){MatAXPY(mat_, 1.0, other.mat_,SAME_NONZERO_PATTERN); return *this;};
    DeprecatedDenseMatrix& operator-=(const DeprecatedDenseMatrix& other){MatAXPY(mat_,-1.0, other.mat_,SAME_NONZERO_PATTERN); return *this;};


    DeprecatedDenseMatrix& operator*=(const PetscScalar& scalar){MatScale(mat_,     scalar); return *this;};
    DeprecatedDenseMatrix& operator/=(const PetscScalar& scalar){MatScale(mat_, 1.0/scalar); return *this;};

    friend DeprecatedDenseMatrix operator*(const PetscScalar& scalar, const DeprecatedDenseMatrix& mat){
        Mat B;
        MatDuplicate(mat.mat_, MAT_COPY_VALUES, &B);
        MatScale(B, scalar);
        return DeprecatedDenseMatrix(B);
    };

    DeprecatedDenseMatrix operator*(const PetscScalar& scalar){
        Mat B;
        MatDuplicate(mat_, MAT_COPY_VALUES, &B);
        MatScale(B, scalar);
        return DeprecatedDenseMatrix(B);
    };

    DeprecatedDenseMatrix operator*(const DeprecatedDenseMatrix& other){
        Mat C;
        MatMatMult(mat_, other.mat_, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &C);
        return DeprecatedDenseMatrix(C);
    };


    Vector operator*(const Vector& x){return linalg::mat_vec_mult(*this, x);};

    DeprecatedDenseMatrix operator/(const PetscScalar& scalar){
        Mat B;
        MatDuplicate(mat_, MAT_COPY_VALUES, &B);
        MatScale(B, 1.0/scalar);
        return DeprecatedDenseMatrix(B);
    };

    DeprecatedDenseMatrix operator+(const DeprecatedDenseMatrix& other){
        Mat B;
        MatDuplicate(mat_, MAT_COPY_VALUES, &B);
        MatAXPY(B, 1.0, other.mat_, SAME_NONZERO_PATTERN);  
        return DeprecatedDenseMatrix(B);
    };

    DeprecatedDenseMatrix operator-(const DeprecatedDenseMatrix& other){
        Mat B;
        MatDuplicate(mat_, MAT_COPY_VALUES, &B);
        MatAXPY(B, -1.0, other.mat_, SAME_NONZERO_PATTERN);
        return DeprecatedDenseMatrix(B);
    };

    //Constructors
    //DeprecatedDenseMatrix(Mat mat): mat_(std::move(mat)){};
    DeprecatedDenseMatrix(PETSc_DENSE_SEQ_Matrix mat): mat_(std::move(mat)){};

    DeprecatedDenseMatrix(PetscInt rows, PetscInt cols){
        MatCreateSeqDense(PETSC_COMM_SELF, rows, cols, PETSC_NULLPTR, &mat_);
    };
    
    DeprecatedDenseMatrix(PetscScalar* data, PetscInt rows, PetscInt cols){
        //owns_data = false;
        MatCreateSeqDense(PETSC_COMM_SELF, rows, cols, data, &mat_);
    };
    
    explicit DeprecatedDenseMatrix(std::ranges::contiguous_range auto const& data, PetscInt extent1, PetscInt extent2){
        //std::cout << "PETSc DeprecatedDenseMatrix Constructor - range const&\n";
        MatCreateSeqDense(PETSC_COMM_SELF, extent1, extent2, std::to_address(data.begin()), &mat_);
    };
    
    explicit DeprecatedDenseMatrix(std::ranges::contiguous_range auto&& data, PetscInt extent1, PetscInt extent2){
        //std::cout << "PETSc DeprecatedDenseMatrix Constructor - range&& \n";
        MatCreateSeqDense(PETSC_COMM_SELF, extent1, extent2, std::to_address(data.begin()), &mat_);
    };
    
    //Copy Constructor
    DeprecatedDenseMatrix(const DeprecatedDenseMatrix& other){
        //std::cout << "PETSc DeprecatedDenseMatrix Copy Constructor\n";
        MatDuplicate(other.mat_, MAT_COPY_VALUES, &mat_);
        };
    
    //Move Constructor
    DeprecatedDenseMatrix(DeprecatedDenseMatrix&& other){
        //std::cout << "PETSc DeprecatedDenseMatrix Move Constructor\n";
        other.mat_ = std::exchange(mat_, other.mat_);
    };
    
    //Copy Assignment
    DeprecatedDenseMatrix& operator=(const DeprecatedDenseMatrix& other){
        //std::cout << "PETSc DeprecatedDenseMatrix Copy Assignment\n";
        MatDuplicate(other.mat_, MAT_COPY_VALUES, &mat_);
        return *this;
    };
    
    //Move Assignment
    DeprecatedDenseMatrix& operator=(DeprecatedDenseMatrix&& other){
         //std::cout << "PETSc DeprecatedDenseMatrix Move Assignment\n";
        other.mat_ = std::exchange(mat_, other.mat_);
        
        //MatDuplicate(other.mat_, MAT_COPY_VALUES, &mat_);
        return *this;
    };

    DeprecatedDenseMatrix(){
        //std::cout << "PETSc DeprecatedDenseMatrix Default Constructor\n";
        MatCreateSeqDense(PETSC_COMM_SELF, 0, 0, PETSC_NULLPTR, &mat_);
        
    }

    ~DeprecatedDenseMatrix(){MatDestroy(&mat_);};
};



#endif