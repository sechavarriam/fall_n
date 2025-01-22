#ifndef FALL_N_VECTOR_WRAPPER 
#define FALL_N_VECTOR_WRAPPER

#include <concepts>
#include <cstdio>
#include <iostream>
#include <array>
#include <initializer_list>
#include <span>

#include <memory>
#include <ranges>

#include <coroutine> 

#include <petscsystypes.h>
#include <petscvec.h>

//#include "Operations.hh"
//#include "DeprecatedDenseMatrix.hh"


class Vector;
class DeprecatedDenseMatrix;
namespace linalg{
    std::floating_point auto dot(const Vector& vec1, const Vector& vec2); 

    std::integral auto mat_vec_mult(const DeprecatedDenseMatrix& A, const Vector& x, Vector& y);
    Vector             mat_vec_mult(const DeprecatedDenseMatrix& A, const Vector& x);
} //namespace linalg


class Vector //Wrapper Around PETSc Seq Vector
{
    using PETSc_Vector = Vec;
    
    public:

        PETSc_Vector vec_;
    

    bool owns_data{true};

    // Accessors
    std::floating_point auto& operator[](std::size_t i){
        PetscScalar* data;
        VecGetArray(vec_, &data);
        return data[i];
    };

    std::span<PetscScalar> data(){
        PetscInt size;
        PetscScalar* p_data;
        VecGetSize(vec_, &size);
        VecGetArray(vec_, &p_data);
        //co_yield std::span<PetscScalar>(p_data, size);
        return std::span<PetscScalar>(p_data, size);
    }

    void print_content(){for(auto i:data()) printf("%f ", i);printf("\n");};

    friend std::floating_point auto linalg::dot(const Vector& vec1, const Vector& vec2);
    friend std::integral       auto linalg::mat_vec_mult(const DeprecatedDenseMatrix& A, const Vector& x, Vector& y);
    friend Vector                   linalg::mat_vec_mult(const DeprecatedDenseMatrix& A, const Vector& x);
    
    //Operators
    Vector& operator+=(const Vector& other)  {VecAXPY(vec_, 1.0, other.vec_);return *this;};
    Vector& operator-=(const Vector& other)  {VecAXPY(vec_,-1.0, other.vec_);return *this;};
    Vector& operator*=(const PetscScalar& scalar){VecScale(vec_,     scalar);return *this;};
    Vector& operator/=(const PetscScalar& scalar){VecScale(vec_, 1.0/scalar);return *this;};

    friend Vector operator*(const PetscScalar& scalar, const Vector& vec){
        Vector result;
        VecDuplicate(vec.vec_, &result.vec_);
        VecCopy(vec.vec_, result.vec_);
        VecScale(result.vec_, scalar);
        return result;
    };

    Vector operator*(const PetscScalar& scalar){
        Vector result;
        VecDuplicate(vec_, &result.vec_);
        VecCopy(vec_, result.vec_);
        VecScale(result.vec_, scalar);
        return result;
    };

    Vector operator+(const Vector& other){
        Vector result;
        VecDuplicate(vec_, &result.vec_);
        VecCopy(vec_, result.vec_);
        VecAXPY(result.vec_, 1.0, other.vec_);
        return result;
    };

    Vector operator-(const Vector& other){
        Vector result;
        VecDuplicate(vec_, &result.vec_);
        VecCopy(vec_, result.vec_);
        VecAXPY(result.vec_, -1.0, other.vec_);
        return result;
    };

    // Constructors
    Vector(){
        //std::cout << "PETSc Vector Default Constructor\n";
        VecCreateSeq(PETSC_COMM_SELF, 0, &vec_);
    };

    explicit Vector(std::ranges::contiguous_range auto const& data) {
        //std::cout << "PETSc Vector Constructor - range const&\n";
        //owns_data = false;
        VecCreateSeq(PETSC_COMM_SELF, data.size(), &vec_);
        VecPlaceArray(vec_, data.begin());
    };

    explicit Vector(std::ranges::contiguous_range auto&& data){
        //std::cout << "PETSc Vector Constructor - range&& \n";
        //owns_data = false;
        VecCreateSeq(PETSC_COMM_SELF, data.size(), &vec_);
        VecPlaceArray(vec_, std::to_address(data.begin()));
    };

    // Copy and Move Constructors and Assignment Operators
    Vector(const Vector& other){
        //std::cout << "PETSc Vector Copy Constructor\n";
        VecDuplicate(other.vec_, &vec_);
        VecCopy(other.vec_, vec_);
    };

    Vector(Vector&& other){
        //std::cout << "PETSc Vector Move Constructor\n";
        VecDuplicate(other.vec_, &vec_);
        VecCopy(other.vec_, vec_);
        //vec_ = other.vec_;
        //other.vec_ = nullptr;
    };

    //Vector& operator=(Vector other) = delete;
    Vector& operator=(const Vector& other)
    {
        //std::cout << "PETSc Vector Copy Assignment\n";
        VecDuplicate(other.vec_, &vec_);
        VecCopy(other.vec_, vec_);
        return *this;
    };

    Vector& operator=(const volatile Vector&& other){
        //std::cout << "PETSc Vector Move Assignment\n";
        VecDuplicate(other.vec_, &vec_);
        VecCopy(other.vec_, vec_);
        return *this;
    };

    // Destructor
    ~Vector(){VecDestroy(&vec_);};
};



#endif