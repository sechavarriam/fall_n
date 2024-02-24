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
//#include "Matrix.hh"


class Vector;
class Matrix;
namespace linalg{
    std::floating_point auto dot(const Vector& vec1, const Vector& vec2); 

    std::integral auto mat_vec_mult(const Matrix& A, const Vector& x, Vector& y);
    Vector             mat_vec_mult(const Matrix& A, const Vector& x);
} //namespace linalg


class Vector //Wrapper Around PETSc Seq Vector
{
    //using namespace linalg;
    using PETSc_Vector = Vec;
    private:
        PETSc_Vector vec_;
    public:

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
    friend std::integral       auto linalg::mat_vec_mult(const Matrix& A, const Vector& x, Vector& y);
    friend Vector                   linalg::mat_vec_mult(const Matrix& A, const Vector& x);
    
    //Operators
    Vector& operator+=(const Vector& other){//Chek sizes
        VecAXPY(vec_, 1.0, other.vec_);
        return *this;
    };

    Vector operator-= (const Vector& other){
        VecAXPY(vec_, -1.0, other.vec_);
        return *this;
    };

    Vector operator*=(const PetscScalar& scalar){
        VecScale(vec_, scalar);
        return *this;
        
    };

    Vector operator/=(const PetscScalar& scalar){
        VecScale(vec_, 1.0/scalar);
        return *this;
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

    // Accessors
    std::floating_point auto operator[](std::size_t i){
        PetscScalar* data;
        VecGetArray(vec_, &data);
        return data[i];
    };

    
    // Constructors
    Vector(){
        VecCreateSeq(PETSC_COMM_SELF, 0, &vec_);
        std::cout << "Default" << std::endl;
    };

    //Vector(std::initializer_list<PetscScalar>data){ //Does weird things
    //    VecCreateSeq(PETSC_COMM_SELF, data.size(), &vec_);
    //    VecPlaceArray(vec_, data.begin());
    //    std::cout << "Initializer List" << std::endl;
    //};

    Vector(std::ranges::contiguous_range auto const& data){
        VecCreateSeq(PETSC_COMM_SELF, data.size(), &vec_);
        VecPlaceArray(vec_, data.begin());
        std::cout << "range auto const&" << std::endl;
    };

    Vector(std::ranges::contiguous_range auto&& data){
        VecCreateSeq(PETSC_COMM_SELF, data.size(), &vec_);
        VecPlaceArray(vec_, std::to_address(data.begin()));
        std::cout << "range auto&&" << std::endl;
    };

    // Copy and Move Constructors and Assignment Operators
    Vector(const Vector& other){
        VecDuplicate(other.vec_, &vec_);
        VecCopy(other.vec_, vec_);
    };

    Vector(Vector&& other){
        vec_ = other.vec_;
        other.vec_ = nullptr;
    };

    Vector& operator=(const Vector& other){
        VecDuplicate(other.vec_, &vec_);
        VecCopy(other.vec_, vec_);
        return *this;
    };

    Vector& operator=(Vector&& other){
        vec_ = other.vec_;
        other.vec_ = nullptr;
        return *this;
    };

    // Destructor
    ~Vector(){VecDestroy(&vec_);};
};



#endif