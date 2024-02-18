#ifndef FALL_N_VECTOR_WRAPPER 
#define FALL_N_VECTOR_WRAPPER


#include "Matrix.hh"

#include <iostream>
#include <array>
#include <initializer_list>

#include <memory>
#include <ranges>


#include <petscsystypes.h>
#include <petscvec.h>


class Vector //Wrapper Around PETSc Vector
{
    using PETSc_Vector = Vec;
    private:
        PETSc_Vector vec_;
    public:

    void print_contents(){
        PetscScalar* data;
        PetscInt size;

        VecGetSize(vec_, &size);
        VecGetArray(vec_, &data);
        for (int i = 0; i < size; i++){
            std::cout << data[i] << std::endl;
        }
        VecRestoreArray(vec_, &data);
    }


    // Constructors
    Vector(std::initializer_list<PetscScalar>data){
        VecCreateSeq(PETSC_COMM_SELF, data.size(), &vec_);
        VecPlaceArray(vec_, data.begin());
        std::cout << "Initializer List" << std::endl;
    }

    Vector(std::ranges::contiguous_range auto const& data){
        VecCreateSeq(PETSC_COMM_SELF, data.size(), &vec_);
        VecPlaceArray(vec_, data.begin());
        std::cout << "range auto const&" << std::endl;
    }

    Vector(std::ranges::contiguous_range auto&& data){
        VecCreateSeq(PETSC_COMM_SELF, data.size(), &vec_);
        VecPlaceArray(vec_, std::to_address(data.begin()));
        std::cout << "range auto&&" << std::endl;
    }
        
};
// Vector defined in terms of the matrix wrapper.
//template<unsigned int N, typename ScalarType = double>
//using Vector = Matrix<N,1,ScalarType> ;



#endif