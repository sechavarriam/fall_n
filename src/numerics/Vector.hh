#ifndef FALL_N_VECTOR_WRAPPER 
#define FALL_N_VECTOR_WRAPPER


#include "Matrix.hh"

#include <concepts>
#include <iostream>
#include <array>
#include <initializer_list>
#include <span>

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

    auto data(){
        PetscInt size;
        PetscScalar* p_data;
        VecGetSize(vec_, &size);
        VecGetArray(vec_, &p_data);
        return std::span<PetscScalar>(p_data, size);
    }


    std::floating_point auto operator[](std::size_t i){
        PetscScalar* data;
        VecGetArray(vec_, &data);
        return data[i];
    };

    void print_contents(){
        for (auto i : data()){
            std::cout << i << std::endl;
        };
    };

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



#endif