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

class DeprecatedSequentialVector;
class DeprecatedDenseMatrix;

namespace linalg
{
    std::floating_point auto dot(const DeprecatedSequentialVector &vec1, const DeprecatedSequentialVector &vec2);

    std::integral auto mat_vec_mult(const DeprecatedDenseMatrix &A, const DeprecatedSequentialVector &x, DeprecatedSequentialVector &y);
    DeprecatedSequentialVector mat_vec_mult(const DeprecatedDenseMatrix &A, const DeprecatedSequentialVector &x);
} // namespace linalg

class DeprecatedSequentialVector // Wrapper Around PETSc Seq DeprecatedSequentialVector
{
    using PETSc_Vector = Vec;

public:
    PETSc_Vector vec_;

    bool owns_data{true};

    // Accessors
    std::floating_point auto &operator[](std::size_t i)
    {
        PetscScalar *data;
        VecGetArray(vec_, &data);
        return data[i];
    };

    std::span<PetscScalar> data()
    {
        PetscInt size;
        PetscScalar *p_data;
        VecGetSize(vec_, &size);
        VecGetArray(vec_, &p_data);
        // co_yield std::span<PetscScalar>(p_data, size);
        return std::span<PetscScalar>(p_data, size);
    }

    void print_content()
    {
        for (auto i : data())
            printf("%f ", i);
        printf("\n");
    };

    friend std::floating_point auto linalg::dot(const DeprecatedSequentialVector &vec1, const DeprecatedSequentialVector &vec2);
    friend std::integral auto linalg::mat_vec_mult(const DeprecatedDenseMatrix &A, const DeprecatedSequentialVector &x, DeprecatedSequentialVector &y);
    friend DeprecatedSequentialVector linalg::mat_vec_mult(const DeprecatedDenseMatrix &A, const DeprecatedSequentialVector &x);

    // Operators
    DeprecatedSequentialVector &operator+=(const DeprecatedSequentialVector &other)
    {
        VecAXPY(vec_, 1.0, other.vec_);
        return *this;
    };
    DeprecatedSequentialVector &operator-=(const DeprecatedSequentialVector &other)
    {
        VecAXPY(vec_, -1.0, other.vec_);
        return *this;
    };
    DeprecatedSequentialVector &operator*=(const PetscScalar &scalar)
    {
        VecScale(vec_, scalar);
        return *this;
    };
    DeprecatedSequentialVector &operator/=(const PetscScalar &scalar)
    {
        VecScale(vec_, 1.0 / scalar);
        return *this;
    };

    friend DeprecatedSequentialVector operator*(const PetscScalar &scalar, const DeprecatedSequentialVector &vec)
    {
        DeprecatedSequentialVector result;
        VecDuplicate(vec.vec_, &result.vec_);
        VecCopy(vec.vec_, result.vec_);
        VecScale(result.vec_, scalar);
        return result;
    };

    DeprecatedSequentialVector operator*(const PetscScalar &scalar)
    {
        DeprecatedSequentialVector result;
        VecDuplicate(vec_, &result.vec_);
        VecCopy(vec_, result.vec_);
        VecScale(result.vec_, scalar);
        return result;
    };

    DeprecatedSequentialVector operator+(const DeprecatedSequentialVector &other)
    {
        DeprecatedSequentialVector result;
        VecDuplicate(vec_, &result.vec_);
        VecCopy(vec_, result.vec_);
        VecAXPY(result.vec_, 1.0, other.vec_);
        return result;
    };

    DeprecatedSequentialVector operator-(const DeprecatedSequentialVector &other)
    {
        DeprecatedSequentialVector result;
        VecDuplicate(vec_, &result.vec_);
        VecCopy(vec_, result.vec_);
        VecAXPY(result.vec_, -1.0, other.vec_);
        return result;
    };

    // Constructors
    DeprecatedSequentialVector()
    {
        // std::cout << "PETSc DeprecatedSequentialVector Default Constructor\n";
        VecCreateSeq(PETSC_COMM_SELF, 0, &vec_);
    };

    explicit DeprecatedSequentialVector(std::ranges::contiguous_range auto const &data)
    {
        // std::cout << "PETSc DeprecatedSequentialVector Constructor - range const&\n";
        // owns_data = false;
        VecCreateSeq(PETSC_COMM_SELF, data.size(), &vec_);
        VecPlaceArray(vec_, data.begin());
    };

    explicit DeprecatedSequentialVector(std::ranges::contiguous_range auto &&data)
    {
        // std::cout << "PETSc DeprecatedSequentialVector Constructor - range&& \n";
        // owns_data = false;
        VecCreateSeq(PETSC_COMM_SELF, data.size(), &vec_);
        VecPlaceArray(vec_, std::to_address(data.begin()));
    };

    // Copy and Move Constructors and Assignment Operators
    DeprecatedSequentialVector(const DeprecatedSequentialVector &other)
    {
        // std::cout << "PETSc DeprecatedSequentialVector Copy Constructor\n";
        VecDuplicate(other.vec_, &vec_);
        VecCopy(other.vec_, vec_);
    };

    DeprecatedSequentialVector(DeprecatedSequentialVector &&other)
    {
        // std::cout << "PETSc DeprecatedSequentialVector Move Constructor\n";
        VecDuplicate(other.vec_, &vec_);
        VecCopy(other.vec_, vec_);
        // vec_ = other.vec_;
        // other.vec_ = nullptr;
    };

    // DeprecatedSequentialVector& operator=(DeprecatedSequentialVector other) = delete;
    DeprecatedSequentialVector &operator=(const DeprecatedSequentialVector &other)
    {
        // std::cout << "PETSc DeprecatedSequentialVector Copy Assignment\n";
        VecDuplicate(other.vec_, &vec_);
        VecCopy(other.vec_, vec_);
        return *this;
    };

    DeprecatedSequentialVector &operator=(const volatile DeprecatedSequentialVector &&other)
    {
        // std::cout << "PETSc DeprecatedSequentialVector Move Assignment\n";
        VecDuplicate(other.vec_, &vec_);
        VecCopy(other.vec_, vec_);
        return *this;
    };

    // Destructor
    ~DeprecatedSequentialVector() { VecDestroy(&vec_); };
};

#endif