#ifndef MATERIAL_STATE_POLICYS_HH
#define MATERIAL_STATE_POLICYS_HH

// Your code goes here


#include <concepts>
#include <array>
#include <vector>
#include <span>
#include <ranges>

#include "Strain.hh"


template<typename T>
class MaterialState{
    private:


        std::unique_ptr<T> state_variable_; // Pointer to state variable in the domain (DOF, Integration Point, etc.)

    public:

        void update_state(const T& q){state_variable_ = q;};


        //copy constructor
        MaterialState(const MaterialState& other){
            state_variable_ = std::make_unique<T>(*other.state_variable_);
        };
        //move constructor
        MaterialState(MaterialState&& other){
            state_variable_ = std::move(other.state_variable_);
        };
        //copy assignment
        MaterialState& operator=(const MaterialState& other){
            state_variable_ = std::make_unique<T>(*other.state_variable_);
            return *this;
        };
        //move assignment
        MaterialState& operator=(MaterialState&& other){
            state_variable_ = std::move(other.state_variable_);
            return *this;
        };

        MaterialState() = default;
        ~MaterialState() =  default;
};


template<typename StrainType>
using ElasticMaterialState = MaterialState<StrainType>;//,StrainType> ; //Non-memory material q

template<typename StrainType>
using MemoryMaterialState = MaterialState<std::vector<StrainType>>;//,StrainType>;  



#endif // MATERIAL_STATE_HH