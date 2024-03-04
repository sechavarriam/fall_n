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
        T state_variable_{}; //Array of pointers? Array of references? Array of values? (vector?)

    public:

        auto current_state() const{
            if constexpr (StrainRange<T>){
                return state_variable_.end();
            }
            else if constexpr (Strain<T>){
                return state_variable_;
            }
            else{
                std::unreachable();
            };
        };


        void update_state(const T& q){state_variable_ = q;};


        MaterialState(){};
        ~MaterialState(){};
};


template<typename StrainType>
using ElasticMaterialState = MaterialState<StrainType>;//,StrainType> ; //Non-memory material q

template<typename StrainType>
using MemoryMaterialState = MaterialState<std::vector<StrainType>>;//,StrainType>;  



#endif // MATERIAL_STATE_HH