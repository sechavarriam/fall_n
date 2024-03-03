#ifndef MATERIAL_STATE_POLICYS_HH
#define MATERIAL_STATE_POLICYS_HH

// Your code goes here


#include <concepts>
#include <array>
#include <vector>
#include <span>

/*
template<typename StateType>
concept MaterialState = requires(StateType s){
//    {s.num_components}->std::convertible_to<std::size_t>;
//    {s.tensor};
//    {s.get_state()};
};
*/

template<typename ContainerType, typename EfectType>
class MaterialState{
    private:
        ContainerType state_{};

        //INTERFASE

    public:
        MaterialState(){};
        ~MaterialState(){};
};


template<typename StrainType>
using ElasticMaterialState = MaterialState<StrainType,StrainType> ; //Non-memory material state

template<typename StrainType>
using MemoryMaterialState = MaterialState<std::vector<StrainType>,StrainType>;  



#endif // MATERIAL_STATE_HH