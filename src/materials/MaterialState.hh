#ifndef MATERIAL_STATE_POLICYS_HH
#define MATERIAL_STATE_POLICYS_HH

#include <concepts>
#include <array>
#include <vector>
#include <span>
#include <ranges>
#include <cstddef>
#include <memory>

#include "MaterialStatePolicy.hh"

#include <tuple>
#include <type_traits>

template <template <typename> class MemoryPolicy, typename... StateVariableType>
class MaterialState
{
    static consteval bool is_multivariable(){return sizeof...(StateVariableType) > 1;};

    static consteval auto multivariable_type(){
        if constexpr (is_multivariable())
            return std::tuple<StateVariableType...>{}; 
        else
            return std::get<0>(std::tuple<StateVariableType...>{}); //Trick: The pack has to be expanded...            
    };
    
public:
    using VariableContainer = std::invoke_result_t<decltype(&MaterialState::multivariable_type)>;

private:
    MemoryPolicy<VariableContainer> value;
public:


    inline auto current_value()   const noexcept {return  value.current_value();}
    inline auto current_value_p() const noexcept {return &value.current_value();}
    
    inline void update(VariableContainer &s) {value.update(s);}

    // Copy and Move constructors

    MaterialState(const MaterialState &s)     : value{s.value} {}
    MaterialState(const VariableContainer &s) : value{s}       {}
    MaterialState(VariableContainer &&s) : value{std::forward<VariableContainer>(s)} {}


    MaterialState() = default;
    ~MaterialState() = default;
    
};





#endif // MATERIAL_STATE_HH