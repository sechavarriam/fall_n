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

template <template <typename> class MemoryPolicy, typename... StateVariableTypes> // TODO: Constrain with MemoryPolicy concept. 
class MaterialState{
    
    static consteval bool is_multivariable(){return sizeof...(StateVariableTypes) > 1;};

    static consteval auto multivariable_type(){
        if constexpr (is_multivariable())
            return std::tuple<StateVariableTypes...>{}; 
        else
            return std::get<0>(std::tuple<StateVariableTypes...>{}); //Trick: The pack has to be expanded...            
    };
    
public:

    using StateVariableT = std::invoke_result_t<decltype(&MaterialState::multivariable_type)>;

private:
    MemoryPolicy<StateVariableT> value;

public:
    inline auto current_value  () const noexcept {return  value.current_value();}
    inline auto current_value_p() const noexcept {return &value.current_value();}
    
    inline void update(const StateVariableT &s) {value.update(s);}


    // Copy and Move constructors
    MaterialState(const MaterialState &s)     : value{s.value} {}
    MaterialState(const StateVariableT &s) : value{s}       {}
    MaterialState(StateVariableT &&s) : value{std::forward<StateVariableT>(s)} {}


    MaterialState() = default;
    ~MaterialState() = default;
    
};





#endif // MATERIAL_STATE_HH