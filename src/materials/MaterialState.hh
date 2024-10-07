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
    
    using VariableContainer = std::invoke_result_t<decltype(&MaterialState::multivariable_type)>;

    MemoryPolicy<VariableContainer> value;

public:

    inline auto current_value() const noexcept {return value.current_value();}
    inline void update_state(const VariableContainer &s){value.update_state(s);}

    MaterialState(auto&&... args) : value{std::forward<decltype(args)>(args)...} {}

    MaterialState() = default;
    ~MaterialState() = default;
    
};





#endif // MATERIAL_STATE_HH