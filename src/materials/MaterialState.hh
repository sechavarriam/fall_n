#ifndef MATERIAL_STATE_POLICYS_HH
#define MATERIAL_STATE_POLICYS_HH

#include <concepts>
#include <array>
#include <vector>
#include <span>
#include <ranges>

#include "Strain.hh"

// TODO: Implement a MemoryPolicy concept to constraint the MemoryPolicy template parameter

// =================================================================================
// MemoryPolicies

template <typename T>
class ElasticState
{
    T value;

public:
    auto current_value() const noexcept { return value; }
    void update_state(const T &s) { value = s; }

    ElasticState() = default;
    ~ElasticState() = default;
};

template <typename T>
class MemoryState
{

    std::vector<T> value;

public:
    auto current_value() const noexcept { return value.back(); }
    void update_state(const T &s) { value.push_back(s); }

    MemoryState() = default;
    ~MemoryState() = default;
};

// There can be more memory policies e.g. with different memory management strategies.

// =================================================================================



template <template <typename> class MemoryPolicy, typename... StateVariableType>
class MaterialState
{
    static consteval bool is_multivariable(){return sizeof...(StateVariableType) > 1;};
    
    static consteval auto multivariable_check(){
        if constexpr (is_multivariable())
            return std::tuple<StateVariableType...>{}; 
        else
            return std::get<0>(std::tuple<StateVariableType...>{});//Trick: The pack has to be expanded...            
    };
    
    using StateVariable = std::invoke_result_t<decltype(&MaterialState::multivariable_check)>;

    MemoryPolicy<StateVariable> state_variable_;

public:
    auto current_value() const noexcept
    {
        return state_variable_.current_value();
    }

    auto update_state(const StateVariable &s)
    {
        state_variable_.update_state(s);
    }


    MaterialState() = default;
    ~MaterialState() = default;
    
};

#endif // MATERIAL_STATE_HH