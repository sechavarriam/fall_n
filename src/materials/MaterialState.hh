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
    static consteval auto variadic_check(){
        if constexpr (sizeof...(StateVariableType) == 1)
            return std::get<0>(std::tuple<StateVariableType...>{}); 
        else
            return std::tuple<StateVariableType...>{};
    };
    

    using StateVariable = std::invoke_result_t<decltype(&MaterialState::variadic_check)>;

    MemoryPolicy<StateVariable> state_variable_;

public:
    auto current_value() const noexcept
    {
        return state_variable_.current_value();
    }

    MaterialState() = default;
    ~MaterialState() = default;
    
};

#endif // MATERIAL_STATE_HH