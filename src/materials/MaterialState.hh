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

template <typename S>
class ElasticState
{
    S value;

public:
    auto current_value() const noexcept { return value; }
    void update(const S &s) { value = s; }

    ElasticState() = default;
    ~ElasticState() = default;
};

template <typename S>
class MemoryState
{
    std::vector<S> value;

public:
    inline auto current_value() const noexcept { return value.back(); }
    inline void update(const S &s) { value.push_back(s); }

    MemoryState() = default;
    ~MemoryState() = default;
};
 
// There can be more memory policies e.g. with different memory management strategies.

// =================================================================================

// MaterialState<ElasticState, Strain, T, M,...> ms1;

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
    
    using StateVariable = std::invoke_result_t<decltype(&MaterialState::multivariable_type)>;

    MemoryPolicy<StateVariable> state_variable_;

public:

    inline auto current_value() const noexcept {return state_variable_.current_value();}
    inline void update_state(const StateVariable &s){state_variable_.update_state(s);}


    MaterialState() = default;
    ~MaterialState() = default;
    
};

#endif // MATERIAL_STATE_HH