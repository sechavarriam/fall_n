#ifndef FALLN_STATEVARIABLE_HH
#define FALLN_STATEVARIABLE_HH

#include <tuple>
#include <type_traits>


// StateVariable<ElasticState, Strain, Temperature, Others...> SV1;

template <template <typename> class MemoryPolicy, typename... StateVariableType>
class StateVariable
{
    static consteval bool is_multivariable(){return sizeof...(StateVariableType) > 1;};

    static consteval auto multivariable_type(){
        if constexpr (is_multivariable())
            return std::tuple<StateVariableType...>{}; 
        else
            return std::get<0>(std::tuple<StateVariableType...>{}); //Trick: The pack has to be expanded...            
    };
    
    using VariableContainer = std::invoke_result_t<decltype(&StateVariable::multivariable_type)>;

    MemoryPolicy<VariableContainer> state_variable_;

public:

    inline auto current_value() const noexcept {return state_variable_.current_value();}
    inline void update_state(const VariableContainer &s){state_variable_.update_state(s);}

    StateVariable() = default;
    ~StateVariable() = default;
    
};




#endif // FALLN_STATEVARIABLE_HH