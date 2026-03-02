#ifndef MATERIAL_STATE_HH
#define MATERIAL_STATE_HH

#include <cstddef>
#include <tuple>
#include <type_traits>

#include "MaterialStatePolicy.hh"


template <template <typename> class MemoryPolicy, typename... StateVariableTypes>
requires (sizeof...(StateVariableTypes) >= 1)
class MaterialState {
private:
    using TupleT  = std::tuple<StateVariableTypes...>;
    using SingleT = std::tuple_element_t<0, TupleT>;

public:
    using StateVariableT =
           std::conditional_t<sizeof...(StateVariableTypes) == 1, SingleT, TupleT>;

    // Verify that the chosen MemoryPolicy satisfies the concept for StateVariableT
    static_assert(MemoryPolicyFor<MemoryPolicy, StateVariableT>,
                  "MemoryPolicy does not satisfy MemoryPolicyFor<StateVariableT>");

private:
    MemoryPolicy<StateVariableT> value_;

public:
    constexpr const StateVariableT& current_value() const noexcept { return value_.current_value(); }
    constexpr const StateVariableT* current_value_p() const noexcept { return value_.current_value_p(); }

    constexpr void update(const StateVariableT& s) { value_.update(s); }
    constexpr void update(StateVariableT&& s) { value_.update(std::move(s)); }

    // --- Rule of zero ---------------------------------------------------------
    constexpr MaterialState() = default;
    constexpr ~MaterialState() = default;

    constexpr MaterialState(const MaterialState&) = default;
    constexpr MaterialState(MaterialState&&) noexcept = default;
    constexpr MaterialState& operator=(const MaterialState&) = default;
    constexpr MaterialState& operator=(MaterialState&&) noexcept = default;

    explicit constexpr MaterialState(const StateVariableT& s) : value_{s} {}
    explicit constexpr MaterialState(StateVariableT&& s) : value_{std::move(s)} {}
};

#endif // MATERIAL_STATE_HH