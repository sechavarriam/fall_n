#ifndef MATERIAL_STATE_HH
#define MATERIAL_STATE_HH

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include "MaterialStatePolicy.hh"


template <template <typename> class StateStoragePolicy, typename... StateVariableTypes>
requires (sizeof...(StateVariableTypes) >= 1)
class MaterialState {
private:
    using TupleT  = std::tuple<StateVariableTypes...>;
    using SingleT = std::tuple_element_t<0, TupleT>;

public:
    using StateVariableT =
           std::conditional_t<sizeof...(StateVariableTypes) == 1, SingleT, TupleT>;

    // Verify that the chosen storage policy satisfies the concept contract.
    static_assert(StateStoragePolicyFor<StateStoragePolicy, StateVariableT>,
                  "StateStoragePolicy does not satisfy StateStoragePolicyFor<StateVariableT>");

private:
    StateStoragePolicy<StateVariableT> value_;

public:
    constexpr const StateVariableT& current_value() const noexcept { return value_.current_value(); }
    constexpr const StateVariableT* current_value_p() const noexcept { return value_.current_value_p(); }

    constexpr void update(const StateVariableT& s) { value_.update(s); }
    constexpr void update(StateVariableT&& s) { value_.update(std::move(s)); }

    [[nodiscard]] constexpr bool empty() const noexcept
        requires requires { { value_.empty() } -> std::convertible_to<bool>; }
    {
        return value_.empty();
    }

    [[nodiscard]] constexpr std::size_t size() const noexcept
        requires requires { { value_.size() } -> std::convertible_to<std::size_t>; }
    {
        return static_cast<std::size_t>(value_.size());
    }

    [[nodiscard]] constexpr const StateVariableT& operator[](std::size_t i) const noexcept
        requires requires { value_[i]; }
    {
        return value_[i];
    }

    constexpr void commit_trial()
        requires requires { value_.commit_trial(); }
    {
        value_.commit_trial();
    }

    constexpr void revert_trial()
        requires requires { value_.revert_trial(); }
    {
        value_.revert_trial();
    }

    [[nodiscard]] constexpr bool has_trial_value() const noexcept
        requires requires { { value_.has_trial_value() } -> std::same_as<bool>; }
    {
        return value_.has_trial_value();
    }

    [[nodiscard]] constexpr const StateVariableT& committed_value() const noexcept
        requires requires { value_.committed_value(); }
    {
        return value_.committed_value();
    }

    [[nodiscard]] constexpr const StateVariableT* committed_value_p() const noexcept
        requires requires { value_.committed_value_p(); }
    {
        return value_.committed_value_p();
    }

    [[nodiscard]] constexpr const StateVariableT& trial_value() const noexcept
        requires requires { value_.trial_value(); }
    {
        return value_.trial_value();
    }

    [[nodiscard]] constexpr const StateVariableT* trial_value_p() const noexcept
        requires requires { value_.trial_value_p(); }
    {
        return value_.trial_value_p();
    }

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
