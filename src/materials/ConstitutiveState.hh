#ifndef FALL_N_CONSTITUTIVE_STATE_HH
#define FALL_N_CONSTITUTIVE_STATE_HH

#include <cstddef>
#include <utility>

#include "MaterialState.hh"
#include "MaterialStatePolicy.hh"

// =============================================================================
// ConstitutiveState<StateStoragePolicy, ...>
// =============================================================================
//
// Semantic layer above MaterialState.  MaterialState remains the low-level
// storage adapter parameterised by a storage policy; ConstitutiveState gives
// that adapter the correct constitutive meaning at a material/section site.
//
// It does not yet separate irreversible internal variables from the law.
// That next step will require an explicit ConstitutiveLaw / ConstitutiveState /
// ConstitutiveIntegrator split.  This wrapper is the bridge toward that
// design.
//
// =============================================================================

template <template <typename> class StateStoragePolicy, typename... StateVariableTypes>
requires (sizeof...(StateVariableTypes) >= 1)
class ConstitutiveState {
    using StorageAdapterT = MaterialState<StateStoragePolicy, StateVariableTypes...>;

public:
    using StateVariableT = typename StorageAdapterT::StateVariableT;

private:
    StorageAdapterT storage_{};

public:
    [[nodiscard]] constexpr const StateVariableT& current_value() const noexcept {
        return storage_.current_value();
    }

    [[nodiscard]] constexpr const StateVariableT* current_value_p() const noexcept {
        return storage_.current_value_p();
    }

    constexpr void update(const StateVariableT& s) {
        storage_.update(s);
    }

    constexpr void update(StateVariableT&& s) {
        storage_.update(std::move(s));
    }

    [[nodiscard]] constexpr bool empty() const noexcept
        requires requires { storage_.empty(); }
    {
        return storage_.empty();
    }

    [[nodiscard]] constexpr std::size_t size() const noexcept
        requires requires { storage_.size(); }
    {
        return storage_.size();
    }

    [[nodiscard]] constexpr const StateVariableT& operator[](std::size_t i) const noexcept
        requires requires { storage_[i]; }
    {
        return storage_[i];
    }

    constexpr void commit_trial()
        requires requires { storage_.commit_trial(); }
    {
        storage_.commit_trial();
    }

    constexpr void revert_trial()
        requires requires { storage_.revert_trial(); }
    {
        storage_.revert_trial();
    }

    [[nodiscard]] constexpr bool has_trial_value() const noexcept
        requires requires { storage_.has_trial_value(); }
    {
        return storage_.has_trial_value();
    }

    [[nodiscard]] constexpr const StateVariableT& committed_value() const noexcept
        requires requires { storage_.committed_value(); }
    {
        return storage_.committed_value();
    }

    [[nodiscard]] constexpr const StateVariableT* committed_value_p() const noexcept
        requires requires { storage_.committed_value_p(); }
    {
        return storage_.committed_value_p();
    }

    [[nodiscard]] constexpr const StateVariableT& trial_value() const noexcept
        requires requires { storage_.trial_value(); }
    {
        return storage_.trial_value();
    }

    [[nodiscard]] constexpr const StateVariableT* trial_value_p() const noexcept
        requires requires { storage_.trial_value_p(); }
    {
        return storage_.trial_value_p();
    }

    [[nodiscard]] constexpr const StorageAdapterT& storage_adapter() const noexcept {
        return storage_;
    }

    [[nodiscard]] constexpr StorageAdapterT& storage_adapter() noexcept {
        return storage_;
    }

    constexpr ConstitutiveState() = default;
    constexpr ~ConstitutiveState() = default;

    constexpr ConstitutiveState(const ConstitutiveState&) = default;
    constexpr ConstitutiveState(ConstitutiveState&&) noexcept = default;
    constexpr ConstitutiveState& operator=(const ConstitutiveState&) = default;
    constexpr ConstitutiveState& operator=(ConstitutiveState&&) noexcept = default;

    explicit constexpr ConstitutiveState(const StateVariableT& s)
        : storage_{s}
    {}

    explicit constexpr ConstitutiveState(StateVariableT&& s)
        : storage_{std::move(s)}
    {}
};

template <typename S>
using CommittedConstitutiveState = ConstitutiveState<CommittedState, S>;

template <typename S>
using HistoryConstitutiveState = ConstitutiveState<HistoryState, S>;

template <typename S, std::size_t Capacity>
using CircularHistoryConstitutiveState =
    ConstitutiveState<CircularHistoryPolicy<Capacity>::template Policy, S>;

template <typename S>
using TrialCommittedConstitutiveState = ConstitutiveState<TrialCommittedState, S>;

#endif // FALL_N_CONSTITUTIVE_STATE_HH
