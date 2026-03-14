#ifndef MATERIALSTATEPOLICY_HH
#define MATERIALSTATEPOLICY_HH

#include <cassert>
#include <array>
#include <cstddef>
#include <concepts>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

// =============================================================================
// State-storage concepts
// =============================================================================
// Constrains any policy P<S> used by MaterialState:
//   - P<S> must expose current_value() → const S&
//   - P<S> must expose current_value_p() → const S*
//   - P<S> must accept update(const S&) and update(S&&)

template <template <typename> class Policy, typename S>
concept StateStoragePolicyFor = requires(Policy<S> p, const Policy<S> cp, const S& lv, S&& rv) {
    { cp.current_value()   } -> std::same_as<const S&>;
    { cp.current_value_p() } -> std::same_as<const S*>;
    p.update(lv);
    p.update(std::move(rv));
};

// Legacy alias kept while the module transitions away from "MemoryPolicy"
// terminology toward the more accurate "state-storage policy".
template <template <typename> class Policy, typename S>
concept MemoryPolicyFor = StateStoragePolicyFor<Policy, S>;

template <typename Storage, typename S>
concept HistoryStorageFor = requires(Storage storage, const Storage cstorage,
                                     const S& lv, S&& rv, std::size_t i) {
    { cstorage.empty() } -> std::convertible_to<bool>;
    { cstorage.size() } -> std::convertible_to<std::size_t>;
    { cstorage.back() } -> std::same_as<const S&>;
    { cstorage.back_p() } -> std::same_as<const S*>;
    storage.push_back(lv);
    storage.push_back(std::move(rv));
    { cstorage[i] } -> std::same_as<const S&>;
};

// =============================================================================
// ElasticState / CommittedState — single committed/current value
// =============================================================================
// Stores a single value of type S. Each update() overwrites the previous one.
// Suitable for path-independent materials (linear elasticity).

template <typename S>
class ElasticState {
    S value_{};

public:
    constexpr const S& current_value()   const noexcept { return  value_; }
    constexpr const S* current_value_p() const noexcept { return &value_; }

    constexpr void update(const S& s) { value_ = s; }
    constexpr void update(S&& s)      { value_ = std::move(s); }

    // --- Rule of zero: all special members defaulted --------------------------
    constexpr ElasticState() = default;
    constexpr ~ElasticState() = default;

    constexpr ElasticState(const ElasticState&) = default;
    constexpr ElasticState(ElasticState&&) noexcept = default;
    constexpr ElasticState& operator=(const ElasticState&) = default;
    constexpr ElasticState& operator=(ElasticState&&) noexcept = default;

    // --- Converting constructors (explicit to prevent implicit conversions) ---
    explicit constexpr ElasticState(const S& s) : value_{s} {}
    explicit constexpr ElasticState(S&& s)      : value_{std::move(s)} {}
};

template <typename S>
using CommittedState = ElasticState<S>;

// =============================================================================
// History storage backends
// =============================================================================

template <typename S, typename ContainerT = std::vector<S>>
class DynamicHistoryStorage {
    ContainerT values_{};

public:
    constexpr void push_back(const S& s) { values_.push_back(s); }
    constexpr void push_back(S&& s) { values_.emplace_back(std::move(s)); }

    [[nodiscard]] constexpr const S& back() const noexcept {
        assert(!values_.empty() && "DynamicHistoryStorage::back() called on empty history");
        return values_.back();
    }

    [[nodiscard]] constexpr const S* back_p() const noexcept {
        return values_.empty() ? nullptr : std::addressof(values_.back());
    }

    [[nodiscard]] constexpr bool empty() const noexcept { return values_.empty(); }
    [[nodiscard]] constexpr std::size_t size() const noexcept { return values_.size(); }

    [[nodiscard]] constexpr const S& operator[](std::size_t i) const noexcept {
        assert(i < values_.size() && "DynamicHistoryStorage index out of bounds");
        return values_[i];
    }

    explicit constexpr DynamicHistoryStorage(const S& s) : values_{s} {}
    explicit constexpr DynamicHistoryStorage(S&& s) { values_.emplace_back(std::move(s)); }

    constexpr DynamicHistoryStorage() = default;
    constexpr ~DynamicHistoryStorage() = default;
    constexpr DynamicHistoryStorage(const DynamicHistoryStorage&) = default;
    constexpr DynamicHistoryStorage(DynamicHistoryStorage&&) noexcept = default;
    constexpr DynamicHistoryStorage& operator=(const DynamicHistoryStorage&) = default;
    constexpr DynamicHistoryStorage& operator=(DynamicHistoryStorage&&) noexcept = default;
};

template <typename S, std::size_t Capacity>
class CircularHistoryStorage {
    static_assert(Capacity > 0, "CircularHistoryStorage requires Capacity > 0");
    static_assert(std::default_initializable<S>,
                  "CircularHistoryStorage currently requires default-initializable state variables");

    std::array<S, Capacity> values_{};
    std::size_t size_{0};
    std::size_t next_{0};

    [[nodiscard]] constexpr std::size_t logical_start_() const noexcept {
        return size_ < Capacity ? 0 : next_;
    }

public:
    constexpr void push_back(const S& s) {
        values_[next_] = s;
        next_ = (next_ + 1) % Capacity;
        if (size_ < Capacity) {
            ++size_;
        }
    }

    constexpr void push_back(S&& s) {
        values_[next_] = std::move(s);
        next_ = (next_ + 1) % Capacity;
        if (size_ < Capacity) {
            ++size_;
        }
    }

    [[nodiscard]] constexpr const S& back() const noexcept {
        assert(size_ > 0 && "CircularHistoryStorage::back() called on empty history");
        const auto index = (next_ + Capacity - 1) % Capacity;
        return values_[index];
    }

    [[nodiscard]] constexpr const S* back_p() const noexcept {
        return size_ == 0 ? nullptr : std::addressof(back());
    }

    [[nodiscard]] constexpr bool empty() const noexcept { return size_ == 0; }
    [[nodiscard]] constexpr std::size_t size() const noexcept { return size_; }
    [[nodiscard]] static consteval std::size_t capacity() noexcept { return Capacity; }
    [[nodiscard]] constexpr std::size_t next_position() const noexcept { return next_; }

    [[nodiscard]] constexpr const S& operator[](std::size_t i) const noexcept {
        assert(i < size_ && "CircularHistoryStorage index out of bounds");
        return values_[(logical_start_() + i) % Capacity];
    }

    explicit constexpr CircularHistoryStorage(const S& s) { push_back(s); }
    explicit constexpr CircularHistoryStorage(S&& s) { push_back(std::move(s)); }

    constexpr CircularHistoryStorage() = default;
    constexpr ~CircularHistoryStorage() = default;
    constexpr CircularHistoryStorage(const CircularHistoryStorage&) = default;
    constexpr CircularHistoryStorage(CircularHistoryStorage&&) noexcept = default;
    constexpr CircularHistoryStorage& operator=(const CircularHistoryStorage&) = default;
    constexpr CircularHistoryStorage& operator=(CircularHistoryStorage&&) noexcept = default;
};

// =============================================================================
// MemoryState / HistoryState — history-tracking policy
// =============================================================================
// Accumulates every state in a storage backend supplied by the caller.
// current_value() returns the most recent entry.
// Suitable for path-dependent materials (plasticity, damage, etc.).

template <typename S, typename StorageT = DynamicHistoryStorage<S>>
requires HistoryStorageFor<StorageT, S>
class MemoryState {
    StorageT value_{};

public:
    constexpr const S& current_value() const noexcept {
        return value_.back();
    }

    constexpr const S* current_value_p() const noexcept {
        return value_.back_p();
    }

    [[nodiscard]] constexpr bool empty() const noexcept { return value_.empty(); }
    [[nodiscard]] constexpr std::size_t size() const noexcept { return value_.size(); }

    [[nodiscard]] constexpr const S& operator[](std::size_t i) const noexcept {
        return value_[i];
    }

    constexpr void update(const S& s) { value_.push_back(s); }
    constexpr void update(S&& s)      { value_.push_back(std::move(s)); }

    // --- Rule of zero ---------------------------------------------------------
    constexpr MemoryState() = default;
    constexpr ~MemoryState() = default;

    constexpr MemoryState(const MemoryState&) = default;
    constexpr MemoryState(MemoryState&&) noexcept = default;
    constexpr MemoryState& operator=(const MemoryState&) = default;
    constexpr MemoryState& operator=(MemoryState&&) noexcept = default;

    // --- Converting constructors ----------------------------------------------
    explicit constexpr MemoryState(const S& s) : value_{s} {}
    explicit constexpr MemoryState(S&& s) : value_{std::move(s)} {}
};

template <typename S>
using HistoryState = MemoryState<S>;

template <typename S, std::size_t Capacity>
using CircularHistoryState = MemoryState<S, CircularHistoryStorage<S, Capacity>>;

template <std::size_t Capacity>
struct CircularHistoryPolicy {
    template <typename S>
    using Policy = CircularHistoryState<S, Capacity>;
};

// =============================================================================
// TrialCommittedState — staged trial + committed storage
// =============================================================================
// Stages trial states explicitly while preserving an independently committed
// value. This policy is not yet the default production path, but it provides
// the right storage semantics for future generic integration algorithms that
// need try/accept/reject workflows without coupling that logic to the law.

template <typename S>
class TrialCommittedState {
    S committed_{};
    S trial_{};
    bool has_trial_{false};

public:
    [[nodiscard]] constexpr const S& current_value() const noexcept {
        return has_trial_ ? trial_ : committed_;
    }

    [[nodiscard]] constexpr const S* current_value_p() const noexcept {
        return std::addressof(current_value());
    }

    [[nodiscard]] constexpr const S& committed_value() const noexcept {
        return committed_;
    }

    [[nodiscard]] constexpr const S* committed_value_p() const noexcept {
        return std::addressof(committed_);
    }

    [[nodiscard]] constexpr const S& trial_value() const noexcept {
        return has_trial_ ? trial_ : committed_;
    }

    [[nodiscard]] constexpr const S* trial_value_p() const noexcept {
        return has_trial_ ? std::addressof(trial_) : nullptr;
    }

    [[nodiscard]] constexpr bool has_trial_value() const noexcept {
        return has_trial_;
    }

    constexpr void update(const S& s) {
        trial_ = s;
        has_trial_ = true;
    }

    constexpr void update(S&& s) {
        trial_ = std::move(s);
        has_trial_ = true;
    }

    constexpr void commit_trial() {
        if (has_trial_) {
            committed_ = trial_;
            has_trial_ = false;
        }
    }

    constexpr void revert_trial() noexcept {
        has_trial_ = false;
    }

    constexpr TrialCommittedState() = default;
    constexpr ~TrialCommittedState() = default;

    constexpr TrialCommittedState(const TrialCommittedState&) = default;
    constexpr TrialCommittedState(TrialCommittedState&&) noexcept = default;
    constexpr TrialCommittedState& operator=(const TrialCommittedState&) = default;
    constexpr TrialCommittedState& operator=(TrialCommittedState&&) noexcept = default;

    explicit constexpr TrialCommittedState(const S& s) : committed_{s} {}
    explicit constexpr TrialCommittedState(S&& s) : committed_{std::move(s)} {}
};

#endif // MATERIALSTATEPOLICY_HH
