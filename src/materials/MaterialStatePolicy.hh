#ifndef MATERIALSTATEPOLICY_HH
#define MATERIALSTATEPOLICY_HH

#include <cassert>
#include <concepts>
#include <type_traits>
#include <vector>

// =============================================================================
// MemoryPolicy concept
// =============================================================================
// Constrains any policy P<S> used by MaterialState:
//   - P<S> must expose current_value() → const S&
//   - P<S> must expose current_value_p() → const S*
//   - P<S> must accept update(const S&) and update(S&&)

template <template <typename> class Policy, typename S>
concept MemoryPolicyFor = requires(Policy<S> p, const Policy<S> cp, const S& lv, S&& rv) {
    { cp.current_value()   } -> std::same_as<const S&>;
    { cp.current_value_p() } -> std::same_as<const S*>;
    p.update(lv);
    p.update(std::move(rv));
};

// =============================================================================
// ElasticState — stateless (current-value-only) policy
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

// =============================================================================
// MemoryState — history-tracking policy
// =============================================================================
// Accumulates every state in a container (default: std::vector<S>).
// current_value() returns the most recent entry.
// Suitable for path-dependent materials (plasticity, damage, etc.).

template <typename S, typename ContainerT = std::vector<S>>
class MemoryState {
    ContainerT value_{};

public:
    constexpr const S& current_value() const noexcept {
        assert(!value_.empty() && "MemoryState::current_value() called on empty history");
        return value_.back();
    }

    constexpr const S* current_value_p() const noexcept {
        return value_.empty() ? nullptr : &value_.back();
    }

    constexpr void update(const S& s) { value_.push_back(s); }
    constexpr void update(S&& s)      { value_.emplace_back(std::move(s)); }

    // --- Rule of zero ---------------------------------------------------------
    constexpr MemoryState() = default;
    constexpr ~MemoryState() = default;

    constexpr MemoryState(const MemoryState&) = default;
    constexpr MemoryState(MemoryState&&) noexcept = default;
    constexpr MemoryState& operator=(const MemoryState&) = default;
    constexpr MemoryState& operator=(MemoryState&&) noexcept = default;

    // --- Converting constructors ----------------------------------------------
    explicit constexpr MemoryState(const S& s) : value_{s} {}
    explicit constexpr MemoryState(S&& s)      { value_.emplace_back(std::move(s)); }
};

#endif // MATERIALSTATEPOLICY_HH