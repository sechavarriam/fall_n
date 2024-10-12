#ifndef MATERIALSTATEPOLICY_HH
#define MATERIALSTATEPOLICY_HH


#include <concepts>
#include <array>
#include <vector>

// MemoryPolicies

// TODO: Implement a MemoryPolicy concept to constraint the MemoryPolicy template parameter in StateVariable template.
template <typename S>
class ElasticState
{
    S value;

public:
    inline constexpr auto current_value()   const noexcept { return  value; }
    inline constexpr auto current_value_p() const noexcept { return &value; }

    inline void update(const S& s) { value = s; }

    // Copy and Move constructors
    
    constexpr ElasticState(const ElasticState &s) : value{s.value} {}
    constexpr ElasticState(const S &s) : value{s} {}
    constexpr ElasticState(auto &&s) : value{std::forward<S>(s)} {}

    constexpr ElasticState() = default;
    constexpr ~ElasticState() = default;
};


template <typename S>
class MemoryState // Dynamic Memory Policy
{
    std::vector<S> value;

public:
    inline constexpr auto current_value() const noexcept { return value.back(); }
    inline constexpr auto current_value_p() const noexcept { return &value; }

    inline constexpr void update(const S &s) { value.push_back(s); }

    // Copy and Move constructors
    constexpr MemoryState(const MemoryState &s) : value{s.value} {}
    constexpr MemoryState(const S &s) : value{s} {}
    constexpr MemoryState(auto &&s) : value{std::forward<S>(s)} {}

    constexpr MemoryState() = default;
    constexpr ~MemoryState() = default;
};

// TODO: if needed...
//template <typename S, std::size_t N>
//class FixedMemoryState // Fixed Memory Policy
//{
//    std::array<S, N> value;  //List? some fixed memory circular buffer(in terms of array)?
//    std::size_t index;
//
//public:
//    inline auto current_value() const noexcept { return value[index]; }
//    inline void update(const S &s) { value[index] = s; }
//
//    FixedMemoryState() : index{0} {}
//    ~FixedMemoryState() = default;
//};

// There can be more memory policies e.g. with different memory management strategies.



#endif // MATERIALSTATEPOLICY_HH