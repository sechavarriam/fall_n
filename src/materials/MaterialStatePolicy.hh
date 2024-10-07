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
    auto current_value() const noexcept { return value; }
    void update(const S &s) { value = s; }

    // Copy and Move constructors
    
    ElasticState(const ElasticState &s) : value{s.value} {}
    ElasticState(const S &s) : value{s} {}
    ElasticState(auto &&s) : value{std::forward<S>(s)} {}

    ElasticState() = default;
    ~ElasticState() = default;
};


template <typename S>
class MemoryState // Dynamic Memory Policy
{
    std::vector<S> value;

public:
    inline auto current_value() const noexcept { return value.back(); }
    inline void update(const S &s) { value.push_back(s); }

    // Copy and Move constructors
    MemoryState(const MemoryState &s) : value{s.value} {}
    MemoryState(const S &s) : value{s} {}
    MemoryState(auto &&s) : value{std::forward<S>(s)} {}

    MemoryState() = default;
    ~MemoryState() = default;
};

// TODO: if needed...
//template <typename S, std::size_t N>
//class FixedMemoryState // Fixed Memory Policy
//{
//    std::array<S, N> value;
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