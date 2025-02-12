#ifndef MATERIALSTATEPOLICY_HH
#define MATERIALSTATEPOLICY_HH


#include <concepts>
#include <array>
#include <vector>

// MemoryPolicies

// TODO: Implement a MemoryPolicy concept to constraint the MemoryPolicy template parameter in StateVariable template.
template <typename S>
class ElasticState{
    S value;

public:
    inline constexpr auto current_value()   const noexcept { 
        //std::cout << "ElasticState::current_value()\n" << value.vector() << std::endl;
        return  value; 
    }
    
    inline constexpr auto current_value_p() const noexcept { return &value; }

    inline void update(const S& s) { value = s; }

    // Copy and Move constructors  
    constexpr ElasticState(const ElasticState &s) : value{s.value} {}
    constexpr ElasticState(const S &s) : value{s} {}
    constexpr ElasticState(auto &&s) : value{std::forward<S>(s)} {}

    constexpr ElasticState() = default;
    constexpr ~ElasticState() = default;
};


template <typename S, typename ContainerT = std::vector<S>>
class MemoryState // Dynamic Memory Policy
{
    ContainerT value;

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


#endif // MATERIALSTATEPOLICY_HH