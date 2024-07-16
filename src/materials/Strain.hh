#ifndef FN_ABSTRACT_STRAIN
#define FN_ABSTRACT_STRAIN

#include <concepts>
#include <ranges>

template <typename StrainType>
concept Strain = requires(StrainType e) {
    { e.num_components } -> std::convertible_to<std::size_t>;
    { e.tensor };
    { e.get_strain() };
};

template <typename T>
concept StrainRange = std::ranges::range<T> && requires(T s) {
    { s.begin().num_components } -> std::convertible_to<std::size_t>;
    { s.begin().tensor };
    { s.begin().get_strain() };
};

// https://stackoverflow.com/questions/64228197/range-concept-for-a-specific-type
template <typename R, typename V>
concept RangeOf = std::ranges::range<R> && std::same_as<std::ranges::range_value_t<R>, V>;

template <std::size_t N> requires(N > 0)
class VoigtStrain{
public:
    static constexpr std::size_t dim{[](){if      constexpr (N == 1) return 1;
                                          else if constexpr (N == 3) return 2;
                                          else if constexpr (N == 6) return 3;
                                          else                       return 0; 
                                          }()};

    static constexpr std::size_t num_components{N};

private:
    std::array<double, N> component_{0};

public:
    Vector tensor{component_};

    constexpr std::span<const double, N> get_strain() const { return component_; };
    constexpr std::floating_point auto   get_strain(std::size_t i) const { return component_[i]; };

    template <typename... S> requires(sizeof...(S) == N)
    constexpr void set_strain(S... s){
        std::size_t i{0};
        ((component_[++i] = s), ...);
    }

    template <typename... S> requires(sizeof...(S) == N)
    constexpr VoigtStrain(S... s) : component_{s...}{}

    constexpr  VoigtStrain() = default;
    constexpr ~VoigtStrain() = default;
};

template <>
class VoigtStrain<1>{
public:
    static constexpr std::size_t dim{1};
    static constexpr std::size_t num_components = 1;

private:
    double component_{0.0};

public:
    double tensor{component_};

    constexpr void set_strain(double e) { component_ = e; };
    constexpr std::floating_point auto get_strain() const { return component_; };

    constexpr  VoigtStrain() = default;
    constexpr ~VoigtStrain() = default;
};

#endif

