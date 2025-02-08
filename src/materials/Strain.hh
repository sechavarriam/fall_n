#ifndef FN_ABSTRACT_STRAIN
#define FN_ABSTRACT_STRAIN

#include <concepts>
#include <ranges>

#include "VoigtVector.hh"

//template <typename StrainType>
//concept StrainC = requires(StrainType e) {
//    { e.num_components } -> std::convertible_to<std::size_t>;
//    { e.vector };
//    { e.get_strain() };
//};
//
//template <typename T>
//concept StrainRange = std::ranges::range<T> && requires(T s) {
//    { s.begin().num_components } -> std::convertible_to<std::size_t>;
//    { s.begin().vector };
//    { s.begin().get_strain() };
//};

// https://stackoverflow.com/questions/64228197/range-concept-for-a-specific-type
template <typename R, typename V>
concept RangeOf = std::ranges::range<R> && std::same_as<std::ranges::range_value_t<R>, V>;



template <std::size_t N> requires(N > 0)
class Strain : public VoigtVector<N> {
    
  public:
    
    using VectorT = Eigen::Vector<double, N>;
    
    static constexpr std::size_t num_components{N};
    static constexpr std::size_t dim{[](){if      constexpr (N == 1) return 1;
                                          else if constexpr (N == 3) return 2;
                                          else if constexpr (N == 6) return 3;
                                          else                       return 0; 
                                          }()}; // Tambien en stress? Subir a VoigtVector?

   
    //VectorT vector() const noexcept { return VoigtVector<N>::vector(); };

    template <typename Derived> //requires std::same_as<Derived, Eigen::Matrix<double, N, 1>>
    constexpr void set_strain(const Eigen::MatrixBase<Derived> &s) { VoigtVector<N>::set_vector(s); };

    

    constexpr  Strain() = default;  
    constexpr ~Strain() = default;
};





//=======================================================================================
// _____           _                               _       _ _ _ 
//|_   _|         | |                             | |     | | | |
//  | | ___     __| | ___ _ __  _ __ ___  ___ __ _| |_ ___| | | |
//  | |/ _ \   / _` |/ _ \ '_ \| '__/ _ \/ __/ _` | __/ _ \ | | |
//  | | (_) | | (_| |  __/ |_) | | |  __/ (_| (_| | ||  __/_|_|_|
//  \_/\___/   \__,_|\___| .__/|_|  \___|\___\__,_|\__\___(_|_|_)
//                       | |                                     
//                       |_|                                     
//=======================================================================================


template <std::size_t N> requires(N > 0)
class StrainDeprecated{
public:
    static constexpr std::size_t dim{[](){if      constexpr (N == 1) return 1;
                                          else if constexpr (N == 3) return 2;
                                          else if constexpr (N == 6) return 3;
                                          else                       return 0; 
                                          }()};

    static constexpr std::size_t num_components{N};

private:
    std::array<double, N> component_{0.0};

public:
    DeprecatedSequentialVector vector{component_};

    constexpr std::span<const double, N> get_strain() const { return component_; };
    constexpr std::floating_point auto   get_strain(std::size_t i) const { return component_[i]; };

    constexpr double& operator[](std::size_t i) { return component_[i]; };

    template <typename... S> requires(sizeof...(S) == N)
    constexpr void set_strain(S... s){
        std::size_t i{0};
        ((component_[++i] = s), ...);
    }

    // =========== CONSTRUCTORS ==========================

    template <typename... S> requires(sizeof...(S) == N)
    constexpr StrainDeprecated(S... s) : component_{s...}{}

    // copy constructor
    constexpr StrainDeprecated(const StrainDeprecated<N> &e) : component_{e.component_} {}
    // move constructor
    constexpr StrainDeprecated(StrainDeprecated<N> &&e) : component_{std::move(e.component_)} {}
    // copy assignment
    constexpr StrainDeprecated &operator=(const StrainDeprecated<N> &e){
        component_ = e.component_;
        return *this;
    }
    // move assignment
    constexpr StrainDeprecated &operator=(StrainDeprecated<N> &&e){
        component_ = std::move(e.component_);
        return *this;
    }

    constexpr  StrainDeprecated() = default;
    constexpr ~StrainDeprecated() = default;
};

template <>
class StrainDeprecated<1>{
public:
    static constexpr std::size_t dim{1};
    static constexpr std::size_t num_components = 1;

private:
    double component_{0.0};

public:
    double vector{component_};

    constexpr void set_strain(double e) { component_ = e; };
    constexpr std::floating_point auto get_strain() const { return component_; };

    constexpr  StrainDeprecated() = default;
    constexpr ~StrainDeprecated() = default;
};

#endif


