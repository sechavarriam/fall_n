#ifndef FN_ABSTRACT_STRAIN
#define FN_ABSTRACT_STRAIN

#include <concepts>
#include <ranges>

#include "VoigtVector.hh"

template <std::size_t N> requires(N > 0)
class Strain : public VoigtVector<N> {
    
  public:
    using VoigtVector<N>::VoigtVector;  // Inherit constructors from VoigtVector
    using VoigtVector<N>::num_components;

    template <typename Derived> //requires std::same_as<Derived, Eigen::Matrix<double, N, 1>>
    constexpr void set_strain(const Eigen::MatrixBase<Derived> &s) { VoigtVector<N>::set_components(s); }

    constexpr  Strain() = default;  
    constexpr ~Strain() = default;
};


#endif


