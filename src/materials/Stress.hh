#ifndef FALL_N_CAUCHY_STRESS
#define FALL_N_CAUCHY_STRESS

//#include "../numerics/Tensor.hh"

# include <array>
# include <concepts>
# include <span>

# include "../numerics/linear_algebra/Vector.hh"

# include "VoigtVector.hh"



template <std::size_t N> requires(N > 0)
class Stress : public VoigtVector<N> {

    public:
    using VoigtVector<N>::VoigtVector;  // Inherit constructors from VoigtVector
    using VoigtVector<N>::num_components;

    template <typename Derived>
    constexpr void set_stress(const Eigen::MatrixBase<Derived> &s) { VoigtVector<N>::set_components(s); }

    constexpr  Stress() = default;
    constexpr ~Stress() = default;
    
};



#endif // FALL_N_CAUCHY_STRESS