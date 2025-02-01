#ifndef FALL_N_CAUCHY_STRESS
#define FALL_N_CAUCHY_STRESS

//#include "../numerics/Tensor.hh"

# include <array>
# include <concepts>
# include <span>

# include "../numerics/linear_algebra/Vector.hh"

# include "VoigtVector.hh"


template<typename StressType>
concept StressC = requires(StressType s){
    {s.num_components}->std::convertible_to<std::size_t>;
    {s.vector};
    {s.get_stress()};
};

template<typename StressType>
class CauchyStress{
    private:
        StressType stress_{};

    public:

    CauchyStress(){};
    ~CauchyStress(){};
};


template <std::size_t N> requires(N > 0)
class Stress : public VoigtVector<N> {

  public:
    static constexpr std::size_t num_components{N};



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


template<std::size_t N> requires (N > 0)
class StressDeprecated{
    private:
        std::array<double, N> component_{0.0}; // Default value is 0.0
    
    public:
        static constexpr std::size_t num_components = N;

        DeprecatedSequentialVector vector{component_};

        constexpr std::span<const double, N> get_stress() const{return component_;};
        constexpr std::floating_point auto get_stress(std::size_t i) const{return component_[i];};

        template<typename... S> requires (sizeof...(S) == N)
        constexpr void set_stress(S... s){
            std::size_t i{0};
            ((component_[++i]=s), ...);
        }

        template<typename... S> requires (sizeof...(S) == N)
        StressDeprecated(S... s) : component_{s...}{}

        StressDeprecated() = default;
        ~StressDeprecated() = default;
};

//Specialization for 1D stress (Uniaxial StressDeprecated) avoiding array overhead
template<>
class StressDeprecated<1>{
    private:
        double component_{0.0};
    
    public:
        static constexpr std::size_t num_components = 1;

        double vector{component_};

        constexpr void set_stress(double s){component_ = s;};

        constexpr std::floating_point auto get_stress() const{return component_;};

        StressDeprecated() = default;
        ~StressDeprecated() = default;
};



 





#endif