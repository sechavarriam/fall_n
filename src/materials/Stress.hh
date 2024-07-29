#ifndef FALL_N_CAUCHY_STRESS
#define FALL_N_CAUCHY_STRESS

//#include "../numerics/Tensor.hh"

# include <array>
# include <concepts>
# include <span>

# include "../numerics/linear_algebra/Vector.hh"


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


template<std::size_t N> requires (N > 0)
class Stress{
    private:
        std::array<double, N> component_;
    
    public:
        static constexpr std::size_t num_components = N;

        Vector vector{component_};

        constexpr std::span<const double, N> get_stress() const{return component_;};
        constexpr std::floating_point auto get_stress(std::size_t i) const{return component_[i];};

        template<typename... S> requires (sizeof...(S) == N)
        constexpr void set_stress(S... s){
            std::size_t i{0};
            ((component_[++i]=s), ...);
        }

        template<typename... S> requires (sizeof...(S) == N)
        Stress(S... s) : component_{s...}{}

        Stress() = default;
        ~Stress() = default;
};

//Specialization for 1D stress (Uniaxial Stress) avoiding array overhead
template<>
class Stress<1>{
    private:
        double component_{0.0};
    
    public:
        static constexpr std::size_t num_components = 1;

        double vector{component_};

        constexpr void set_stress(double s){component_ = s;};

        constexpr std::floating_point auto get_stress() const{return component_;};

        Stress() = default;
        ~Stress() = default;
};



 





#endif