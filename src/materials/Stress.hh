#ifndef FALL_N_CAUCHY_STRESS
#define FALL_N_CAUCHY_STRESS

//#include "../numerics/Tensor.hh"

# include <array>
# include <concepts>
# include <span>

# include "../numerics/linear_algebra/Vector.hh"


template<typename StressType>
concept Stress = requires(StressType s){
    {s.num_components}->std::convertible_to<std::size_t>;
    {s.tensor};
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
class VoigtStress{
    private:
        std::array<double, N> component_;
    
    public:
        static constexpr std::size_t num_components = N;

        Vector tensor{component_};

        constexpr std::span<const double, N> get_stress() const{return component_;};
        constexpr std::floating_point auto get_stress(std::size_t i) const{return component_[i];};

        template<typename... S> requires (sizeof...(S) == N)
        constexpr void set_stress(S... s){
            std::size_t i{0};
            ((component_[++i]=s), ...);
        };

        template<typename... S> requires (sizeof...(S) == N)
        VoigtStress(S... s) : component_{s...}{};

        VoigtStress() = default;
        ~VoigtStress() = default;
};

//Specialization for 1D stress (Uniaxial Stress) avoiding array overhead
template<>
class VoigtStress<1>{
    private:
        double component_{0.0};
    
    public:
        static constexpr std::size_t num_components = 1;

        double tensor{component_};

        constexpr void set_stress(double s){component_ = s;};

        constexpr std::floating_point auto get_stress() const{return component_;};

        VoigtStress() = default;
        ~VoigtStress() = default;
};



/*

class UniaxialStress{
    private:
        double component_{0.0};

    public:
        static constexpr std::size_t num_components = 1;


        double tensor{component_};

        constexpr void set_stress(double s){component_ = s;};

    UniaxialStress(){};
    ~UniaxialStress(){};
};

class PlaneStress{

    private:
        std::array<double, 3> component_{0.0, 0.0, 0.0};
    public:
        static constexpr std::size_t num_components{3};

        Vector tensor{component_}; //TODO: TensorPolicy (Vector = Voigt Tensor, Matrix = Matrix Tensor, etc.)

        constexpr std::span<const double, 3> get_stress() const{return component_;};
        constexpr std::floating_point auto get_stress(std::size_t i) const{return component_[i];};

        constexpr void set_stress(double s11, double s22, double s12){
            component_[0] = s11; component_[1] = s22; // Normal stresses
            component_[2] = s12;                      // Shear stress
        };

    PlaneStress(double s11, double s22, double s12) : component_{s11, s22, s12}{};

    PlaneStress(){};
    ~PlaneStress(){};
};


class ContinuumStress{
    private:
        
        std::array<double, 6> component_{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};


    public:

        Vector tensor{component_};

        constexpr std::span<const double, 6> get_stress() const{return component_;};
        constexpr std::floating_point auto get_stress(std::size_t i) const{return component_[i];};

        constexpr void set_stress(double s11, double s22, double s33, double s12, double s23, double s13){
            component_[0] = s11; component_[1] = s22; component_[2] = s33; // Normal stresses
            component_[3] = s12; component_[4] = s23; component_[5] = s13; // Shear  stresses
        };

    ContinuumStress(double s11, double s22, double s33, double s12, double s23, double s13) : component_{s11, s22, s33, s12, s23, s13}{};

    ContinuumStress(){};
    ~ContinuumStress(){};
};

*/









#endif