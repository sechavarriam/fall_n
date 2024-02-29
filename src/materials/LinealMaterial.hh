
#ifndef FALL_N_LINEAL_MATERIAL
#define FALL_N_LINEAL_MATERIAL


#include <cstddef>
#include <type_traits>
#include <concepts>

#include "Stress.hh"
#include "Strain.hh"

# include "../numerics/linear_algebra/Matrix.hh"
# include "../utils/index.hh"



namespace material{
}



//Some concept
template<Stress StressPolicy, Strain StrainPolicy> //Continuum, Uniaxial, Plane, etc. 
class LinealMaterial{ //Or materialBase

    static constexpr std::size_t num_stresses_ = StressPolicy::num_components;
    static constexpr std::size_t num_strains_  = StrainPolicy::num_components;

    static constexpr std::size_t total_parameters_ = num_stresses_*num_strains_;

    std::array<double, total_parameters_> material_parameters_{0.0};

  public:

    constexpr void set_parameter(std::size_t i, std::size_t j, double value){
        material_parameters_[utils::md_index_2_list<num_stresses_,num_strains_>(i,j)] = value;
    };

    Matrix compliance_tensor{material_parameters_}; //elasticity tensor or material stiffness matrix 

    //Constructor
    //LinealMaterial(std::array<double, total_parameters_> material_parameters) : material_parameters_{material_parameters}{};

    LinealMaterial() = default;
    ~LinealMaterial() = default;
};


typedef LinealMaterial<VoigtStress<1>, VoigtStrain<1>> UniaxialMaterial;
typedef LinealMaterial<VoigtStress<3>, VoigtStrain<3>> ContinuumMaterial2D;
typedef LinealMaterial<VoigtStress<6>, VoigtStrain<6>> ContinuumMaterial3D;



class IsotropicMaterial : public ContinuumMaterial3D{
    
    // https://stackoverflow.com/questions/9864125/c11-how-to-alias-a-function
    double young_modulus_{0.0};
    double poisson_ratio_{0.0};

    //using E = young_modulus_;
    //using v = poisson_ratio_;
    
    public:
    constexpr double E() const{return young_modulus_;};
    constexpr double v() const{return poisson_ratio_;};

    constexpr void set_E(double E){young_modulus_ = E;};
    constexpr void set_v(double v){poisson_ratio_ = v;};

    constexpr double G()      const{return young_modulus_/(2.0*(1.0+poisson_ratio_));};     //Shear Modulus
    constexpr double K()      const{return young_modulus_/(3.0*(1.0-2.0*poisson_ratio_));}; //Bulk Modulus
    constexpr double lambda() const{return young_modulus_*poisson_ratio_/((1.0+poisson_ratio_)*(1.0-2.0*poisson_ratio_));}; //Lamé's first parameter
    constexpr double mu()     const{return young_modulus_/(2.0*(1.0+poisson_ratio_));}; //Lamé's second parameter

    constexpr void update_stiffness_matrix(){ //ensamblar con threads 
        set_parameter(0,0,lambda()+2.0*mu());
        set_parameter(1,1,lambda()+2.0*mu());
        set_parameter(2,2,lambda()+2.0*mu());
        set_parameter(3,3,mu());
        set_parameter(4,4,mu());
        set_parameter(5,5,mu());
        set_parameter(0,1,lambda());
        set_parameter(0,2,lambda());
        set_parameter(1,0,lambda());
        set_parameter(1,2,lambda());
        set_parameter(2,0,lambda());
        set_parameter(2,1,lambda());
    };


};
#endif // FALL_N_LINEAL_MATERIAL