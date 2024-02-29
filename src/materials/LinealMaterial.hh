
#ifndef FALL_N_LINEAL_MATERIAL
#define FALL_N_LINEAL_MATERIAL


#include <cstddef>
#include <type_traits>
#include <concepts>

#include "Stress.hh"
#include "Strain.hh"

# include "../numerics/linear_algebra/Matrix.hh"
# include "../utilis/index.hh"



namespace material{
}



//Some concept
template<Stress StressPolicy, Strain StrainPolicy> //Continuum, Uniaxial, Plane, etc. 
class LinealMaterial{ //Or materialBase

    static constexpr std::size_t total_parameters_ = StressPolicy::num_components*StrainPolicy::num_components;

    std::array<double, total_parameters_> material_parameters_{0.0};

  public:

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

    constexpr update_stiffness_matrix(){
        material_parameters_[utils::md_index_2_list<6,6>(0,0)] = lambda()+ 2.0*mu_();
        material_parameters_[utils::md_index_2_list<6,6>(1,1)] = lambda()+ 2.0*mu_();
        material_parameters_[utils::md_index_2_list<6,6>(2,2)] = lambda()+ 2.0*mu_();
        material_parameters_[utils::md_index_2_list<6,6>(3,3)] = mu_();
        material_parameters_[utils::md_index_2_list<6,6>(4,4)] = mu_();
        material_parameters_[utils::md_index_2_list<6,6>(5,5)] = mu_()
        material_parameters_[utils::md_index_2_list<6,6>(0,1)] = lambda();
        material_parameters_[utils::md_index_2_list<6,6>(0,2)] = lambda();
        material_parameters_[utils::md_index_2_list<6,6>(1,2)] = lambda();
        material_parameters_[utils::md_index_2_list<6,6>(1,0)] = lambda();
        material_parameters_[utils::md_index_2_list<6,6>(2,0)] = lambda();
        material_parameters_[utils::md_index_2_list<6,6>(2,1)] = lambda();
    };

        //proportionality_matrix = material_parameters_;
    }

    

    //static constexpr std::size_t num_parameters = 2;

    




};
#endif // FALL_N_LINEAL_MATERIAL