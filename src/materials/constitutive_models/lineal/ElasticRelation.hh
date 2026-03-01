
#ifndef FALL_N_CONSTUTUTIVE_LINEAL_RELATION
#define FALL_N_CONSTUTUTIVE_LINEAL_RELATION


#include <iostream>
#include <cstddef>
#include <type_traits>
#include <concepts>

#include "../../MaterialPolicy.hh"
#include "../../MaterialState.hh"

#include "../../../numerics/linear_algebra/Matrix.hh"
#include "../../../utils/index.hh"

// TODO: ElasticRelation<KinematicMeasure, TensionalConjugate>
template<class MaterialPolicy>
class ElasticRelation{

  public:
    using StrainT        = typename MaterialPolicy::StrainT;
    using StressT        = typename MaterialPolicy::StressT;
    
    using MaterialStateT = MaterialState<ElasticState,StrainT>;
    using StateVariableT = typename MaterialStateT::StateVariableT; //Esto es una tupla! o un strain solamente.
                                                                    //En este caso el StateVariableT es un StrainT.
                                                                    //Poner un consteval assert!
    
    using MatrixT = Eigen::Matrix<double, StrainT::num_components, StressT::num_components>;

    static constexpr std::size_t dim               = StrainT::dim;
    static constexpr std::size_t num_strains_      = StrainT::num_components;
    static constexpr std::size_t num_stresses_     = StressT::num_components;
    static constexpr std::size_t total_parameters_ = num_stresses_*num_strains_;

  private:
    std::array<double, total_parameters_> compliance_parameters_{0.0};
  
  public:

    MatrixT compliance_matrix = MatrixT::Zero();

    StressT compute_stress(const StrainT& strain){ 
        StressT stress;  
        stress.set_components(compliance_matrix*strain.vector()); // sigma = C*epsilon
        return stress;
    };

    void compute_stress(StressT& stress, const StrainT& strain){ ///Esto debe ser del relation o del material con el update_strategy?
        stress.set_components(compliance_matrix*strain.vector());    // sigma = C*epsilon
    };

    // =========== CONSTRUCTORS ==========================  
    constexpr ElasticRelation(){};
    constexpr ~ElasticRelation() = default;

};
    
// ============================================================================================================
// ========================== SPECIALIZATIONS =================================================================
// ============================================================================================================



template<> //Specialization for 1D stress (Uniaxial Stress) avoiding array overhead
class ElasticRelation<UniaxialMaterial>{

  public:
  using MaterialPolicy = UniaxialMaterial;
  
  using StrainT = Strain<1>;
  using StressT = Stress<1>;

  using MaterialStateT = MaterialState<ElasticState,StrainT>;
  using StateVariableT = StrainT;

  static constexpr std::size_t dim               = StrainT::dim;
  static constexpr std::size_t num_strains_      = StrainT::num_components;
  static constexpr std::size_t num_stresses_     = StressT::num_components;
  static constexpr std::size_t total_parameters_ = num_stresses_*num_strains_;
  
private:

    double E_{0.0}; // E

public:

    void compute_stress(const Strain<1>& strain, Stress<1>& stress){
        stress.set_components(E_ * strain.components()); // sigma = E*epsilon
    };

    
    constexpr inline void set_parameter    (double value)        {E_ = value;};
    constexpr inline void update_elasticity(double young_modulus){E_ = young_modulus;};

    constexpr ElasticRelation(double young_modulus) : E_{young_modulus}{};

    // =========== CONSTRUCTORS ==========================
    constexpr  ElasticRelation() = default;
    constexpr ~ElasticRelation() = default;

    // =========== TESTING FUNCTIONS ============================
    void print_constitutive_parameters() const{
        std::cout << "Proportionality Compliance Parameter (Young): " << E_ << std::endl;
    };

};

#endif // FALL_N_LINEAL_MATERIAL