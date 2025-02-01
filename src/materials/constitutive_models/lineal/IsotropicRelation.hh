
#ifndef FALL_N_CONSTUTUTIVE_ISOTROPIC_LINEAL_RELATION
#define FALL_N_CONSTUTUTIVE_ISOTROPIC_LINEAL_RELATION


#include "ElasticRelation.hh"

class ContinuumIsotropicRelation : public ElasticRelation<ThreeDimensionalMaterial> {

  public:
    using StrainType = typename ElasticRelation<ThreeDimensionalMaterial>::StrainType;
    using StressType = typename ElasticRelation<ThreeDimensionalMaterial>::StressType;

    using ConstitutiveModel = ElasticRelation<ThreeDimensionalMaterial>;
    using ConstitutiveModel::compliance_matrix;

  private:

  public:

    
    // https://stackoverflow.com/questions/9864125/c11-how-to-alias-a-function
    double E{0.0};
    double v{0.0};

    constexpr inline void set_E(double E_){E = E_;};
    constexpr inline void set_v(double v_){v = v_;};
    
    //constexpr inline double E() const{return E;};
    //constexpr inline double v() const{return v;};

    constexpr inline double c11 () const{return E*(1-v)/((1.0+v)*(1.0-2.0*v));}
    constexpr inline double c12 () const{return E*   v /((1.0+v)*(1.0-2.0*v));}

    
    constexpr inline double G()      const{return E/(2.0*(1.0+v));};             //Shear Modulus
    constexpr inline double k()      const{return E/(3.0*(1.0-2.0*v));};         //Bulk Modulus
    constexpr inline double lambda() const{return E*v/((1.0+v)*(1.0-2.0*v));}; //Lamé's first parameter
    constexpr inline double mu()     const{return E/(2.0*(1.0+v));};             //Lamé's second parameter


    constexpr void update_elasticity()
    {
      this->compliance_matrix(0,0) = c11();             //lambda()+2.0*mu());
      this->compliance_matrix(1,1) = c11();             //lambda()+2.0*mu());
      this->compliance_matrix(2,2) = c11();             //lambda()+2.0*mu());
      this->compliance_matrix(3,3) = (c11()-c12())/2;   //mu()             );
      this->compliance_matrix(4,4) = (c11()-c12())/2;   //mu()             );
      this->compliance_matrix(5,5) = (c11()-c12())/2;   //mu()             );
      this->compliance_matrix(0,1) = c12();//0.000000         ); // ONLY FOR TESTING !REMOVE when done!
      this->compliance_matrix(0,2) = c12();//0.000000         ); // ONLY FOR TESTING !REMOVE when done!
      this->compliance_matrix(1,0) = c12();//0.000000         ); // ONLY FOR TESTING !REMOVE when done!
      this->compliance_matrix(1,2) = c12();//0.000000         ); // ONLY FOR TESTING !REMOVE when done!
      this->compliance_matrix(2,0) = c12();//0.000000         ); // ONLY FOR TESTING !REMOVE when done!
      this->compliance_matrix(2,1) = c12();//0.000000         ); // ONLY FOR TESTING !REMOVE when done! 
    }

    //constexpr void update_elasticity(){ //TODO: use threads. 
    //    set_parameter(0,0,  c11());             //lambda()+2.0*mu());
    //    set_parameter(1,1,  c11());             //lambda()+2.0*mu());
    //    set_parameter(2,2,  c11());             //lambda()+2.0*mu());
    //    set_parameter(3,3,  (c11()-c12())/2);   //mu()             );
    //    set_parameter(4,4,  (c11()-c12())/2);   //mu()             );
    //    set_parameter(5,5,  (c11()-c12())/2);   //mu()             );
    //    set_parameter(0,1,  c12());//0.000000         ); // ONLY FOR TESTING !REMOVE when done! 
    //    set_parameter(0,2,  c12());//0.000000         ); // ONLY FOR TESTING !REMOVE when done!
    //    set_parameter(1,0,  c12());//0.000000         ); // ONLY FOR TESTING !REMOVE when done!
    //    set_parameter(1,2,  c12());//0.000000         ); // ONLY FOR TESTING !REMOVE when done!
    //    set_parameter(2,0,  c12());//0.000000         ); // ONLY FOR TESTING !REMOVE when done!
    //    set_parameter(2,1,  c12());//0.000000         ); // ONLY FOR TESTING !REMOVE when done!
    //    //set_parameter(0,1, lambda()         );
    //    //set_parameter(0,2, lambda()         );
    //    //set_parameter(1,0, lambda()         );
    //    //set_parameter(1,2, lambda()         );
    //    //set_parameter(2,0, lambda()         );
    //    //set_parameter(2,1, lambda()         );
    //};

    constexpr void update_elasticity(double young_modulus, double poisson_ratio){
        E = young_modulus;
        v = poisson_ratio;
        update_elasticity();
    };

    constexpr ContinuumIsotropicRelation(double young_modulus, double poisson_ratio) : E{young_modulus}, v{poisson_ratio}{
        update_elasticity();
    };
};


typedef ElasticRelation<UniaxialMaterial> UniaxialIsotropicRelation;


#endif // FALL_N_CONSTUTUTIVE_ISOTROPIC_LINEAL_RELATION
