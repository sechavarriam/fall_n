
#ifndef FALL_N_CONSTUTUTIVE_ISOTROPIC_LINEAL_RELATION
#define FALL_N_CONSTUTUTIVE_ISOTROPIC_LINEAL_RELATION


#include "LinealRelation.hh"

//template<std::size_t dim> requires topology::EmbeddableInSpace<dim>
//consteval std::size_t Voigt_Dim(){
//    if      constexpr(dim==1){return 1;}
//    else if constexpr(dim==2){return 3;}
//    else if constexpr(dim==3){return 6;};
//};
//
//
//template<std::size_t N>
//class IsotropicRelation : public LinealRelation<VoigtStress<N>, VoigtStrain<N>>{
//};



class ContinuumIsotropicRelation : public LinealRelation<VoigtStress<6>, VoigtStrain<6>> {
    
    // https://stackoverflow.com/questions/9864125/c11-how-to-alias-a-function
    double E_{0.0};
    double v_{0.0};

    public:
    constexpr inline void set_E(double E){E_ = E;};
    constexpr inline void set_v(double v){v_ = v;};
    
    constexpr inline double E() const{return E_;};
    constexpr inline double v() const{return v_;};
    
    constexpr inline double G()      const{return E_/(2.0*(1.0+v_));};             //Shear Modulus
    constexpr inline double k()      const{return E_/(3.0*(1.0-2.0*v_));};         //Bulk Modulus
    constexpr inline double lambda() const{return E_*v_/((1.0+v_)*(1.0-2.0*v_));}; //Lamé's first parameter
    constexpr inline double mu()     const{return E_/(2.0*(1.0+v_));};             //Lamé's second parameter

    constexpr void update_elasticity(){ //TODO: use threads. 
        set_parameter(0,0, lambda()+2.0*mu());
        set_parameter(1,1, lambda()+2.0*mu());
        set_parameter(2,2, lambda()+2.0*mu());
        set_parameter(3,3, mu()             );
        set_parameter(4,4, mu()             );
        set_parameter(5,5, mu()             );
        set_parameter(0,1, lambda()         );
        set_parameter(0,2, lambda()         );
        set_parameter(1,0, lambda()         );
        set_parameter(1,2, lambda()         );
        set_parameter(2,0, lambda()         );
        set_parameter(2,1, lambda()         );
    };

    constexpr void update_elasticity(double young_modulus, double poisson_ratio){
        E_ = young_modulus;
        v_ = poisson_ratio;
        update_elasticity();
    };

    constexpr ContinuumIsotropicRelation(double young_modulus, double poisson_ratio) : E_{young_modulus}, v_{poisson_ratio}{
        update_elasticity();
    };
};


typedef LinealRelation<VoigtStress<1>, VoigtStrain<1>> UniaxialIsotropicRelation;



#endif // FALL_N_CONSTUTUTIVE_ISOTROPIC_LINEAL_RELATION
