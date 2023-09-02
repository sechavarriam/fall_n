#ifndef FN_CONSTITUTIVE_RELATION
#define FN_CONSTITUTIVE_RELATION




#include <Eigen/Dense>

typedef unsigned short ushort;
typedef unsigned int   uint  ;

template<ushort nVars> // Dim?
class ConstitutiveRelation{

    // nVars: Number of state variables (strains)
    private:
        Eigen::Matrix<double, nVars, 1> strain_ = Eigen::Matrix<double, nVars, 1>::Zero();

    public:
        
    //TODO: OVERLOAD OPERATOR()

};


#endif