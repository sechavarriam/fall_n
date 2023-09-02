#ifndef FN_ABSTRACT_STRAIN
#define FN_ABSTRACT_STRAIN

#include <Eigen/Dense>

typedef unsigned short ushort;
typedef unsigned int   uint  ;


    // Derivatives of displacement Field
    // nVars: Number of model state variables (strains)

template<ushort nVars> // Dim?
class Strain{
    private:
        //Container.
        Eigen::Matrix<double, nVars, 1> data_ = Eigen::Matrix<double, nVars, 1>::Zero();

    public:
        

};


#endif