#ifndef FN_ABSTRACT_STRAIN
#define FN_ABSTRACT_STRAIN


#include "../numerics/Tensor.h"

// Derivatives of displacement Field (DoF)
// nVars: Number of model state variables (strains)

template<unsigned short nVars> // Dim?
class Strain{
    private:
        //Container.
        //Eigen::Matrix<double, nVars, 1> data_ = Eigen::Matrix<double, nVars, 1>::Zero();

    public:
        
    Strain(){};
    virtual ~Strain(){};
};


#endif