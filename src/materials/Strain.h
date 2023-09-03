#ifndef FN_ABSTRACT_STRAIN
#define FN_ABSTRACT_STRAIN

#include <Eigen/Dense>


// Derivatives of displacement Field (DoF)
// nVars: Number of model state variables (strains)

template<unsigned short nVars> // Dim?
class Strain{
    private:
        //Container.
        Eigen::Matrix<double, nVars, 1> data_ = Eigen::Matrix<double, nVars, 1>::Zero();

    public:
        

};


#endif