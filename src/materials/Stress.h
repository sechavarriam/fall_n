#ifndef FN_ABSTRACT_STRESS
#define FN_ABSTRACT_STRESS

#include <Eigen/Dense>


// Derivatives of displacement Field (DoF)
// nVars: Number of model state variables (strains)

template<unsigned short nVars> // Dim?
class Stress{
    private:
        //Container.
        Eigen::Matrix<double, nVars, 1> data_ = Eigen::Matrix<double, nVars, 1>::Zero();

    public:
        
    Stress(){};
    virtual ~Stress(){};
};


#endif