#ifndef FN_MATERIAL
#define FN_MATERIAL


#include <Eigen/Dense>
#include <vector>

#include "../numerics/Tensor.hh"
#include "Strain.hh"

typedef unsigned short ushort;
typedef unsigned int   uint  ;

template<ushort Dim, ushort Order> // Dim?
class Material{

    // nVars: Number of state variables (strains)
    private:
        
        Tensor<Dim,Order> strain;

        std::vector<Tensor<Dim,Order>> strain_t_; //Store history of strains in simulation for memory materials


    public:
        void preallocate_strain_t_(){};
    
        

};


#endif