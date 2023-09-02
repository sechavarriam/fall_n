#ifndef FN_MATERIAL
#define FN_MATERIAL




#include <Eigen/Dense>
#include <vector>

#include "Strain.h"

typedef unsigned short ushort;
typedef unsigned int   uint  ;

template<ushort nVars> // Dim?
class Material{

    // nVars: Number of state variables (strains)
    private:
        std::vector<Strain<nVars>> strains_t_; //Store history of strains in simulation for memory materials

    public:

    
        

};


#endif