
#ifndef FALL_N_LINEAL_MATERIAL
#define FALL_N_LINEAL_MATERIAL


#include <cstddef>
#include <type_traits>
#include <concept>

#include "Stress.hh"
#include "Strain.hh"


namespace material{


//Some concept
template<Stress StressPolicy, Strain StrainPolicy> //Continuum, Uniaxial, Plane, etc. 
class LinealMaterial{ //Or materialBase


    private:
        std::array components_{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        

};


}



#endif // FALL_N_LINEAL_MATERIAL