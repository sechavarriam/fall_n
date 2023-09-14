#ifndef FN_TENSOR
#define FN_TENSOR

#

#include "../domain/Topology.h"

#include <Eigen/Dense>

typedef unsigned short ushort;
typedef unsigned int   uint  ;


// Compile time swich case.
template<ushort Dim,  ushort Order> requires Topology::EmbeddableInSpace<Dim>
consteval ushort Voigth_Dim1(){
    if constexpr (Dim==1){
        return 1; 
    }else if constexpr(Dim==2){
        return 3;
    }else if constexpr(Dim==3){
        return 6;
    };
};

template<ushort Dim,  ushort Order> requires Topology::EmbeddableInSpace<Dim>
consteval ushort Voigth_Dim2(){
    if constexpr (Dim==1){
        return 1; 
    }else if constexpr(Dim==2){
        return 1;
    }else if constexpr(Dim==3){
        return 6;
    };
};

// ========================================================================
// Voight notation tensor second order tensor (Vector Form)

template<ushort Dim,  ushort Order> 
using VoightTensorContainer = Eigen::Matrix<double, Voigth_Dim1<Dim,Order>(), Voigth_Dim2<Dim,Order>()> ;

template<ushort Dim,  ushort Order> 
class Tensor : VoightTensorContainer<Dim, Order>{
  private:
    
    // More things.

  public:
    template <typename... Args> Tensor(Args... args) : VoightTensorContainer<Dim, Order>(args...) {}
    ~Tensor(){};
};




#endif