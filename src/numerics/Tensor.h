#ifndef FN_TENSOR
#define FN_TENSOR

#include <utility>
#include <concepts>

#include "../geometry/Topology.h"

#include "Matrix.h"
#include "Vector.h"


typedef unsigned short ushort;
typedef unsigned int   uint  ;

// Compile time swich case.
template<ushort Dim,  ushort Order> requires topology::EmbeddableInSpace<Dim>
consteval ushort Voigth_Dim1(){
    if constexpr (Dim==1){
        return 1; 
    }else if constexpr(Dim==2){
        return 3;
    }else if constexpr(Dim==3){
        return 6;
    };
};

template<ushort Dim,  ushort Order> requires topology::EmbeddableInSpace<Dim>
consteval ushort Voigth_Dim2(){
    if constexpr (Order==1){
        return 1; 
    }else if constexpr(Order==2){
        return 1;
    }else if constexpr(Order==3){
        return 6;
    };
};

template<ushort Dim,  ushort Order> 
using VoightTensorContainer = Matrix<Voigth_Dim1<Dim,Order>(), Voigth_Dim2<Dim,Order>()> ;
// ========================================================================
// Voight notation tensor second order tensor (Vector Form)


template<ushort Dim,  ushort Order> 
class Tensor: public VoightTensorContainer<Dim, Order>{
  private:

    // More things.

  public:

    //This method allows you to assign Eigen expressions to Tensor
    //template<typename EigenDerived>
    //Tensor& operator=(const Eigen::MatrixBase <EigenDerived>& other){
    //    this->VoightTensorContainer<Dim, Order>::operator=(other);
    //    return *this;
    //}//https://eigen.tuxfamily.org/dox-devel/TopicCustomizing_InheritingMatrix.html

    using VoightTensorContainer<Dim, Order>::operator=;

    template <typename... Args>
    Tensor(Args&&... args):VoightTensorContainer<Dim, Order>(std::forward<Args>(args)...)
    {};

    ~Tensor(){};
};




#endif