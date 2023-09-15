#ifndef FN_TENSOR
#define FN_TENSOR


#include <utility>
#include <concepts>
#include <type_traits>


#include "../domain/Topology.h"

#include <Eigen/Dense>
#include <Eigen/Core>

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
    if constexpr (Order==1){
        return 1; 
    }else if constexpr(Order==2){
        return 1;
    }else if constexpr(Order==3){
        return 6;
    };
};


template<ushort Dim,  ushort Order> 
using VoightTensorContainer 
    = Eigen::Matrix<double, Voigth_Dim1<Dim,Order>(), Voigth_Dim2<Dim,Order>()> ;
// ========================================================================
// Voight notation tensor second order tensor (Vector Form)



// https://eigen.tuxfamily.org/dox-devel/TopicCustomizing_InheritingMatrix.html

template<ushort Dim,  ushort Order> 
class Tensor: public VoightTensorContainer<Dim, Order>{
  private:

    // More things.

  public:


    //===========================================================================
    // Constructor
    Tensor(void):VoightTensorContainer<Dim, Order>(){}
    // This constructor allows you to construct Tensor from Eigen expressions

    template<typename OtherDerived>
    Tensor(const Eigen::MatrixBase<OtherDerived>& other)
        : VoightTensorContainer<Dim, Order>(other)
    { }
 
    // This method allows you to assign Eigen expressions to Tensor
    template<typename OtherDerived>
    Tensor& operator=(const Eigen::MatrixBase <OtherDerived>& other)
    {
        this->VoightTensorContainer<Dim, Order>::operator=(other);
        return *this;
    }
    //===========================================================================

    template <typename... Args>
    Tensor(Args&&... args):VoightTensorContainer<Dim, Order>(std::forward<Args>(args)...)
    {};

    ~Tensor(){};
};




#endif