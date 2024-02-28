#ifndef FN_TENSOR
#define FN_TENSOR

#include <utility>
#include <concepts>

#include "../geometry/Topology.hh"

#include "linear_algebra/Matrix.hh"
#include "linear_algebra/Vector.hh"


// Compile time swich case.
template<std::size_t dim> requires topology::EmbeddableInSpace<dim>
consteval std::size_t Voigth_Dim1(){
    if      constexpr(dim==1){return 1;}
    else if constexpr(dim==2){return 3;}
    else if constexpr(dim==3){return 6;};
};

template<std::size_t order> requires topology::EmbeddableInSpace<order>
consteval std::size_t Voigth_Dim2(){
    if      constexpr(order==1){return 1;}  //Order 0 tensor
    else if constexpr(order==2){return 1;}  //Order 2 tensor
    else if constexpr(order==3){return 6;}; //Order 4 tensor
};

/*
template<std::size_t dim,  std::size_t order> 
using VoightTensorContainer = Matrix<Voigth_Dim1<dim>(), Voigth_Dim2<order>()> ;
// ========================================================================
// Voight notation tensor second order tensor (Vector Form)


template<std::size_t dim,  std::size_t order> 
class Tensor: public VoightTensorContainer<dim, order>{
  private:

    // More things.

  public:

    //This method allows you to assign Eigen expressions to Tensor
    //template<typename EigenDerived>
    //Tensor& operator=(const Eigen::MatrixBase <EigenDerived>& other){
    //    this->VoightTensorContainer<dim, order>::operator=(other);
    //    return *this;
    //}//https://eigen.tuxfamily.org/dox-devel/TopicCustomizing_InheritingMatrix.html

    using VoightTensorContainer<dim, order>::operator=;

    template <typename... Args>
    Tensor(Args&&... args):VoightTensorContainer<dim, order>(std::forward<Args>(args)...)
    {};

    ~Tensor(){};
};
*/ 



#endif