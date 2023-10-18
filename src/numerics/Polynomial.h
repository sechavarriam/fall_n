#ifndef FALL_N_POLYNOMIAL
#define FALL_N_POLYNOMIAL

#include <array>
#include <iostream>
#include <type_traits>
#include <ranges>
#include <concepts>

#include "Vector.h"
#include "Matrix.h"

template<typename T,typename X>
concept Multipliable = requires(T t, X x){
    {t*x} -> std::convertible_to<X>;
};

template<typename T,typename X>
concept Addable = requires(T t, X x){
    {t+x} -> std::convertible_to<X>;
};

// P(x) = cn x^n + cn-1 x^n-1 + ... + c1 x + c0
template<typename T, T cn, T ... coefs> requires (std::is_arithmetic_v<T>)
class Polynomial {
  private:
    static constexpr unsigned short order_ = sizeof...(coefs);
    static constexpr std::array<T, order_+1> coeff_ = {cn,coefs...};
  
  public:
    Polynomial(){};
    ~Polynomial(){};

    template<typename X> 
    //requires std::is_arithmetic_v<X> 
    constexpr X operator()(X x){//Horner's Method
        if constexpr (std::is_arithmetic_v<X>){
            auto fx = coeff_[0]*X(1);
            for(const auto& c : coeff_|std::views::drop(1)) fx = fx*x + c*X(1);
            return fx;
        }else{ //Eigen Types
            X fx = coeff_[0]*X::Ones();
            for(const auto& c : coeff_|std::views::drop(1)) fx = fx.cwiseProduct(x)+c*X::Ones();
            return fx;
        }
    }
    
};



//template <typename X>
//consteval auto identity(){
//    if constexpr ( std::is_pointer_interconvertible_base_of_v<X,Matrix<1,1>>){
//        return X::Identity();
//    }else{
//        return X(1);
//    }
//};


#endif