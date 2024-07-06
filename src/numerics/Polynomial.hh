#ifndef FALL_N_POLYNOMIAL
#define FALL_N_POLYNOMIAL

#include <array>
#include <iostream>
#include <iterator>
#include <type_traits>
#include <ranges>
#include <concepts>
#include <algorithm>
#include <numeric>


#include <variant>
#include <any>

//#include "Vector.hh"
//#include "Matrix.hh"

template<typename T,typename X>
concept Multipliable = requires(T t, X x){
    {t*x} -> std::convertible_to<X>;
};

template<typename T,typename X>
concept Addable = requires(T t, X x){
    {t+x} -> std::convertible_to<X>;
};

template<typename T>
concept Scalar = std::is_arithmetic_v<T>; //|| std::is_pointer_interconvertible_base_of_v<T,Matrix<1,1>>;

template<Scalar T, Scalar... U>
struct largest_scalar_type{
    using type = std::common_type_t<T,U...>;
};

// P(x) = c0 + c1 x + c2 x^2 + ... + cn x^n
template<Scalar T, T... coefs>
class Polynomial {
  private:
    static constexpr std::array coeff_{coefs...};
  public:

    //default constructor
    constexpr Polynomial(){};
    ~Polynomial(){};

    //move constructor
    Polynomial(Polynomial&&) = default;
    //copy constructor
    Polynomial(const Polynomial&) = default;
    //move assignment
    Polynomial& operator=(Polynomial&&) = default;
    //copy assignment
    Polynomial& operator=(const Polynomial&) = default;


    template<typename X> 
    constexpr X operator()(X x){//Horner's Method
        if constexpr (std::is_arithmetic_v<X>){
            return std::accumulate( //From last to first
                std::next(coeff_.rbegin()),   // Skip the first element ............... (The last.... cn)
                coeff_.rend(),                // End of the range ..................... (The first... c0)
                static_cast<X>(coeff_.back()),// Initial value casted to X type ........(cn)
                [&x](const X& fx, const auto& c){return fx*x + c;} //Lambda function, Horner's method (captures x)
            );
        }
        //else{ //Eigen Types (TODO: Impreve efficiency!)
        //    X fx = coeff_.back()*X::Ones();
        //    for(const auto& c : coeff_|std::views::drop(1)) fx = fx.cwiseProduct(x)+c*X::Ones();
        //    return fx-X::Ones(); //(TODO: Impreve efficiency!)
        //}
    }
};


// Function evaluation without creating a Polynomial Functor (Doestn't work for Eigen Types yet)
// Doest't store the coefficients.

// P(x) = c0 + c1 x + c2 x^2 + ... + cn x^n
//template<typename X, Scalar T>
//inline constexpr X poly_eval(X x, T cn){return X(cn);}

template<typename X, Scalar T, Scalar... Ts>
inline constexpr X poly_eval(X x, T cn, Ts... coefs){ 
    return cn + (x*poly_eval(x, coefs...));
    }

#endif