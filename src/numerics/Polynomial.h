#ifndef FALL_N_POLYNOMIAL
#define FALL_N_POLYNOMIAL

#include <array>
#include <type_traits>
#include <ranges>
#include <concepts>

template<typename T,typename X>
concept Multipliable = requires(T t, X x){
    {t*x} -> std::convertible_to<T>;
};

//Compile time version
template<typename T, T c0, T ... coefs> requires (std::is_arithmetic_v<T>)
class Polynomial {
  private:

    static constexpr unsigned short order_ = sizeof...(coefs);
    static constexpr std::array<T, order_+1> coeff_ = {c0,coefs...};
  
  public:
    Polynomial(){};
    ~Polynomial(){};

    template<typename X> 
    requires Multipliable<T,X>
    constexpr auto operator()(X x){//Horner's Method
        T fx = coeff_[0];
        for(const auto& c : coeff_ | std::views::drop(1)){
            fx = fx*x + c;
        }
        return fx;
    }
};


#endif