#ifndef FN_GENERIC_INTERPOLATION_FUNCTION_INTERFASE
#define FN_GENERIC_INTERPOLATION_FUNCTION_INTERFASE

#include <memory>

#include<concepts>



namespace interpolation{

//Concept of generic interpolant
template <typename Funtor, typename PointType, typename ReturnType>
concept Interpolant = requires(Funtor f, PointType x)
{
    {f(x)} -> std::convertible_to<ReturnType>;
    f.points();
    f.values();
};
\

}


#endif