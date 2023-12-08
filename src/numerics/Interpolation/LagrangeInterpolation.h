
#ifndef FN_LAGRANGE_INTERPOLATION_H
#define FN_LAGRANGE_INTERPOLATION_H

#include <array>
#include <ranges>
#include <numeric>

template <unsigned short nPoints>
class LagrangianInterpolant{
    
    std::array<double,nPoints> xPoints_;
    std::array<double,nPoints> yValues_;

  public:
    
    constexpr double points() const {return xPoints_;};
    constexpr double values() const {return yValues_;};

    constexpr LagrangianInterpolant(std::array<double,nPoints> xPoints, std::array<double,nPoints> yValues)
        : xPoints_(xPoints), yValues_(yValues){};

    constexpr double operator()(double x) const
    {
        return std::transform_reduce(std::ranges::begin(xPoints_), std::ranges::end(xPoints_), 0.0, std::plus{}, [this, x](double xi) {
            auto L = std::transform_reduce(std::ranges::begin(xPoints_), std::ranges::end(xPoints_), 1.0, std::multiplies{}, [x, xi](double xj) {
                return xi != xj ? (x - xj) / (xi - xj) : 1.0;
            });
            return yValues_[std::ranges::distance(std::ranges::begin(xPoints_), std::ranges::find(xPoints_, xi))] * L;
        });
    }

    constexpr ~LagrangianInterpolant(){};

};



#endif // __LAGRANGE_INTERPOLATION_H__
