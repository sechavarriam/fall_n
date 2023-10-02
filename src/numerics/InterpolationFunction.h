#ifndef FN_ABSTRACT_INTERPOLATION_FUNCTION
#define FN_ABSTRACT_INTERPOLATION_FUNCTION

#include <functional>
#include "../domain/Point.h"


template <typename ReturnType, unsigned short Dim, unsigned short nPoints>
class InterpolationFunction{

  private:

    std::function<ReturnType(Point<Dim>)> function_;

    //std::array<Point<Dim>,nPoints> points_;
    //std::array<ReturnType,nPoints> values_;

  public:  
    InterpolationFunction(){};
    virtual ~InterpolationFunction(){};
};


#endif