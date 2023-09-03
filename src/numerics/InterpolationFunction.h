#ifndef FN_ABSTRACT_INTERPOLATION_FUNCTION
#define FN_ABSTRACT_INTERPOLATION_FUNCTION

template<typename T> //T should be Point<Dim> or derived.
class InterpolationFunction{

  public:

    virtual double operator()(T point) = 0;

    InterpolationFunction(){};
    virtual ~InterpolationFunction(){};
};


#endif