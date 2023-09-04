#ifndef FN_ABSTRACT_INTERPOLATION_FUNCTION
#define FN_ABSTRACT_INTERPOLATION_FUNCTION

template<typename T, typename P> //T should be Point<Dim> or derived.
class InterpolationFunction{

  public:
    virtual T operator()(P point) = 0;

    

    InterpolationFunction(){};
    virtual ~InterpolationFunction(){};
};


#endif