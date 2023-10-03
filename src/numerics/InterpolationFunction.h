#ifndef FN_ABSTRACT_INTERPOLATION_FUNCTION
#define FN_ABSTRACT_INTERPOLATION_FUNCTION

#include <array>
#include <functional>
#include "../domain/Point.h"

template <unsigned short Dim, 
          unsigned short nPoints, 
          typename       PointType,
          typename       DataType  = double, 
          typename       ReturnType= double>
class Interpolant{ // Interpolation Function

  private:
    std::function<ReturnType(PointType)> F_; //Implemented in derived classes (e.g. Element Shape Functions).

    // For optimazation purposes, the points and values shouln't be stored in the interpolant.    
    //std::array<Point<Dim>,nPoints> points_;
    //std::array<ReturnType,nPoints> values_;
  public:      
    ReturnType operator()(PointType x) const {return F_(x);};

    //Constructors

    Interpolant(const std::function<ReturnType(PointType)>& F){
      F_ = F; // Copy F 
    };

    Interpolant(std::array<PointType,nPoints> SampleNodes, std::array<DataType,nPoints> SampleValues){
      // Construct F_ from array of nodes and values.
    };

    Interpolant(const PointType& SampleNodes, const DataType& SampleValues){
      // Construct F_ from SampleNode and SampleValue references (could be the nodes and values of an element)
    };
    
    Interpolant(PointType&& SampleNodes, DataType&& SampleValues){
      // Construct F_ forwarding SampleNodes and SampleValues
    };

    virtual ~Interpolant(){};
};


#endif