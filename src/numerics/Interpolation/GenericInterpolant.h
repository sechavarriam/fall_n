#ifndef FN_GENERIC_INTERPOLATION_FUNCTION_INTERFASE
#define FN_GENERIC_INTERPOLATION_FUNCTION_INTERFASE

#include <array>
#include <memory>
#include <functional>

//template <unsigned short Dim, 
//          unsigned short nPoints, 
//          typename       PointType,
//          typename       DataType  = double, 
//          typename       ReturnType= double>
class Interpolant{ // Type Erased Interpolation Function
    
    class InterpolantConcept{
      public:
      virtual std::unique_ptr<InterpolantConcept> clone() const = 0;
      virtual ~InterpolantConcept() = default;

      //virtual ReturnType operator()(PointType x) const = 0;
    };

    template <typename InterpolantType>
    class InterpolantModel : public InterpolantConcept{
    
      public:
        InterpolantType interpolant_;

        std::unique_ptr<InterpolantConcept> clone() const override {
          return std::make_unique<InterpolantModel<InterpolantType>>(*this);
        };

      InterpolantModel(InterpolantType&& interpolant)
       : interpolant_(std::forward<InterpolantType>(interpolant)){};

      ~InterpolantModel(){};
    
    };

    std::unique_ptr<InterpolantConcept> p_interpolant_impl;

  public:

    template <typename InterpolantType>
    Interpolant(InterpolantType&& interpolant)
      : p_interpolant_impl(
        std::make_unique<InterpolantModel<InterpolantType>>(
          std::forward<InterpolantType>(interpolant)
          )
        ){};

    Interpolant(const Interpolant& other)
      : p_interpolant_impl(other.p_interpolant_impl->clone()){};

    Interpolant& operator=(const Interpolant& other){
      other.p_interpolant_impl->clone().swap(p_interpolant_impl);
      return *this;
    };  

    Interpolant(Interpolant&& other) noexcept = default;
    Interpolant& operator=(Interpolant&& other) noexcept = default;

    
    
    ~Interpolant() = default;

    //OVERLOAD () OPERATOR
   // Hidden implementation Friends
};




    //std::function<ReturnType(PointType)> F_; //Implemented in derived classes (e.g. Element Shape Functions).

    // For optimazation purposes, the points and values shouln't be stored in the interpolant.    
    //std::array<Point<Dim>,nPoints> points_;
    //std::array<ReturnType,nPoints> values_;
//  public:     
//    ReturnType operator()(PointType x) const {return F_(x);};
//
//    //Constructors
//
//    Interpolant(const std::function<ReturnType(PointType)>& F){
//      F_ = F; // Copy F 
//    };
//
//    Interpolant(std::array<PointType,nPoints> SampleNodes, std::array<DataType,nPoints> SampleValues){
//      // Construct F_ from array of nodes and values.
//    };
//
//    Interpolant(const PointType& SampleNodes, const DataType& SampleValues){
//      // Construct F_ from SampleNode and SampleValue references (could be the nodes and values of an element)
//    };
//    
//    Interpolant(PointType&& SampleNodes, DataType&& SampleValues){
//      // Construct F_ forwarding SampleNodes and SampleValues
//    };
//
//    virtual ~Interpolant(){};


#endif