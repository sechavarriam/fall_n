#ifndef FN_INTEGRATION_POINT
#define FN_INTEGRATION_POINT

#include <iostream> 

#include "../geometry/Point.hh"
#include "../geometry/Topology.hh"

template<std::size_t dim> 
class IntegrationPoint : public geometry::Point<dim>{
  
    using Point = geometry::Point<dim>;

  private:

  public:

    IntegrationPoint(){};  

    template<typename... Args>
    IntegrationPoint(Args&&... args) : Point(std::forward<Args>(args)...){}

    ~IntegrationPoint(){} 
};



#endif