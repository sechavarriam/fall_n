#ifndef FN_INTEGRATION_POINT
#define FN_INTEGRATION_POINT

#include <initializer_list>
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
    IntegrationPoint(Args&&... args):Point(std::forward<Args>(args)...){};



    // Copy, move constructors and assignment operators
    // IntegrationPoint(const IntegrationPoint& other):Point(other){};
    // IntegrationPoint& operator=(const IntegrationPoint& other){Point::operator=(other); return *this;};
    //IntegrationPoint(IntegrationPoint&& other) noexcept:Point(std::move(other)){};
    //IntegrationPoint& operator=(IntegrationPoint&& other) noexcept{Point::operator=(std::move(other)); return *this;}



    ~IntegrationPoint(){} 
};



#endif