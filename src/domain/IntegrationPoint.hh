#ifndef FN_INTEGRATION_POINT
#define FN_INTEGRATION_POINT

#include <concepts>
#include <initializer_list>
#include <iostream> 
#include <memory>

#include "../geometry/Point.hh"
#include "../geometry/Topology.hh"
#include "../materials/Material.hh"

template<std::size_t dim> 
class IntegrationPoint{// : public geometry::Point<dim>{
  
    using Point = geometry::Point<dim>;

    Point Point_; // auto is_derived from Point (or something like that... )

    std::shared_ptr<Material> material_; // O tal vez no un punterno sino una instancia.
                                         // El puntero es util en caso de un material lineal... Puesto que no se repetirían instancias.
                                        
                                         // Another type erased class with strategies (this!)   - 

    
      
      
          //std::unique_ptr<Material> material_;
  public:
   
   double coord(std::size_t i) const {return Point_.coord( i );};


    // Constructors

    IntegrationPoint(){};  

    template<std::floating_point... Args>
    IntegrationPoint(Args&&... args) : Point_(std::forward<Args>(args)...){};



    // Copy, move constructors and assignment operators
    // IntegrationPoint(const IntegrationPoint& other):Point(other){};
    // IntegrationPoint& operator=(const IntegrationPoint& other){Point::operator=(other); return *this;};
    //IntegrationPoint(IntegrationPoint&& other) noexcept:Point(std::move(other)){};
    //IntegrationPoint& operator=(IntegrationPoint&& other) noexcept{Point::operator=(std::move(other)); return *this;}



    ~IntegrationPoint(){} 
};



#endif