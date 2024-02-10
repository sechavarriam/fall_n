#ifndef FN_INTEGRATION_POINT
#define FN_INTEGRATION_POINT

#include <concepts>
#include <initializer_list>
#include <iostream> 
#include <memory>

#include "../geometry/Point.hh"
#include "../geometry/Topology.hh"
#include "../materials/Material.hh"


//https://www.sandordargo.com/blog/2023/04/12/vector-of-unique-pointers

template<std::size_t dim> 
class IntegrationPoint{// : public geometry::Point<dim>{
  
    using Point = geometry::Point<dim>;

    std::unique_ptr<Point> Point_;
    std::shared_ptr<Material> material_; // O tal vez no un punterno sino una instancia.
                                         // El puntero es util en caso de un material lineal... Puesto que no se repetirían instancias.
                                        
                                         // Another type erased class with strategies (this!)   - 

          //std::unique_ptr<Material> material_;
  public:
    double coord(std::size_t i) const {return Point_->coord( i );};


    template<std::floating_point... Args>
    IntegrationPoint(Args&&... args) : Point_(std::make_unique<Point>(std::forward<Args>(args)...)){};


    // Point and Material provided
    IntegrationPoint(Point&& p, Material&& m):
         Point_   (std::make_unique<Point>   (std::move(p))),
         material_(std::make_shared<Material>(std::move(m))){};

    IntegrationPoint(const Point& p, const Material& m):
         Point_   (std::make_unique<Point>   (p)),
         material_(std::make_shared<Material>(m)){};
    
    IntegrationPoint(std::unique_ptr<Point> p, Material m):
         Point_   (std::move(p)),
         material_(std::make_shared<Material>(std::move(m))){};

    //Only Point provided
    IntegrationPoint(Point&& p):
         Point_(std::make_unique<Point>(std::move(p))){};

    IntegrationPoint(const Point& p):
         Point_(std::make_unique<Point>(p)){};

    IntegrationPoint(std::unique_ptr<Point> p):
         Point_(std::move(p)){};

    //Other IntegrarionPoint provided (copy and move constructors and assignment operators)
    IntegrationPoint(const IntegrationPoint& other):
            Point_   (std::make_unique<Point>()),
            material_(other.material_){};

    IntegrationPoint(IntegrationPoint&& other) = default;

    IntegrationPoint& operator=(const IntegrationPoint& other){
        Point_    = std::make_unique<Point>();
        material_ = other.material_;
        return *this;
    };

    IntegrationPoint& operator=(IntegrationPoint&& other) = default;

    // Constructors

    IntegrationPoint() =  default;  

    
    
    //std::forward<Args>(args)...){};



    // Copy, move constructors and assignment operators
    // IntegrationPoint(const IntegrationPoint& other):Point(other){};
    // IntegrationPoint& operator=(const IntegrationPoint& other){Point::operator=(other); return *this;};
    //IntegrationPoint(IntegrationPoint&& other) noexcept:Point(std::move(other)){};
    //IntegrationPoint& operator=(IntegrationPoint&& other) noexcept{Point::operator=(std::move(other)); return *this;}

    ~IntegrationPoint() = default;
};



#endif