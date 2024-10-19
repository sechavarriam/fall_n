#ifndef FN_ELEMENT_H
#define FN_ELEMENT_H

#include <array>
#include <cstddef>
#include <iostream>
#include <utility>
#include <memory>
#include <vector>
#include <functional>
#include <concepts>

#include <span>
#include <ranges>

#include "../Node.hh"


// HEADER FILES FOR ELEMENTS.
#include "LagrangeElement.hh"

namespace impl
{ // Implementation details

    template <std::size_t dim>
    class ElementGeometryConcept
    { // Define the minimum interface for all element types (of any kind)
    using Array = std::array<double, dim>;
    public:
        virtual ~ElementGeometryConcept() = default;

        virtual std::unique_ptr<ElementGeometryConcept> clone()    const = 0;    // To allow copy construction of the wrapper
        virtual void clone(ElementGeometryConcept *memory_address) const = 0; // Instead of returning a newly instatiated ElementGeometry
                                                                      // we pass the memory address of the ElementGeometry to be
                                                                      // constructed.
    public:
        constexpr virtual void print_nodes_info() const = 0;

        constexpr virtual std::size_t num_nodes() const = 0;
        constexpr virtual std::size_t id() const = 0;

        constexpr virtual Node<dim>& node(std::size_t i) const = 0;

        constexpr virtual std::size_t num_integration_points() const = 0;

        constexpr virtual double H    (std::size_t i,                const Array &X) const = 0;
        constexpr virtual double dH_dx(std::size_t i, std::size_t j, const Array &X) const = 0;

        constexpr virtual double integrate(std::function<double(Array)>&& f) const = 0;
        constexpr virtual Vector integrate(std::function<Vector(Array)>&& f) const = 0;
        constexpr virtual Matrix integrate(std::function<Matrix(Array)>&& f) const = 0; 

    };

    template <typename ElementType, typename IntegrationStrategy> 
    class NON_OwningModel_ElementGeometry;            // Forward declaration                     

    template <typename ElementType, typename IntegrationStrategy> // External Polymorfism Design Pattern
    class OwningModel_ElementGeometry : public ElementGeometryConcept<ElementType::dim>
    { // Wrapper for all element types (of any kind)
        using Array = std::array<double, ElementType::dim>;
        static constexpr auto num_integration_points_ = IntegrationStrategy::num_integration_points;

        ElementType         element_   ; // Stores the ElementGeometry object
        IntegrationStrategy integrator_; // Stores the Integration Strategy object (Spacial integration strategy)
    public:

        static constexpr auto dim = ElementType::dim;
        

        explicit OwningModel_ElementGeometry(ElementType const &element, IntegrationStrategy const &integrator) :
            element_   (element   ),
            integrator_(integrator) {};

        explicit OwningModel_ElementGeometry(ElementType &&element, IntegrationStrategy &&integrator) : 
            element_(std::forward<ElementType>(element)),
            integrator_(std::forward<IntegrationStrategy>(integrator)) {};

        std::unique_ptr<ElementGeometryConcept<dim>> clone() const override{
            return std::make_unique<OwningModel_ElementGeometry<ElementType, IntegrationStrategy>>(*this);
        };

        void clone(ElementGeometryConcept<dim> *memory_address) const override{
            using Model = NON_OwningModel_ElementGeometry<ElementType const, IntegrationStrategy const>;
            std::construct_at(static_cast<Model *>(memory_address), element_, integrator_);
        };

    public: // Implementation of the virtual operations derived from ElementGeometryConcept

        constexpr void print_nodes_info() const override { element_.print_nodes_info(); };

        constexpr std::size_t num_nodes() const override { return element_.num_nodes(); };
        constexpr std::size_t id()        const override { return element_.id(); };

        constexpr Node<dim>& node(std::size_t i) const override { return element_.node(i); };
        
        constexpr std::size_t num_integration_points() const override { return num_integration_points_; };

        constexpr double H(std::size_t i, const Array& X) const override {
             return element_.H(i, X); 
             };
        constexpr double dH_dx(std::size_t i, std::size_t j, const Array& X) const override {
             return element_.dH_dx(i, j, X); 
             };

        constexpr double integrate (std::function<double(Array)>&& f) const override {
            return integrator_(element_,std::forward<std::function<double(Array)>>(f));
        };

        Vector integrate (std::function<Vector(Array)>&& f) const override {
            return integrator_(element_,std::forward<std::function<Vector(Array)>>(f));
        };

        Matrix integrate (std::function<Matrix(Array)>&& f) const override {
            return integrator_(element_,std::forward<std::function<Matrix(Array)>>(f));
        };

    };

    template <typename ElementType, typename IntegrationStrategy> // External Polymorfism Design Pattern
    class NON_OwningModel_ElementGeometry : public ElementGeometryConcept<ElementType::dim>
    { // Reference semantic version of OwningModel_ElementGeometry.
        using Array = std::array<double, ElementType::dim>;
        static constexpr auto num_integration_points_ = IntegrationStrategy::num_integration_points;

        ElementType         *element_   {nullptr}; // Only stores a pointer to the ElementGeometry object (aka NonOwning)
        IntegrationStrategy *integrator_{nullptr}; //
    public:
        
        static constexpr auto dim = ElementType::dim;
        
        
        NON_OwningModel_ElementGeometry(ElementType &element, IntegrationStrategy &integrator)
            : element_   {std::addressof(element)}, // &element
              integrator_{std::addressof(integrator)} {}; // &integrator

        std::unique_ptr<ElementGeometryConcept<dim>> clone() const override{
            using Model = OwningModel_ElementGeometry<ElementType, IntegrationStrategy>;
            return std::make_unique<Model>(*element_, *integrator_);
        };

        void clone(ElementGeometryConcept<dim> *memory_address) const override{
            std::construct_at(static_cast<NON_OwningModel_ElementGeometry *>(memory_address), *this);
        };

    public: // Implementation of the virtual operations derived from ElementGeometryConcept (Accesing pointer members)
        constexpr void print_nodes_info() const override { element_->print_nodes_info(); };
        
        constexpr std::size_t num_nodes() const override { return element_->num_nodes(); };
        constexpr std::size_t id()        const override { return element_->id(); };
        
        constexpr std::size_t num_integration_points() const override { return num_integration_points_; };

        constexpr Node<dim>& node(std::size_t i) const override { return element_->node(i); };

        constexpr double H(std::size_t i, const Array &X) const override { 
            return element_->H(i, X); 
            };

        constexpr double dH_dx(std::size_t i, std::size_t j, const Array& X) const override {
            return element_->dH_dx(i, j, X); 
            };
        
        constexpr double integrate (std::function<double(Array)>&& f) const override {
            return (*integrator_)(*element_,std::forward<std::function<double(Array)>>(f));
        };

        Vector integrate (std::function<Vector(Array)>&& f) const override {
            return (*integrator_)(*element_,std::forward<std::function<Vector(Array)>>(f));
        };

        Matrix integrate (std::function<Matrix(Array)>&& f) const override {
            return (*integrator_)(*element_,std::forward<std::function<Matrix(Array)>>(f));
        };

        //template <std::invocable<Array> F> 
        //constexpr auto integrate(F&& f) const -> std::invoke_result_t<F, Array>{return (*integrator_)(*element_,f);};

    };
} // impl

template<std::size_t dim> class ElementGeometry; // Forward declaration

template<std::size_t dim> 
class ElementGeometryConstRef
{
    friend class ElementGeometry<dim>;
    
    using Array = std::array<double, dim>;

    // Expected size of a model instantiation: sizeof(ShapeT*) + sizeof(DrawStrategy*) + sizeof(vptr)
    static constexpr std::size_t MODEL_SIZE = 3;            // The 3 pointers of the NON_OwningModel_ElementGeometry,
    alignas(void *) std::array<std::byte, MODEL_SIZE> raw_; // Raw storage (Aligned Byte array)

    impl::ElementGeometryConcept<dim>       *pimpl()       { return reinterpret_cast<impl::ElementGeometryConcept<dim> *     >(raw_.data()); };
    impl::ElementGeometryConcept<dim> const *pimpl() const { return reinterpret_cast<impl::ElementGeometryConcept<dim> const*>(raw_.data()); };

public:

    constexpr std::size_t id()        const { return pimpl()->id(); };
    constexpr std::size_t num_nodes() const { return pimpl()->num_nodes(); };

    constexpr Node<dim>& node(std::size_t i) const { return pimpl()->node(i); };

    //constexpr auto nodes() const { return std::span<Node<dim>>{pimpl()->node(0), pimpl()->num_nodes()};}; // UNTESTED // No debe servir. Asume nodos contiguos en memoria.
    
    constexpr std::size_t num_integration_points() const { return pimpl()->num_integration_points; };

    constexpr double H    (std::size_t i, const Array &X) const {return pimpl()->H(i, X);};
    constexpr double dH_dx(std::size_t i, std::size_t j, Array &X ) const {return pimpl()->dH_dx(i, j, X);};

    //constexpr auto integrate(std::invocable<Array> auto&& F) const {return pimpl()->integrator_(std::forward<decltype(F)>(F));};
    constexpr double integrate(std::function<double(Array)>&& f) const {return pimpl()->integrate(std::forward<std::function<double(Array)>>(f));};
    Vector           integrate(std::function<Vector(Array)>&& f) const {return pimpl()->integrate(std::forward<std::function<Vector(Array)>>(f));};
    Matrix           integrate(std::function<Matrix(Array)>&& f) const {return pimpl()->integrate(std::forward<std::function<Matrix(Array)>>(f));};

    template <typename ElementType, typename IntegrationStrategy>
    constexpr ElementGeometryConstRef(ElementType &element, IntegrationStrategy &integrator)
    {
        using Model = impl::NON_OwningModel_ElementGeometry<const ElementType, const IntegrationStrategy>;

        static_assert( sizeof(Model) == MODEL_SIZE, "Invalid Model size"); //(<= ?)
        static_assert(alignof(Model) == alignof(void *), "Model Misaligned");

        std::construct_at(static_cast<Model *>(pimpl()), element, integrator);
    }

    constexpr ElementGeometryConstRef(ElementGeometryConstRef const &other) { other.pimpl()->clone(pimpl()); };

    ElementGeometryConstRef(ElementGeometry<dim> &other);       // Implicit conversion from ElementGeometry to ElementGeometryConstRef
    ElementGeometryConstRef(ElementGeometry<dim> const &other); //

    ElementGeometryConstRef &operator=(ElementGeometryConstRef const &other)
    {
        ElementGeometryConstRef copy{other};
        raw_.swap(copy.raw_);
        return *this;
    };
    constexpr ~ElementGeometryConstRef() { std::destroy_at(pimpl()); }; // OR: ~ElementGeometryConstRef(){pimpl()->~ElementGeometryConcept();};
    // Move operations explicitly not declared

private: // Operations with references

    constexpr inline friend std::size_t id       (ElementGeometryConstRef const &element) { return element.pimpl()->id()       ; };
    constexpr inline friend std::size_t num_nodes(ElementGeometryConstRef const &element) { return element.pimpl()->num_nodes(); };
    constexpr inline friend void print_nodes_info(ElementGeometryConstRef const &element) { element.pimpl()->print_nodes_info(); };

    constexpr inline friend 
    auto integrate(ElementGeometryConstRef const &element, std::invocable<Array> auto&& F) {
        return element.pimpl()->integrate(std::forward<decltype(F)>(F));
        };  
};

template<std::size_t dim>
class ElementGeometry
{
    friend class ElementGeometryConstRef<dim>;
    using Array = std::array<double, dim>;
    
    std::unique_ptr<impl::ElementGeometryConcept<dim>> pimpl_; // Bridge to implementation details (compiler generated).

public:
    constexpr std::size_t id()        const { return pimpl_->id(); };
    constexpr std::size_t num_nodes() const { return pimpl_->num_nodes(); };

    constexpr Node<dim>& node(std::size_t i) const { return pimpl_->node(i); };
    


    constexpr std::size_t num_integration_points() const { return pimpl_->num_integration_points(); };

    constexpr double H    (std::size_t i, const Array &X) const { return pimpl_->H(i, X);};
    constexpr double dH_dx(std::size_t i, std::size_t j,const Array &X) const { return pimpl_->dH_dx(i, j, X);};

    //constexpr auto integrate(std::invocable<Array> auto&& F) const {return pimpl_->integrate(std::forward<decltype(F)>(F));};
    
    constexpr double integrate(std::function<double(Array)>&& f) const {return pimpl_->integrate(std::forward<std::function<double(Array)>>(f));};
    Vector           integrate(std::function<Vector(Array)>&& f) const {return pimpl_->integrate(std::forward<std::function<Vector(Array)>>(f));};
    Matrix           integrate(std::function<Matrix(Array)>&& f) const {return pimpl_->integrate(std::forward<std::function<Matrix(Array)>>(f));};

    template <typename ElementType, typename IntegrationStrategy> // CAN BE CONSTRAINED WITH CONCEPTS!
    constexpr ElementGeometry(ElementType element, IntegrationStrategy integrator)
    {
        using Model = impl::OwningModel_ElementGeometry<ElementType, IntegrationStrategy>;
        pimpl_ = std::make_unique<Model>(
            std::move(element),     // forward perhaphs?=
            std::move(integrator)); //
    }

    ElementGeometry(ElementGeometry const &other) : pimpl_{other.pimpl_->clone()} {};
    ElementGeometry(ElementGeometryConstRef<dim> const &other) : pimpl_{other.pimpl()->clone()} {};

    ElementGeometry &operator=(const ElementGeometryConstRef<dim> &other)
    {
        ElementGeometry copy{other};
        pimpl_.swap(copy.pimpl_);
        return *this;
    };

    constexpr ~ElementGeometry() = default;
    constexpr ElementGeometry(ElementGeometry &&) = default;
    constexpr ElementGeometry &operator=(ElementGeometry &&) = default;

private:
    // -----------------------------------------------------------------------------------------------
    // Hidden Friends (Free Functions)
    // This functions trigger the polymorfic behaviour of the wrapper. They takes the pimpl
    // and call the virtual function do_action() on it. This function is implemented in the
    // ElementModel class, THE REAL ElementType BEHAVIOUR of the erased type.
    // friend void action(ElementGeometry const& element /*, Args.. args*/){element.p_element_impl->do_action(/* args...*/);
    //};

    constexpr inline friend std::size_t        id(ElementGeometry const& element) { return element.pimpl_->id()       ; };
    constexpr inline friend std::size_t num_nodes(ElementGeometry const& element) { return element.pimpl_->num_nodes(); };
    constexpr inline friend void print_nodes_info(ElementGeometry const& element) { element.pimpl_->print_nodes_info(); };

    constexpr inline friend
    auto integrate(ElementGeometry const &element, std::invocable<Array> auto&& F) {
        return element.pimpl_->integrate(std::forward<decltype(F)>(F));
        };

    // -----------------------------------------------------------------------------------------------
};
template<std::size_t dim>
inline ElementGeometryConstRef<dim>::ElementGeometryConstRef(ElementGeometry<dim> &other) { other.pimpl_->clone(pimpl()); };

template<std::size_t dim>
inline ElementGeometryConstRef<dim>::ElementGeometryConstRef(ElementGeometry<dim> const &other) { other.pimpl_->clone(pimpl()); };




#endif // FN_ELEMENT_H
