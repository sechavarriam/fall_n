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

#include "../../integrator/MaterialIntegrator.hh"

// HEADER FILES FOR ELEMENTS.
#include "LagrangeElement.hh"

namespace impl

{ // Implementation details

    class ElementGeometryConcept
    { // Define the minimum interface for all element types (of any kind)
    public:
        virtual ~ElementGeometryConcept() = default;
        // virtual void do_action(/*Args.. args*/) const = 0;
        
        //virtual double compute_integral(std::function<double(double)>)             const = 0;
        /*  
        virtual double compute_integral(std::function<double(geometry::Point<3>  )>) const = 0;
        virtual double compute_integral(std::function<double(std::array<double,3>)>) const = 0;      
        */

        virtual std::unique_ptr<ElementGeometryConcept> clone() const = 0;    // To allow copy construction of the wrapper
        virtual void clone(ElementGeometryConcept *memory_address) const = 0; // Instead of returning a newly instatiated ElementGeometry
                                                                      // we pass the memory address of the ElementGeometry to be
                                                                      // constructed.
    public:
        constexpr virtual void print_nodes_info() const = 0;
        constexpr virtual std::size_t num_nodes() const = 0;
        constexpr virtual std::size_t id() const = 0;
        // constexpr virtual void set_num_dofs() const = 0;
    };

    template <typename ElementType, typename IntegrationStrategy> //, typename MaterialPolicy> ???
    class NON_OwningModel_ElementGeometry;                                 // Forward declaration

    template <typename ElementType, typename IntegrationStrategy> // External Polymorfism Design Pattern
    class OwningModel_ElementGeometry : public ElementGeometryConcept
    { // Wrapper for all element types (of any kind)

        ElementType         element_   ; // Stores the ElementGeometry object
        IntegrationStrategy integrator_; // Stores the Integration Strategy object (Spacial integration strategy)

    public:
        /*
                double compute_integral(std::function<double(double)> F) const override {return integrator_(element_, F);};
                double compute_integral(std::function<double(geometry::Point<3>  )> F) const override {return integrator_(element_, F);};
                double compute_integral(std::function<double(std::array<double,3>)> F) const override {return integrator_(element_, F);};
        */

        explicit OwningModel_ElementGeometry(ElementType const &element, IntegrationStrategy const &integrator) : element_(element),
                                                                                                         integrator_(integrator) {};

        explicit OwningModel_ElementGeometry(ElementType &&element, IntegrationStrategy &&integrator) : element_(std::forward<ElementType>(element)),
                                                                                               integrator_(std::forward<IntegrationStrategy>(integrator)) {};

        std::unique_ptr<ElementGeometryConcept> clone() const override
        {
            return std::make_unique<OwningModel_ElementGeometry<ElementType, IntegrationStrategy>>(*this);
        };

        void clone(ElementGeometryConcept *memory_address) const override
        {
            using Model = NON_OwningModel_ElementGeometry<ElementType const, IntegrationStrategy const>;
            std::construct_at(static_cast<Model *>(memory_address), element_, integrator_);
        };

    public: // Implementation of the virtual operations derived from ElementGeometryConcept
        constexpr std::size_t num_nodes() const override { return element_.num_nodes(); };
        constexpr std::size_t id()        const override { return element_.id(); };
        constexpr void print_nodes_info() const override { element_.print_nodes_info(); };
        // void set_material_integrator() const override {element_.set_material_integrator();};
    };

    template <typename ElementType, typename IntegrationStrategy> // External Polymorfism Design Pattern
    class NON_OwningModel_ElementGeometry : public ElementGeometryConcept
    { // Reference semantic version of OwningModel_ElementGeometry.

        ElementType         *element_   {nullptr}; // Only stores a pointer to the ElementGeometry object (aka NonOwning)
        IntegrationStrategy *integrator_{nullptr}; //
    public:
        // void compute_integral(std::invocable auto F) const {return (*integrator_)(*element_,F);};
        // double compute_integral(std::function<double()> F) const {return (*integrator_)(*element_,F);};
        /*
                double compute_integral(std::function<double(double)> F) const override               {return (*integrator_)(*element_,F);};
                double compute_integral(std::function<double(geometry::Point<3>  )> F) const override {return (*integrator_)(*element_,F);};
                double compute_integral(std::function<double(std::array<double,3>)> F) const override {return (*integrator_)(*element_,F);};
        */

        NON_OwningModel_ElementGeometry(ElementType &element, IntegrationStrategy &integrator)
            : element_   {std::addressof(element)}, // &element
              integrator_{std::addressof(integrator)} {}; // &integrator

        std::unique_ptr<ElementGeometryConcept> clone() const override
        {
            using Model = OwningModel_ElementGeometry<ElementType, IntegrationStrategy>;
            return std::make_unique<Model>(*element_, *integrator_);
        };

        void clone(ElementGeometryConcept *memory_address) const override
        {
            std::construct_at(static_cast<NON_OwningModel_ElementGeometry *>(memory_address), *this);
        };

    public: // Implementation of the virtual operations derived from ElementGeometryConcept (Accesing pointer members)
        constexpr std::size_t num_nodes() const override { return element_->num_nodes(); };
        constexpr std::size_t id()        const override { return element_->id(); };
        constexpr void print_nodes_info() const override { element_->print_nodes_info(); };

        // void set_material_integrator() const override {element_->set_material_integrator();};
        // std::size_t const* nodes() const override {return element_->nodes();};
    };
} // impl

class ElementGeometry; // Forward declaration
class ElementGeometryConstRef
{
    friend class ElementGeometry;

    // Expected size of a model instantiation: sizeof(ShapeT*) + sizeof(DrawStrategy*) + sizeof(vptr)
    static constexpr std::size_t MODEL_SIZE = 3;            // The 3 pointers of the NON_OwningModel_ElementGeometry,
    alignas(void *) std::array<std::byte, MODEL_SIZE> raw_; // Raw storage (Aligned Byte array)

    impl::ElementGeometryConcept       *pimpl()       { return reinterpret_cast<impl::ElementGeometryConcept *      >(raw_.data()); };
    impl::ElementGeometryConcept const *pimpl() const { return reinterpret_cast<impl::ElementGeometryConcept const *>(raw_.data()); };

public:
    template <typename ElementType, typename IntegrationStrategy>
    ElementGeometryConstRef(ElementType &element, IntegrationStrategy &integrator)
    {
        using Model = impl::NON_OwningModel_ElementGeometry<const ElementType, const IntegrationStrategy>;

        static_assert( sizeof(Model) == MODEL_SIZE, "Invalid Model size"); //(<= ?)
        static_assert(alignof(Model) == alignof(void *), "Model Misaligned");

        std::construct_at(static_cast<Model *>(pimpl()), element, integrator);
    }

    ElementGeometryConstRef(ElementGeometryConstRef const &other) { other.pimpl()->clone(pimpl()); };

    ElementGeometryConstRef(ElementGeometry &other);       // Implicit conversion from ElementGeometry to ElementGeometryConstRef
    ElementGeometryConstRef(ElementGeometry const &other); //

    ElementGeometryConstRef &operator=(ElementGeometryConstRef const &other)
    {
        ElementGeometryConstRef copy{other};
        raw_.swap(copy.raw_);
        return *this;
    };
    ~ElementGeometryConstRef() { std::destroy_at(pimpl()); }; // OR: ~ElementGeometryConstRef(){pimpl()->~ElementGeometryConcept();};
                                                      // Move operations explicitly not declared

private: // Operations with references
    // friend void integrate(ElementGeometryConstRef const& element){
    //     element.pimpl()->compute_integral(element,/*args...*/);
    //     std::cout << "ElementGeometry " << element.pimpl()->id() << " integrated" << std::endl;
    //     };

    /*
        friend double integrate(ElementGeometryConstRef const& element, std::function<double(double)> F){return element.pimpl()->compute_integral(F);};
        friend double integrate(ElementGeometryConstRef const& element, std::function<double(geometry::Point<3>  )> F){return element.pimpl()->compute_integral(F);};
        friend double integrate(ElementGeometryConstRef const& element, std::function<double(std::array<double,3>)> F){return element.pimpl()->compute_integral(F);};
    */

    friend std::size_t id       (ElementGeometryConstRef const &element) { return element.pimpl()->id()       ; };
    friend std::size_t num_nodes(ElementGeometryConstRef const &element) { return element.pimpl()->num_nodes(); };
    friend void print_nodes_info(ElementGeometryConstRef const &element) { element.pimpl()->print_nodes_info(); };
};

class ElementGeometry
{
    friend class ElementGeometryConstRef;
    std::unique_ptr<impl::ElementGeometryConcept> pimpl_; // Bridge to implementation details (compiler generated).

    std::size_t id() const { return pimpl_->id(); };
    std::size_t num_nodes() const { return pimpl_->num_nodes(); };
    std::size_t num_dofs() const { return num_nodes() * 3; };

public:
    template <typename ElementType, typename IntegrationStrategy> // CAN BE CONSTRAINED WITH CONCEPTS!
    ElementGeometry(ElementType element, IntegrationStrategy integrator)
    {
        using Model = impl::OwningModel_ElementGeometry<ElementType, IntegrationStrategy>;
        pimpl_ = std::make_unique<Model>(
            std::move(element),     // forward perhaphs?=
            std::move(integrator)); //
    }

    ElementGeometry(ElementGeometry const &other) : pimpl_{other.pimpl_->clone()} {};
    ElementGeometry(ElementGeometryConstRef const &other) : pimpl_{other.pimpl()->clone()} {};

    ElementGeometry &operator=(const ElementGeometryConstRef &other)
    {
        ElementGeometry copy{other};
        pimpl_.swap(copy.pimpl_);
        return *this;
    };

    ~ElementGeometry() = default;
    ElementGeometry(ElementGeometry &&) = default;
    ElementGeometry &operator=(ElementGeometry &&) = default;

private:
    // -----------------------------------------------------------------------------------------------
    // Hidden Friends (Free Functions)
    // This functions trigger the polymorfic behaviour of the wrapper. They takes the pimpl
    // and call the virtual function do_action() on it. This function is implemented in the
    // ElementModel class, THE REAL ElementType BEHAVIOUR of the erased type.
    // friend void action(ElementGeometry const& element /*, Args.. args*/){element.p_element_impl->do_action(/* args...*/);
    //};

    /*
        friend double integrate(ElementGeometry const& element, std::function<double(double)> F){return element.pimpl_->compute_integral(F);};
        friend double integrate(ElementGeometry const& element, std::function<double(geometry::Point<3>  )> F){return element.pimpl_->compute_integral(F);};
        friend double integrate(ElementGeometry const& element, std::function<double(std::array<double,3>)> F){return element.pimpl_->compute_integral(F);};
    */

    friend std::size_t id(ElementGeometry const &element) { return element.pimpl_->id(); };
    friend std::size_t num_nodes(ElementGeometry const &element) { return element.pimpl_->num_nodes(); };
    friend void print_nodes_info(ElementGeometry const &element) { element.pimpl_->print_nodes_info(); };

    // -----------------------------------------------------------------------------------------------
};

inline ElementGeometryConstRef ::ElementGeometryConstRef(ElementGeometry &other) { other.pimpl_->clone(pimpl()); };
inline ElementGeometryConstRef ::ElementGeometryConstRef(ElementGeometry const &other) { other.pimpl_->clone(pimpl()); };

#endif // FN_ELEMENT_H
