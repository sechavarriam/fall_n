#ifndef FN_ELEMENT_H
#define FN_ELEMENT_H

#include <array>
#include <cstddef>
#include <iostream>
#include <utility>
#include <memory>
#include <vector>
#include <functional>

#include <span>
#include <ranges>

typedef unsigned short ushort;
typedef unsigned int   uint  ;

namespace impl{ //Implementation details

    class ElementConcept{
    public:
        virtual ~ElementConcept() = default;  
        //virtual void do_action(/*Args.. args*/) const = 0;
        
        virtual void compute_integral(/*function2integrate?*/) const = 0; //Maybe double? auto?

        virtual std::unique_ptr<ElementConcept> clone() const = 0;   // To allow copy construction of the wrapper
        virtual void clone(ElementConcept* memory_address) const = 0;// Instead of returning a newly instatiated Element
                                                                    // we pass the memory address of the Element to be
                                                                    // constructed. 
    public: 
        constexpr virtual ushort get_num_nodes() const = 0;
        constexpr virtual ushort get_num_dof()   const = 0;

        virtual uint          get_id()    const = 0;
        virtual ushort const* get_nodes() const = 0;
    };  

    template <typename ElementType, typename IntegrationStrategy> 
    class NON_OwningElementModel; //Forward declaration

    template <typename ElementType, typename IntegrationStrategy> //External Polymorfism Design Pattern
    class OwningElementModel: public ElementConcept{ // Wrapper for all element types (of any kind) 
        
        ElementType         element_;    // Stores the Element object
        IntegrationStrategy integrator_; // Stores the Integration Strategy object (Spacial integration strategy)

    public:
        void compute_integral(/*function2integrate?*/) const override {integrator_(element_);}; //Maybe double?

        explicit OwningElementModel(ElementType element, IntegrationStrategy integrator) 
            : element_   (std::move(element)),
              integrator_(std::move(integrator)){};

        std::unique_ptr<ElementConcept> clone() const override {
            return std::make_unique<OwningElementModel<ElementType,IntegrationStrategy>>(*this);
        };

        void clone(ElementConcept* memory_address) const override{ // override{ ? 
            using Model = NON_OwningElementModel<ElementType const,IntegrationStrategy const>;
            std::construct_at(static_cast<Model*>(memory_address),element_,integrator_);
        };

    public: // Implementation of the virtual operations derived from ElementConcept
        ushort constexpr get_num_nodes()const override {return element_.num_nodes();};
        ushort constexpr get_num_dof()  const override {return element_.num_dof()  ;};
        
        uint          get_id()    const override {return element_.id()   ;};
        ushort const* get_nodes() const override {return element_.nodes();};
    };

    template <typename ElementType, typename IntegrationStrategy> //External Polymorfism Design Pattern
    class NON_OwningElementModel: public ElementConcept{ // Reference semantic version of OwningElementModel.
        ElementType*         element_{nullptr};    // Only stores a pointer to the Element object (aka NonOwning)
        IntegrationStrategy* integrator_{nullptr}; // 

    public:
        void compute_integral(/*function2integrate?*/) const override {(*integrator_)(*element_);}; //Maybe double?

        NON_OwningElementModel(ElementType& element, IntegrationStrategy& integrator) 
            : element_   {std::addressof(element)  }     // &element
            , integrator_{std::addressof(integrator)}{}; // &integrator

        std::unique_ptr<ElementConcept> clone() const override {
            using Model = OwningElementModel<ElementType,IntegrationStrategy>;
            return std::make_unique<Model>(*element_,*integrator_);
        };

        void clone( ElementConcept* memory_address) const override{
            std::construct_at(static_cast<NON_OwningElementModel*>(memory_address),*this);      
        };

    public:  // Implementation of the virtual operations derived from ElementConcept (Accesing pointer members)
        ushort constexpr get_num_nodes()const override {return element_->num_nodes();};
        ushort constexpr get_num_dof()  const override {return element_->num_dof()  ;};
        
        uint          get_id()    const override {return element_->id()   ;};
        ushort const* get_nodes() const override {return element_->nodes();};
    };
} //impl

class Element; // Forward declaration
class ElementConstRef{
    friend class Element;

    // Expected size of a model instantiation: sizeof(ShapeT*) + sizeof(DrawStrategy*) + sizeof(vptr)
    static constexpr std::size_t MODEL_SIZE = 3; // The 3 pointers of the NON_OwningElementModel,
    alignas(void*) std::array<std::byte,MODEL_SIZE> raw_; //Raw storage (Aligned Byte array)

    impl::ElementConcept* pimpl(){
        return reinterpret_cast<impl::ElementConcept*>(raw_.data());
    };

    impl::ElementConcept const* pimpl() const{
        return reinterpret_cast<impl::ElementConcept const*>(raw_.data());
    };

  public:
    template<typename ElementType, typename IntegrationStrategy>
    ElementConstRef(ElementType& element, IntegrationStrategy& integrator){
        using Model = impl::NON_OwningElementModel<const ElementType,const IntegrationStrategy>;
        
        static_assert(sizeof(Model)  == MODEL_SIZE    , "Invalid Model size"); //(<= ?) 
        static_assert(alignof(Model) == alignof(void*), "Model Misaligned"  ); 

        std::construct_at(static_cast<Model*>(pimpl()),element,integrator);
    };

    ElementConstRef(ElementConstRef const& other){other.pimpl()->clone(pimpl());};

    ElementConstRef(Element& other);      // Implicit conversion from Element to ElementConstRef
    ElementConstRef(Element const& other);//  

    ElementConstRef& operator=(ElementConstRef const& other){
        ElementConstRef copy{other};
        raw_.swap(copy.raw_);
        return *this;
    };

    ~ElementConstRef(){std::destroy_at(pimpl());}; // OR: ~ElementConstRef(){pimpl()->~ElementConcept();};
    // Move operations explicitly not declared

  private: //Operations with references
    friend void integrate(ElementConstRef const& element){
        element.pimpl()->compute_integral(/*args...*/);
        std::cout << "Element " << element.pimpl()->get_id() << " integrated" << std::endl;
        };

};

class Element{
    friend class ElementConstRef;
    std::unique_ptr<impl::ElementConcept> pimpl_; // Bridge to implementation details (compiler generated).

  public:
    template<typename ElementType, typename IntegrationStrategy>
    Element(ElementType element, IntegrationStrategy integrator){
        using Model = impl::OwningElementModel<ElementType,IntegrationStrategy>;
        pimpl_ = std::make_unique<Model>(
            std::move(element),
            std::move(integrator));
    };

    Element( Element         const& other) : pimpl_{other.pimpl_ ->clone()} {};
    Element( ElementConstRef const& other) : pimpl_{other.pimpl()->clone()} {};

    Element& operator=(const ElementConstRef& other){
        Element copy{other};
        pimpl_.swap(copy.pimpl_);
        return *this;
    };

    ~Element() = default;
    Element(Element &&) = default;
    Element& operator=(Element &&) = default;

  private:
    // -----------------------------------------------------------------------------------------------
    // Hidden Friends (Free Functions)

    friend void integrate(Element const& element){
        element.pimpl_->compute_integral(/*args...*/);
        std::cout << "Element " << element.pimpl_->get_id() << " integrated" << std::endl;
        };

    //friend void action(Element const& element /*, Args.. args*/){element.p_element_impl->do_action(/* args...*/);
    //};
    friend uint   id       (Element const& element){return element.pimpl_->get_id()       ;};
    friend ushort num_nodes(Element const& element){return element.pimpl_->get_num_nodes();};
    friend ushort num_dof  (Element const& element){return element.pimpl_->get_num_dof()  ;};
     
    friend auto nodes(Element const& element){
        return std::span{
            element.pimpl_->get_nodes(),    //Pointer to data
            element.pimpl_->get_num_nodes() //Size of the span
            };
        };
     
    // This functions trigger the polymorfic behaviour of the wrapper. They takes the pimpl
    // and call the virtual function do_action() on it. This function is implemented in the
    // ElementModel class, THE REAL ElementType BEHAVIOUR of the erased type.  
    // -----------------------------------------------------------------------------------------------
};

inline ElementConstRef :: ElementConstRef(Element&       other) {other.pimpl_->clone(pimpl());};
inline ElementConstRef :: ElementConstRef(Element const& other) {other.pimpl_->clone(pimpl());};


#endif //FN_ELEMENT_H
