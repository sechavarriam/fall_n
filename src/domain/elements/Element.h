#ifndef FN_ELEMENT_H
#define FN_ELEMENT_H

#include <array>
#include <cstddef>
#include <utility>
#include <memory>
#include <vector>
#include <functional>

#include <span>
#include <ranges>

typedef unsigned short ushort;
typedef unsigned int   uint  ;

/*
/////class Element{
/////
/////    // External hiyerarchy for element types 
/////    class ElementConcept{
/////        public:
/////        //https://stackoverflow.com/questions/3141087/what-is-meant-with-const-at-end-of-function-declaration  
/////        //                                                |        
/////        //                                                v   
/////        virtual std::unique_ptr<ElementConcept> clone() const = 0; // To allow copy construction of the wrapper
/////                                                                   // (Prototype Design Pattern)
/////        virtual ~ElementConcept() = default;
/////
/////        ////VIRTUAL OPERATIONS =======================================
/////        //virtual void do_action (/*Args.. args) const = 0; 
/////
/////        constexpr virtual ushort get_num_nodes() const = 0;
/////        constexpr virtual ushort get_num_dof()   const = 0;
/////
/////        virtual uint          get_id()    const = 0;
/////        virtual ushort const* get_nodes() const = 0;
/////        ////==========================================================
/////    };
/////    template <typename ElementType> //(External Polymorfism Design Pattern).
/////    class ElementModel: public ElementConcept{ // Wrapper for all element types (of any kind)
/////        public:
/////
/////        ElementType element_; // Stores the Element object
/////
/////        std::unique_ptr<ElementConcept> clone() const override {
/////            return std::make_unique<ElementModel<ElementType>>(*this);
/////        };        
/////
/////        ElementModel(ElementType&& element) : element_(std::forward<ElementType>(element)){}; // to test.
/////        ~ElementModel(){};
/////
/////        // IMPLEMENTATION OF VIRTUAL OPERATIONS ===============================================
/////        // void do_action (/*Args.. args*///) const override{action(element_/*, args...*/);};
/////
/////        ushort constexpr get_num_nodes()const override {return element_.num_nodes();};
/////        ushort constexpr get_num_dof()  const override {return element_.num_dof()  ;};
/////        
/////        uint get_id() const override {return element_.id()   ;};
/////
/////        ushort const* get_nodes() const override {return element_.nodes();};
/////
/////        //unsigned short* get_nodes() const override {return element_.nodes();};
/////        //=====================================================================================
/////    };
/////    // -----------------------------------------------------------------------------------------------
/////    // Hidden Friends (Free Functions)
/////    //friend void action(Element const& element /*, Args.. args*/){element.p_element_impl->do_action(/* args...*/);
/////    //};
/////    friend uint   id       (Element const& element){return element.p_element_impl->get_id()       ;};
/////    friend ushort num_nodes(Element const& element){return element.p_element_impl->get_num_nodes();};
/////    friend ushort num_dof  (Element const& element){return element.p_element_impl->get_num_dof()  ;};
/////
/////    friend auto nodes(Element const& element){
/////        return std::span{
/////            element.p_element_impl->get_nodes(),    //Pointer to data
/////            element.p_element_impl->get_num_nodes() //Size of the span
/////            };
/////        };
/////
/////    // This functions trigger the polymorfic behaviour of the wrapper. They takes the pimpl
/////    // and call the virtual function do_action() on it. This function is implemented in the
/////    // ElementModel class, THE REAL ElementType BEHAVIOUR of the erased type.  
/////    // -----------------------------------------------------------------------------------------------
/////
/////    // Pointer to base. (Bridge Design Pattern)
/////    std::unique_ptr<ElementConcept> p_element_impl; //pimpl (Pointer to implementation) idiom
/////  
/////  public:
/////  // Templated constructor which takes ANY kind of element and deduces the type!
/////    template <typename ElementType> //Bridge to possible impementation details (compiler generated).
/////    Element(ElementType&& element) : p_element_impl{
/////        std::make_unique<ElementModel<ElementType>>(std::forward<ElementType>(element))
/////    } {};
/////    
/////    // This constructors take and element of ElementType and build an ElementModel, 
/////    // in which the element is stored "erasing the type of the element". We only have 
/////    // a pointer to base (ElementConcept)
/////
/////    // -----------------------------------------------------------------------------------------------
/////    // Copy Operations (in terms of clone)
/////    Element(Element const& other)
/////        : p_element_impl{other.p_element_impl->clone()} {};
/////
/////    Element& operator=(Element const& other){ //Copy and Swap idiom
/////        other.p_element_impl->clone().swap(p_element_impl);
/////        return *this;
/////        };
/////
/////    // Move Operations (Thera are several possible implementations) 
/////    Element(Element&& other) noexcept = default;
/////    Element& operator=(Element&& other) noexcept = default;
/////    // -----------------------------------------------------------------------------------------------
/////};


// REFACTOR

namespace impl{ //Implementation details

class ElementConcept{
  public:
    virtual ~ElementConcept() = default;  
    //virtual void do_action(/*Args.. args*/) const = 0;

    virtual std::unique_ptr<ElementConcept> clone() const = 0;   // To allow copy construction of the wrapper
    virtual void clone(ElementConcept* memory_address) const = 0;// Instead of returning a newly instatiated Element
                                                                 // we pass the memory address of the Element to be
                                                                 // constructed. 
};  

template <typename ElementType, typename IntegrationStrategy> class NonOwningElementModel; //Forward declaration

template <typename ElementType, typename IntegrationStrategy> //External Polymorfism Design Pattern
class OwningElementModel: public ElementConcept{ // Wrapper for all element types (of any kind)
  
    ElementType         element_;    // Stores the Element object
    IntegrationStrategy integrator_; // Stores the Integration Strategy object (Spacial integration strategy)

  public:
    explicit OwningElementModel(ElementType&& element, IntegrationStrategy&& integrator) 
        : element_   (std::forward<ElementType>        (element)),
          integrator_(std::forward<IntegrationStrategy>(integrator)){}; // to test.

    std::unique_ptr<ElementConcept> clone() const override {
        return std::make_unique<OwningElementModel<ElementType,IntegrationStrategy>>(*this);
    };

    void clone(ElementConcept* memory_address) const{ // override{ ? 
        using Model = NonOwningElementModel<ElementType const,IntegrationStrategy const>;

        std::construct_at(static_cast<Model*>(memory_address),element_,integrator_);
//or:// auto* ptr = const_cast<void*>(static_cast<void const volatile*>(memory));
     // ::new (ptr) Model( element_, integrator_ );
    }
};

template <typename ElementType, typename IntegrationStrategy> //External Polymorfism Design Pattern
class NonOwningElementModel: public ElementConcept{ // Reference semantic version of OwningElementModel.
    ElementType*         element_{nullptr};    // Only stores a pointer to the Element object (aka NonOwning)
    IntegrationStrategy* integrator_{nullptr}; // 

  public:
    NonOwningElementModel(ElementType& element, IntegrationStrategy& integrator) 
        : element_   {std::addressof(element)  }     // &element
        , integrator_{std::addressof(integrator)}{}; // &integrator

    std::unique_ptr<ElementConcept> clone() const override {
        using Model = OwningElementModel<ElementType,IntegrationStrategy>;
        return std::make_unique<Model>(*element_,*integrator_);
    };

    void clone( ElementConcept* memory_address) const override{
        std::construct_at(static_cast<NonOwningElementModel*>(memory_address),*this);      
//or:// auto* ptr = const_cast<void*>(static_cast<void const volatile*>(memory));
     // ::new (ptr) NonOwningElementModel( *this );
    };
};

} //impl

class Element; // Forward declaration

class ElementConstRef{
    friend class Element;

    // Expected size of a model instantiation: sizeof(ShapeT*) + sizeof(DrawStrategy*) + sizeof(vptr)
    static constexpr std::size_t MODEL_SIZE = 3; // The 3 pointers of the NonOwningElementModel,
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
        using Model = impl::NonOwningElementModel<const ElementType,const IntegrationStrategy>;
        
        static_assert(sizeof(Model)  == MODEL_SIZE    , "Invalid Model size"); //(<= ?) 
        static_assert(alignof(Model) == alignof(void*), "Model Misaligned"  ); 

        std::construct_at(static_cast<Model*>(pimpl()),element,integrator);
    };

    ElementConstRef(ElementConstRef const& other){other.pimpl()->clone(pimpl());};

    ElementConstRef(Element& other);      // Forwarding constructor. Implicit conversion from Element to ElementConstRef
    ElementConstRef(Element const& other);// Forwarding constructor.  

    ElementConstRef& operator=(ElementConstRef const& other){
        ElementConstRef copy{other};
        raw_.swap(copy.raw_);
        return *this;
    };

    ~ElementConstRef(){std::destroy_at(pimpl());}; // OR: ~ElementConstRef(){pimpl()->~ElementConcept();};
    // Move operations explicitly not declared


  private:
  //friends implementations
};

class Element{
    friend class ElementConstRef;

    std::unique_ptr<impl::ElementConcept> pimpl_; // Bridge to implementation details (compiler generated).

  public:
    template<typename ElementType, typename IntegrationStrategy>
    Element(ElementType&& element, IntegrationStrategy&& integrator)
        : pimpl_{std::make_unique<impl::OwningElementModel<ElementType,IntegrationStrategy>>(
            std::forward<ElementType>(element),
            std::forward<IntegrationStrategy>(integrator)
        )} {};

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
    // Friend, implementations
};

ElementConstRef::ElementConstRef(Element&       other) {other.pimpl_->clone(pimpl());};
ElementConstRef::ElementConstRef(Element const& other) {other.pimpl_->clone(pimpl());};




















//void do_on_element(SomeElementType const& element, /*Args.. args*/){
//    //Do something with the element
//};
//
//void action_on_containter_of_elements(std::vector<Element> const& elems){
//    for (auto const& e: elems){
//        do_on_element(some_element); // some_element of SomeElementType.
//    }
//};











#endif


















