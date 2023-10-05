#ifndef FN_ELEMENT_H
#define FN_ELEMENT_H

#include <array>
#include <cstddef>
#include <utility>
#include <memory>
#include <vector>
#include <functional>

typedef unsigned short ushort;
typedef unsigned int   uint  ;

class Element{
    // External hiyerarchy for element types 
    class ElementConcept{
        public:
        //https://stackoverflow.com/questions/3141087/what-is-meant-with-const-at-end-of-function-declaration  
        //                                                |        
        //                                                v   
        virtual std::unique_ptr<ElementConcept> clone() const = 0; // To allow copy construction of the wrapper
                                                                   // (Prototype Design Pattern)
        virtual ~ElementConcept() = default;

        ////VIRTUAL OPERATIONS =======================================
        //virtual void do_action (/*Args.. args*/) const = 0; 

        constexpr virtual ushort get_num_nodes() const = 0;
        constexpr virtual ushort get_num_dof()   const = 0;

        virtual uint          get_id()    const = 0;
        virtual ushort const& get_nodes() const = 0;
        ////==========================================================
    };
    template <typename ElementType> //(External Polymorfism Design Pattern).
    class ElementModel: public ElementConcept{ // Wrapper for all element types (of any kind)
        public:

        // POSIBLE CAMINO"""""""""
        static constexpr ushort num_nodes = ElementType::Num_nodes;
        // =========================

        ElementType element_; // Stores the Element object

        std::unique_ptr<ElementConcept> clone() const override {
            return std::make_unique<ElementModel<ElementType>>(*this);
        };        

        //ElementModel(ElementType   element) : element_{std::move(element)}{};
        ElementModel(ElementType&& element) : element_(std::forward<ElementType>(element)){}; // to test.
        ~ElementModel(){};

        // IMPLEMENTATION OF VIRTUAL OPERATIONS ==================
        // void do_action (/*Args.. args*/) const override {
        //     action(element_/*, args...*/);  
        // };

        ushort constexpr get_num_nodes()const override {return element_.num_nodes();};
        ushort constexpr get_num_dof()  const override {return element_.num_dof()  ;};
        
        uint get_id() const override {return element_.id()   ;};

        ushort const& get_nodes() const override {return element_.nodes();};

        //unsigned short* get_nodes() const override {return element_.nodes();};
        //========================================================
    };
    // -----------------------------------------------------------------------------------------------
    // Hidden Friends (Free Functions)
    //friend void action(Element const& element /*, Args.. args*/){
    //    element.p_element_impl->do_action(/* args...*/);
    //};
    friend uint   id       (Element const& element){return element.p_element_impl->get_id()       ;};
    friend constexpr ushort num_nodes(Element const& element){return element.p_element_impl->get_num_nodes();};
    friend constexpr ushort num_dof  (Element const& element){return element.p_element_impl->get_num_dof()  ;};

    //template<std::size_t N>
    //friend std::array<ushort,N> nodes(Element const& element){
    //    return std::array{element.p_element_impl->get_nodes()[element.p_element_impl->get_num_nodes()]};
    //    };

    friend auto nodes(Element const& element){
        ushort N = element.p_element_impl->get_num_nodes();
        return element.p_element_impl->get_nodes();
        };

    //friend unsigned short* nodes(Element const& element){
    //    return element.p_element_impl->get_nodes();
    //};
    // This functions trigger the polymorfic behaviour of the wrapper. They takes the pimpl
    // and call the virtual function do_action() on it. This function is implemented in the
    // ElementModel class, THE REAL ElementType BEHAVIOUR of the erased type.  
    // -----------------------------------------------------------------------------------------------

    // Pointer to base. (Bridge Design Pattern)
    std::unique_ptr<ElementConcept> p_element_impl; //pimpl (Pointer to implementation) idiom
  
  public:
  // Templated constructor which takes ANY kind of element and deduces the type!
    template <typename ElementType> //Bridge to possible impementation details (compiler generated).
    Element(ElementType&& element) : p_element_impl{
        std::make_unique<ElementModel<ElementType>>(std::forward<ElementType>(element))
    } {};

    /*
    template <typename ElementType>
    Element(ElementType element) : p_element_impl{
            std::make_unique<ElementModel<ElementType>>(std::move(element))
            }{};
    */
    //    ^ ^ ^ 
    //    | | | 
    // This constructors take and element of ElementType and construct an ElementModel, 
    // in which the element is stored "erasing the type of the element". We only have 
    // a pointer to base (ElementConcept)

    // -----------------------------------------------------------------------------------------------
    // Copy Operations (in terms of clone)
    Element(Element const& other)
        : p_element_impl{other.p_element_impl->clone()} {};

    Element& operator=(Element const& other){ //Copy and Swap idiom
        other.p_element_impl->clone().swap(p_element_impl);
        return *this;
        };

    // Move Operations (Thera are several possible implementations) 
    Element(Element&& other) noexcept = default;
    Element& operator=(Element&& other) noexcept = default;
    // -----------------------------------------------------------------------------------------------
};

// PARTICULAR IMPLEMENTATIONS OF VIRTUAL OPERATIONS ==========
//
// Example: (implement in .cpp file)  
// Only EXTERNAL POLYMORFISM is needed.
//void action_on_ALLelements(std::vector<std::unique_ptr<ElementConcept>> const& elems){
//    for (auto const& e: elems){
//        e->do_action(/*args...*/);
//    }
//};
//

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


















