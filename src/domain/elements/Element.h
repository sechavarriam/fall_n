#ifndef FN_ELEMENT_H
#define FN_ELEMENT_H

#include <utility>
#include <memory>
#include <vector>


// Wrapper for element base class
class Element{

    int id_ ; //tag   
    double measure_ = 0; // Length: for topological 1D element (like truss or beam).
                         // Area  : for topological 2D element (like shell or plate).
                         // Volume: for topological 3D element (like brick element).
    protected:
    virtual void set_id (int t){id_=t;}
    virtual void set_tag(int t){id_=t;}

    public:
    virtual int id() {return id_;};
    virtual int tag(){return id_;};
    
    Element(){};
    Element(int tag):id_(tag){};

    virtual ~Element(){};
};


class ElementWrapper{
    
    // External hiyerarchy for element types 
    class ElementConcept{
        public:

        //https://stackoverflow.com/questions/3141087/what-is-meant-with-const-at-end-of-function-declaration  
        //                                                |        
        //                                                v   
        virtual std::unique_ptr<ElementConcept> clone() const = 0; // To allow copy construction of the wrapper
                                                                   // (Prototype Design Pattern)
        virtual ~ElementConcept() = default;

        //VIRTUAL OPERATIONS =======================================
        virtual void do_action (/*Args.. args*/) const = 0;

        //==========================================================

    };

    template <typename ElementType> //(External Polymorfism Design Pattern).
    class ElementModel: public ElementConcept{ // Wrapper for all element types (of any kind)
        public:

        ElementType element_; // Stores the Element object

        std::unique_ptr<ElementConcept> clone() const override {
            return std::make_unique<ElementModel<ElementType>>(*this);
        };        

        ElementModel(ElementType   element) : element_{std::move(element)}{};
        ElementModel(ElementType&& element) : element_(std::forward<ElementType>(element)){}; // to test.

        ~ElementModel(){};

        // IMPLEMENTATION OF VIRTUAL OPERATIONS ==================
        void do_action (/*Args.. args*/) const override {
            action(element_/*, args...*/);  
        };
        //========================================================
    };

    // -----------------------------------------------------------------------------------------------
    // Hidden Friends (Free Functions)
    friend void action(ElementWrapper const& element /*, Args.. args*/){
        element.p_element_impl->do_action(/* args...*/);
    };

    // This functions trigger the polymorfic behaviour of the wrapper. They takes the pimpl
    // and call the virtual function do_action() on it. This function is implemented in the
    // ElementModel class, THE REAL ElementType BEHAVIOUR of the erased type.  
    // -----------------------------------------------------------------------------------------------

    // Pointer to base. (Bridge Design Pattern)
    std::unique_ptr<ElementConcept> p_element_impl; //pimpl (Pointer to implementation) idiom
  
  public:
  // Templated constructor which takes ANY kind of element and deduces the type!
    template <typename ElementType> //Bridge to possible impementation details (compiler generated).
    ElementWrapper(ElementType&& element)
        : p_element_impl{
            std::make_unique<ElementModel<ElementType>>(std::forward<ElementType>(element))
            } {};

    template <typename ElementType>
    ElementWrapper(ElementType element)
        : p_element_impl{
            std::make_unique<ElementModel<ElementType>>(std::move(element))
            }{};

    // ^ ^ ^ 
    // | | | 
    // This constructors take and element of ElementType and construct an ElementModel, in which the element is stored
    // "erasing the type of the element". We only have a pointer to base (ElementConcept)

    // -----------------------------------------------------------------------------------------------
    // Copy Operations (in terms of clone)
    ElementWrapper(ElementWrapper const& other)
        : p_element_impl{other.p_element_impl->clone()} {};

    ElementWrapper& operator=(ElementWrapper const& other){ //Copy and Swap idiom
        other.p_element_impl->clone().swap(p_element_impl);
        return *this;
        };

    // Move Operations (Thera are several possible implementations) 
    ElementWrapper(ElementWrapper&& other) noexcept = default;
    ElementWrapper& operator=(ElementWrapper&& other) noexcept = default;
    // -----------------------------------------------------------------------------------------------

};

// PARTICULAR IMPLEMENTATIONS OF VIRTUAL OPERATIONS ==========
//
// Example: (implement in .cpp file)  
// Only EXTERNAL POLYMORFISM is needed.
// void action_on_ALLelements(std::vector<std::unique_ptr<ElementConcept>> const& elems){
//     for (auto const& e: elems){
//         e->do_action(/*args...*/);
//     }
// };
//

//void do_on_element(SomeElementType const& element, /*Args.. args*/){
//    //Do something with the element
//};
//
//void action_on_containter_of_elements(std::vector<ElementWrapper> const& elems){
//    for (auto const& e: elems){
//        do_on_element(some_element); // some_element of SomeElementType.
//    }
//};


#endif


















