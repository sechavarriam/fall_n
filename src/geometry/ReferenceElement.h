#ifndef FN_REFERENCES_ELEMENT_H
#define FN_REFERENCES_ELEMENT_H

#include <array>
#include <cstddef>
#include <iostream>
#include <utility>
#include <memory>


typedef unsigned short ushort;
typedef unsigned int   uint  ;


namespace impl{ //Implementation details

    class ref_ElementConcept{
    public:
        virtual ~ref_ElementConcept() = default;  
        
        virtual void compute_integral(/*function2integrate?*/) const = 0; //Maybe double? auto?

        virtual std::unique_ptr<ref_ElementConcept> clone()   const = 0;
        virtual void clone(ref_ElementConcept* memory_address)const = 0;

    public: 
        //constexpr virtual ushort get_num_nodes() const = 0;
        //constexpr virtual ushort get_num_dof()   const = 0;
        //virtual uint          get_id()    const = 0;
        //virtual ushort const* get_nodes() const = 0;
    };  

    template <typename ElementType, typename IntegrationStrategy> 
    class NON_Owning_ref_ElementModel; //Forward declaration

    template <typename ElementType, typename IntegrationStrategy> 
    class Owning_ref_ElementModel: public ref_ElementConcept{ // Wrapper for all element types (of any kind) 
        
        ElementType         element_;    // Stores the ReferenceElement object
        IntegrationStrategy integrator_; // Stores the Integration Strategy object (Spacial integration strategy)

    public:
        void compute_integral(/*function2integrate?*/) const override {integrator_(element_);}; //Maybe double?

        explicit Owning_ref_ElementModel(ElementType element, IntegrationStrategy integrator) 
            : element_   (std::move(element)),
              integrator_(std::move(integrator)){};

        std::unique_ptr<ref_ElementConcept> clone() const override {
            return std::make_unique<Owning_ref_ElementModel<ElementType,IntegrationStrategy>>(*this);
        };

        void clone(ref_ElementConcept* memory_address) const override{ // override{ ? 
            using Model = NON_Owning_ref_ElementModel<ElementType const,IntegrationStrategy const>;
            std::construct_at(static_cast<Model*>(memory_address),element_,integrator_);
        };

    public: // Implementation of the virtual operations derived from ref_ElementConcept

        //ushort constexpr get_num_nodes()const override {return element_.num_nodes();};
        //ushort constexpr get_num_dof()  const override {return element_.num_dof()  ;};
        //uint          get_id()    const override {return element_.id()   ;};
        //ushort const* get_nodes() const override {return element_.nodes();};
    };

    template <typename ElementType, typename IntegrationStrategy> 
    class NON_Owning_ref_ElementModel: public ref_ElementConcept{ 
        ElementType*         element_{nullptr};    // Only stores a pointer to the ReferenceElement object (aka NonOwning)
        IntegrationStrategy* integrator_{nullptr}; // 

    public:
        void compute_integral(/*function2integrate?*/) const override {(*integrator_)(*element_);}; //Maybe double?

        NON_Owning_ref_ElementModel(ElementType& element, IntegrationStrategy& integrator) 
            : element_   {std::addressof(element)  }     // &element
            , integrator_{std::addressof(integrator)}{}; // &integrator

        std::unique_ptr<ref_ElementConcept> clone() const override {
            using Model = Owning_ref_ElementModel<ElementType,IntegrationStrategy>;
            return std::make_unique<Model>(*element_,*integrator_);
        };

        void clone( ref_ElementConcept* memory_address) const override{
            std::construct_at(static_cast<NON_Owning_ref_ElementModel*>(memory_address),*this);      
        };

    public:  // Implementation of the virtual operations derived from ref_ElementConcept (Accesing pointer members)
 
        //ushort constexpr get_num_nodes()const override {return element_->num_nodes();};
        //ushort constexpr get_num_dof()  const override {return element_->num_dof()  ;};   
        //uint          get_id()    const override {return element_->id()   ;};
        //ushort const* get_nodes() const override {return element_->nodes();};
    };
} //impl

class ReferenceElement; // Forward declaration
class ReferenceElementConstRef{
    friend class ReferenceElement;

    // Expected size of a model instantiation: sizeof(ShapeT*) + sizeof(DrawStrategy*) + sizeof(vptr)
    static constexpr std::size_t MODEL_SIZE = 3; //The 3 pointers of the NON_Owning_ref_ElementModel,
    alignas(void*) std::array<std::byte,MODEL_SIZE> raw_; //Raw storage (Aligned Byte array)

    impl::ref_ElementConcept* pimpl(){
        return reinterpret_cast<impl::ref_ElementConcept*>(raw_.data());
    };

    impl::ref_ElementConcept const* pimpl() const{
        return reinterpret_cast<impl::ref_ElementConcept const*>(raw_.data());
    };

  public:
    template<typename ElementType, typename IntegrationStrategy>
    ReferenceElementConstRef(ElementType& element, IntegrationStrategy& integrator){
        using Model = impl::NON_Owning_ref_ElementModel<const ElementType,const IntegrationStrategy>;
        
        static_assert(sizeof(Model)  == MODEL_SIZE    , "Invalid Model size"); //(<= ?) 
        static_assert(alignof(Model) == alignof(void*), "Model Misaligned"  ); 

        std::construct_at(static_cast<Model*>(pimpl()),element,integrator);
    };

    ReferenceElementConstRef(ReferenceElementConstRef const& other){other.pimpl()->clone(pimpl());};

    ReferenceElementConstRef(ReferenceElement& other);      // Implicit conversion from ReferenceElement to ReferenceElementConstRef
    ReferenceElementConstRef(ReferenceElement const& other);//  

    ReferenceElementConstRef& operator=(ReferenceElementConstRef const& other){
        ReferenceElementConstRef copy{other};
        raw_.swap(copy.raw_);
        return *this;
    };
    ~ReferenceElementConstRef(){std::destroy_at(pimpl());}; // OR: ~ReferenceElementConstRef(){pimpl()->~ref_ElementConcept();};
    // Move operations explicitly not declared

  private: //Operations with references
    friend void integrate(ReferenceElementConstRef const& element){
        element.pimpl()->compute_integral(/*args...*/);
        //std::cout << "ReferenceElement " << element.pimpl()->get_id() << " integrated" << std::endl;
        };
};

class ReferenceElement{
    friend class ReferenceElementConstRef;
    std::unique_ptr<impl::ref_ElementConcept> pimpl_; // Bridge to implementation details (compiler generated).

  public:
    template<typename ElementType, typename IntegrationStrategy>
    ReferenceElement(ElementType element, IntegrationStrategy integrator){
        using Model = impl::Owning_ref_ElementModel<ElementType,IntegrationStrategy>;
        pimpl_ = std::make_unique<Model>(
            std::move(element),
            std::move(integrator));
    };

    ReferenceElement( ReferenceElement         const& other) : pimpl_{other.pimpl_ ->clone()} {};
    ReferenceElement( ReferenceElementConstRef const& other) : pimpl_{other.pimpl()->clone()} {};

    ReferenceElement& operator=(const ReferenceElementConstRef& other){
        ReferenceElement copy{other};
        pimpl_.swap(copy.pimpl_);
        return *this;
    };

    ~ReferenceElement() = default;
    ReferenceElement(ReferenceElement &&) = default;
    ReferenceElement& operator=(ReferenceElement &&) = default;

  private:
    // -----------------------------------------------------------------------------------------------
    // Hidden Friends (Free Functions)

    friend void integrate(ReferenceElement const& element){
        element.pimpl_->compute_integral(/*args...*/);
    //    std::cout << "ReferenceElement " << element.pimpl_->get_id() << " integrated" << std::endl;
        };
        
    //friend uint   id       (ReferenceElement const& element){return element.pimpl_->get_id()       ;};
    //friend ushort num_nodes(ReferenceElement const& element){return element.pimpl_->get_num_nodes();};
    //friend ushort num_dof  (ReferenceElement const& element){return element.pimpl_->get_num_dof()  ;};
    // 
    //friend auto nodes(ReferenceElement const& element){
    //    return std::span{
    //        element.pimpl_->get_nodes(),    //Pointer to data
    //        element.pimpl_->get_num_nodes() //Size of the span
    //        };
    //    };
};

inline ReferenceElementConstRef :: ReferenceElementConstRef(ReferenceElement&       other) {other.pimpl_->clone(pimpl());};
inline ReferenceElementConstRef :: ReferenceElementConstRef(ReferenceElement const& other) {other.pimpl_->clone(pimpl());};

#endif //FN_ELEMENT_H
