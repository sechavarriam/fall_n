#ifndef MODEL_BUILDER_HH
#define MODEL_BUILDER_HH


#include <memory>
#include <cstddef>

#include "../domain/Domain.hh"
#include "Model.hh"


// Initially, this class realizes the linlking process between the Domain and the Model. That is, 
// sets the default num_dofs_ for the nodes in the domain and link the interfases to the Model (dof_vector).

template<std::size_t dim>
class ModelBuilder{

    template<std::size_t D>
    using Domain = domain::Domain<D>;

    std::size_t defaulted_num_dofs_{0};

    std::shared_ptr<Domain<dim>> domain_;
    std::shared_ptr<Model <dim>> model_;

  public:

    void set_model(Model<dim>& model){
        model_ = std::make_shared<Model<dim>>(model);
    };

    void set_domain(Domain<dim>& domain){
        domain_ = std::make_shared<Domain<dim>>(domain);
    };

    void set_node_num_dofs(std::size_t n){
        defaulted_num_dofs_ = n;
        for (auto node : domain_->nodes()){
            node.set_num_dof(n);
        }
    };

    void update_model(){}; //This function shound be called if some element changes his defaulted num_dofs.

    
    ModelBuilder(Model<dim>& model) : model_{std::make_shared<Model<dim>>(model)}{};
    
    ModelBuilder(Model<dim>& model,Domain<dim>& domain) : 
        domain_{std::make_shared<Domain<dim>>(domain)},
        model_ {std::make_shared<Model<dim>>(model)}
        {};



    //ModelBuilder(std::size_t ndofs) : 
    //    defaulted_num_dofs_{ndofs},
    //    model_{std::make_shared<Model<dim>>()}
    //    {};
    
    ModelBuilder() = default;
    ~ModelBuilder() = default;
};




#endif // MODEL_BUILDER_HH