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

    Model <dim>* model_;
    Domain<dim>* domain_; 

    std::size_t default_num_dofs_{0}; //Default num_dofs for the nodes in the domain.

    public:
    



    void setup_model(){
        
        // 1. set dof_vector_ size in the model.
        model_->set_total_dofs(domain_->num_nodes() * default_num_dofs_);

        auto dof_position = model_->dof_vector_.begin();

        // 2. set dof_vector_ size in the nodes and link the dofs to the model.
        for (auto& node : domain_->nodes()){
            node.set_num_dof(default_num_dofs_);

            for(auto dof : node.dofs()){
                dof = std::addressof(*dof_position);
                dof_position ++;
            }
        }

        // 3.
    };
    

    void update_model(){
        // 1. If domain changed, update the model (put observer in domain).
    };


    void set_default_num_dofs_per_node(std::size_t n){
        default_num_dofs_ = n;
        for (auto& node : domain_->nodes())node.set_num_dof(default_num_dofs_);
    };

    //void set_model_dofs(){
    //    for (auto& node : domain_->nodes())model_->add_dofs(node.dofs_);
    //};

    void link_dofs_to_model(){
        
    };



    ModelBuilder(Model<dim>& model,Domain<dim>& domain):model_{&model},domain_{&domain}{};

    ModelBuilder(Model<dim>& model,Domain<dim>& domain, std::size_t def_ndof) : 
        model_{&model},
        domain_{&domain},
        default_num_dofs_{def_ndof}
        {
            setup_model();
        };


    
    ModelBuilder() = default;
    ~ModelBuilder() = default;
};




#endif // MODEL_BUILDER_HH