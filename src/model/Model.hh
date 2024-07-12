#ifndef FALL_N_MODEL_HH
#define FALL_N_MODEL_HH

#include <cstddef>
#include <memory>

#include "../domain/Domain.hh"

template<std::size_t dim> //Revisar como hacer un Wrapper para evitarse este parámetro.
class Model {
private:
public:

    Domain<dim>* domain_; 

    std::size_t dofsXnode{0};
    std::vector<double>  dof_vector_; 


    void set_default_num_dofs_per_node(std::size_t n){
        for(auto& node : domain_->nodes()) node.set_num_dof(n);
    }

    void link_dofs_to_node(){
        auto pos = dof_vector_.front();
        for(auto& node : domain_->nodes()){
            for(auto& dof : node.dofs()){
                dof = std::addressof(dof_vector_[++pos]);
            }
        }
    }


    Model(Domain<dim>& domain, std::size_t num_dofs) : domain_(std::addressof(domain)), dofsXnode(num_dofs)
    {
        dof_vector_.resize(domain_->num_nodes()*num_dofs, 0.0); //Set capacity to avoid reallocation (and possible dangling pointers)
        set_default_num_dofs_per_node(num_dofs);           //Set default number of dofs per node in Dof_Interface
        link_dofs_to_node();                                //Link Dof_Interface to Node
    }

    Model() = delete;
    ~Model() = default;
    

    // Other members
};

#endif // FALL_N_MODEL_HH

