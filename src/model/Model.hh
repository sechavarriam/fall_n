#ifndef FALL_N_MODEL_HH
#define FALL_N_MODEL_HH

#include <cstddef>
#include <memory>
#include <type_traits>



#include "../domain/Domain.hh"

#include "../materials/Material.hh"
#include "../materials/LinealRelation.hh"

#include "MaterialPoint.hh"


// https://stackoverflow.com/questions/872675/policy-based-design-and-best-practices-c

using LinealElastic3D = LinealRelation<Stress<6>, Strain<6>>;
using LinealElastic2D = LinealRelation<Stress<3>, Strain<3>>;
using LinealElastic1D = LinealRelation<Stress<1>, Strain<1>>;

// The MaterialPolicy defines the constitutive relation and the number of dimensions
template </*TOOD: typename KinematicPolicy,*///Kinematic Policy (e.g. Static, pseudo-static, dynamic...) 
    typename MaterialPolicy,                 //Constitutive relation (e.g. LinealRelation, NeoHookeanRelation, PlasticRelation)
    std::size_t ndofs = MaterialPolicy::dim  //Default: Solid Model with "dim" displacements per node. 
    >
class Model
{

public:
    static constexpr std::size_t dim{MaterialPolicy::dim};

private:
    Domain<dim> *domain_;
    std::size_t dofsXnode{ndofs};
public:
    
    std::vector<double> dof_vector_;
 
    void set_default_num_dofs_per_node(std::size_t n)
    {
        for (auto &node : domain_->nodes())
            node.set_num_dof(n);
    }

    void link_dofs_to_node()
    {
        auto pos = 0;
        for (auto &node : domain_->nodes())
        {
            for (auto &dof : node.dofs())
            {
                dof = std::addressof(dof_vector_[pos++]);
            }
        }
    }

    Model(Domain<dim> &domain) : domain_(std::addressof(domain))
    {
        dof_vector_.resize(domain_->num_nodes() * ndofs, 0.0); // Set capacity to avoid reallocation (and possible dangling pointers)
        set_default_num_dofs_per_node(ndofs);                  // Set default number of dofs per node in Dof_Interface
        link_dofs_to_node();                                   // Link Dof_Interface to Node

        // Fill for testing
        auto idx = 0;
        for (auto &dof : dof_vector_)
            dof = idx++;
    }

    Model() = delete;
    ~Model() = default;

    // Other members
};

#endif // FALL_N_MODEL_HH
