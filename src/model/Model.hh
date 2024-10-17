#ifndef FALL_N_MODEL_HH
#define FALL_N_MODEL_HH

#include <cstddef>
#include <memory>
#include <type_traits>

#include "../domain/Domain.hh"

#include "../materials/Material.hh"
#include "../materials/MaterialPolicy.hh"

#include "../elements/ContinuumElement.hh" // Se usa este por ahora mientras se define la interfaz del wrapper.

#include "MaterialPoint.hh"


// https://stackoverflow.com/questions/872675/policy-based-design-and-best-practices-c

//using LinealElastic3D = ElasticRelation<ThreeDimensionalMaterial>;
//using LinealElastic2D = ElasticRelation<PlaneMaterial>;
//using LinealElastic1D = ElasticRelation<UniaxialMaterial>;

// The MaterialPolicy defines the constitutive relation and the number of dimensions
template </*TOOD: typename KinematicPolicy,*///Kinematic Policy (e.g. Static, pseudo-static, dynamic...) 
    typename MaterialPolicy,                 //Considerar la definici[on de un ModelPolicy que encapsule estaticamente el MaterialPolicy
    std::size_t ndofs = MaterialPolicy::dim  //Default: Solid Model with "dim" displacements per node. 
    >
class Model
{

public:
    
    using Material = Material<MaterialPolicy>;
    using FEM_Element = ContinuumElement<MaterialPolicy, ndofs>; // Aca ira en adelante el wrapper de Element FEM_Element
    
    static constexpr std::size_t dim{MaterialPolicy::dim};

private:
    Domain<dim> *domain_;
    std::size_t dofsXnode{ndofs};
public:
    
    std::vector<double>      dof_vector_;
    std::vector<Material>    materials_;
    std::vector<FEM_Element> elements_;
    

    void set_default_num_dofs_per_node(std::size_t n){
        for (auto &node : domain_->nodes())
            node.set_num_dof(n);
    }

    void link_dofs_to_node(){
        auto pos = 0;
        for (auto &node : domain_->nodes()){
            for (auto &dof : node.dofs()){
                dof = std::addressof(dof_vector_[pos++]);
            }
        }
    }

    Model(Domain<dim> &domain) : domain_(std::addressof(domain)){
        elements_.reserve(domain_->num_elements()); // El dominio ya debe tener TODOS LOS ELEMENTOS GEOMETRICOS CREADOS!
        dof_vector_.resize(domain_->num_nodes() * ndofs, 0.0); // Set capacity to avoid reallocation (and possible dangling pointers) // TODO: PUT AN OBSERVER!
        set_default_num_dofs_per_node(ndofs);                  // Set default number of dofs per node in Dof_Interface
        link_dofs_to_node();                                   // Link Dof_Interface to Node

        // Fill for testing
        // auto idx = 0;
        // for (auto &dof : dof_vector_)
        //     dof = idx++;
    }

    Model() = delete;
    ~Model() = default;

    // Other members
};

#endif // FALL_N_MODEL_HH
