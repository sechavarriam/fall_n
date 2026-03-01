#ifndef FALL_STRUCTURAL_ELEMENT_ABSTRACTION
#define FALL_STRUCTURAL_ELEMENT_ABSTRACTION

// Concept constrained material policy (force-deformation relation) and kinematic policy (static, pseudo-static, dynamic, etc.) to define the structural element behavior.

#include "element_geometry/ElementGeometry.hh"
#include "Section.hh"


template <typename MaterialPolicy, std::size_t ndof>
class StructuralElement
{
    using ElementGeometryT = ::ElementGeometry<MaterialPolicy::dim>;
    using NodeSection     = NodeSection<Node<MaterialPolicy::dim>>;
    // degenerated geometry

    ElementGeometryT* geometry_ ; //ya incluye los nodos y la estrategia de integracion )
    std::vector<NodeSection> node_sections_; // This is the section of the element, it contains the nodes and the dof information. It can be used to define the boundary conditions and the loads.
    



    public :



    // El constructor debe verificar que la dimensi[on de le geometria sea menor a la dimension del especio, provista por la politica cinematica,

    StructuralElement() = delete; 

    StructuralElement(ElementGeometryT* degenerated_geometry) : geometry_{degenerated_geometry} 
    {
        for (std::size_t i = 0; i < geometry_->num_nodes(); ++i){
            node_sections_.emplace_back(NodeSection(&geometry_->node_p(i)));
        }
    }; 
    

};

#endif // FALL_STRUCTURAL_ELEMENT_ABSTRACTION