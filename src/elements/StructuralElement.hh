#ifndef FALL_STRUCTURAL_ELEMENT_ABSTRACTION
#define FALL_STRUCTURAL_ELEMENT_ABSTRACTION




// Concept constrained material policy (force-deformation relation) and kinematic policy (static, pseudo-static, dynamic, etc.) to define the structural element behavior.

template <typename MaterialPolicy, std::size_t ndof>
class StructuralElement
{
    using ElementGeometry = ElementGeometry<MaterialPolicy::dim>;
    // degenerated geometry

    ElementGeometry* geometry_ ; //ya incluye los nodos y la estrategia de integracion )

    public:




    // El constructor debe verificar que la dimensi[on de le geometria sea menor a la dimension del especio, provista por la politica cinematica,

    StructuralElement() = delete; 

    StructuralElement(ElementGeometry* degenerated_geometry) : geometry_{degenerated_geometry} 
    {
        // Assert // Metodo para setear materiales debe ser llamado despues de la creacion de los elementos.
    }; 
    

};

#endif // FALL_STRUCTURAL_ELEMENT_ABSTRACTION