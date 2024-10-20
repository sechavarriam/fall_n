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


// https://www.dealii.org/current/doxygen/deal.II/namespacePETScWrappers.html

// https://petsc.org/release/manual/mat/#sec-matsparse
// https://petsc.org/release/manual/mat/#block-matrices

// https://stackoverflow.com/questions/872675/policy-based-design-and-best-practices-c

//using LinealElastic3D = ElasticRelation<ThreeDimensionalMaterial>;
//using LinealElastic2D = ElasticRelation<PlaneMaterial>;
//using LinealElastic1D = ElasticRelation<UniaxialMaterial>;

// The MaterialPolicy defines the constitutive relation and the number of dimensions
template </*TOOD: typename KinematicPolicy,*///Kinematic Policy (e.g. Static, pseudo-static, dynamic...) 
    typename MaterialPolicy,                 //Considerar la definici[on de un ModelPolicy que encapsule estaticamente el MaterialPolicy
    std::size_t ndofs = MaterialPolicy::dim  //Default: Solid Model with "dim" displacements per node. 
    >
class Model{

    using PETScMatrix = Mat; //TODO: Use PETSc Matrix
    using PETScVector = Vec; //TODO: Use PETSc Vector

public:
    
    using Material = Material<MaterialPolicy>;
    using FEM_Element = ContinuumElement<MaterialPolicy, ndofs>; // Aca ira en adelante el wrapper de Element FEM_Element
    
    static constexpr std::size_t dim{MaterialPolicy::dim};

private:
    Domain<dim> *domain_;
    std::size_t dofsXnode{ndofs}; // Esto es necesario???? No creo.
    std::size_t num_dofs_{0}; // Total number of dofs in the model.
public:
      
    //std::vector<Material>    materials_; // De momento este catalogo de materiales no es requerido.
    std::vector<double>      dof_vector_; // Esto obliga a ser secuencial!!!!
    std::vector<FEM_Element> elements_;
    
    std::size_t num_dofs() const {return num_dofs_;};

    PETScMatrix K; // Global Stiffness Matrix
    PETScVector F; // Global Load Vector
    PETScVector U; // Global Displacement Vector


    // Methods
    // 1. Apply boundary conditions. Constrain or Fix Dofs.
    // 2. Apply loads. (Construct the load vector - PETSc Vector (could be parallel)). Also dof_vector_ could be parallel....

private:

    void init_vector(PETScVector &v){ // To init PETSc Vectors F and U
        VecCreate(PETSC_COMM_WORLD, &v);
        VecSetSizes(v, PETSC_DECIDE, num_dofs_);
        VecSetType(v, VECSTANDARD);

        VecSetOption(v, VEC_IGNORE_NEGATIVE_INDICES, PETSC_TRUE);

    }

    void init_K(PetscInt max_num_nodesXelement = 0){
        auto N = domain_->num_nodes() * dofsXnode; // Total number of dofs in the model (ASUMIDO PARA TODOS LOS NODOS CON EL MISMO NUMERO DE DOFS).

        PetscInt nz_upper_bound; // Estimador provisional para nz (POR FILAAAAA!!!!) asumiendo malla estructurada 3D de celdas. 
        nz_upper_bound = 8*max_num_nodesXelement*dofsXnode;//*N; // Para prueba! Debe ser inyectado (calculado).
                                                                // Cota superior asumida para mallas estructuradas 3D.
                                                                // Esto es un estimador para el espacio de memoria requerido

        MatCreateSeqAIJ(PETSC_COMM_WORLD, N, N, nz_upper_bound, PETSC_NULLPTR, &K);
    }

    void apply_node_force(std::size_t node_idx, [[maybe_unused]] auto force){
        auto& node = domain_->nodes()[node_idx];
        
        // get indices

    }

    void set_default_num_dofs_per_node(std::size_t n){for (auto &node : domain_->nodes()) node.set_num_dof(n);}

    void set_dof_index(){
        std::size_t pos = 0;
        for (auto &node : domain_->nodes()){
            node.set_dof_index(std::ranges::iota_view{pos, pos + dofsXnode}); //Esto podría ser otro RANGE con los índices óptimos luego de un reordenamiento tipo Cuthill-McKee.
            pos += dofsXnode;
        }
        num_dofs_ = pos;
    };

    //void link_dofs_to_node(){
    //    auto pos = 0;
    //    for (auto &node : domain_->nodes()){
    //        for (auto &dof : node.dofs()){
    //            dof = std::addressof(dof_vector_[pos++]);
    //        }
    //    }
    //}

    // https://petsc.org/release/manual/profiling/
    void assembly_K(){// Assembly Global Stiffness Matrix
        MatAssemblyBegin(K, MAT_FINAL_ASSEMBLY);
        
        for (auto &element : elements_)  element.inject_K(K);
        
        MatAssemblyEnd(K, MAT_FINAL_ASSEMBLY);
    }


public:

    // Constructors

    Model(Domain<dim> &domain, Material default_mat) : domain_(std::addressof(domain)){

        elements_.reserve(domain_->num_elements()); // El dominio ya debe tener TODOS LOS ELEMENTOS GEOMETRICOS CREADOS!

        for (auto &element : domain_->elements()){                                     //By now all elements are ContinuumElements
            elements_.emplace_back(FEM_Element{std::addressof(element), default_mat}); //By default, all elements have the same material.
        }

        dof_vector_.resize(domain_->num_nodes() * ndofs, 0.0); // Set capacity to avoid reallocation (and possible dangling pointers) // TODO: PUT AN OBSERVER!
        set_default_num_dofs_per_node(ndofs);                  // Set default number of dofs per node in Dof_Interface
        //link_dofs_to_node();                                   // Link Dof_Interface to Node
        set_dof_index();                                       // Set Dof Indexes

        // Fill for testing
        // Print Warning 
        std::cout << "================================================================================" << std::endl;
        std::cout << "==> WARNING: DoF vector indexed from 0 to " << dof_vector_.size() - 1 << " for testing purposes." << std::endl; 
        auto idx = 0;
        for (auto &dof : dof_vector_)
            dof = idx++;
        std::cout << "================================================================================" << std::endl;

        // No es requerido por el constructor. Pero para propositos de prueba se deja. 
        // El metodo assembly_K() debe ser llamado explicitamente desde el solver (analisis) para ensamblar la matriz de rigidez global.
        
        init_K(elements_[0].num_nodes()); // PROVISIONAL! Asume todos los elementos con el mismo numero de nodos. Por eso se toma como ref el primero. 
        // Initialize Global Stiffness Matrix // DEBE SER LUEGO DE SETEAR LOS DOFS (y conocer el tipo de los elementos: num_element_nodes)!
        assembly_K(); // Assembly Global Stiffness Matrix

        init_vector(F); // Initialize Global Load Vector
        init_vector(U); // Initialize Global Displacement Vector

        // Spy view (draw)
        MatView(K, PETSC_VIEWER_DRAW_WORLD);

    }

    Model() = delete;
    ~Model() = default;

    // Other members
};

#endif // FALL_N_MODEL_HH
