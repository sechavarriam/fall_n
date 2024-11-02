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

#include <petsc.h>


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

    friend class Analysis; // Por ahora. Para no exponer publicamentge el dominio.

    using PETScMatrix = Mat; //TODO: Use PETSc Matrix
    using PETScVector = Vec; //TODO: Use PETSc Vector

public:
    
    using Material = Material<MaterialPolicy>;
    using FEM_Element = ContinuumElement<MaterialPolicy, ndofs>; // Aca ira en adelante el wrapper de Element FEM_Element
    
    static constexpr std::size_t dim{MaterialPolicy::dim};

private:
    Domain<dim> *domain_;
    //std::size_t dofsXnode{ndofs}; // Esto es necesario???? No creo.
    //std::size_t num_dofs_{0}; // Total number of dofs in the model.
public:
      
    
    //std::vector<Material>    materials_; // De momento este catalogo de materiales no es requerido.
    std::vector<FEM_Element> elements_;
    
    //std::size_t num_dofs() const {return num_dofs_;};

    Mat K; // Global Stiffness Matrix
    Vec F; // Global Load Vector
    Vec U; // Global Displacement Vector

    PetscSection dof_section; // PETSc Section

    // Methods
    // 1. Apply boundary conditions. Constrain or Fix Dofs.
    // 2. Apply loads. (Construct the load vector - PETSc Vector (could be parallel)). Also dof_vector_ could be parallel....

private:

    void set_num_dofs_in_elements(){ for (auto &element : elements_) element.set_num_dof_in_nodes();}

    void set_dof_index(){
        PetscInt ndof, offset;

        for (auto &node : domain_->nodes()){

            PetscSectionGetDof   (dof_section, node.sieve_id.value(), &ndof  );
            PetscSectionGetOffset(dof_section, node.sieve_id.value(), &offset);

            node.set_dof_index(std::ranges::iota_view{offset, offset + ndof});
            //Esto podría ser otro RANGE con los índices óptimos luego de un reordenamiento tipo Cuthill-McKee.
        }
    };

    //// https://petsc.org/release/manual/profiling/
    //void assembly_K(){// Assembly Global Stiffness Matrix
    //    MatAssemblyBegin(K, MAT_FINAL_ASSEMBLY);
    //    for (auto &element : elements_)  element.inject_K(K);        
    //    MatAssemblyEnd(K, MAT_FINAL_ASSEMBLY);
    //}

public:

    auto get_plex(){return domain_->mesh.dm;};

    // Apply boundary conditions. Constrain or Fix Dofs.    
    void fix_node_dofs(std::size_t node_idx, auto... dofs_to_constrain)
    requires (std::is_convertible_v<decltype(dofs_to_constrain), PetscInt> && ...)
    {
        auto& node = domain_->node(node_idx);
        auto  num_dofs = node.num_dof();
        auto  plex_id  = node.sieve_id.value();

        if (sizeof...(dofs_to_constrain) > num_dofs) throw std::runtime_error("Dofs to constrain exceed the number of dofs in the node.");        
    
        PetscSectionSetConstraintIndices(dof_section, plex_id, std::array{dofs_to_constrain...}.data());
        PetscSectionSetUp(dof_section);
        //for (auto dof : {dofs_to_constrain...}) node.fix_dof(dof);

    }

    void fix_node(std::size_t node_idx)
    {
        auto& node = domain_->node(node_idx);
        auto  num_dofs = node.num_dof();
        auto  plex_id  = node.sieve_id.value();
        
        PetscSectionSetConstraintDof(dof_section, plex_id, num_dofs);
        PetscSectionSetUp(dof_section);

    }

    // Apply forces to nodes
    void apply_node_force(std::size_t node_idx, std::ranges::contiguous_range auto&& force)
    requires std::convertible_to<std::ranges::range_value_t<decltype(force)>, double>
    {
        PetscInt offset;

        auto& node     = domain_->node(node_idx);
        auto num_dofs  = node.num_dof();
        auto plex_id   = node.sieve_id.value();

        if (std::ranges::size(force) != num_dofs) throw std::runtime_error("Force vector size mismatch."); 

        PetscSectionGetOffset(dof_section, plex_id, &offset); 
        
        for (auto i = 0; i < num_dofs; ++i){
            VecSetValues(F, 1, &offset, &force[i], INSERT_VALUES);
            offset++;
        }

        VecAssemblyBegin(F);
        VecAssemblyEnd(F);       
    }

    void apply_node_force(std::size_t node_idx, auto... force_components)
    requires (std::is_convertible_v<decltype(force_components), PetscScalar> && ...)
    {
        auto num_dofs = domain_->node(node_idx).num_dof();
        if (sizeof...(force_components) != num_dofs) throw std::runtime_error("Force components size mismatch.");
        
        auto dofs = domain_->node(node_idx).dof_index().data();

        const PetscScalar force[] = {force_components...};

        
        VecSetValues(this->F, num_dofs, dofs, force, INSERT_VALUES);

        VecAssemblyBegin(this->F);
        VecAssemblyEnd(this->F);       
    }

    void set_sieve_layout (){ // ONLY CELL-VERTEX MESHES SUPPORTED BY NOW! 

        PetscInt pStart, pEnd, cStart, cEnd, vStart, vEnd;

        PetscSectionCreate(PETSC_COMM_WORLD, &dof_section); // Create the section (PetscSection)
        //DMSetGlobalSection(domain_->mesh.dm, dof_section); // Set the global section for the mesh

        DMSetSection(domain_->mesh.dm, dof_section); // Set the section for the mesh

        DMPlexGetChart(domain_->mesh.dm, &pStart, &pEnd);
        DMPlexGetHeightStratum(domain_->mesh.dm, 0, &cStart, &cEnd);   // cells
        DMPlexGetHeightStratum(domain_->mesh.dm, 1, &vStart, &vEnd);   // vertices, equivalent to DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);
                                                                       // https://petsc.org/release/manualpages/DMPlex/DMPlexStratify/
        // For cell-vertex meshes, vertices are depth 0 and cells are depth 1. For fully interpolated meshes, depth 0 for vertices, 
        // 1 for edges, and so on  until cells have depth equal to the dimension of the mesh.

        PetscSectionSetChart(dof_section, vStart, vEnd); // Set the chart for the section (PetscSection)

        for (auto &node: domain_->nodes()){
            PetscSectionSetDof(dof_section, node.sieve_id.value(), node.num_dof());
        }
        PetscSectionSetUp(dof_section);

        set_dof_index(); // Set the dof index for each node in the domain.
                         // Si hay un reorder en el plex se debe volver a llamar... Poner observador?

        DMSetUp(domain_->mesh.dm); // Setup the mesh

        DMCreateLocalVector (domain_->mesh.dm, &F); // Create the local force vector for the mesh.
        DMCreateGlobalVector(domain_->mesh.dm, &U); // Create the global vector for the mesh (Solution Vector) - Move to analysis!!!!
        //VecSetSizes(U, PETSC_DECIDE, total_dofs); // Set the size of the global displacement vector.

        //https://lists.mcs.anl.gov/mailman/htdig/petsc-users/2016-March/028797.html
        DMCreateMatrix(domain_->mesh.dm, &K);
        DMSetMatType(domain_->mesh.dm, MATAIJ); // Set the matrix type for the mesh. 

        for (auto &element : elements_) element.inject_K(K);
        MatAssemblyBegin(K, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd  (K, MAT_FINAL_ASSEMBLY);
        MatView(K, PETSC_VIEWER_DRAW_WORLD); // Draw the matrix

        domain_->mesh.view(); // View the mesh
    };


    // Constructors
    Model(Domain<dim> &domain, Material default_mat) : domain_(std::addressof(domain)){

        DMSetVecType(domain_->mesh.dm, VECSTANDARD);        
        DMSetDimension(domain_->mesh.dm, dim); // Set the dimension of the mesh
        DMSetBasicAdjacency(domain_->mesh.dm, PETSC_FALSE, PETSC_TRUE); // Set the adjacency information for the FEM mesh.

        PetscSectionCreate(PETSC_COMM_WORLD, &dof_section);

        elements_.reserve(domain_->num_elements()); // El dominio ya debe tener TODOS LOS ELEMENTOS GEOMETRICOS CREADOS!

        for (auto &element : domain_->elements()){                                     //By now all elements are ContinuumElements
            elements_.emplace_back(FEM_Element{std::addressof(element), default_mat}); //By default, all elements have the same material.
        }

        set_num_dofs_in_elements(); // Set the number of dofs per node in the elements (FEM_Elements)
        set_sieve_layout(); // Set the sieve layout for the mesh

    }


    Model() = delete;
    ~Model() {
        VecDestroy(&F);
        //VecDestroy(&U);
        MatDestroy(&K);
        PetscSectionDestroy(&dof_section);
    }

    // Other members
};

#endif // FALL_N_MODEL_HH
