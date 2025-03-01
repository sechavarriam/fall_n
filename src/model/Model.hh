#ifndef FALL_N_MODEL_HH
#define FALL_N_MODEL_HH

#include <cstddef>
#include <memory>
#include <type_traits>

#include "../domain/Domain.hh"

#include "../materials/Material.hh"
#include "../materials/MaterialPolicy.hh"

#include "../elements/ContinuumElement.hh" // Se usa este por ahora mientras se define la interfaz del wrapper.

#include "../post-processing/VTK/VTKdataContainer.hh"

#include "MaterialPoint.hh"

#include <petsc.h>


// https://www.dealii.org/current/doxygen/deal.II/namespacePETScWrappers.html
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

public:    
    using MaterialT         = Material<MaterialPolicy>;
    using FEM_Element       = ContinuumElement<MaterialPolicy, ndofs>; // Aca ira en adelante el wrapper de Element FEM_Element
    using ConstraintDofInfo = std::map<PetscInt, std::pair<std::vector<PetscInt>, std::vector<PetscScalar>>>; 
    
    static constexpr std::size_t dim{MaterialPolicy::dim};

private:
    Domain<dim>*      domain_;
    ConstraintDofInfo constraints_; 
    PetscSection      dof_section; 
public:

    bool is_bc_updated{false}; // Flag to check if the model has been updated (global-local sixes changed).

    //std::vector<Material>    materials_; // De momento este catalogo de materiales no es requerido.
    std::vector<FEM_Element> elements;
    
    Vec nodal_forces; 
    Vec global_imposed_solution; // Global Displacement Vector (coordinate sense and parallel sense)

    Vec current_state;           // Current state of the system (Displacements, velocities, accelerations, etc.)

    Mat Kt; // Global Stiffness Matrix

    auto& get_domain(){return *domain_;};
    auto  get_plex()  {return domain_->mesh.dm;};

private:

    void set_num_dofs_in_elements(){ for (auto &element : elements) element.set_num_dof_in_nodes();}

    void set_dof_index(){
        PetscSection local_section;
        PetscInt ndof, offset;

        DMGetLocalSection(domain_->mesh.dm, &local_section);
        for (auto &node : domain_->nodes()){

            PetscSectionGetDof   (local_section, node.sieve_id.value(), &ndof  );
            PetscSectionGetOffset(local_section, node.sieve_id.value(), &offset);

            node.set_dof_index(std::ranges::iota_view{offset, offset + ndof});//Esto podría ser otro RANGE con los índices óptimos luego de un reordenamiento tipo Cuthill-McKee.
        }
    };

public:

    void set_sieve_layout (){ // ONLY CELL-VERTEX MESHES SUPPORTED BY NOW! 

        PetscInt pStart, pEnd, cStart, cEnd, vStart, vEnd;

        PetscSectionCreate(PETSC_COMM_WORLD, &dof_section); // Create the section (PetscSection)

        DMPlexGetChart        (domain_->mesh.dm,    &pStart, &pEnd);
        DMPlexGetHeightStratum(domain_->mesh.dm, 0, &cStart, &cEnd);   // cells
        DMPlexGetHeightStratum(domain_->mesh.dm, 1, &vStart, &vEnd);   // vertices, equivalent to DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);
                                                                       // https://petsc.org/release/manualpages/DMPlex/DMPlexStratify/
        // For cell-vertex meshes, vertices are depth 0 and cells are depth 1. For fully interpolated meshes, depth 0 for vertices, 
        // 1 for edges, and so on  until cells have depth equal to the dimension of the mesh.

        PetscSectionSetChart(dof_section, vStart, vEnd); // Set the chart for the section (PetscSection)
        for (auto &node: domain_->nodes()) {
            PetscSectionSetDof(dof_section, node.sieve_id.value(), node.num_dof()); 
            }
    };

    void setup_boundary_conditions(){
        for (auto &[plex_id, idx] : constraints_){
            PetscSectionAddConstraintDof(this->dof_section, plex_id,static_cast<PetscInt>(idx.first.size()));
            }

        PetscSectionSetUp(this->dof_section);
        
        for (const auto &[plex_id, idx] : constraints_){
            PetscSectionSetConstraintIndices(this->dof_section, plex_id, idx.first.data());
            }

        DMSetLocalSection (domain_->mesh.dm, dof_section); // Set the local section for the mesh
        DMSetUp(domain_->mesh.dm); // Setup the mesh

        set_dof_index(); // Set the dof index for each node in the domain IN EACH NODE OBJECT.
    }

    void setup_vectors(){
        DMCreateLocalVector(domain_->mesh.dm, &nodal_forces);
        DMCreateLocalVector(domain_->mesh.dm, &global_imposed_solution);
        VecDuplicate(global_imposed_solution, &current_state);

        VecSet(nodal_forces           , 0.0);
        VecSet(global_imposed_solution, 0.0);
        VecSet(current_state          , 0.0);
        
    };

    void setup_matrix(){
        DMCreateMatrix(domain_->mesh.dm, &Kt);
        DMSetMatType  (domain_->mesh.dm, MATAIJ); // Set the matrix type for the mesh
        DMSetUp(domain_->mesh.dm);
        MatZeroEntries(Kt);
    };

    void setup(){
        setup_boundary_conditions();
        setup_vectors();
        setup_matrix();
    };

    void fix_node(std::size_t node_idx) noexcept{
        auto& node = domain_->node(node_idx);
        auto  num_dofs = node.num_dof();
        auto  plex_id  = node.sieve_id.value();

        auto dofs_idx = std::vector<PetscInt>(num_dofs);
        std::iota(dofs_idx.begin(), dofs_idx.end(), 0);

        constraints_.insert({plex_id, {dofs_idx, std::vector<PetscScalar>(num_dofs, 0.0)}});

        is_bc_updated = false;
    }

    void fix_orthogonal_plane(const int i, const double val, const double tol = 1.0e-6) noexcept{
        for (auto &node : domain_->nodes()){
            if (std::abs(node.coord(i) - val) < tol){ //std::cout << "Fixing node: " << node.id() << " ----> "<< node.sieve_id.value() << std::endl;
                fix_node(node.id());
            }
        }
    }

    void fix_x (const double val, const double tol = 1.0e-6) noexcept {fix_orthogonal_plane(0, val, tol);} // ok
    void fix_y (const double val, const double tol = 1.0e-6) noexcept {fix_orthogonal_plane(1, val, tol);} // ???
    void fix_z (const double val, const double tol = 1.0e-6) noexcept {fix_orthogonal_plane(2, val, tol);} // NOT WORKING???? WHY???

    void apply_node_force(std::size_t node_idx, auto... force_components) //REFACTOR EN TERMINOS DEL PLEX
    requires (std::is_convertible_v<decltype(force_components), PetscScalar> && ...){
        
        const PetscScalar force[] = {static_cast<PetscScalar>(force_components)...};
        
        auto num_dofs = domain_->node(node_idx).num_dof();
    
        if (sizeof...(force_components) != num_dofs) throw std::runtime_error("Force components size mismatch.");        
        
        auto dofs = domain_->node(node_idx).dof_index().data(); // TODO: Poner en terminos del plex para no depender del puntero al nodo.

        VecSetValuesLocal(this->nodal_forces, num_dofs, dofs, force, ADD_VALUES);

        VecAssemblyBegin (this->nodal_forces);
        VecAssemblyEnd   (this->nodal_forces);
    }   

    // Only For Testing Purposes! This thing is not accurate.
    void _force_orthogonal_plane(const int d, const double val, auto... force_components) noexcept{
        std::size_t count = 0;
        double tol = 1.0e-6;

        for (auto &node : domain_->nodes()) if (std::abs(node.coord(d) - val) < tol) count++;
        
        if (count){
            for (auto &node : domain_->nodes()){
                if (std::abs(node.coord(d) - val) < tol){
                    apply_node_force(node.id(), std::forward<PetscScalar>(force_components/double(count))...);   
                }
            }
        }
    }       

    void inject_K(Mat& analysis_K){
        for (auto &element : elements) element.inject_K(analysis_K);

        MatAssemblyBegin(analysis_K, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd  (analysis_K, MAT_FINAL_ASSEMBLY);
    }


    void update_elements_state(){
        for (auto &element : elements) {
            element.set_material_point_state(*this);
        }
    }



    void record_gauss_strains(VTKDataContainer &recorder){

        std::size_t N = domain_->num_integration_points() * MaterialPolicy::StrainT::num_components;
        
        std::vector<PetscScalar> strains; 
        strains.reserve(N);

        for (auto &element : elements){
            for (auto &gauss_point : element.material_points()){
                auto strain = gauss_point.current_state().vector().data();
                for (std::size_t i = 0; i < MaterialPolicy::StrainT::num_components; i++){
                    strains.push_back(strain[i]);
                }
            }
        }

        recorder.load_gauss_tensor_field("Cauchy Strain", strains.data(), domain_->num_integration_points());  

    }






    // Constructors
    Model(Domain<dim> &domain, MaterialT default_mat) : domain_(std::addressof(domain)){
        elements.reserve(domain_->num_elements()); // El dominio ya debe tener TODOS LOS ELEMENTOS GEOMETRICOS CREADOS!

        // GEOMETRIC ELEMENT WRAPPING        
        for (auto &element : domain_->elements()){                                     //By now all elements are ContinuumElements
            elements.emplace_back(FEM_Element{std::addressof(element), default_mat}); //By default, all elements have the same material
        }

        // PETSC SETUP
        DMSetVecType(domain_->mesh.dm, VECSTANDARD);        
        DMSetDimension(domain_->mesh.dm, dim); // Set the dimension of the mesh
        DMSetBasicAdjacency(domain_->mesh.dm, PETSC_FALSE, PETSC_TRUE); // Set the adjacency information for the FEM mesh.

        set_num_dofs_in_elements(); // 1. Set the number of dofs per node in the elements (FEM_Elements) // TODO: AVOID THIS!.
        set_sieve_layout();         // 2. Set the sieve layout for the mesh
    }

    Model() = delete;
    ~Model() {
        VecDestroy(&nodal_forces);
        VecDestroy(&global_imposed_solution);
        //PetscSectionDestroy(&dof_section);
    }

};

    //void inject_K(){
    //    for (auto &element : elements) element.inject_K(K);
    //    MatAssemblyBegin(K, MAT_FINAL_ASSEMBLY);
    //    MatAssemblyEnd  (K, MAT_FINAL_ASSEMBLY);
    //    //MatView(K, PETSC_VIEWER_DRAW_WORLD); // Draw the matrix
    //    //domain_->mesh.view();                // View the mesh
    //}

    //auto get_node_solution(PetscInt node_plex_id) // move to analysis?
    //{
    //    PetscInt offset, num_dofs;
    //    const PetscScalar *u;
    //    PetscSectionGetOffset(dof_section, node_plex_id, &offset); 
    //    PetscSectionGetDof   (dof_section, node_plex_id, &num_dofs);
    //    VecGetArrayRead(U, &u);
    //    auto dofs = std::span<const PetscScalar>(u + offset, num_dofs);
    //    VecRestoreArrayRead(U, &u);
    //    return dofs; 
    //}

#endif // FALL_N_MODEL_HH
