#ifndef FALL_N_MODEL_HH
#define FALL_N_MODEL_HH

#include <cstddef>
#include <map>
#include <memory>
#include <optional>
#include <type_traits>

#include "../domain/Domain.hh"

#include "../materials/Material.hh"
#include "../materials/MaterialPolicy.hh"

#include "../elements/ContinuumElement.hh" // Se usa este por ahora mientras se define la interfaz del wrapper.
#include "../elements/ElementPolicy.hh"
#include "../elements/SurfaceLoad.hh"

#include "../continuum/KinematicPolicy.hh"

#include "MaterialPoint.hh"

#include <petsc.h>


// https://www.dealii.org/current/doxygen/deal.II/namespacePETScWrappers.html
// https://stackoverflow.com/questions/872675/policy-based-design-and-best-practices-c

//using LinealElastic3D = ElasticRelation<ThreeDimensionalMaterial>;
//using LinealElastic2D = ElasticRelation<PlaneMaterial>;
//using LinealElastic1D = ElasticRelation<UniaxialMaterial>;

// The MaterialPolicy defines the constitutive relation and the number of dimensions.
// The KinematicPolicy selects the kinematic formulation (SmallStrain by default).
template <
    typename MaterialPolicy,
    typename KinematicPolicy = continuum::SmallStrain,
    std::size_t ndofs = MaterialPolicy::dim, //Default: Solid Model with "dim" displacements per node.
    typename ElemPolicy = SingleElementPolicy<ContinuumElement<MaterialPolicy, ndofs, KinematicPolicy>>
    >
class Model{
    friend class Analysis; // Por ahora. Para no exponer publicamentge el dominio.

public:    
    using MaterialT         = Material<MaterialPolicy>;
    using element_type      = typename ElemPolicy::element_type;
    using container_type    = typename ElemPolicy::container_type;
    using ConstraintDofInfo = std::map<PetscInt, std::pair<std::vector<PetscInt>, std::vector<PetscScalar>>>; 
    
    static constexpr std::size_t dim{MaterialPolicy::dim};

private:
    Domain<dim>*      domain_;
    ConstraintDofInfo constraints_; 
    PetscSection      dof_section{nullptr}; 

    container_type elements_;

    Vec nodal_forces_           {nullptr}; 
    Vec global_imposed_solution_{nullptr};
    Vec current_state_          {nullptr};

    Mat Kt_{nullptr};

public:

    // ── Element access ───────────────────────────────────────────────
    const container_type& elements() const noexcept { return elements_; }
          container_type& elements()       noexcept { return elements_; }

    std::size_t num_elements() const noexcept { return elements_.size(); }

    // ── PETSc handle access (non-owning views) ──────────────────────
    Vec  state_vector()        const noexcept { return current_state_; }
    Vec  imposed_solution()    const noexcept { return global_imposed_solution_; }
    Vec  force_vector()        const noexcept { return nodal_forces_; }
    Mat  stiffness_matrix()    const noexcept { return Kt_; }

    auto& get_domain()       { return *domain_; }
    const auto& get_domain() const { return *domain_; }
    auto  get_plex()         { return domain_->mesh.dm; };

private:

    void set_num_dofs_in_elements(){ for (auto &element : elements_) element.set_num_dof_in_nodes();}

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

        // Use the full DM chart [pStart, pEnd) instead of [vStart, vEnd).
        // Workaround: PETSc ≤3.24 PetscSectionSetConstraintIndices()
        // accesses atlasDof[point] instead of atlasDof[point - pStart],
        // causing out-of-bounds reads when pStart > 0 and pStart > (pEnd-pStart).
        // Setting pStart = 0 makes the raw index coincide with the adjusted one.
        // Elements simply keep 0 DOFs.
        PetscSectionSetChart(dof_section, pStart, pEnd);
        std::cout << "  [SIEVE-DBG] section_ptr=" << (void*)dof_section
                  << " chart=[" << pStart << "," << pEnd << ")" << std::endl;
        for (auto &node: domain_->nodes()) {
            PetscSectionSetDof(dof_section, node.sieve_id.value(), node.num_dof()); 
            }

        // ── DEBUG: verify section DOFs ───
        {
            PetscInt sec_pStart, sec_pEnd;
            PetscSectionGetChart(dof_section, &sec_pStart, &sec_pEnd);
            int mismatches = 0;
            for (auto &node: domain_->nodes()) {
                PetscInt ndof;
                PetscSectionGetDof(dof_section, node.sieve_id.value(), &ndof);
                if (ndof != static_cast<PetscInt>(node.num_dof())) {
                    if (mismatches < 10)
                        std::cerr << "  [SIEVE-DBG] MISMATCH sieve=" << node.sieve_id.value()
                                  << " node_id=" << node.id()
                                  << " section_dof=" << ndof
                                  << " node_dof=" << node.num_dof()
                                  << " chart=[" << sec_pStart << "," << sec_pEnd << ")\n";
                    ++mismatches;
                }
            }
            if (mismatches > 0)
                std::cerr << "  [SIEVE-DBG] Total mismatches: " << mismatches << "\n";
            else
                std::cerr << "  [SIEVE-DBG] All " << domain_->nodes().size() << " section DOFs OK\n";
        }
    };

    void setup_boundary_conditions(){
        for (auto &[plex_id, idx] : constraints_){
            PetscSectionAddConstraintDof(this->dof_section, plex_id,static_cast<PetscInt>(idx.first.size()));
            }

        PetscErrorCode ierr = PetscSectionSetUp(this->dof_section);
        if (ierr) std::cerr << "  [BC-ERR] PetscSectionSetUp returned " << ierr << std::endl;

        // ── DEBUG: verify section DOFs before SetConstraintIndices ───
        {
            std::cout << "  [BC-DBG] Constraints count: " << constraints_.size() << std::endl;
            for (const auto &[plex_id, idx] : constraints_){
                PetscInt ndof = -999;
                PetscSectionGetDof(this->dof_section, plex_id, &ndof);
                std::cout << "  [BC-DBG] plex_id=" << plex_id
                          << " section_dof=" << ndof
                          << " constraint_size=" << idx.first.size() << std::endl;
            }
            std::cout << std::flush;
        }
        
        for (const auto &[plex_id, idx] : constraints_){
            ierr = PetscSectionSetConstraintIndices(this->dof_section, plex_id, idx.first.data());
            if (ierr) std::cerr << "  [BC-ERR] SetConstraintIndices for plex_id=" << plex_id << " returned " << ierr << std::endl;
            }

        ierr = DMSetLocalSection (domain_->mesh.dm, dof_section); // Set the local section for the mesh
        if (ierr) std::cerr << "  [BC-ERR] DMSetLocalSection returned " << ierr << std::endl;
        ierr = DMSetUp(domain_->mesh.dm); // Setup the mesh
        if (ierr) std::cerr << "  [BC-ERR] DMSetUp returned " << ierr << std::endl;

        set_dof_index(); // Set the dof index for each node in the domain IN EACH NODE OBJECT.
    }

    void setup_vectors(){
        PetscErrorCode verr;
        verr = DMCreateLocalVector(domain_->mesh.dm, &nodal_forces_);
        if (verr) std::cerr << "  [VEC-ERR] DMCreateLocalVector(nf) returned " << verr << std::endl;
        verr = DMCreateLocalVector(domain_->mesh.dm, &global_imposed_solution_);
        if (verr) std::cerr << "  [VEC-ERR] DMCreateLocalVector(gis) returned " << verr << std::endl;
        VecDuplicate(global_imposed_solution_, &current_state_);

        VecSet(nodal_forces_           , 0.0);
        VecSet(global_imposed_solution_, 0.0);
        VecSet(current_state_          , 0.0);
        
    };

    void setup_matrix(){
        DMCreateMatrix(domain_->mesh.dm, &Kt_);
        // Allow dynamic allocation for entries missed by DMPlex preallocation
        // (unstructured simplex meshes may have adjacency underestimates).
        MatSetOption(Kt_, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
        MatZeroEntries(Kt_);
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

        VecSetValuesLocal(this->nodal_forces_, num_dofs, dofs, force, ADD_VALUES);

        VecAssemblyBegin (this->nodal_forces_);
        VecAssemblyEnd   (this->nodal_forces_);
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

    // ── Consistent surface traction (proper Neumann BC) ─────────────────
    //
    //  Applies uniform traction  t ∈ ℝ^dim  on a boundary surface group.
    //  The consistent nodal forces  f_I = ∫_Γ N_I t dA  are computed by
    //  Gauss quadrature over each surface element and assembled into the
    //  PETSc local nodal_forces vector.
    //
    //  group_name: physical group name from the Gmsh mesh (e.g., "Load")
    //
    void apply_surface_traction(const std::string& group_name, auto... traction_components)
    requires (sizeof...(traction_components) == dim
              && (std::is_convertible_v<decltype(traction_components), double> && ...))
    {
        auto surf_elems = domain_->boundary_elements(group_name);
        if (surf_elems.empty()) {
            std::cerr << "Warning: No boundary elements found for group '" 
                      << group_name << "'\n";
            return;
        }

        std::array<double, dim> traction = {static_cast<double>(traction_components)...};
        surface_load::apply_traction<dim>(surf_elems, traction, this->nodal_forces_);
    }       

    void inject_K(Mat& analysis_K){
        for (auto &element : elements_) element.inject_K(analysis_K);

        MatAssemblyBegin(analysis_K, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd  (analysis_K, MAT_FINAL_ASSEMBLY);
    }

    // ── Mass matrix assembly (for dynamics) ──────────────────────────
    //
    //  Assembles the global consistent mass matrix M by summing
    //  element-level contributions:  M = Σ_e M_e
    //
    //  Elements must have had their density set (via set_density)
    //  before calling this.

    void assemble_mass_matrix(Mat M) {
        MatZeroEntries(M);
        for (auto& element : elements_) {
            if constexpr (requires { element.inject_mass(M); }) {
                element.inject_mass(M);
            }
        }
        MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);
    }

    // ── Set density on all elements ──────────────────────────────────

    void set_density(double rho) {
        for (auto& element : elements_) {
            if constexpr (requires { element.set_density(rho); }) {
                element.set_density(rho);
            }
        }
    }

    // ── Set density per physical group ───────────────────────────────

    void set_density(const std::map<std::string, double>& density_map,
                     double default_density = 0.0)
    {
        for (auto& element : elements_) {
            double rho = default_density;
            if constexpr (requires { element.physical_group(); }) {
                if (element.has_physical_group()) {
                    auto it = density_map.find(element.physical_group());
                    if (it != density_map.end()) rho = it->second;
                }
            }
            if constexpr (requires { element.set_density(rho); }) {
                element.set_density(rho);
            }
        }
    }


    void update_elements_state(){
        if constexpr (requires(element_type e) { e.set_material_point_state(*static_cast<Model*>(nullptr)); }) {
            for (auto &element : elements_) {
                element.set_material_point_state(*this);
            }
        }
        // For structural elements, commit_material_state already handles state update.
    }


    // Constructors

    // Constructor 1: auto-creates elements from domain geometries + material.
    //   Works for ContinuumElement, BeamElement, or any concrete element type
    //   constructible from (ElementGeometry<dim>*, MaterialT).
    Model(Domain<dim> &domain, MaterialT default_mat)
        requires (std::constructible_from<element_type, ElementGeometry<dim>*, MaterialT>)
        : domain_(std::addressof(domain))
    {
        elements_.reserve(domain_->num_elements()); // El dominio ya debe tener TODOS LOS ELEMENTOS GEOMETRICOS CREADOS!

        // GEOMETRIC ELEMENT WRAPPING        
        for (auto &element : domain_->elements()){ //By now all elements are ContinuumElements
            elements_.emplace_back(element_type{std::addressof(element), default_mat}); //By default, all elements have the same material
        }

        // PETSC SETUP
        DMSetVecType(domain_->mesh.dm, VECSTANDARD);        
        DMSetDimension(domain_->mesh.dm, dim); // Set the dimension of the mesh
        DMSetBasicAdjacency(domain_->mesh.dm, PETSC_FALSE, PETSC_TRUE); // Set the adjacency information for the FEM mesh.

        set_num_dofs_in_elements(); // 1. Set the number of dofs per node in the elements (FEM_Elements) // TODO: AVOID THIS!.
        set_sieve_layout();         // 2. Set the sieve layout for the mesh
    }

    // Constructor 2: takes pre-built elements (for type-erased wrappers
    //   like StructuralElement, or any case where element construction
    //   differs from the simple (geometry*, material) pattern).
    Model(Domain<dim> &domain, container_type pre_built)
        : domain_(std::addressof(domain)),
          elements_(std::move(pre_built))
    {
        // PETSC SETUP
        DMSetVecType(domain_->mesh.dm, VECSTANDARD);        
        DMSetDimension(domain_->mesh.dm, dim);
        DMSetBasicAdjacency(domain_->mesh.dm, PETSC_FALSE, PETSC_TRUE);

        set_num_dofs_in_elements();
        set_sieve_layout();
    }

    // Constructor 3: multi-material — assigns material per physical group.
    //
    //   material_map: physical group name → material instance.
    //   Each element's geometry must carry a physical_group() set by the
    //   mesh builder; elements whose group is not in the map receive
    //   default_mat (if provided) or trigger an error.
    //
    //   Usage:
    //     std::map<std::string, MaterialT> mats{
    //         {"Steel",    Material<...>{...}},
    //         {"Concrete", Material<...>{...}},
    //     };
    //     Model M{domain, mats};
    //
    Model(Domain<dim> &domain, const std::map<std::string, MaterialT>& material_map,
          std::optional<MaterialT> default_mat = std::nullopt)
        requires (std::constructible_from<element_type, ElementGeometry<dim>*, MaterialT>)
        : domain_(std::addressof(domain))
    {
        elements_.reserve(domain_->num_elements());

        for (auto &geom : domain_->elements()) {
            const auto& group = geom.physical_group();

            if (auto it = material_map.find(group); it != material_map.end()) {
                elements_.emplace_back(element_type{std::addressof(geom), it->second});
            } else if (default_mat.has_value()) {
                elements_.emplace_back(element_type{std::addressof(geom), *default_mat});
            } else {
                throw std::runtime_error(
                    "Model: no material for physical group '" + group
                    + "' and no default material provided.");
            }
        }

        // PETSC SETUP
        DMSetVecType(domain_->mesh.dm, VECSTANDARD);
        DMSetDimension(domain_->mesh.dm, dim);
        DMSetBasicAdjacency(domain_->mesh.dm, PETSC_FALSE, PETSC_TRUE);

        set_num_dofs_in_elements();
        set_sieve_layout();
    }

    Model() = delete;
    ~Model() {
        PetscErrorCode e;
        e = MatDestroy(&Kt_); if (e) std::cerr << "[~Model] MatDestroy: " << e << "\n";
        e = VecDestroy(&current_state_); if (e) std::cerr << "[~Model] VecDestroy(cs): " << e << "\n";
        e = VecDestroy(&nodal_forces_); if (e) std::cerr << "[~Model] VecDestroy(nf): " << e << "\n";
        e = VecDestroy(&global_imposed_solution_); if (e) std::cerr << "[~Model] VecDestroy(gis): " << e << "\n";
        e = PetscSectionDestroy(&dof_section); if (e) std::cerr << "[~Model] SectionDestroy: " << e << "\n";
    }

};

#endif // FALL_N_MODEL_HH
