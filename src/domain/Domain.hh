#ifndef FN_DOMAIN
#define FN_DOMAIN

#include <cstddef>
#include <functional>
#include <utility>
#include <optional>
#include <format>
#include <iostream>
#include <memory>
#include <vector>
#include <map>
#include <string>
#include <set>
#include <unordered_set>
#include <cmath>
#include <algorithm>
#include <numeric>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "../geometry/Topology.hh"
#include "../mesh/Mesh.hh"

#include "../elements/Node.hh"

#include "../elements/element_geometry/ElementGeometry.hh"

#include "../mesh/gmsh/ReadGmsh.hh"
// #include "../mesh/gmsh/GmshDomainBuilder.hh"

template <std::size_t dim> requires topology::EmbeddableInSpace<dim>
class Domain
{ // Spacial (Phisical) Domain. Where the simulation takes place
    static constexpr std::size_t dim_ = dim;

    std::vector<Node<dim>>            nodes_   ;
    std::vector<ElementGeometry<dim>> elements_;

    // ── Boundary (surface) element storage ──────────────────────────────
    //  Keyed by physical group name (e.g., "Fixed", "Load").
    //  Each entry holds a vector of surface ElementGeometry objects
    //  whose nodes are pointers to the same Node<dim> objects in nodes_.
    std::map<std::string, std::vector<ElementGeometry<dim>>> boundary_elements_;

    // ── O(1) node lookup: direct-address table (id → pointer) ──────────
    //  Built once via build_node_index(). Since Gmsh/PETSc IDs are dense
    //  non-negative integers, a flat vector is optimal: O(1) access,
    //  perfect cache locality, ~8 bytes per node.
    std::vector<Node<dim>*> node_by_id_;
    bool node_index_built_ = false;

    std::optional<std::size_t>        num_integration_points_;

public:

    Mesh mesh;

    inline std::size_t num_nodes()    const { return nodes_.size(); };
    inline std::size_t num_elements() const { return elements_.size(); };
    
    std::size_t num_integration_points(){
        if (!num_integration_points_.has_value()){
            std::size_t n = 0;
            for (auto &e : elements_){
                n += e.num_integration_points();
            }
            //set optional value
            num_integration_points_ = n;
        }
        //std::cout << "Number of integration points: " << num_integration_points_.value() << std::endl;
        return num_integration_points_.value();
    };

    // ── Build the O(1) node index ────────────────────────────────────
    void build_node_index() {
        if (nodes_.empty()) return;
        std::size_t max_id = 0;
        for (const auto& n : nodes_) max_id = std::max(max_id, n.id());
        node_by_id_.assign(max_id + 1, nullptr);
        for (auto& n : nodes_) node_by_id_[n.id()] = &n;
        node_index_built_ = true;
    }

    // ── Node accessors: O(1) via direct-address table ────────────────
    Node<dim>* node_p(std::size_t id) {
        return node_by_id_[id];
    }

    Node<dim>& node(std::size_t id) {
        return *node_by_id_[id];
    }

    std::span<Node<dim>>            nodes()    { return std::span<Node<dim>>           (nodes_);    };  
    std::span<ElementGeometry<dim>> elements() { return std::span<ElementGeometry<dim>>(elements_); };
    ElementGeometry<dim>&           element(std::size_t i) { return elements_[i]; };

    // ── Boundary access ─────────────────────────────────────────────────
    bool has_boundary_group(const std::string& name) const {
        return boundary_elements_.count(name) > 0;
    }

    std::span<ElementGeometry<dim>> boundary_elements(const std::string& name) {
        auto it = boundary_elements_.find(name);
        if (it == boundary_elements_.end()) return {};
        return std::span<ElementGeometry<dim>>(it->second);
    }

    std::vector<std::string> boundary_group_names() const {
        std::vector<std::string> names;
        for (const auto& [name, _] : boundary_elements_) names.push_back(name);
        return names;
    }

    template <typename ElementType, typename IntegrationStrategy>
    auto& make_boundary_element(const std::string& group_name,
                                IntegrationStrategy&& integrator,
                                std::size_t tag, PetscInt node_ids[]) {
        auto& vec = boundary_elements_[group_name];
        vec.emplace_back(
            ElementGeometry<dim>(
                ElementType(
                    std::forward<std::size_t>(tag),
                    std::span<PetscInt>(node_ids, ElementType::num_nodes)),
                std::forward<IntegrationStrategy>(integrator)));
        return vec.back();
    }

    // ── Create boundary elements from a coordinate plane ────────────────
    //
    //  Finds all volume element faces whose nodes all lie on the plane
    //  x_d = val (within tolerance).  Creates surface ElementGeometry
    //  objects by querying each element's subentity topology (generic —
    //  no hard-coded face tables).
    //
    //  This is useful when the mesh doesn't define boundary physical groups
    //  for all surfaces of interest.
    //
    //  Must be called AFTER assemble_sieve() (nodes need to be bound).
    //
    void create_boundary_from_plane(const std::string& group_name,
                                    int d, double val,
                                    double tol = 1.0e-6)
    {
        if (!node_index_built_) build_node_index();

        // 1. Collect IDs of nodes on the plane — O(1) lookup via unordered_set
        std::unordered_set<PetscInt> plane_node_ids;
        plane_node_ids.reserve(nodes_.size() / 4); // heuristic
        for (const auto& nd : nodes_) {
            if (std::abs(nd.coord(d) - val) < tol) {
                plane_node_ids.insert(static_cast<PetscInt>(nd.id()));
            }
        }
        if (plane_node_ids.empty()) return;

        // 2. For each volume element, check each face via generic subentity topology
        std::size_t surf_tag = 900000;

        for (auto& elem : elements_) {
            const auto nf = elem.num_faces();

            for (std::size_t f = 0; f < nf; ++f) {
                auto local_indices = elem.face_node_indices(f);
                const auto fn = local_indices.size();

                // Check if ALL face nodes lie on the plane
                bool all_on_plane = true;
                for (std::size_t k = 0; k < fn; ++k) {
                    PetscInt nid = static_cast<PetscInt>(
                        elem.node_p(local_indices[k]).id());
                    if (plane_node_ids.find(nid) == plane_node_ids.end()) {
                        all_on_plane = false;
                        break;
                    }
                }

                if (all_on_plane) {
                    // Collect global node IDs for the face
                    std::vector<PetscInt> face_node_ids(fn);
                    for (std::size_t k = 0; k < fn; ++k) {
                        face_node_ids[k] = static_cast<PetscInt>(
                            elem.node_p(local_indices[k]).id());
                    }

                    // Create the surface element via the generic factory
                    auto& vec = boundary_elements_[group_name];
                    vec.push_back(elem.make_face_geometry(
                        f, surf_tag++,
                        std::span<PetscInt>(face_node_ids.data(), fn)));

                    // Bind nodes via O(1) lookup
                    auto& new_elem = vec.back();
                    for (std::size_t i = 0; i < fn; ++i) {
                        new_elem.bind_node(i, node_by_id_[new_elem.node(i)]);
                    }
                }
            }
        }
    }

    // ===========================================================================================================

    void setup_integration_points(){
        const auto ne = elements_.size();

        // 1. Pre-compute per-element offsets via prefix sum (serial — O(n) integers)
        std::vector<std::size_t> offsets(ne);
        std::size_t total = 0;
        for (std::size_t i = 0; i < ne; ++i) {
            offsets[i] = total;
            total += elements_[i].num_integration_points();
        }

        // 2. Initialize integration points in parallel (each element is independent)
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (std::size_t i = 0; i < ne; ++i) {
            elements_[i].setup_integration_points(offsets[i]);
        }

        std::cout << "Number of integration points: " << num_integration_points() << std::endl;
        std::cout << "offset: " << total << std::endl;
    }

    void link_nodes_to_elements(){
        if (!node_index_built_) build_node_index();

        // Volume elements — O(1) lookup, parallelizable (each element is independent)
        const auto ne = elements_.size();
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (std::size_t e = 0; e < ne; ++e) {
            for (std::size_t i = 0; i < elements_[e].num_nodes(); ++i) {
                elements_[e].bind_node(i, node_by_id_[elements_[e].node(i)]);
            }
        }

        // Boundary (surface) elements — O(1) lookup
        for (auto& [name, surf_elems] : boundary_elements_) {
            for (auto& elem : surf_elems) {
                for (std::size_t i = 0; i < elem.num_nodes(); ++i) {
                    elem.bind_node(i, node_by_id_[elem.node(i)]);
                }
            }
        }
    }

    void assemble_sieve() {
        link_nodes_to_elements();
        setup_integration_points();

        // Uninterpoleated topology by now (no edges or faces).
        mesh.set_size(PetscInt(num_nodes() + num_elements())); // Number of DAG points = nodes + edges + faces + cells
        
        PetscInt sieve_point_idx = 0;

        for (auto &e : elements_){
            e.set_sieve_id(sieve_point_idx);
            mesh.set_sieve_cone_size(sieve_point_idx, e.num_nodes());
            ++sieve_point_idx;
        }

        for (auto &n : nodes_){  // Para esto se requiere haber linkeado los nodos a los elementos primero.
            n.set_sieve_id(sieve_point_idx);
            ++sieve_point_idx;
        }

        mesh.setup(sieve_point_idx);

        // Set the sieve cone for each entity (only elements by now). 
        // The nodes doesn't cover any entity of lower dimension.
        // Stack-allocated buffer avoids per-element heap allocation.
        constexpr std::size_t MAX_NODES_PER_ELEM = 64; // ≥ max(27 hex27, 64 hex4⁴)
        for (auto &e : elements_){
            PetscInt cone[MAX_NODES_PER_ELEM];
            const auto nn = e.num_nodes();
            for (std::size_t i = 0; i < nn; ++i){
                cone[i] = e.node_p(i).sieve_id.value();
            }
            mesh.set_sieve_cone(e.sieve_id.value(), cone);
        }
        mesh.symmetrize_sieve();
    }

    template <typename ElementType, typename IntegrationStrategy>
    auto& make_element(IntegrationStrategy &&integrator, std::size_t tag, PetscInt node_ids[]){
        elements_.emplace_back(
            ElementGeometry<dim>(
                ElementType( // Forward this?
                    std::forward<std::size_t>(tag),
                    std::span<PetscInt>(node_ids, ElementType::num_nodes)),
                std::forward<IntegrationStrategy>(integrator)));

        return elements_.back(); 
    }

    template <typename ElementType, typename IntegrationStrategy>
    auto& make_element(IntegrationStrategy &&integrator, std::size_t tag, PetscInt node_ids[], PetscInt local_ordering[]){
        elements_.emplace_back(
            ElementGeometry<dim>(
                ElementType( // Forward this?
                    std::forward<std::size_t>(tag),
                    std::span<PetscInt>(node_ids      , ElementType::num_nodes()),
                    std::span<PetscInt>(local_ordering, ElementType::num_nodes())),
                std::forward<IntegrationStrategy>(integrator)));
        return elements_.back();
    }
    

    void add_node(std::size_t tag, std::floating_point auto... coords) 
    requires (sizeof...(coords) == dim){
        nodes_.emplace_back(Node<dim>(tag, coords...));
    };

    Node<dim> *add_node(Node<dim> &&node){ 
        nodes_.emplace_back(std::forward<Node<dim>>(node));
        return &nodes_.back();
    };

    // Reserve capacity before any node insertion (avoids repeated reallocation).
    void preallocate_node_capacity(std::size_t n, double margin = 1.20) {
        if (nodes_.empty()) {
            nodes_.reserve(static_cast<std::size_t>(n * margin));
        }
    }

    // Constructors
    // Copy Constructor
    Domain(const Domain &other) = default;
    // Move Constructor
    Domain(Domain &&other) = default;
    // Copy Assignment
    Domain &operator=(const Domain &other) = default;
    // Move Assignment
    Domain &operator=(Domain &&other) = default;

    Domain() = default;
    ~Domain() = default;
};

#endif


    //template <typename ElementType, typename IntegrationStrategy>
    //void make_element(IntegrationStrategy &&integrator, std::size_t &&tag, std::vector<Node<3> *> nodeAdresses)
    //{
    //    elements_.emplace_back(
    //        ElementGeometry<dim>(
    //            ElementType( // Forward this?
    //                std::forward<std::size_t>(tag),
    //                std::forward<std::vector<Node<3> *>>(nodeAdresses)),
    //            std::forward<IntegrationStrategy>(integrator)));
    //}