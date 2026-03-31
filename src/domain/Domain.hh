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
#include <string_view>
#include <set>
#include <span>
#include <unordered_set>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <tuple>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "../geometry/Topology.hh"
#include "../geometry/Vertex.hh"
#include "../mesh/Mesh.hh"

#include "../elements/Node.hh"

#include "../elements/element_geometry/ElementGeometry.hh"

template <std::size_t dim> requires topology::EmbeddableInSpace<dim>
class Domain
{ // Spatial domain = geometry/topology owner with analysis-node cache.
  //
  // Canonical ownership is moving toward Vertex<dim>, not Node<dim>.
  // The Node container below is kept as a transitional analysis cache so that
  // Model, ElementGeometry, and existing element kernels continue to work
  // while the rest of the codebase is migrated.

    static constexpr std::size_t dim_ = dim;
    static constexpr std::string_view point_role_label_name_{"fall_n.point_role"};
    static constexpr std::string_view topological_dim_label_name_{"fall_n.topological_dim"};
    static constexpr std::string_view physical_group_label_name_{"fall_n.physical_group_id"};

    std::vector<geometry::Vertex<dim>> vertices_;
    std::vector<Node<dim>>            nodes_cache_;
    std::vector<ElementGeometry<dim>> elements_;

    // ── Boundary (surface) element storage ──────────────────────────────
    //  Keyed by physical group name (e.g., "Fixed", "Load").
    //  Each entry holds a vector of surface ElementGeometry objects
    //  whose geometry is bound to the same Vertex<dim> objects as the
    //  volume mesh and whose Node<dim> binding is optional.
    std::map<std::string, std::vector<ElementGeometry<dim>>> boundary_elements_;

    // ── O(1) lookup tables: direct-address tables (id → pointer) ───────
    //  vertices_ is the canonical geometric storage.
    //  nodes_cache_ is the analysis-level materialization derived from it.
    std::vector<geometry::Vertex<dim>*> vertex_by_id_;
    bool vertex_index_built_ = false;

    std::vector<Node<dim>*> node_by_id_;
    bool node_cache_valid_ = false;
    bool node_index_built_ = false;

    std::map<std::string, PetscInt, std::less<>> physical_group_ids_;
    mutable std::optional<std::size_t>        num_integration_points_;

    enum class PlexPointRole : PetscInt {
        vertex = 0,
        cell   = 1
    };

public:

    Mesh mesh;

    inline std::size_t num_vertices() const { return vertices_.size(); };
    inline std::size_t num_nodes()    const { return vertices_.size(); };
    inline std::size_t num_elements() const { return elements_.size(); };
    inline std::size_t plex_dimension() const {
        std::size_t max_dim = 0;
        for (const auto& element : elements_) {
            max_dim = std::max(max_dim, element.topological_dimension());
        }
        return max_dim;
    }

    static constexpr std::string_view point_role_label_name() noexcept { return point_role_label_name_; }
    static constexpr std::string_view topological_dimension_label_name() noexcept { return topological_dim_label_name_; }
    static constexpr std::string_view physical_group_label_name() noexcept { return physical_group_label_name_; }

    std::optional<PetscInt> physical_group_id(std::string_view group) const {
        if (auto it = physical_group_ids_.find(group); it != physical_group_ids_.end()) {
            return it->second;
        }
        return std::nullopt;
    }
    
    std::size_t num_integration_points() const {
        if (!num_integration_points_.has_value()){
            std::size_t n = 0;
            for (const auto &e : elements_){
                n += e.num_integration_points();
            }
            num_integration_points_ = n;
        }
        return num_integration_points_.value();
    };

private:
    static Node<dim> make_node_from_vertex(const geometry::Vertex<dim>& vertex) {
        return Node<dim>(vertex.id(), vertex.coord_ref());
    }

    void invalidate_vertex_index() noexcept {
        vertex_by_id_.clear();
        vertex_index_built_ = false;
    }

    void invalidate_node_cache() noexcept {
        nodes_cache_.clear();
        node_by_id_.clear();
        node_cache_valid_ = false;
        node_index_built_ = false;
        num_integration_points_.reset();
    }

    void rebuild_physical_group_ids() {
        physical_group_ids_.clear();
        PetscInt next_id = 1;
        for (const auto& element : elements_) {
            if (!element.has_physical_group()) continue;
            const auto& group = element.physical_group();
            if (!physical_group_ids_.contains(group)) {
                physical_group_ids_.emplace(group, next_id++);
            }
        }
    }

    void label_mesh_points() {
        rebuild_physical_group_ids();

        for (const auto& element : elements_) {
            const auto sieve_id = element.sieve_id();
            mesh.set_label_value(point_role_label_name_, sieve_id,
                                 static_cast<PetscInt>(PlexPointRole::cell));
            mesh.set_label_value(topological_dim_label_name_, sieve_id,
                                 static_cast<PetscInt>(element.topological_dimension()));

            if (element.has_physical_group()) {
                if (const auto group_id = physical_group_ids_.find(element.physical_group());
                    group_id != physical_group_ids_.end()) {
                    mesh.set_label_value(physical_group_label_name_, sieve_id, group_id->second);
                }
            }
        }

        for (const auto& node : nodes_cache_) {
            const auto sieve_id = node.sieve_id.value();
            mesh.set_label_value(point_role_label_name_, sieve_id,
                                 static_cast<PetscInt>(PlexPointRole::vertex));
            mesh.set_label_value(topological_dim_label_name_, sieve_id, 0);
        }
    }

    void ensure_node_cache() {
        if (node_cache_valid_) return;

        nodes_cache_.clear();
        nodes_cache_.reserve(vertices_.size());
        for (const auto& vertex : vertices_) {
            nodes_cache_.emplace_back(make_node_from_vertex(vertex));
        }

        node_cache_valid_ = true;
        node_index_built_ = false;
    }

public:

    // ── Build the O(1) vertex index ──────────────────────────────────
    void build_vertex_index() {
        if (vertices_.empty()) return;
        std::size_t max_id = 0;
        for (const auto& v : vertices_) max_id = std::max(max_id, v.id());
        vertex_by_id_.assign(max_id + 1, nullptr);
        for (auto& v : vertices_) vertex_by_id_[v.id()] = &v;
        vertex_index_built_ = true;
    }

    // ── Build the O(1) node index ────────────────────────────────────
    void build_node_index() {
        ensure_node_cache();
        if (nodes_cache_.empty()) return;
        std::size_t max_id = 0;
        for (const auto& n : nodes_cache_) max_id = std::max(max_id, n.id());
        node_by_id_.assign(max_id + 1, nullptr);
        for (auto& n : nodes_cache_) node_by_id_[n.id()] = &n;
        node_index_built_ = true;
    }

    // ── Vertex accessors: O(1) via direct-address table ──────────────
    geometry::Vertex<dim>* vertex_p(std::size_t id) {
        if (!vertex_index_built_) build_vertex_index();
        return vertex_by_id_[id];
    }
    const geometry::Vertex<dim>* vertex_p(std::size_t id) const {
        auto* self = const_cast<Domain*>(this);
        if (!self->vertex_index_built_) self->build_vertex_index();
        return self->vertex_by_id_[id];
    }

    geometry::Vertex<dim>& vertex(std::size_t id) {
        return *vertex_p(id);
    }
    const geometry::Vertex<dim>& vertex(std::size_t id) const {
        return *vertex_p(id);
    }

    // ── Node accessors: O(1) via direct-address table ────────────────
    Node<dim>* node_p(std::size_t id) {
        if (!node_index_built_) build_node_index();
        return node_by_id_[id];
    }
    const Node<dim>* node_p(std::size_t id) const {
        auto* self = const_cast<Domain*>(this);
        if (!self->node_index_built_) self->build_node_index();
        return self->node_by_id_[id];
    }

    Node<dim>& node(std::size_t id) {
        if (!node_index_built_) build_node_index();
        return *node_by_id_[id];
    }
    const Node<dim>& node(std::size_t id) const {
        return *node_p(id);
    }

    std::span<geometry::Vertex<dim>>                vertices()       { return std::span<geometry::Vertex<dim>>(vertices_); }
    std::span<const geometry::Vertex<dim>>          vertices() const { return std::span<const geometry::Vertex<dim>>(vertices_); }

    std::span<Node<dim>>                    nodes() {
        ensure_node_cache();
        return std::span<Node<dim>>(nodes_cache_);
    }
    std::span<const Node<dim>>              nodes() const {
        auto* self = const_cast<Domain*>(this);
        self->ensure_node_cache();
        return std::span<const Node<dim>>(self->nodes_cache_);
    }
    std::span<ElementGeometry<dim>>         elements()       { return std::span<ElementGeometry<dim>>(elements_); };
    std::span<const ElementGeometry<dim>>   elements() const { return std::span<const ElementGeometry<dim>>(elements_); };
    ElementGeometry<dim>&           element(std::size_t i) { return elements_[i]; };
    const ElementGeometry<dim>&     element(std::size_t i) const { return elements_[i]; };
    [[nodiscard]] bool has_materialized_nodes() const noexcept { return node_cache_valid_; }
    [[nodiscard]] bool has_vertex_index() const noexcept { return vertex_index_built_; }
    [[nodiscard]] bool has_node_index() const noexcept { return node_index_built_; }

    void sort_vertices_by_id() {
        std::sort(vertices_.begin(), vertices_.end(),
                  [](const geometry::Vertex<dim>& a, const geometry::Vertex<dim>& b) {
                      return a.id() < b.id();
                  });
        invalidate_vertex_index();
        invalidate_node_cache();
    }

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
                    tag,
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
    //  This operation is geometry-first: it can run before the analysis-node
    //  cache exists.  If nodes are already materialized they are rebound too.
    //
    void create_boundary_from_plane(const std::string& group_name,
                                    int d, double val,
                                    double tol = 1.0e-6)
    {
        // 1. Collect IDs of nodes on the plane — O(1) lookup via unordered_set
        std::unordered_set<PetscInt> plane_node_ids;
        plane_node_ids.reserve(vertices_.size() / 4); // heuristic
        for (const auto& vertex : vertices_) {
            if (std::abs(vertex.coord(d) - val) < tol) {
                plane_node_ids.insert(static_cast<PetscInt>(vertex.id()));
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
                    PetscInt nid = elem.node(local_indices[k]);
                    if (plane_node_ids.find(nid) == plane_node_ids.end()) {
                        all_on_plane = false;
                        break;
                    }
                }

                if (all_on_plane) {
                    // Collect global node IDs for the face
                    std::vector<PetscInt> face_node_ids(fn);
                    for (std::size_t k = 0; k < fn; ++k) {
                        face_node_ids[k] = elem.node(local_indices[k]);
                    }

                    // Create the surface element via the generic factory
                    auto& vec = boundary_elements_[group_name];
                    vec.push_back(elem.make_face_geometry(
                        f, surf_tag++,
                        std::span<PetscInt>(face_node_ids.data(), fn)));

                    // Bind geometry immediately; node/DoF binding remains optional.
                    auto& new_elem = vec.back();
                    for (std::size_t i = 0; i < fn; ++i) {
                        new_elem.bind_point(i, vertex_by_id_[new_elem.node(i)]);
                        if (node_cache_valid_) {
                            if (!node_index_built_) build_node_index();
                            new_elem.bind_node(i, node_by_id_[new_elem.node(i)]);
                        }
                    }
                }
            }
        }
    }

    // ===========================================================================================================

    void setup_integration_points(){
        link_geometry_to_elements();
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
    }

    // Bind the purely geometric view first so mapping/Jacobians do not depend
    // on the analysis-node cache.
    void link_geometry_to_elements() {
        if (!vertex_index_built_) build_vertex_index();

        const auto ne = elements_.size();
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (std::size_t e = 0; e < ne; ++e) {
            for (std::size_t i = 0; i < elements_[e].num_nodes(); ++i) {
                elements_[e].bind_point(i, vertex_by_id_[elements_[e].node(i)]);
            }
        }

        for (auto& [name, surf_elems] : boundary_elements_) {
            for (auto& elem : surf_elems) {
                for (std::size_t i = 0; i < elem.num_nodes(); ++i) {
                    elem.bind_point(i, vertex_by_id_[elem.node(i)]);
                }
            }
        }
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
        ensure_node_cache();
        link_geometry_to_elements();
        link_nodes_to_elements();
        setup_integration_points();

        // Non-interpolated DMPlex by design for now: cells connect directly to
        // vertices. Mixed cell families are distinguished by labels rather
        // than by inserting explicit edges/faces into the DAG.
        mesh.set_dimension(static_cast<PetscInt>(plex_dimension()));
        mesh.set_size(PetscInt(num_nodes() + num_elements())); // Number of DAG points = nodes + edges + faces + cells
        
        PetscInt sieve_point_idx = 0;

        for (auto &e : elements_){
            e.set_sieve_id(sieve_point_idx);
            mesh.set_sieve_cone_size(sieve_point_idx, e.num_nodes());
            ++sieve_point_idx;
        }

        for (auto &n : nodes_cache_){  // Para esto se requiere haber linkeado los nodos a los elementos primero.
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
                cone[i] = node_by_id_[e.node(i)]->sieve_id.value();
            }
            mesh.set_sieve_cone(e.sieve_id(), cone);
        }
        mesh.symmetrize_sieve();
        label_mesh_points();
    }

    template <typename ElementType, typename IntegrationStrategy>
    auto& make_element(IntegrationStrategy &&integrator, std::size_t tag, PetscInt node_ids[]){
        elements_.emplace_back(
            ElementGeometry<dim>(
                ElementType( // Forward this?
                    tag,
                    std::span<PetscInt>(node_ids, ElementType::num_nodes)),
                std::forward<IntegrationStrategy>(integrator)));
        num_integration_points_.reset();

        return elements_.back(); 
    }

    template <typename ElementType, typename IntegrationStrategy>
    auto& make_element(IntegrationStrategy &&integrator, std::size_t tag, PetscInt node_ids[], PetscInt local_ordering[]){
        elements_.emplace_back(
            ElementGeometry<dim>(
                ElementType( // Forward this?
                    tag,
                    std::span<PetscInt>(node_ids      , ElementType::num_nodes),
                    std::span<PetscInt>(local_ordering, ElementType::num_nodes)),
                std::forward<IntegrationStrategy>(integrator)));
        num_integration_points_.reset();
        return elements_.back();
    }
    

    void add_vertex(std::size_t tag, std::floating_point auto... coords)
    requires (sizeof...(coords) == dim) {
        vertices_.emplace_back(geometry::Vertex<dim>(tag, coords...));
        invalidate_vertex_index();
        invalidate_node_cache();
    };

    geometry::Vertex<dim>* add_vertex(geometry::Vertex<dim>&& vertex){
        vertices_.emplace_back(std::forward<geometry::Vertex<dim>>(vertex));
        invalidate_vertex_index();
        invalidate_node_cache();
        return &vertices_.back();
    };

    void add_node(std::size_t tag, std::floating_point auto... coords)
    requires (sizeof...(coords) == dim){
        add_vertex(tag, coords...);
    };

    Node<dim> *add_node(Node<dim> &&node){
        vertices_.emplace_back(geometry::Vertex<dim>(node.id(), node.coord_ref()));
        invalidate_vertex_index();
        invalidate_node_cache();
        ensure_node_cache();
        if (!node_index_built_) build_node_index();
        return node_p(node.id());
    };

    // Reserve capacity before any node insertion (avoids repeated reallocation).
    void preallocate_node_capacity(std::size_t n, double margin = 1.20) {
        if (vertices_.empty()) {
            vertices_.reserve(static_cast<std::size_t>(n * margin));
        }
    }

    void preallocate_vertex_capacity(std::size_t n, double margin = 1.20) {
        preallocate_node_capacity(n, margin);
    }

    // Constructors
    Domain() = default;
    ~Domain() = default;

    // Move (transfers DM ownership via Mesh move ctor)
    Domain(Domain &&other) = default;
    Domain &operator=(Domain &&other) = default;

    // Copy is deleted because the Mesh member owns a DM handle.
    Domain(const Domain &other) = delete;
    Domain &operator=(const Domain &other) = delete;
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
