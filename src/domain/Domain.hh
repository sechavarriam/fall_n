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
#include <cmath>

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

    // Getters
    //Node<dim> *node_p(std::size_t i) { return &nodes_[i];};
    //Node<dim>& node  (std::size_t i) { return  nodes_[i];}; //Esto no es la iesima posicion!!!! debe devolver el nodo con id i.
                                                              //Esto se podria hacer asi si el vector de nodos fuera desde cero hasta el max id, y dejando espacios vacios.
                                                              //Otra opcion es almacenar un position_index en el nodo referente al dominio. Para llamar directamente sin buscar.
                                                              //Otra opcion es un unordered_map con los ids como keys.
                                                              //Por ahora se hara con un find_if, y luego se cambiara por alguna de las anteriores luego de un profiling.
    
    Node<dim> *node_p(std::size_t i) {
        // El iterador inicial podria tener un atajo si se ordenan los nodos por id.
        auto pos = std::find_if(nodes_.begin(), nodes_.end(), [&](auto &node){return node.id() == i;});
        
        std::cout << "Node id: " << pos->id() << std::endl;
        };


    Node<dim>& node  (std::size_t i) {
        // El iterador inicial podria tener un atajo si se ordenan los nodos por id.
        return *std::find_if(nodes_.begin(), nodes_.end(), [&](auto &node){return node.id() == i;});
        };

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

    // TODO: LO SIGUIENTE ES UN PARCHE! DEBE SER SACADO DE ACÁ Y MOVIDO A LAGRANGE ELEMENTS POR EJEMPLO!
    // LA IDEA ES QUE CADA ELEMENTO SEPA CUAL ES SU FRONTERA. EN LOS LAGRANGE ACTUALES 
    // PUEDE SER INFERIDO, SIN EMBARGO SE PUEDE HACER OTRA IMPLEMENTACION BASADA EN INDEXACIÓN
    // TOPOLÓGICA (ESQUINAS, ARISTAS, CARAS, INTERIORES,... )
    //
    // ── Create boundary elements from a coordinate plane ────────────────
    //
    //  Finds all volume element faces whose nodes all lie on the plane
    //  x_d = val (within tolerance).  Creates surface ElementGeometry
    //  objects (quad4 for hex8, quad9 for hex27) and stores them under
    //  the given group_name.
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
        // 1. Collect IDs of nodes on the plane
        std::set<PetscInt> plane_node_ids;
        for (auto& node : nodes_) {
            if (std::abs(node.coord(d) - val) < tol) {
                plane_node_ids.insert(static_cast<PetscInt>(node.id()));
            }
        }
        if (plane_node_ids.empty()) return;

        // 2. For each volume element, check each face
        std::size_t surf_tag = 900000; // arbitrary starting tag for generated elements

        for (auto& elem : elements_) {
            const auto n_nodes = elem.num_nodes();

            if (n_nodes == 27 && dim == 3) {
                // Hex27 face definitions: each face is 9 nodes.
                // Internal ordering (flat = ix + 3*iy + 9*iz):
                //   ξ = -1 (ix=0):  { 0, 3, 6,  9,12,15, 18,21,24}
                //   ξ = +1 (ix=2):  { 2, 5, 8, 11,14,17, 20,23,26}
                //   η = -1 (iy=0):  { 0, 1, 2,  9,10,11, 18,19,20}
                //   η = +1 (iy=2):  { 6, 7, 8, 15,16,17, 24,25,26}
                //   ζ = -1 (iz=0):  { 0, 1, 2,  3, 4, 5,  6, 7, 8}
                //   ζ = +1 (iz=2):  {18,19,20, 21,22,23, 24,25,26}
                static constexpr std::size_t hex27_faces[6][9] = {
                    { 0, 3, 6,  9,12,15, 18,21,24},  // ξ = -1
                    { 2, 5, 8, 11,14,17, 20,23,26},  // ξ = +1
                    { 0, 1, 2,  9,10,11, 18,19,20},  // η = -1
                    { 6, 7, 8, 15,16,17, 24,25,26},  // η = +1
                    { 0, 1, 2,  3, 4, 5,  6, 7, 8},  // ζ = -1
                    {18,19,20, 21,22,23, 24,25,26},  // ζ = +1
                };

                for (int f = 0; f < 6; ++f) {
                    bool all_on_plane = true;
                    for (int k = 0; k < 9; ++k) {
                        PetscInt nid = static_cast<PetscInt>(
                            elem.node_p(hex27_faces[f][k]).id());
                        if (plane_node_ids.find(nid) == plane_node_ids.end()) {
                            all_on_plane = false;
                            break;
                        }
                    }

                    if (all_on_plane) {
                        std::array<PetscInt, 9> face_node_ids;
                        for (int k = 0; k < 9; ++k) {
                            face_node_ids[k] = static_cast<PetscInt>(
                                elem.node_p(hex27_faces[f][k]).id());
                        }

                        auto integrator = GaussLegendreCellIntegrator<3,3>{};
                        auto& vec = boundary_elements_[group_name];
                        vec.emplace_back(
                            ElementGeometry<dim>(
                                LagrangeElement<3,3,3>(
                                    std::forward<std::size_t>(surf_tag++),
                                    std::span<PetscInt>(face_node_ids.data(), 9)),
                                std::move(integrator)));

                        // Bind nodes immediately
                        auto& new_elem = vec.back();
                        for (std::size_t i = 0; i < 9; ++i) {
                            auto pos = std::find_if(nodes_.begin(), nodes_.end(),
                                [&](auto& node){ return PetscInt(node.id()) == new_elem.node(i); });
                            new_elem.bind_node(i, std::addressof(*pos));
                        }
                    }
                }
            }
            else if (n_nodes == 8 && dim == 3) {
                // Hex8 face definitions: each face is 4 nodes.
                // Internal ordering (flat = ix + 2*iy + 4*iz):
                //   ξ = -1 (ix=0):  {0, 2, 4, 6}
                //   ξ = +1 (ix=1):  {1, 3, 5, 7}
                //   η = -1 (iy=0):  {0, 1, 4, 5}
                //   η = +1 (iy=1):  {2, 3, 6, 7}
                //   ζ = -1 (iz=0):  {0, 1, 2, 3}
                //   ζ = +1 (iz=1):  {4, 5, 6, 7}
                static constexpr std::size_t hex8_faces[6][4] = {
                    {0, 2, 4, 6}, {1, 3, 5, 7},
                    {0, 1, 4, 5}, {2, 3, 6, 7},
                    {0, 1, 2, 3}, {4, 5, 6, 7},
                };

                for (int f = 0; f < 6; ++f) {
                    bool all_on_plane = true;
                    for (int k = 0; k < 4; ++k) {
                        PetscInt nid = static_cast<PetscInt>(
                            elem.node_p(hex8_faces[f][k]).id());
                        if (plane_node_ids.find(nid) == plane_node_ids.end()) {
                            all_on_plane = false;
                            break;
                        }
                    }

                    if (all_on_plane) {
                        std::array<PetscInt, 4> face_node_ids;
                        for (int k = 0; k < 4; ++k) {
                            face_node_ids[k] = static_cast<PetscInt>(
                                elem.node_p(hex8_faces[f][k]).id());
                        }

                        auto integrator = GaussLegendreCellIntegrator<2,2>{};
                        auto& vec = boundary_elements_[group_name];
                        vec.emplace_back(
                            ElementGeometry<dim>(
                                LagrangeElement<3,2,2>(
                                    std::forward<std::size_t>(surf_tag++),
                                    std::span<PetscInt>(face_node_ids.data(), 4)),
                                std::move(integrator)));

                        auto& new_elem = vec.back();
                        for (std::size_t i = 0; i < 4; ++i) {
                            auto pos = std::find_if(nodes_.begin(), nodes_.end(),
                                [&](auto& node){ return PetscInt(node.id()) == new_elem.node(i); });
                            new_elem.bind_node(i, std::addressof(*pos));
                        }
                    }
                }
            }
        }
    }

    // ===========================================================================================================

    void setup_integration_points(){
        std::size_t gauss_point_counter = 0;

        for (auto &e : elements_){
            gauss_point_counter = e.setup_integration_points(gauss_point_counter);
        }

        std::cout << "Number of integration points: " << num_integration_points() << std::endl;
        std::cout << "offset: " << gauss_point_counter << std::endl;
    }

    void link_nodes_to_elements(){
        for (auto &e : elements_){
            for (std::size_t i = 0; i < e.num_nodes(); i++){
                // Find position of node i in domain   
                auto pos = std::find_if(nodes_.begin(), nodes_.end(), [&](auto &node){return PetscInt(node.id()) == e.node(i);});
                e.bind_node(i, std::addressof(*pos));
            }
        }
        // Also link boundary (surface) element nodes
        for (auto& [name, surf_elems] : boundary_elements_) {
            for (auto& e : surf_elems) {
                for (std::size_t i = 0; i < e.num_nodes(); i++) {
                    auto pos = std::find_if(nodes_.begin(), nodes_.end(),
                        [&](auto &node){ return PetscInt(node.id()) == e.node(i); });
                    e.bind_node(i, std::addressof(*pos));
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
        for (auto &e : elements_){
            std::vector<PetscInt> cone(e.num_nodes());

            for (std::size_t i = 0; i < e.num_nodes(); i++){
                cone[i] = e.node_p(i).sieve_id.value();
            }
            mesh.set_sieve_cone(e.sieve_id.value(), cone.data());
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

    // Tol increases capacity by default in 20%.
    void preallocate_node_capacity(std::size_t n, double tol = 1.20)
    { // Use Try and Catch to allow this operation if the container is empty.
        try{
            if (nodes_.empty())
                nodes_.reserve(n * tol);
            else
                throw nodes_.empty();
        }catch (bool NotEmpty){
            std::cout << "Preallocation should be done only before any node definition. Doing nothing." << std::endl;
        }
    };

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