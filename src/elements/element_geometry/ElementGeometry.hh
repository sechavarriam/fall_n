#ifndef FN_ELEMENT_H
#define FN_ELEMENT_H

#include <Eigen/Dense>

#include <array>
#include <cmath>
#include <cstddef>
#include <concepts>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef __clang__
  #include <format>
  #include <print>
#endif

#include "../Node.hh"

#include "../../geometry/IntegrationPoint.hh"
#include "../../geometry/Point.hh"
#include "../../numerics/numerical_integration/QuadratureStrategy.hh"

// HEADER FILES FOR ELEMENTS.
#include "LagrangeElement.hh"
#include "SerendipityElement.hh"
#include "SimplexElement.hh"

namespace impl
{ // Implementation details

    template <std::size_t dim> // Spatial dimension (topological dimension is runtime via virtual)
    class ElementGeometryConcept
    { // Define the minimum interface for all element types (of any kind)
    public:
    using LocalCoordView  = std::span<const double>;
    using SpatialArray    = std::array<double, dim>;
    // Stack-allocated Jacobian: fixed rows = dim, dynamic cols (max = dim).
    // Avoids heap allocation when crossing the virtual boundary.
    using JacobianMatrix  = Eigen::Matrix<double, dim, Eigen::Dynamic, Eigen::ColMajor, dim, dim>;

        virtual ~ElementGeometryConcept() = default;

        virtual std::unique_ptr<ElementGeometryConcept<dim>> clone() const = 0;    // To allow copy construction of the wrapper

    public:

        constexpr virtual void print_info() const = 0;

        constexpr virtual std::size_t topological_dimension() const = 0;

        constexpr virtual std::size_t num_nodes() const = 0;
        constexpr virtual std::size_t id() const = 0;

        constexpr virtual PetscInt node(std::size_t i) const = 0;
        constexpr virtual const geometry::Point<dim>& point_p(std::size_t i) const = 0;
        constexpr virtual Node<dim>& node_p(std::size_t i) const = 0;
 
        constexpr virtual void bind_point(std::size_t i, const geometry::Point<dim>* point) = 0;
        constexpr virtual void bind_node(std::size_t i, Node<dim> *node) = 0;

        constexpr virtual std::size_t num_integration_points() const = 0;

        constexpr virtual SpatialArray map_local_point(LocalCoordView x) const = 0;

        // Raw Jacobian matrix (dim × topological_dim). Stack-allocated.
        virtual JacobianMatrix evaluate_jacobian(LocalCoordView X) const = 0;

        // Scalar differential measure, computed from J according to
        // the relationship between topological_dim and dim:
        //   topo_dim == dim          : |det(J)|          — volume element
        //   topo_dim == 1            : ‖J‖ (col norm)     — arc length
        //   topo_dim == 2, dim == 3  : ‖J₁ × J₂‖          — surface area
        // Implemented with if constexpr in OwningModel (zero-cost dispatch).
        virtual double differential_measure(LocalCoordView X) const = 0;

        constexpr virtual double H    (std::size_t i,                LocalCoordView X) const = 0;
        constexpr virtual double dH_dx(std::size_t i, std::size_t j, LocalCoordView X) const = 0;

        constexpr virtual LocalCoordView reference_integration_point(std::size_t i) const = 0;
        constexpr virtual double weight(std::size_t i) const = 0;

        // ── Subentity topology (codim-1 faces) ──────────────────────────
        virtual std::size_t              num_faces()                        const = 0;
        virtual std::size_t              face_num_nodes(std::size_t f)      const = 0;
        virtual std::vector<std::size_t> face_node_indices(std::size_t f)   const = 0;

        // Create a type-erased ElementGeometry for the f-th face of this element.
        // The returned geometry has spatial dim = dim but topological_dim = sizeof...(N)-1.
        // node_ids are the *global* PETSc node IDs for the face.
        virtual std::unique_ptr<ElementGeometryConcept<dim>>
        make_face_geometry(std::size_t f, std::size_t tag,
                           std::span<PetscInt> face_global_node_ids) const = 0;

    };
             
    template <typename ElementType, QuadratureStrategy IntegrationStrategy> // External Polymorfism Design Pattern
    class OwningModel_ElementGeometry : public ElementGeometryConcept<ElementType::dim>
    { // Wrapper for all element types (of any kind)
        
        static constexpr auto topological_dim = []() constexpr {
            if constexpr (requires { ElementType::topological_dim; }) {
                return ElementType::topological_dim;
            } else {
                return ElementType::dim; // Default to spatial dimension if topological dimension is not defined
            }
        }();

        using LocalCoordView = std::span<const double>;
        using NaturalArray   = std::array<double, topological_dim>;
        using SpatialArray   = std::array<double, ElementType::dim>;

        static constexpr auto num_integration_points_ = IntegrationStrategy::num_integration_points;
        
        ElementType         element_   ; // Stores the ElementGeometry object
        IntegrationStrategy integrator_; // Stores the Integration Strategy object (Spacial integration strategy)

        // Convert type-erased span to statically-sized natural-coordinate array.
        constexpr NaturalArray to_natural(LocalCoordView X) const noexcept {
            NaturalArray xi{};
            for (std::size_t k = 0; k < topological_dim; ++k) xi[k] = X[k];
            return xi;
        }
    
     public:

        static constexpr auto dim = ElementType::dim;
        
        explicit OwningModel_ElementGeometry(ElementType const &element, IntegrationStrategy const &integrator) :
            element_   (element   ),
            integrator_(integrator) {};

        explicit OwningModel_ElementGeometry(ElementType &&element, IntegrationStrategy &&integrator) : 
            element_(std::forward<ElementType>(element)),
            integrator_(std::forward<IntegrationStrategy>(integrator)) {};

        std::unique_ptr<ElementGeometryConcept<dim>> clone() const override{
            return std::make_unique<OwningModel_ElementGeometry<ElementType, IntegrationStrategy>>(*this);
        };

    public: // Implementation of the virtual operations derived from ElementGeometryConcept

        constexpr void print_info() const override { element_.print_info(); };

        constexpr std::size_t topological_dimension() const override { return topological_dim; };

        constexpr std::size_t num_nodes()           const override { return ElementType::num_nodes; };
        constexpr std::size_t id()                  const override { return element_.id(); };
        constexpr PetscInt    node  (std::size_t i) const override { return element_.node(i)  ;}; // renombrar como node_idx
        constexpr const geometry::Point<dim>& point_p(std::size_t i) const override { return element_.point_p(i); };
        constexpr Node<dim>&  node_p(std::size_t i) const override { return element_.node_p(i);};

        constexpr void bind_point(std::size_t i, const geometry::Point<dim>* point) override { element_.bind_point(i, point); };
        constexpr void bind_node(std::size_t i, Node<dim> *node) override { element_.bind_node(i, node); };
        
        constexpr std::size_t num_integration_points() const override { return num_integration_points_; };

        constexpr SpatialArray map_local_point(LocalCoordView x) const override {
            return element_.map_local_point(to_natural(x));
        };

        using JacobianMatrix = typename ElementGeometryConcept<dim>::JacobianMatrix;

        JacobianMatrix evaluate_jacobian(LocalCoordView X) const override {
            return element_.evaluate_jacobian(to_natural(X));
        };

        double differential_measure(LocalCoordView X) const override {
            auto J = element_.evaluate_jacobian(to_natural(X)); // Static-size Eigen matrix

            if constexpr (topological_dim == dim) {
                return std::abs(J.determinant());
            } else if constexpr (topological_dim == 1) {
                return J.col(0).norm();           // arc-length: ‖∂x/∂ξ‖
            } else if constexpr (topological_dim == 2 && dim == 3) {
                return J.col(0).cross(J.col(1)).norm(); // surface metric
            } else { // General: sqrt(det(Jᵀ J))
                return std::sqrt((J.transpose() * J).determinant());
            }
        };

        constexpr double H(std::size_t i, LocalCoordView X) const override {
            return element_.H(i, to_natural(X));
        };

        constexpr double dH_dx(std::size_t i, std::size_t j, LocalCoordView X) const override {
            return element_.dH_dx(i, j, to_natural(X));
        };

        constexpr LocalCoordView reference_integration_point(std::size_t i) const override {return integrator_.reference_integration_point(i);};
        constexpr double weight                    (std::size_t i) const override {return integrator_.weight(i);};

        // ── Subentity topology overrides ────────────────────────────────
        //
        //  Delegate to the compile-time SubentityDescriptor living inside
        //  the ReferenceCell of a LagrangeElement.  For non-Lagrange
        //  element types we fall back to returning 0 / empty.
        //

        std::size_t num_faces() const override {
            if constexpr (requires { typename ElementType::ReferenceCell; ElementType::ReferenceCell::num_faces; }) {
                // topological_dim >= 1 guaranteed by LagrangianCell
                return ElementType::ReferenceCell::num_faces;
            } else {
                return 0;
            }
        }

        std::size_t face_num_nodes(std::size_t f) const override {
            if constexpr (is_SimplexElement<ElementType> || is_SerendipityElement<ElementType>) {
                return ElementType::ReferenceCell::face_num_nodes(f);
            } else if constexpr (requires { typename ElementType::ReferenceCell; }) {
                using Faces = typename ElementType::ReferenceCell::Faces;
                auto lookup = [&]<std::size_t... I>(std::index_sequence<I...>)
                    -> std::size_t
                {
                    static constexpr std::array<std::size_t, sizeof...(I)> table = {
                        Faces::subentity_num_nodes(I)...
                    };
                    return table[f];
                };
                return lookup(std::make_index_sequence<ElementType::ReferenceCell::num_faces>{});
            } else {
                (void)f;
                return 0;
            }
        }

        std::vector<std::size_t> face_node_indices(std::size_t f) const override {
            if constexpr (is_SimplexElement<ElementType> || is_SerendipityElement<ElementType>) {
                const auto entry = ElementType::ReferenceCell::face_node_indices(f);
                return {entry.indices.begin(), entry.indices.begin() + entry.size};
            } else if constexpr (requires { typename ElementType::ReferenceCell; }) {
                using Faces = typename ElementType::ReferenceCell::Faces;
                // Faces::node_indices(f) is constexpr but f is a runtime argument,
                // so we build a small lookup table and dispatch.
                // For a modest number of faces (≤12 in 3D) this is cheap.
                auto lookup = [&]<std::size_t... I>(std::index_sequence<I...>) 
                    -> std::vector<std::size_t>
                {
                    using FI = typename Faces::NodeIndices;
                    // Build constexpr array of all face-node-index lists
                    static constexpr std::array<FI, sizeof...(I)> table = {
                        Faces::node_indices(I)...
                    };
                    const auto& entry = table[f];
                    return {entry.indices.begin(), entry.indices.begin() + entry.size};
                };
                return lookup(std::make_index_sequence<ElementType::ReferenceCell::num_faces>{});
            } else {
                (void)f;
                return {};
            }
        }

        // ── make_face_geometry: create a type-erased surface element for face f ──
        //
        //  For each compile-time face index I, we know the free-axis dimensions
        //  at compile time via SubentityDescriptor.  We build a dispatch table
        //  of factory functions (one per face) that instantiate the correct
        //  LagrangeElement<Dim, FreeDim0, FreeDim1, ...> and integrator.
        //
        std::unique_ptr<ElementGeometryConcept<dim>> make_face_geometry(
            std::size_t f, std::size_t tag,
            std::span<PetscInt> face_global_node_ids) const override
        {
            if constexpr (is_SimplexElement<ElementType>) {
                // ── Simplex path ────────────────────────────────────────
                // All faces of a D-simplex are (D-1)-simplices of the same order.
                if constexpr (ElementType::topological_dim > 1) {
                    constexpr std::size_t face_top_dim = ElementType::topological_dim - 1;
                    constexpr std::size_t face_order   = ElementType::order;
                    using FaceElem  = SimplexElement<dim, face_top_dim, face_order>;
                    using FaceInteg = SimplexIntegrator<face_top_dim, face_order>;
                    using FaceModel = OwningModel_ElementGeometry<FaceElem, FaceInteg>;
                    return std::make_unique<FaceModel>(
                        FaceElem(tag, face_global_node_ids),
                        FaceInteg{});
                } else {
                    // TopDim == 1 → faces are points, not meaningful
                    (void)f; (void)tag; (void)face_global_node_ids;
                    return nullptr;
                }
            }
            else if constexpr (is_SerendipityElement<ElementType>) {
                if constexpr (ElementType::topological_dim > 1) {
                    constexpr std::size_t face_top_dim = ElementType::topological_dim - 1;
                    constexpr std::size_t face_order   = ElementType::order;
                    constexpr std::size_t q            = face_order + 1;
                    using FaceElem = SerendipityElement<dim, face_top_dim, face_order>;

                    if constexpr (face_top_dim == 2) {
                        using FaceInteg = GaussLegendreCellIntegrator<q, q>;
                        using FaceModel = OwningModel_ElementGeometry<FaceElem, FaceInteg>;
                        return std::make_unique<FaceModel>(
                            FaceElem(tag, face_global_node_ids),
                            FaceInteg{});
                    } else if constexpr (face_top_dim == 1) {
                        using FaceInteg = GaussLegendreCellIntegrator<q>;
                        using FaceModel = OwningModel_ElementGeometry<FaceElem, FaceInteg>;
                        return std::make_unique<FaceModel>(
                            FaceElem(tag, face_global_node_ids),
                            FaceInteg{});
                    } else {
                        (void)f; (void)tag; (void)face_global_node_ids;
                        return nullptr;
                    }
                } else {
                    (void)f; (void)tag; (void)face_global_node_ids;
                    return nullptr;
                }
            }
            else if constexpr (requires { typename ElementType::ReferenceCell; }) {
                using RC = typename ElementType::ReferenceCell;

                // Factory function type: creates a concept pointer from (tag, node_ids)
                using FactoryFn = std::unique_ptr<ElementGeometryConcept<dim>>(*)(
                    std::size_t, std::span<PetscInt>);

                // Build dispatch table: one factory per face index
                auto table = [&]<std::size_t... I>(std::index_sequence<I...>)
                    -> std::array<FactoryFn, sizeof...(I)>
                {
                    return { &make_face_factory<I>... };
                }(std::make_index_sequence<RC::num_faces>{});

                return table[f](tag, face_global_node_ids);
            } else {
                (void)f; (void)tag; (void)face_global_node_ids;
                return nullptr;
            }
        }

    private:
        // Compile-time factory for face FaceIdx of this element type.
        // Uses sub_dimensions(FaceIdx) to deduce the correct LagrangeElement
        // and GaussLegendreCellIntegrator template parameters.
        // Only valid for tensor-product (Lagrange) element families.
        template <std::size_t FaceIdx>
        static std::unique_ptr<ElementGeometryConcept<dim>>
        make_face_factory(std::size_t tag, std::span<PetscInt> node_ids)
            requires (requires { typename ElementType::ReferenceCell; } && !is_SimplexElement<ElementType>)
        {
            using RC    = typename ElementType::ReferenceCell;
            using Faces = typename RC::Faces;
            constexpr auto sd = Faces::sub_dimensions(FaceIdx);

            // Dispatch on the number of free dimensions of the face
            if constexpr (sd.num_free == 2) {
                // Face of a 3D element → surface element
                constexpr auto D0 = sd.free_dims[0];
                constexpr auto D1 = sd.free_dims[1];
                using FaceElem  = LagrangeElement<dim, D0, D1>;
                using FaceInteg = GaussLegendreCellIntegrator<D0, D1>;
                using FaceModel = OwningModel_ElementGeometry<FaceElem, FaceInteg>;
                return std::make_unique<FaceModel>(
                    FaceElem(tag, node_ids),
                    FaceInteg{});
            } else if constexpr (sd.num_free == 1) {
                // Face of a 2D element → edge element
                constexpr auto D0 = sd.free_dims[0];
                using FaceElem  = LagrangeElement<dim, D0>;
                using FaceInteg = GaussLegendreCellIntegrator<D0>;
                using FaceModel = OwningModel_ElementGeometry<FaceElem, FaceInteg>;
                return std::make_unique<FaceModel>(
                    FaceElem(tag, node_ids),
                    FaceInteg{});
            } else {
                // Face of a 1D element → point (not useful for boundary elements)
                (void)tag; (void)node_ids;
                return nullptr;
            }
        }

    };
} // impl

template<std::size_t dim>
class ElementGeometry
{
    using LocalCoordView = std::span<const double>;
    using SpatialArray = std::array<double, dim>;

    struct Metadata {
        std::vector<IntegrationPoint<dim>> integration_points{};
        std::optional<PetscInt>            sieve_id{};
        std::string                        physical_group{};
    };

    std::unique_ptr<impl::ElementGeometryConcept<dim>> pimpl_; // Bridge to implementation details (compiler generated).
    Metadata                                            metadata_{};

    static auto clone_pimpl(const ElementGeometry& other)
        -> std::unique_ptr<impl::ElementGeometryConcept<dim>>
    {
        if (!other.pimpl_) {
            return nullptr;
        }
        return other.pimpl_->clone();
    }

public:

    constexpr void print_info() const { pimpl_->print_info(); };
    constexpr void bind_point(std::size_t i, const geometry::Point<dim> *point) { pimpl_->bind_point(i, point); };
    constexpr void bind_node(std::size_t i, Node<dim> *node) { pimpl_->bind_node(i, node); };

    constexpr std::size_t topological_dimension() const { return pimpl_->topological_dimension(); };

    constexpr std::size_t id()        const { return pimpl_->id(); };
    constexpr std::size_t num_nodes() const { return pimpl_->num_nodes(); };

    constexpr PetscInt   node  (std::size_t i) const { return pimpl_->node  (i); };
    constexpr const geometry::Point<dim>& point_p(std::size_t i) const { return pimpl_->point_p(i); };
    constexpr Node<dim>& node_p(std::size_t i) const { return pimpl_->node_p(i); };

    constexpr std::size_t num_integration_points() const { return pimpl_->num_integration_points(); };

    constexpr SpatialArray map_local_point(LocalCoordView x) const { return pimpl_->map_local_point(x); };

    // Raw Jacobian matrix (dim × topological_dim). Stack-allocated.
    auto evaluate_jacobian(LocalCoordView X) const { return pimpl_->evaluate_jacobian(X); };

    // Scalar differential measure — topology-aware, computed via if constexpr
    // inside the OwningModel (static-size J, zero-cost dispatch).
    double differential_measure(LocalCoordView X) const { return pimpl_->differential_measure(X); };

    constexpr double H    (std::size_t i, LocalCoordView X) const { return pimpl_->H(i, X);};
    constexpr double dH_dx(std::size_t i, std::size_t j, LocalCoordView X) const { return pimpl_->dH_dx(i, j, X);};

    constexpr LocalCoordView reference_integration_point(std::size_t i) const {return pimpl_->reference_integration_point(i);};
    constexpr double weight                     (std::size_t i) const {return pimpl_->weight(i);};

    // ── Subentity topology (codim-1 faces) ──────────────────────────────
    std::size_t              num_faces()                      const { return pimpl_->num_faces(); }
    std::size_t              face_num_nodes(std::size_t f)    const { return pimpl_->face_num_nodes(f); }
    std::vector<std::size_t> face_node_indices(std::size_t f) const { return pimpl_->face_node_indices(f); }

    // Create a new type-erased ElementGeometry for the f-th face.
    // face_global_node_ids are the PETSc global node IDs for the face nodes.
    ElementGeometry make_face_geometry(std::size_t f, std::size_t tag,
                                       std::span<PetscInt> face_global_node_ids) const {
        auto face_pimpl = pimpl_->make_face_geometry(f, tag, face_global_node_ids);
        if (!face_pimpl) {
            throw std::logic_error("ElementGeometry cannot create a null face geometry.");
        }
        return ElementGeometry(std::move(face_pimpl));
    }

    // ---- Non-virtual template: numerical integration over the geometry ----
    // Computes  ∫ f(ξ) · dΩ  =  ∑ w_i · |measure(ξ_i)| · f(ξ_i)
    // using the injected quadrature rule.
    //
    // f is a template parameter (no std::function, no heap allocation).
    // Works for scalar and Eigen return types.
    template<std::invocable<LocalCoordView> F>
    auto integrate(F&& f) const {
        using R = decltype(f(std::declval<LocalCoordView>()));
        const auto n = num_integration_points();

        if constexpr (std::is_arithmetic_v<std::decay_t<R>>) {
            double result = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                auto xi = reference_integration_point(i);
                result += weight(i) * differential_measure(xi) * f(xi);
            }
            return result;
        } else {
            // Eigen or matrix-like return type
            auto xi_0  = reference_integration_point(0);
            auto result = (f(xi_0) * (weight(0) * differential_measure(xi_0))).eval();
            for (std::size_t i = 1; i < n; ++i) {
                auto xi = reference_integration_point(i);
                result += f(xi) * (weight(i) * differential_measure(xi));
            }
            return result;
        }
    };

    // Own methods
    constexpr void set_sieve_id(PetscInt id) noexcept { metadata_.sieve_id = id; };
    constexpr bool has_sieve_id() const noexcept { return metadata_.sieve_id.has_value(); }
    constexpr PetscInt sieve_id() const { return metadata_.sieve_id.value(); }
    constexpr const std::optional<PetscInt>& optional_sieve_id() const noexcept { return metadata_.sieve_id; }

    void set_physical_group(const std::string& name) { metadata_.physical_group = name; }
    const std::string& physical_group() const noexcept { return metadata_.physical_group; }
    bool has_physical_group() const noexcept { return !metadata_.physical_group.empty(); }

    constexpr auto& integration_points() noexcept { return metadata_.integration_points; }
    constexpr const auto& integration_points() const noexcept { return metadata_.integration_points; }

    constexpr void allocate_integration_points(){
        if (!metadata_.integration_points.empty()) return; // Already allocated — idempotent guard.
        const auto N = pimpl_->num_integration_points();
        metadata_.integration_points.reserve(N);
        for (std::size_t i = 0; i < N; ++i){
            metadata_.integration_points.emplace_back(IntegrationPoint<dim>{});
        }
    };

    constexpr std::size_t setup_integration_points(std::size_t offset) noexcept {
        allocate_integration_points();
        std::size_t x{0};

        for(auto& gauss_point : metadata_.integration_points){
            const auto gp_index = x++;
            gauss_point.set_coord(pimpl_->map_local_point(pimpl_->reference_integration_point(gp_index)));
            gauss_point.set_id(offset++);
            gauss_point.set_weight(pimpl_->weight(gp_index));
        }
        return offset;
    };

    // Constructors
    template <typename ElementType, typename IntegrationStrategy>
        requires (!std::same_as<std::remove_cvref_t<ElementType>, ElementGeometry> &&
                  QuadratureStrategy<std::remove_cvref_t<IntegrationStrategy>>)
    constexpr ElementGeometry(ElementType&& element, IntegrationStrategy&& integrator){
        using ConcreteElement = std::remove_cvref_t<ElementType>;
        using ConcreteStrategy = std::remove_cvref_t<IntegrationStrategy>;
        using Model = impl::OwningModel_ElementGeometry<ConcreteElement, ConcreteStrategy>;
        pimpl_ = std::make_unique<Model>(
            std::forward<ElementType>(element),
            std::forward<IntegrationStrategy>(integrator));
    }

    ElementGeometry(ElementGeometry const &other)
        : pimpl_{clone_pimpl(other)}, metadata_{other.metadata_} {}

    ElementGeometry &operator=(ElementGeometry const &other) {
        if (this == &other) {
            return *this;
        }
        pimpl_ = clone_pimpl(other);
        metadata_ = other.metadata_;
        return *this;
    }
    constexpr ~ElementGeometry() = default;
    constexpr ElementGeometry(ElementGeometry &&) = default;
    constexpr ElementGeometry &operator=(ElementGeometry &&) = default;

private:

    // Private constructor from a pre-built pimpl (used by make_face_geometry)
    explicit ElementGeometry(std::unique_ptr<impl::ElementGeometryConcept<dim>> p)
        : pimpl_(std::move(p)) {}

    constexpr inline friend std::size_t        id(ElementGeometry const& element) { return element.pimpl_->id()       ; };
    constexpr inline friend std::size_t num_nodes(ElementGeometry const& element) { return element.pimpl_->num_nodes(); };

    // -----------------------------------------------------------------------------------------------
};

#endif // FN_ELEMENT_H
