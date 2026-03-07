#ifndef FN_ELEMENT_H
#define FN_ELEMENT_H

#include <Eigen/Dense>

#include <array>
#include <cstddef>
#include <iostream>
#include <utility>
#include <memory>
#include <vector>
#include <functional>
#include <concepts>

#include <span>
#include <ranges>

// in clang, print
#ifdef __clang__ 
    #include <format>
    #include <print>
#endif

#include <vtkType.h>

#include "../Node.hh"

#include "../../geometry/IntegrationPoint.hh"

// HEADER FILES FOR ELEMENTS.
#include "LagrangeElement.hh"

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

        constexpr virtual unsigned int VTK_cell_type() const = 0;

        constexpr virtual std::span<vtkIdType> VTK_ordered_node_ids() const = 0;

        constexpr virtual std::size_t topological_dimension() const = 0;

        constexpr virtual std::size_t num_nodes() const = 0;
        constexpr virtual std::size_t id() const = 0;

        constexpr virtual PetscInt node(std::size_t i) const = 0;
        constexpr virtual Node<dim>& node_p(std::size_t i) const = 0;

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

    };
             
    template <typename ElementType, typename IntegrationStrategy> // External Polymorfism Design Pattern
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

        constexpr unsigned int         VTK_cell_type()        const override { return ElementType::VTK_cell_type; };
        constexpr std::span<vtkIdType> VTK_ordered_node_ids() const override { return element_.get_VTK_ordered_node_ids(); };

        constexpr std::size_t topological_dimension() const override { return topological_dim; };

        constexpr std::size_t num_nodes()           const override { return ElementType::num_nodes; };
        constexpr std::size_t id()                  const override { return element_.id(); };
        constexpr PetscInt    node  (std::size_t i) const override { return element_.node(i)  ;}; // renombrar como node_idx
        constexpr Node<dim>&  node_p(std::size_t i) const override { return element_.node_p(i);};

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

    };
} // impl

template<std::size_t dim>
class ElementGeometry
{
    using LocalCoordView = std::span<const double>;
    using SpatialArray = std::array<double, dim>;

    std::unique_ptr<impl::ElementGeometryConcept<dim>> pimpl_; // Bridge to implementation details (compiler generated).

public:

    std::vector<IntegrationPoint<dim>> integration_point_;
    std::optional<PetscInt>            sieve_id;  // Optional sieve id for the element inside DMPlex Mesh

    constexpr void print_info() const { pimpl_->print_info(); };
    constexpr void bind_node(std::size_t i, Node<dim> *node) { pimpl_->bind_node(i, node); };

    constexpr unsigned int         VTK_cell_type()        const { return pimpl_->VTK_cell_type(); };
    constexpr std::span<vtkIdType> VTK_ordered_node_ids() const { return pimpl_->VTK_ordered_node_ids(); };

    constexpr std::size_t topological_dimension() const { return pimpl_->topological_dimension(); };

    constexpr std::size_t id()        const { return pimpl_->id(); };
    constexpr std::size_t num_nodes() const { return pimpl_->num_nodes(); };

    constexpr PetscInt   node  (std::size_t i) const { return pimpl_->node  (i); };
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
    constexpr void set_sieve_id(PetscInt id){sieve_id = id;};

    constexpr void allocate_integration_points(){
        if (!integration_point_.empty()) return; // Already allocated — idempotent guard.
        const auto N = pimpl_->num_integration_points();
        integration_point_.reserve(N);
        for (std::size_t i = 0; i < N; ++i){
            integration_point_.emplace_back(IntegrationPoint<dim>{});
        }
    };

    constexpr std::size_t setup_integration_points(std::size_t offset) noexcept {
        allocate_integration_points();
        std::size_t x{0};

        for(auto& gauss_point : integration_point_){
            gauss_point.set_coord(pimpl_->map_local_point(pimpl_->reference_integration_point(x++)));
            gauss_point.set_id(offset++);
        }
        return offset;
    };

    // Constructors
    template <typename ElementType, typename IntegrationStrategy> // CAN BE CONSTRAINED WITH CONCEPTS!
    constexpr ElementGeometry(ElementType element, IntegrationStrategy integrator){
        using Model = impl::OwningModel_ElementGeometry<ElementType, IntegrationStrategy>;
        pimpl_ = std::make_unique<Model>(
            std::move(element),     // forward perhaphs?=
            std::move(integrator)); //
                //set_integration_point_coordinates();  //ESTO LO DEBE HACER EL DOMINIO!
    }

    ElementGeometry(ElementGeometry const &other) : pimpl_{other.pimpl_->clone()} {};
    ElementGeometry &operator=(ElementGeometry const &other) {
        pimpl_ = other.pimpl_->clone();
        return *this;
    }
    constexpr ~ElementGeometry() = default;
    constexpr ElementGeometry(ElementGeometry &&) = default;
    constexpr ElementGeometry &operator=(ElementGeometry &&) = default;

private:

    constexpr inline friend std::size_t        id(ElementGeometry const& element) { return element.pimpl_->id()       ; };
    constexpr inline friend std::size_t num_nodes(ElementGeometry const& element) { return element.pimpl_->num_nodes(); };
    constexpr inline friend void print_nodes_info(ElementGeometry const& element) { element.pimpl_->print_nodes_info(); };

    // -----------------------------------------------------------------------------------------------
};

#endif // FN_ELEMENT_H
