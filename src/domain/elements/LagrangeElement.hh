#ifndef FALL_N_LAGRANGIAN_FINITE_ELEMENT
#define FALL_N_LAGRANGIAN_FINITE_ELEMENT

#include <array>
#include <concepts>
#include <cstddef>
#include <iterator>
#include <memory>
#include <ranges>
#include <span>
#include <tuple>
#include <vector>
#include <type_traits>
#include <ranges>
#include <span>

#include "../Node.hh"

#include "../IntegrationPoint.hh"

#include "../../geometry/Topology.hh"
#include "../../geometry/Cell.hh"
#include "../../geometry/Point.hh"


#include "../../utils/small_math.hh"
#include "../../numerics/numerical_integration/CellQuadrature.hh"

#include "../../integrator/MaterialIntegrator.hh"

template<typename T>
concept private_Lagrange_check_ = requires(T t){
  requires std::same_as<decltype(t._is_LagrangeElement()), bool>;
};

template<typename T> 
struct LagrangeConceptTester{
  static inline constexpr bool _is_in_Lagrange_Family = private_Lagrange_check_<T>;  
};

template<typename T>
concept is_LagrangeElement = LagrangeConceptTester<T>::_is_in_Lagrange_Family;



template <std::size_t... N> requires(topology::EmbeddableInSpace<sizeof...(N)>) 
class LagrangeElement {
  
  template<typename T> friend struct LagrangeConceptTester;  
  static inline constexpr bool _is_LagrangeElement(){return true;};

  using ReferenceCell  = geometry::cell::LagrangianCell<N...>;

public:

  static inline constexpr std::size_t dim = sizeof...(N);
  static inline constexpr std::size_t num_nodes_ = (... * N);
  static inline constexpr ReferenceCell reference_element_{};

  using pNodeArray     = std::array<Node<dim> *, num_nodes_>;
  using JacobianMatrix = std::array<std::array<double, dim>, dim>;

  std::size_t tag_;

  pNodeArray nodes_;

  //std::unique_ptr<MaterialIntegrator> material_integrator_;
  //std::vector <IntegrationPoint<dim>> integration_points_;  

public:
  
  //std::vector <IntegrationPoint<dim>> integration_points_;  
  auto num_nodes() const noexcept { return num_nodes_; };
  auto id()        const noexcept { return tag_      ; };
  
  auto node(std::size_t i) const noexcept { return nodes_[i]; };

  void set_id(std::size_t id) noexcept { tag_ = id; };

  //void set_material_integrator(std::unique_ptr<MaterialIntegrator>&& integrator) noexcept {
  //  material_integrator_ = std::move(integrator);
  //};

  void print_nodes_info() const noexcept {
    for (auto node : nodes_) {
      std::cout << "Node ID: " << node->id() << " ";
      for (std::size_t x=0; x<dim; ++x) {
        printf("%f ", node->coord(x));
      }
      printf("\n");
    }
  };

  // TODO: REPEATED CODE: Template and constrain with concept (coodinate type or something like that)
  auto evaluate_jacobian(const geometry::Point<dim>& X) const noexcept { //Thread Candidate
    JacobianMatrix J;
    for (std::size_t i = 0; i < dim; ++i) {
      for (std::size_t j = 0; j < dim; ++j) {
        for (std::size_t k = 0; k < num_nodes_; ++k) {
          J[i][j] += nodes_[k]->coord(i) 
                   * reference_element_.basis.shape_function_derivative(k, j)(X.coord());
        }
      }
    }
    return J;
  };  

  auto evaluate_jacobian(const std::array<double,dim>& X) const noexcept { 
    JacobianMatrix J;
    for (std::size_t i = 0; i < dim; ++i) {  //Thread Candidate
      for (std::size_t j = 0; j < dim; ++j) {//Thread Candidate
        for (std::size_t k = 0; k < num_nodes_; ++k) {
          J[i][j] += nodes_[k]->coord(i) 
                   * std::invoke(
                          reference_element_.basis.shape_function_derivative(k, j),
                          X);
        }
      }
    }
    return J;
  };

  // TODO: REPEATED CODE: Template and constrain with concept (coodinate type or something like that)
  double detJ(const geometry::Point<dim>&    X) const noexcept {return utils::det(evaluate_jacobian(X));};
  double detJ(      std::array<double,dim>&& X) const noexcept {return utils::det(evaluate_jacobian(X));};
  double detJ(const std::array<double,dim>&  X) const noexcept {return utils::det(evaluate_jacobian(X));};

  // TODO: Refactor with std::format 
  void print_node_coords() noexcept {
    for (auto node : nodes_) {for (auto coord : node->coord()) {printf("%f ", coord);}; printf("\n");};
  };

  //Constructor
  LagrangeElement()  = default;
  LagrangeElement(pNodeArray nodes) : nodes_{std::forward<pNodeArray>(nodes)}{};
  
  LagrangeElement(std::size_t& tag, std::ranges::range auto& node_references) : tag_{tag}
  {
    std::copy(node_references.begin(), node_references.end(), nodes_.begin());
  };

  LagrangeElement(std::size_t&& tag, std::ranges::range auto&& node_references) : tag_{tag}
  {
    std::move(node_references.begin(), node_references.end(), nodes_.begin());
  };

  // Copy and Move Constructors and Assignment Operators
  LagrangeElement(const LagrangeElement& other) : 
    tag_                {other.tag_},
    nodes_              {other.nodes_}//,
    //material_integrator_{std::make_unique<MaterialIntegrator>()}
    {};

  LagrangeElement(LagrangeElement&& other) = default;

  LagrangeElement& operator=(const LagrangeElement& other){
    tag_                 = other.tag_;
    nodes_               = other.nodes_;
    //material_integrator_ = std::make_unique<MaterialIntegrator>();
    return *this;
  };

  LagrangeElement& operator=(LagrangeElement&& other) = default;

  ~LagrangeElement() = default;

};

// =================================================================================================
// =================================================================================================
// =================================================================================================

template <std::size_t... N>
class GaussLegendreCellIntegrator{ // : public MaterialIntegrator {
    using  CellQuadrature = GaussLegendre::CellQuadrature<N...>;

    //std::array<IntegrationPoint<sizeof...(N)>, (N*...)> integration_points_{};
    static constexpr CellQuadrature integrator_{};

  public:

    //bool is_initialized_{false};

    //double operator()
    //(const is_LagrangeElement auto& element, std::function<double(geometry::Point<sizeof...(N)>)> f) const noexcept {
    //    return integrator_([&](double x){
    //        return f(x) * element.detJ(x);
    //    });
    //};
//
    //double operator()
    //(const is_LagrangeElement auto& element, std::function<double(std::array<double,sizeof...(N)>)> f) const noexcept {
    //    return integrator_([&](double x){
    //        return f(x) * element.detJ(x);
    //    });
    //};
//
    //double operator()
    //(const is_LagrangeElement auto& element, std::function<double(double)> f) const noexcept {
    //    return integrator_([&](double x){
    //        return f(x) * element.detJ(x);
    //    });
    //};

    double operator()
    (const is_LagrangeElement auto& element, std::invocable<geometry::Point<sizeof...(N)>> auto&& f) const noexcept {
        return integrator_([&](geometry::Point<sizeof...(N)> x){
            return f(x) * element.detJ(x);
        });
    };

    //auto integration_point(std::size_t i) const noexcept {
    //    return &integration_points_[i];
    //};
    //std::span<IntegrationPoint<sizeof...(N)>,(N*...)> integration_points() const noexcept {
    //    return integration_points_;
    //};
    //constexpr auto set_integration_points(is_LagrangeElement auto& element) const noexcept {
    //    element.set_integration_points(integrator_.evalPoints_);
    //    };

    //constructors
    //copy and move constructors and assignment operators


    constexpr GaussLegendreCellIntegrator() noexcept = default;
    constexpr ~GaussLegendreCellIntegrator() noexcept = default;

};

// =================================================================================================
// =================================================================================================
// ================================================================================================= 

#endif