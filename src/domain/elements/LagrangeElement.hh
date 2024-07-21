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

#include "../../numerics/linear_algebra/LinalgOperations.hh"

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
  using Point          = geometry::Point<sizeof...(N)>;
  using Array          = std::array<double, sizeof...(N)>;
  using JacobianMatrix = std::array<Array,sizeof...(N)>;

public:
  static inline constexpr std::size_t dim = sizeof...(N);
  static inline constexpr std::size_t num_nodes_ = (... * N);
  static inline constexpr ReferenceCell reference_element_{};

  using pNodeArray     = std::array<Node<dim> *, num_nodes_>;
  
  std::size_t tag_;
  pNodeArray nodes_;

public:

  auto num_nodes() const noexcept { return num_nodes_; };
  auto id()        const noexcept { return tag_      ; };

  auto node  (std::size_t i ) const noexcept {return nodes_[i];};
  void set_id(std::size_t id)       noexcept {tag_ = id; };

  constexpr void print_nodes_info() const noexcept {
    for (auto node : nodes_) {
      std::cout << "Node ID: " << node->id() << " ";
      for (std::size_t x=0; x<dim; ++x) {
        printf("%f ", node->coord(x));
      }
      printf("\n");
    }
  };

  constexpr inline double H(std::size_t i, const Array& X) const noexcept {
    return reference_element_.basis.shape_function(i)(X);
  };
  
  constexpr inline double H(std::size_t i, const Point& X) const noexcept {
    return H(i, X.coord());
  };

  constexpr inline double dH_dx(std::size_t i, std::size_t j, const Array& X) const noexcept {
    return reference_element_.basis.shape_function_derivative(i, j)(X);
  };
  
  constexpr inline double dH_dx(std::size_t i, std::size_t j, const Point& X) const noexcept {
    return dH_dx(i, j, X.coord());                          
  };

  auto inline constexpr evaluate_jacobian(const Array& X) const noexcept { 
    JacobianMatrix J{{{0}}}; 
    for (std::size_t i = 0; i < dim; ++i) {  //Thread Candidate
      for (std::size_t j = 0; j < dim; ++j) {//Thread Candidate
        for (std::size_t k = 0; k < num_nodes_; ++k) {
          J[i][j] += nodes_[k]->coord(i)*dH_dx(k, j, X); //*std::invoke(reference_element_.basis.shape_function_derivative(k, j),X);
        }
      }
    }
    return J;
  };

  auto evaluate_jacobian(const Point& X) const noexcept {return evaluate_jacobian(X.coord());};  

  // TODO: REPEATED CODE: Template and constrain with concept (coodinate type or something like that)
  double detJ(const geometry::Point<dim>&    X) const noexcept {return utils::det(evaluate_jacobian(X));};
  double detJ(      std::array<double,dim>&& X) const noexcept {return utils::det(evaluate_jacobian(X));};
  double detJ(const std::array<double,dim>&  X) const noexcept {return utils::det(evaluate_jacobian(X));};


  // TODO: Refactor with std::format 
  void print_node_coords() noexcept {
    for (auto node : nodes_) {for (auto coord : node->coord()) {printf("%f ", coord);}; printf("\n");};
  };


  //Constructor
  constexpr LagrangeElement()  = default;
  constexpr LagrangeElement(pNodeArray nodes) : nodes_{std::forward<pNodeArray>(nodes)}{};
  
  constexpr LagrangeElement(std::size_t& tag, std::ranges::range auto& node_references) : tag_{tag}{
    std::copy(node_references.begin(), node_references.end(), nodes_.begin());
  };

  constexpr LagrangeElement(std::size_t&& tag, std::ranges::range auto&& node_references) : tag_{tag}{
    std::move(node_references.begin(), node_references.end(), nodes_.begin());
  };

  // Copy and Move Constructors and Assignment Operators
  constexpr LagrangeElement(const LagrangeElement& other) : 
    tag_                {other.tag_},
    nodes_              {other.nodes_}
    {};

  constexpr LagrangeElement(LagrangeElement&& other) = default;

  constexpr LagrangeElement& operator=(const LagrangeElement& other){
    tag_                 = other.tag_;
    nodes_               = other.nodes_;
    return *this;
  };

  constexpr LagrangeElement& operator=(LagrangeElement&& other) = default;
  constexpr ~LagrangeElement() = default;

};

// =================================================================================================
// =================================================================================================
// =================================================================================================

template <std::size_t... N>
class GaussLegendreCellIntegrator{ // : public MaterialIntegrator {
    using Array = std::array<double, sizeof...(N)>;
    using Point = geometry::Point<sizeof...(N)>; 
    using CellQuadrature = GaussLegendre::CellQuadrature<N...>;

    static constexpr CellQuadrature integrator_{};

  public:

    //bool is_initialized_{false};

    //constexpr 
    auto operator()
    (const is_LagrangeElement auto& element, std::invocable<Array> auto&& f) const noexcept {
        return integrator_([&](const Array& x){
            return f(x) * element.detJ(x);
            //return element.detJ(x) * f(x);
        });
    };

    //constructors
    //copy and move constructors and assignment operators

    constexpr GaussLegendreCellIntegrator() noexcept = default;
    constexpr ~GaussLegendreCellIntegrator() noexcept = default;

};

// =================================================================================================
// =================================================================================================
// ================================================================================================= 

#endif