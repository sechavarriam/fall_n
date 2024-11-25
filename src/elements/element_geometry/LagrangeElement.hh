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

#include <vtkType.h>

#include "../Node.hh"

#include "../../geometry/Topology.hh"
#include "../../geometry/Cell.hh"
#include "../../geometry/Point.hh"

#include "../../utils/small_math.hh"
#include "../../numerics/numerical_integration/CellQuadrature.hh"

#include "../../numerics/linear_algebra/LinalgOperations.hh"

template <typename T>
concept private_Lagrange_check_ = requires(T t) {
  requires std::same_as<decltype(t._is_LagrangeElement()), bool>;
};

template <typename T>
struct LagrangeConceptTester
{
  static inline constexpr bool _is_in_Lagrange_Family = private_Lagrange_check_<T>;
};

template <typename T>
concept is_LagrangeElement = LagrangeConceptTester<T>::_is_in_Lagrange_Family;

template <std::size_t... N>
  requires(topology::EmbeddableInSpace<sizeof...(N)>)
class LagrangeElement
{
  template <typename T>
  friend struct LagrangeConceptTester;
  static inline constexpr bool _is_LagrangeElement() { return true; };

  using ReferenceCell = geometry::cell::LagrangianCell<N...>;
  using Point = geometry::Point<sizeof...(N)>;
  using Array = std::array<double, sizeof...(N)>;
  using JacobianMatrix = std::array<Array, sizeof...(N)>;

public:
  static inline constexpr std::size_t dim = sizeof...(N);
  static inline constexpr std::size_t num_nodes_ = (... * N);
  static inline constexpr ReferenceCell reference_element_{};
  
  static inline constexpr auto VTK_cell_type = reference_element_.VTK_cell_type();

  using pNodeArray = std::optional<std::array<Node<dim> *, num_nodes_>>;

  std::size_t tag_;
  pNodeArray nodes_p;

  std::array<PetscInt , num_nodes_> nodes_;
  std::array<vtkIdType, num_nodes_> vtk_nodes_{0};

public:

  static constexpr auto num_nodes()             noexcept { return num_nodes_; };
  static constexpr auto get_VTK_cell_type()     noexcept { return VTK_cell_type; };
  static constexpr auto get_VTK_node_ordering() noexcept { return reference_element_.VTK_node_ordering(); };
  
  void set_VTK_node_order() noexcept {
    for (std::size_t i = 0; i < num_nodes_; ++i) vtk_nodes_[i] = static_cast<vtkIdType>(node(get_VTK_node_ordering()[i])) ;
  };

  std::span<vtkIdType> get_VTK_ordered_node_ids() const noexcept { //TODO: check if its ordered and if the VTK_node_ordering is correctly set.
    return std::span<vtkIdType>(const_cast<vtkIdType*>(vtk_nodes_.data()), num_nodes_);
  };


  auto id() const noexcept { return tag_; };

  PetscInt node(std::size_t i) const noexcept { return nodes_[i]; };
    
  std::span<PetscInt> nodes() const noexcept { return std::span<PetscInt>(nodes_); };

  Node<dim>& node_p(std::size_t i) const noexcept{
    return *nodes_p.value()[i];
  };

  void bind_node (std::size_t i, Node<dim> *node) noexcept {
    if (nodes_p.has_value()){ nodes_p.value()[i] = node;} 
    else { //set and assign
      nodes_p = std::array<Node<dim> *, num_nodes_>{};
      nodes_p.value()[i] = node;
    }  
  };
    
  //void set_node_pointer(std::size_t i, Node<dim> *node) noexcept { nodes_p.value()[i] = node;};

  void set_id(std::size_t id) noexcept { tag_ = id; };

  constexpr inline double H(std::size_t i, const Array &X) const noexcept {
    return reference_element_.basis.shape_function(i)(X);
  };

  constexpr inline double H(std::size_t i, const Point &X) const noexcept{return H(i, X.coord());};

  constexpr inline double dH_dx(std::size_t i, std::size_t j, const Array &X) const noexcept{
    return reference_element_.basis.shape_function_derivative(i, j)(X);
  };

  constexpr inline double dH_dx(std::size_t i, std::size_t j, const Point &X) const noexcept{
    return dH_dx(i, j, X.coord());
  };

  auto inline constexpr evaluate_jacobian(const Array &X) const noexcept{
    if (nodes_p.has_value()){
      JacobianMatrix J{{{0}}};
      for (std::size_t i = 0; i < dim; ++i){ // Thread Candidate
        for (std::size_t j = 0; j < dim; ++j){ // Thread Candidate
          for (std::size_t k = 0; k < num_nodes_; ++k){
            J[i][j] += nodes_p.value()[k]->coord(i) * dH_dx(k, j, X); //*std::invoke(reference_element_.basis.shape_function_derivative(k, j),X);
          }
        }
      }
      return J;
    } 
    else {
    std::runtime_error("LagrangeElement: Nodes are not linked yet to the element geoemtry");
    return JacobianMatrix{{{0}}};
    }
  };

  auto evaluate_jacobian(const Point &X) const noexcept { return evaluate_jacobian(X.coord()); };

  // TODO: REPEATED CODE: Template and constrain with concept (coodinate type or something like that)
  double detJ(const     geometry::Point<dim> &X) const noexcept { return utils::det(evaluate_jacobian(X)); };
  double detJ(      std::array<double, dim> &&X) const noexcept { return utils::det(evaluate_jacobian(X)); };
  double detJ(const std::array<double, dim>  &X) const noexcept { return utils::det(evaluate_jacobian(X)); };

  // Constructor
  constexpr LagrangeElement() = default;
  constexpr LagrangeElement(pNodeArray nodes) : nodes_p{std::forward<pNodeArray>(nodes)} 
  {
    //std::cout << "node pointer constructor" << std::endl;
    set_VTK_node_order();
  };

  constexpr LagrangeElement(std::size_t &tag, const std::ranges::range auto &node_ids)
    requires(std::same_as<std::ranges::range_value_t<decltype(node_ids)>, PetscInt>)
      : tag_{tag}
  {
    std::copy(node_ids.begin(), node_ids.end(), nodes_.begin());
    //std::cout << "copy index range constructor" << std::endl;
    set_VTK_node_order();
  };

  constexpr LagrangeElement(std::size_t &&tag, std::ranges::range auto &&node_ids)
    requires(std::same_as<std::ranges::range_value_t<decltype(node_ids)>, PetscInt>)
      : tag_{tag}
  {
    std::move(node_ids.begin(), node_ids.end(), nodes_.begin());
    //std::cout << "move index range constructor" << std::endl;
    set_VTK_node_order();
  };


  // Copy and Move Constructors and Assignment Operators
  constexpr LagrangeElement(const LagrangeElement &other) = default;

  constexpr LagrangeElement(LagrangeElement &&other) = default;

  constexpr LagrangeElement &operator=(const LagrangeElement &other) = default;

  constexpr LagrangeElement &operator=(LagrangeElement &&other) = default;
  constexpr ~LagrangeElement() = default;
};

// =================================================================================================
// =================================================================================================
// =================================================================================================

template <std::size_t... N>
class GaussLegendreCellIntegrator
{
  using Array = std::array<double, sizeof...(N)>;
  using Point = geometry::Point<sizeof...(N)>;
  using CellQuadrature = GaussLegendre::CellQuadrature<N...>;

  static constexpr CellQuadrature integrator_{};

public:
  static constexpr std::size_t num_integration_points = CellQuadrature::num_points;

  constexpr auto operator()(const is_LagrangeElement auto &element, std::invocable<Array> auto &&f) const noexcept
  {
    return integrator_([&](const Array &x)
                       {
                         return f(x) * element.detJ(x);
                         // return element.detJ(x) * f(x);
                       });
  };

  constexpr GaussLegendreCellIntegrator() noexcept = default;
  constexpr ~GaussLegendreCellIntegrator() noexcept = default;
};

// =================================================================================================
// =================================================================================================
// =================================================================================================

#endif