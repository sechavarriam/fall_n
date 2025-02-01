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

  std::array<PetscInt, num_nodes_> nodes_; // Global node numbers in Plex
  std::array<vtkIdType, num_nodes_> vtk_nodes_{-1};

public:
  // =================================================================================================
  // === INFO FOR DEBUG and TESTING ==================================================================
  // =================================================================================================

  void print_info() const noexcept
  {

    // std::format fmt = "Element Tag: {0}\nNumber of Nodes: {1}\nNodes: {2}\n";
    std::cout << "Element Tag    : " << tag_ << std::endl;
    std::cout << "Number of Nodes: " << num_nodes_ << std::endl;
    std::cout << "Nodes ID       : ";
    for (std::size_t i = 0; i < num_nodes_; ++i)
      std::cout << nodes_[i] << " ";
    std::cout << std::endl;

    #ifdef __clang__ 
    // TALBE [index, local coord..., global coord...] // Using std::format and std::print
      for (std::size_t i = 0; i < num_nodes_; ++i)
      {
        std::print("Node: {0:>3} Id: {1:>3} local coord: {2:>5.2f} {3:>5.2f} {4:>5.2f} | global coord: {5:>5.2f} {6:>5.2f} {7:>5.2f}\n",
                   i,
                   nodes_p.value()[i]->id(),
                   reference_element_.reference_nodes[i].coord()[0],
                   reference_element_.reference_nodes[i].coord()[1],
                   reference_element_.reference_nodes[i].coord()[2],
                   nodes_p.value()[i]->coord(0),
                   nodes_p.value()[i]->coord(1),
                   nodes_p.value()[i]->coord(2));
      }
    #endif
  };

  // =================================================================================================
  // === VTK THINGS ==================================================================================
  // =================================================================================================
  static constexpr auto num_nodes() noexcept { return num_nodes_; };
  static constexpr auto get_VTK_cell_type() noexcept { return VTK_cell_type; };
  static constexpr auto get_VTK_node_ordering() noexcept { return reference_element_.VTK_node_ordering(); };

  void set_VTK_node_order() noexcept
  { // std::cout << "Setting VTK node order for element " << tag_ << std::endl;
    for (std::size_t i = 0; i < num_nodes_; ++i)
    {
      vtk_nodes_[i] = static_cast<vtkIdType>(node(get_VTK_node_ordering()[i])); // std::cout << vtk_nodes_[i] << " ";
    } // std::cout << std::endl;
  };

  std::span<vtkIdType> get_VTK_ordered_node_ids() const noexcept
  { // TODO: check if its ordered and if the VTK_node_ordering is correctly set.
    return std::span<vtkIdType>(const_cast<vtkIdType *>(vtk_nodes_.data()), num_nodes_);
  };

  // =================================================================================================

  auto id() const noexcept { return tag_; };
  void set_id(std::size_t id) noexcept { tag_ = id; };

  PetscInt            node  (std::size_t i) const noexcept { return nodes_[i]; };
  std::span<PetscInt> nodes ()              const noexcept { return std::span<PetscInt>(nodes_); };
  Node<dim>          &node_p(std::size_t i) const noexcept { return *nodes_p.value()[i]; };

  void bind_node(std::size_t i, Node<dim> *node) noexcept{
    if (nodes_p.has_value()){
      nodes_p.value()[i] = node;
    }
    else{ // set and assign
      nodes_p = std::array<Node<dim> *, num_nodes_>{};
      nodes_p.value()[i] = node;
    }
  };

  constexpr inline double H(std::size_t i, const Array &X) const noexcept // Quien es en realidad i?
  {
    return reference_element_.basis.shape_function(i)(X);
  };

  constexpr inline double H(std::size_t i, const Point &X) const noexcept { return H(i, X.coord()); }; 

  //template <std::size_t j> // TODO: REMOVE THIS? It is not used in this way
  //constexpr inline double aux_dH_dx(std::size_t i, const Array &X) const noexcept{
  //  return reference_element_.basis.aux_shape_function_derivative<j>(i)(X);
  //};

  constexpr inline double dH_dx(std::size_t i, std::size_t j, const Array &X) const noexcept
  {
    return reference_element_.basis.shape_function_derivative(i, j)(X);
  };

  constexpr inline double dH_dx(std::size_t i, std::size_t j, const Point &X) const noexcept
  {
    return dH_dx(i, j, X.coord());
  };

  constexpr inline auto coord_array() const noexcept
  {
    using CoordArray = std::array<std::array<double, num_nodes_>, dim>;
    std::size_t i, j;

    CoordArray coords{};
    for (i = 0; i < dim; ++i){
      for (j = 0; j < num_nodes_; ++j){
        coords[i][j] = nodes_p.value()[j]->coord(i);
      }
    }
    return coords;
  };

  // map from reference
  constexpr inline Array map_local_point(const Array &x) const noexcept
  {

    std::size_t i;
    Array X{0};
    auto F = coord_array();

    for (i = 0; i < dim; ++i)
      X[i] = reference_element_.basis.interpolate(F[i], x);
    return X;
  };

  constexpr inline auto map_local_point(const Point &x) const noexcept { return map_local_point(x.coord()); };

  constexpr inline auto evaluate_jacobian_V2(const Array &X) const noexcept
  {
    Eigen::Matrix<double, dim, dim> J;
    J.setZero();
    Array x{0.0};

    for (std::size_t k = 0; k < num_nodes_; ++k){
      x = nodes_p.value()[k]->coord();
      for (std::size_t i = 0; i < dim; ++i){
        for (std::size_t j = 0; j < dim; ++j){
          J(i, j) += (x[i] * dH_dx(k, j, X));
        }
      }
    }
    return J;
  };


  constexpr inline auto evaluate_jacobian_V2(const Point &X) const noexcept { return evaluate_jacobian_V2(X.coord()); };
  constexpr inline auto detJ_V2             (const Array &X) const noexcept { return evaluate_jacobian_V2(X).determinant(); };


  constexpr inline auto evaluate_jacobian(const Array &X) const noexcept{
    if (nodes_p.has_value())
    {
      Array x{0.0};
      JacobianMatrix J{{{0.0}}};

      for (std::size_t k = 0; k < num_nodes_; ++k)
      {
        x = nodes_p.value()[k]->coord();
        // std::println(" node {0} coords = {1:> 2.8e} {2:> 2.8e} {3:> 2.8e}\n", k, x[0], x[1], x[2]);
        for (std::size_t i = 0; i < dim; ++i)
        { // Thread Candidate
          for (std::size_t j = 0; j < dim; ++j)
          {                                     // Thread Candidate
            J[i][j] += (x[i] * dH_dx(k, j, X)); //*std::invoke(reference_element_.basis.shape_function_derivative(k, j),X);

            // std::println(" i = {0} j = {1} -> J[{0}][{1}] = {3} --> dH_dx(k, j, X) = {5}\n",
            //              i, j, k, J[i][j], nodes_p.value()[k]->coord(i), dH_dx(k, j, X));
          }
        }
      }
      // std::println("Jacobian Matrix: {0:> 2.8e} {1:> 2.8e} {2:> 2.8e}\n"
      //              "                 {3:> 2.8e} {4:> 2.8e} {5:> 2.8e}\n"
      //              "                 {6:> 2.8e} {7:> 2.8e} {8:> 2.8e}\n",
      //              J[0][0], J[0][1], J[0][2],
      //              J[1][0], J[1][1], J[1][2],
      //              J[2][0], J[2][1], J[2][2]);
      // std::println("Determinant of Jacobian Matrix: {0:> 2.5f}\n", utils::det(J));
      return J;
    }
    else
    {
      std::runtime_error("LagrangeElement: Nodes are not linked yet to the element geoemtry");
      return JacobianMatrix{{{0.0}}};
    }
  };

  auto evaluate_jacobian(const Point &X) const noexcept { return evaluate_jacobian(X.coord()); };

  // TODO: REPEATED CODE: Template and constrain with concept (coodinate type or something like that)
  double detJ(const geometry::Point<dim> &X) const noexcept { return utils::det<dim>(evaluate_jacobian(X)); };
  double detJ(std::array<double, dim> &&X) const noexcept { return utils::det<dim>(evaluate_jacobian(X)); };
  double detJ(const std::array<double, dim> &X) const noexcept { return utils::det<dim>(evaluate_jacobian(X)); };

  // Constructor
  constexpr LagrangeElement() = default;
  constexpr LagrangeElement(pNodeArray nodes) : nodes_p{std::forward<pNodeArray>(nodes)}
  {
    // std::cout << "node pointer constructor" << std::endl;
    set_VTK_node_order();
  };

  constexpr LagrangeElement(std::size_t &tag, const std::ranges::range auto &node_ids)
    requires(std::same_as<std::ranges::range_value_t<decltype(node_ids)>, PetscInt>)
      : tag_{tag}
  {
    std::copy(node_ids.begin(), node_ids.end(), nodes_.begin());
    // std::cout << "copy index range constructor" << std::endl;
    set_VTK_node_order();
  };

  constexpr LagrangeElement(std::size_t &&tag, std::ranges::range auto &&node_ids)
    requires(std::same_as<std::ranges::range_value_t<decltype(node_ids)>, PetscInt>)
      : tag_{tag}
  {
    std::move(node_ids.begin(), node_ids.end(), nodes_.begin());
    // std::cout << "move index range constructor" << std::endl;
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
  using CellQuadrature = GaussLegendre::CellQuadrature<N...>;

  static constexpr CellQuadrature integrator_{};

public:
  static constexpr std::size_t num_integration_points = CellQuadrature::num_points;

  static constexpr auto reference_integration_point(std::size_t i) noexcept{
    return integrator_.get_point_coords(i);
  };

  static constexpr auto weight(std::size_t i) noexcept {
    return integrator_.get_point_weight(i);
  };

  constexpr auto operator()([[maybe_unused]] const is_LagrangeElement auto &element, std::invocable<Array> auto &&f) const noexcept{
    return integrator_([&](const Array &x){
        //std::println("At Gauss Legendre Integrator: x = {0:> 2.5e} {1:> 2.5e} {2:> 2.5e} ----> detJ = {3:> 2.2f}\n", x[0], x[1], x[2], element.detJ(x));  
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