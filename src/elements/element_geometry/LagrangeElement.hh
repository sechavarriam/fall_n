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
struct LagrangeConceptTester{
  static inline constexpr bool _is_in_Lagrange_Family = private_Lagrange_check_<T>;
};

template <typename T>
concept is_LagrangeElement = LagrangeConceptTester<T>::_is_in_Lagrange_Family;

template <std::size_t... N> requires(topology::EmbeddableInSpace<sizeof...(N)>)
class LagrangeElement
{
  template <typename T> friend struct LagrangeConceptTester;
  static inline constexpr bool _is_LagrangeElement() { return true; };

  using ReferenceCell = geometry::cell::LagrangianCell<N...>;
  using Point = geometry::Point<sizeof...(N)>;
  using Array = std::array<double, sizeof...(N)>;

public:
  static inline constexpr std::size_t dim = sizeof...(N);
  static inline constexpr std::size_t num_nodes = (... * N);
  static inline constexpr ReferenceCell reference_element_{};

  static inline constexpr auto VTK_cell_type = reference_element_.VTK_cell_type();

  using pNodeArray = std::optional<std::array<Node<dim> *, num_nodes>>;


  std::size_t tag_   ;
  pNodeArray  nodes_p;

  std::array<PetscInt , num_nodes> nodes_; // Global node numbers in Plex
  std::array<vtkIdType, num_nodes> vtk_nodes_{-1};

private:

  std::array<PetscInt , num_nodes> local_index_{ // default = 0, 1, 2, 3,..., num_nodes - 1
      []<std::size_t... I>(std::index_sequence<I...>){return std::array{static_cast<PetscInt>(I)...};
    }(std::make_index_sequence<num_nodes>{})};

  void set_local_index(const PetscInt idxs[]) noexcept {
    for (std::size_t i = 0; i < num_nodes; ++i) local_index_[i] = idxs[i];
    };  

public:

  // === INFO FOR DEBUG and TESTING ==================================================================
  // =================================================================================================

  void print_info() const noexcept{
    // std::format fmt = "Element Tag: {0}\nNumber of Nodes: {1}\nNodes: {2}\n";
    std::cout << "Element Tag    : " << tag_ << std::endl;
    std::cout << "Number of Nodes: " << num_nodes << std::endl;
    std::cout << "Nodes ID       : ";
    for (std::size_t i = 0; i < num_nodes; ++i)
      std::cout << nodes_[i] << " ";
    std::cout << std::endl;

    #ifdef __clang__ 
    // TALBE [index, local coord..., global coord...] // Using std::format and std::print
      for (std::size_t i = 0; i < num_nodes; ++i)
      {
        std::print("Node: {0:>3} Id: {1:>3} local coord: {2:>5.2f} {3:>5.2f} {4:>5.2f} | global coord: {5:>5.2f} {6:>5.2f} {7:>5.2f} \n",
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
  //static constexpr auto num_nodes() noexcept { return num_nodes; };
  static constexpr auto get_VTK_cell_type() noexcept { return VTK_cell_type; };

  constexpr auto get_VTK_node_ordering() noexcept {
    return reference_element_.VTK_node_ordering();
    //return std::array{ // Esto es para alterar el orden de los nodos en el VTK de aceurdo con la numeracion local
    //  [this]<std::size_t... I>(std::index_sequence<I...>){
    //    return std::array{reference_element_.VTK_node_ordering()[local_index_[I]]...};
    //  }(std::make_index_sequence<num_nodes>{})
    //};
  };

  void set_VTK_node_order() noexcept{ // std::cout << "Setting VTK node order for element " << tag_ << std::endl;
    for (std::size_t i = 0; i < num_nodes; ++i){
      vtk_nodes_[i] = static_cast<vtkIdType>(node(get_VTK_node_ordering()[i])); // std::cout << vtk_nodes_[i] << " ";
    } // std::cout << std::endl;
  };



  std::span<vtkIdType> get_VTK_ordered_node_ids() const noexcept{ // TODO: check if its ordered and if the VTK_node_ordering is correctly set.
    return std::span<vtkIdType>(const_cast<vtkIdType *>(vtk_nodes_.data()), num_nodes);
  };

  // =================================================================================================

  auto id()             const noexcept { return tag_; };
  void set_id(std::size_t id) noexcept { tag_ = id; };

  PetscInt            node  (std::size_t i) const noexcept { return nodes_[i]; };
  std::span<PetscInt> nodes ()              const noexcept { return std::span<PetscInt>(nodes_); };
  Node<dim>          &node_p(std::size_t i) const noexcept { return *nodes_p.value()[i]; };

  void bind_node(std::size_t i, Node<dim> *node) noexcept{
    if (nodes_p.has_value()){
      nodes_p.value()[i] = node;
    }
    else{ // set and assign
      nodes_p = std::array<Node<dim> *, num_nodes>{};
      nodes_p.value()[i] = node;
    }
  };


  // =================================================================================================

  constexpr inline double H(std::size_t i, const Array &X) const noexcept {
    return reference_element_.basis.shape_function(local_index_[i])(X);
  };
  
  constexpr inline double dH_dx(std::size_t i, std::size_t j, const Array &X) const noexcept{
    return reference_element_.basis.shape_function_derivative(local_index_[i], j)(X);
  };

  constexpr inline auto coord_array() const noexcept{
    using CoordArray = std::array<std::array<double, num_nodes>, dim>;
    std::size_t i, j;

    CoordArray coords{};
    for (i = 0; i < dim; ++i){
      for (j = 0; j < num_nodes; ++j){
        coords[i][j] = nodes_p.value()[j]->coord(i);
      }
    }
    return coords;
  };

  // map from reference
  constexpr inline Array map_local_point(const Array &x) const noexcept{

    std::size_t i;
    Array X{0};
    auto F = coord_array();

    for (i = 0; i < dim; ++i)
      X[i] = reference_element_.basis.interpolate(F[i], x);
    return X;
  };

  constexpr inline auto map_local_point(const Point &x) const noexcept { return map_local_point(x.coord()); };

  constexpr inline auto evaluate_jacobian(const Array &X) const noexcept
  {
    Eigen::Matrix<double, dim, dim> J = Eigen::Matrix<double, dim, dim>::Zero();
    Array x{0.0};

    for (std::size_t k = 0; k < num_nodes; ++k){
      x = nodes_p.value()[k]->coord();
      for (std::size_t i = 0; i < dim; ++i){
        for (std::size_t j = 0; j < dim; ++j){
          J(i, j) += (x[i] * dH_dx(k, j, X));
        }
      }
    }
    return J;
  };

  constexpr inline auto evaluate_jacobian(const Point &X) const noexcept { return evaluate_jacobian(X.coord()); };

  constexpr inline auto detJ(const Array &X) const noexcept{return evaluate_jacobian(X).determinant();};

  // Constructor
  constexpr LagrangeElement() = default;
  constexpr LagrangeElement(pNodeArray nodes) : nodes_p{std::forward<pNodeArray>(nodes)}{
    set_VTK_node_order();
  };

  constexpr LagrangeElement(std::size_t &tag, const std::ranges::range auto &node_ids)
    requires(std::same_as<std::ranges::range_value_t<decltype(node_ids)>, PetscInt>) : tag_{tag} 
  {
    //std::cout << "LagrangeElement(std::size_t &tag, const std::ranges::range auto &node_ids)" << std::endl;
    std::copy(node_ids.begin(), node_ids.end(), nodes_.begin());
    set_VTK_node_order();
  };

  constexpr LagrangeElement(std::size_t &&tag, std::ranges::range auto &&node_ids)
    requires(std::same_as<std::ranges::range_value_t<decltype(node_ids)>, PetscInt>) : tag_{tag} {
    //std::cout << "LagrangeElement(std::size_t &&tag, std::ranges::range auto &&node_ids)" << std::endl;
    std::move(node_ids.begin(), node_ids.end(), nodes_.begin());
    set_VTK_node_order();
  };

  constexpr LagrangeElement(std::size_t &&tag, std::ranges::range auto &&node_ids, std::ranges::range auto &&local_ordering)
    requires(std::same_as<std::ranges::range_value_t<decltype(node_ids)>, PetscInt>) : tag_{tag} {
    //std::cout << "LagrangeElement(std::size_t &&tag, std::ranges::range auto &&node_ids, std::ranges::range auto &&local_ordering)" << std::endl;
    std::move(node_ids.begin(), node_ids.end(), nodes_.begin());
    set_local_index(local_ordering.data());
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

  constexpr decltype(auto) operator()([[maybe_unused]] const is_LagrangeElement auto &element, std::invocable<Array> auto &&f) const noexcept{
    using ReturnType = std::invoke_result_t<decltype(f), Array>;
    
    if constexpr (std::is_base_of_v<Eigen::MatrixBase<ReturnType>, ReturnType>){ // If is EigenType
      return integrator_([&](const Array &x) -> ReturnType{
        return (f(x) * element.detJ(x)).eval(); // This temporary has to be created to avoid dangling references!
      });
    } 
    else
    return integrator_([&](const Array &x){  
        return f(x) * element.detJ(x);
      });
  };

  constexpr GaussLegendreCellIntegrator() noexcept = default;
  constexpr ~GaussLegendreCellIntegrator() noexcept = default;
};

// =================================================================================================
// =================================================================================================
// =================================================================================================

#endif