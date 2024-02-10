#ifndef FALL_N_LAGRANGIAN_FINITE_ELEMENT
#define FALL_N_LAGRANGIAN_FINITE_ELEMENT

#include <array>
#include <cstddef>
#include <iterator>
#include <memory>
#include <tuple>


#include "../Node.hh"

#include "../IntegrationPoint.hh"

#include "../../geometry/Topology.hh"
#include "../../geometry/Cell.hh"
#include "../../geometry/Point.hh"


#include "../../utils/small_math.hh"


template <std::size_t... N> requires(topology::EmbeddableInSpace<sizeof...(N)>) 
class LagrangeElement {

  using ReferenceCell  = geometry::cell::LagrangianCell<N...>;
  
  static inline constexpr std::size_t dim = sizeof...(N);

  static inline constexpr std::size_t num_nodes_ = (... * N);
  static inline constexpr ReferenceCell reference_element_{};

  using pNodeArray = std::array<Node<dim> *, num_nodes_>;
  using JacobianMatrix = std::array<std::array<double, dim>, dim>;



  std::size_t tag_;

  pNodeArray nodes_;

  std::vector <IntegrationPoint<dim>> integration_points_;  

public:

  
  
  auto num_nodes() const noexcept { return num_nodes_; };
  auto id()        const noexcept { return tag_      ; };
  
  auto node(std::size_t i) const noexcept { return nodes_[i]; };

  //std::size_t num_nodes() const { return num_nodes; };

  void set_id(std::size_t id) noexcept { tag_ = id; };

  void set_num_integration_points(std::size_t num) noexcept {
    integration_points_.resize(num);
  };
  
  void set_integration_points(std::vector <IntegrationPoint<dim>> points) noexcept {
    integration_points_ = std::move(points);
  };


  // TODO: REPEATED CODE: Template and constrain with concept (coodinate type or something like that)
  auto evaluate_jacobian(const geometry::Point<dim>& X) noexcept { //Thread Candidate
    JacobianMatrix J;
    for (auto i = 0; i < dim; ++i) {
      for (auto j = 0; j < dim; ++j) {
        for (auto k = 0; k < num_nodes_; ++k) {
          J[i][j] += nodes_[k]->coord(i) 
                   * std::invoke(
                          reference_element_.basis.shape_function_derivative(k, j),
                          X.coord());
        }
      }
    }
    return J;
  };  

  auto evaluate_jacobian(const std::array<double,dim>& X) noexcept { 
    JacobianMatrix J;
    for (auto i = 0; i < dim; ++i) {  //Thread Candidate
      for (auto j = 0; j < dim; ++j) {//Thread Candidate
        for (auto k = 0; k < num_nodes_; ++k) {
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
  auto detJ(const geometry::Point<dim>&    X) noexcept {return utils::det(evaluate_jacobian(X));};
  auto detJ(      std::array<double,dim>&& X) noexcept {return utils::det(evaluate_jacobian(X));};
  auto detJ(const std::array<double,dim>&  X) noexcept {return utils::det(evaluate_jacobian(X));};


  // TODO: Refactor with std::format 
  void print_node_coords() noexcept {
    for (auto node : nodes_) {
      for (auto j = 0; j < dim; ++j) {
        printf("%f ", node->coord(j));
      };
      printf("\n");
    }
  };

  // LagrangeElement() = default;

  LagrangeElement(std::array<Node<dim> *, num_nodes_> nodes)
      : nodes_{std::forward<std::array<Node<dim> *, num_nodes_>>(nodes)}{};

  LagrangeElement();

  ~LagrangeElement() = default;
};

#endif