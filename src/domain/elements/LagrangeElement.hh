#ifndef FALL_N_LAGRANGIAN_FINITE_ELEMENT
#define FALL_N_LAGRANGIAN_FINITE_ELEMENT

#include <array>
#include <cstddef>
#include <iterator>
#include <memory>


#include "../Node.hh"

#include "../../geometry/Topology.hh"
#include "../../geometry/Cell.hh"



template<std::size_t dim> requires  topology::EmbeddableInSpace<dim>
inline constexpr auto det(std::array<std::array<double, dim>, dim> A) noexcept
{
  if constexpr (dim == 1){
    return A[0][0];}
  else if constexpr (dim == 2){
    return A[0][0] * A[1][1] - A[0][1] * A[1][0];}
  else if constexpr (dim == 3){
    return A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) 
         - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) 
         + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);}
  else {std::unreachable();}
};

template <std::size_t... N> requires(topology::EmbeddableInSpace<sizeof...(N)>) 
class LagrangeElement {

  using ReferenceCell = geometry::cell::LagrangianCell<N...>;

  static inline constexpr std::size_t dim = sizeof...(N);
  static inline constexpr std::size_t num_nodes = (... * N);

  static inline constexpr ReferenceCell reference_element{};

  std::array<Node<dim> *, num_nodes> nodes_;

public:

  

  auto Jacobian() const noexcept {
    //return reference_element.Jacobian();
  };


  //static constexpr std::shared_ptr<ReferenceCell> reference_element_{std::make_shared<ReferenceCell>()};


  
  void print_node_coords() noexcept {
    for (auto node : nodes_) {
      for (auto j : node->coord()) {
        std::cout << j << " ";
      };
      printf("\n");
    }
  };

  // LagrangeElement() = default;

  LagrangeElement(std::array<Node<dim> *, num_nodes> nodes)
      : nodes_{std::forward<std::array<Node<dim> *, num_nodes>>(nodes)} {};

  LagrangeElement();

  ~LagrangeElement() = default;
};

#endif