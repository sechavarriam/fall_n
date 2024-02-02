#ifndef FALL_N_LAGRANGIAN_FINITE_ELEMENT
#define FALL_N_LAGRANGIAN_FINITE_ELEMENT

#include <array>
#include <cstddef>
#include <memory>


#include "../Node.hh"

#include "../../geometry/Cell.hh"



template <std::size_t... N> class LagrangeElement {
  // using Node_ID       = std::variant<Node<sizeof...(N)>*, std::size_t>;
  using ReferenceCell = geometry::cell::LagrangianCell<N...>;

  static inline constexpr std::size_t dim = sizeof...(N);
  static inline constexpr std::size_t num_nodes = (... * N);

  static inline constexpr ReferenceCell reference_element{};

  std::array<Node<dim> *, num_nodes> nodes_;

public:

  auto detJ() const noexcept {
    //return reference_element.detJ();
  };

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