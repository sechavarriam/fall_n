#ifndef FN_ABSTRACT_STRUCTURAL_ELEM_SECTION
#define FN_ABSTRACT_STRUCTURAL_ELEM_SECTION

#include <concepts>


#include "../../geometry/Point.hh"
#include "../Node.hh"


//template <typename T>
//concept SectionT = requires(T point){
//    //point.some_method(); // Define the required interface for the Section concept here.
//};


template <NodeT Node>
class NodeSection{

  Node* node_p_;
  
  public:

  template <typename... Args>
  explicit NodeSection(Node* node_p, [[maybe_unused]] Args&&... args) : node_p_{node_p} {
    // Initialize the section with the provided arguments if needed.
  }

  NodeSection() = delete;
  ~NodeSection() = default;

};






















#endif