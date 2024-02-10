
#ifndef CELL_INTEGRATION_HH
#define CELL_INTEGRATION_HH

#include <cstddef>

template <std::size_t... N>
class CellIntegrator {

  
  public:

    CellIntegrator() = default;
    ~CellIntegrator() = default;

    template <typename F>
    constexpr auto integrate(F&& f) const noexcept {
      return f();
    }
};


#endif // CELL_INTEGRATION_HH
