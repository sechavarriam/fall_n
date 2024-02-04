
#ifndef CELL_INTEGRATION_HH
#define CELL_INTEGRATION_HH

#include <cstddef>

template <std::size_t... N>
class Cell_Integration {
  public:
    Cell_Integration() = default;
    ~Cell_Integration() = default;

    template <typename F>
    constexpr auto integrate(F&& f) const noexcept {
      return f();
    }
};


#endif // CELL_INTEGRATION_HH
