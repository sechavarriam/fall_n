#ifndef FALL_N_GAUSS_LOBATTO_NODES_HH
#define FALL_N_GAUSS_LOBATTO_NODES_HH

#include <array>
#include <concepts>

namespace GaussLobatto {

template <std::size_t>
inline constexpr bool unsupported_gauss_lobatto_order = false;

template<std::size_t n> requires (n > 0)
static consteval std::array<double, n> evaluation_points() {
    if constexpr (n == 1) {
        return std::array<double, n>{0.00000000000000000000000};
    } else if constexpr (n == 2) {
        return std::array<double, n>{-1.00000000000000000000000, 1.00000000000000000000000};
    } else if constexpr (n == 3) {
        return std::array<double, n>{-1.00000000000000000000000, 0.00000000000000000000000,
                                     1.00000000000000000000000};
    } else if constexpr (n == 4) {
        return std::array<double, n>{-1.00000000000000000000000, -0.44721359549995793928183,
                                     0.44721359549995793928183, 1.00000000000000000000000};
    } else if constexpr (n == 5) {
        return std::array<double, n>{-1.00000000000000000000000, -0.65465367070797714379829,
                                     0.00000000000000000000000, 0.65465367070797714379829,
                                     1.00000000000000000000000};
    } else if constexpr (n == 6) {
        return std::array<double, n>{-1.00000000000000000000000, -0.76505532392946469285100,
                                     -0.28523151648064509631415, 0.28523151648064509631415,
                                     0.76505532392946469285100, 1.00000000000000000000000};
    } else if constexpr (n == 7) {
        return std::array<double, n>{-1.00000000000000000000000, -0.83022389627856692987203,
                                     -0.46884879347071421380377, 0.00000000000000000000000,
                                     0.46884879347071421380377, 0.83022389627856692987203,
                                     1.00000000000000000000000};
    } else if constexpr (n == 8) {
        return std::array<double, n>{-1.00000000000000000000000, -0.87174014850960661533745,
                                     -0.59170018143314230214451, -0.20929921790247886876866,
                                     0.20929921790247886876866, 0.59170018143314230214451,
                                     0.87174014850960661533745, 1.00000000000000000000000};
    } else if constexpr (n == 9) {
        return std::array<double, n>{-1.00000000000000000000000, -0.89975799541146015731235,
                                     -0.67718627951073775344589, -0.36311746382617815871075,
                                     0.00000000000000000000000, 0.36311746382617815871075,
                                     0.67718627951073775344589, 0.89975799541146015731235,
                                     1.00000000000000000000000};
    } else if constexpr (n == 10) {
        return std::array<double, n>{-1.00000000000000000000000, -0.91953390816645881382893,
                                     -0.73877386510550507500311, -0.47792494981044449566118,
                                     -0.16527895766638697874914, 0.16527895766638697874914,
                                     0.47792494981044449566118, 0.73877386510550507500311,
                                     0.91953390816645881382893, 1.00000000000000000000000};
    } else {
        static_assert(unsupported_gauss_lobatto_order<n>,
                      "Gauss-Lobatto nodes are implemented only for orders 1 through 10.");
    }
}

} // namespace GaussLobatto

#endif // FALL_N_GAUSS_LOBATTO_NODES_HH
