#ifndef FALL_N_GAUSS_LOBATTO_WEIGHTS_HH
#define FALL_N_GAUSS_LOBATTO_WEIGHTS_HH

#include <array>
#include <concepts>

namespace GaussLobatto {

template <std::size_t>
inline constexpr bool unsupported_gauss_lobatto_weight_order = false;

template<std::size_t n> requires (n > 0)
static consteval std::array<double, n> weights() {
    if constexpr (n == 1) {
        return std::array<double, n>{2.000000000000000000000000};
    } else if constexpr (n == 2) {
        return std::array<double, n>{1.000000000000000000000000, 1.000000000000000000000000};
    } else if constexpr (n == 3) {
        return std::array<double, n>{0.333333333333333333333333, 1.333333333333333333333333,
                                     0.333333333333333333333333};
    } else if constexpr (n == 4) {
        return std::array<double, n>{0.166666666666666666666667, 0.833333333333333333333333,
                                     0.833333333333333333333333, 0.166666666666666666666667};
    } else if constexpr (n == 5) {
        return std::array<double, n>{0.100000000000000000000000, 0.544444444444444444444444,
                                     0.711111111111111111111111, 0.544444444444444444444444,
                                     0.100000000000000000000000};
    } else if constexpr (n == 6) {
        return std::array<double, n>{0.066666666666666666666667, 0.378474956297846980316612,
                                     0.554858377035486353016721, 0.554858377035486353016721,
                                     0.378474956297846980316612, 0.066666666666666666666667};
    } else if constexpr (n == 7) {
        return std::array<double, n>{0.047619047619047619047619, 0.276826047361565948010700,
                                     0.431745381209862623417871, 0.487619047619047619047619,
                                     0.431745381209862623417871, 0.276826047361565948010700,
                                     0.047619047619047619047619};
    } else if constexpr (n == 8) {
        return std::array<double, n>{0.035714285714285714285714, 0.210704227143506039382991,
                                     0.341122692483504364764240, 0.412458794658703881567052,
                                     0.412458794658703881567052, 0.341122692483504364764240,
                                     0.210704227143506039382991, 0.035714285714285714285714};
    } else if constexpr (n == 9) {
        return std::array<double, n>{0.027777777777777777777778, 0.165495361560805525046339,
                                     0.274538712500161735280705, 0.346428510973046345115131,
                                     0.371519274376417233560091, 0.346428510973046345115131,
                                     0.274538712500161735280705, 0.165495361560805525046339,
                                     0.027777777777777777777778};
    } else if constexpr (n == 10) {
        return std::array<double, n>{0.022222222222222222222222, 0.133305990851070111126227,
                                     0.224889342063126452119457, 0.292042683679683757875582,
                                     0.327539761183897456656510, 0.327539761183897456656510,
                                     0.292042683679683757875582, 0.224889342063126452119457,
                                     0.133305990851070111126227, 0.022222222222222222222222};
    } else {
        static_assert(unsupported_gauss_lobatto_weight_order<n>,
                      "Gauss-Lobatto weights are implemented only for orders 1 through 10.");
    }
}

} // namespace GaussLobatto

#endif // FALL_N_GAUSS_LOBATTO_WEIGHTS_HH
