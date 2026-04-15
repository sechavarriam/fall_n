#ifndef FALL_N_GAUSS_RADAU_NODES_HH
#define FALL_N_GAUSS_RADAU_NODES_HH

#include <array>
#include <concepts>

namespace GaussRadau {

enum class Endpoint {
    Left,
    Right
};

template <std::size_t>
inline constexpr bool unsupported_gauss_radau_order = false;

template<std::size_t n> requires (n > 0)
static consteval std::array<double, n> left_evaluation_points() {
    if constexpr (n == 1) {
        return std::array<double, n>{0.00000000000000000000000};
    } else if constexpr (n == 2) {
        return std::array<double, n>{-1.00000000000000000000000, 0.33333333333333333333333};
    } else if constexpr (n == 3) {
        return std::array<double, n>{-1.00000000000000000000000, -0.28989794855663561963946,
                                     0.68989794855663561963946};
    } else if constexpr (n == 4) {
        return std::array<double, n>{-1.00000000000000000000000, -0.57531892352169411205048,
                                     0.18106627111853057827015, 0.82282408097459210520891};
    } else if constexpr (n == 5) {
        return std::array<double, n>{-1.00000000000000000000000, -0.72048027131243889569583,
                                     -0.16718086473783364011340, 0.44631397272375234463991,
                                     0.88579160777096463561376};
    } else if constexpr (n == 6) {
        return std::array<double, n>{-1.00000000000000000000000, -0.80292982840234714775300,
                                     -0.39092854670727218902923, 0.12405037950522771198997,
                                     0.60397316425278365492842, 0.92038028589706251531839};
    } else if constexpr (n == 7) {
        return std::array<double, n>{-1.00000000000000000000000, -0.85389134263948222970375,
                                     -0.53846772406010900183377, -0.11734303754310026416279,
                                     0.32603061943769140180589, 0.70384280066303141630005,
                                     0.94136714568043021605590};
    } else if constexpr (n == 8) {
        return std::array<double, n>{-1.00000000000000000000000, -0.88747487892615570706870,
                                     -0.63951861652621527002484, -0.29475056577366072525590,
                                     0.09430725266111076600290, 0.46842035443082106304642,
                                     0.77064189367819153618072, 0.95504122712257500378235};
    } else if constexpr (n == 9) {
        return std::array<double, n>{-1.00000000000000000000000, -0.91073208942006029853376,
                                     -0.71126748591570885702956, -0.42635048571113896210263,
                                     -0.09037336960685329806492, 0.25613567083345539513829,
                                     0.57138304120873848328492, 0.81735278420041208773843,
                                     0.96444016970527309637359};
    } else if constexpr (n == 10) {
        return std::array<double, n>{-1.00000000000000000000000, -0.92748437423358107811767,
                                     -0.76384204242000259961543, -0.52564603037007918431959,
                                     -0.23623446939058804532124, 0.07605919783797813023371,
                                     0.38066484014472436588076, 0.64776668767400943627365,
                                     0.85122522058160791072816, 0.97117518070224690273434};
    } else {
        static_assert(unsupported_gauss_radau_order<n>,
                      "Gauss-Radau nodes are implemented only for orders 1 through 10.");
    }
}

template<std::size_t n, Endpoint endpoint = Endpoint::Left> requires (n > 0)
static consteval std::array<double, n> evaluation_points() {
    if constexpr (endpoint == Endpoint::Left) {
        return left_evaluation_points<n>();
    } else {
        auto left = left_evaluation_points<n>();
        std::array<double, n> right{};
        for (std::size_t i = 0; i < n; ++i) {
            right[i] = -left[n - 1 - i];
        }
        return right;
    }
}

} // namespace GaussRadau

#endif // FALL_N_GAUSS_RADAU_NODES_HH
