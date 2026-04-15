#ifndef FALL_N_GAUSS_RADAU_WEIGHTS_HH
#define FALL_N_GAUSS_RADAU_WEIGHTS_HH

#include <array>
#include <concepts>

#include "GaussRadauNodes.hh"

namespace GaussRadau {

template <std::size_t>
inline constexpr bool unsupported_gauss_radau_weight_order = false;

template<std::size_t n> requires (n > 0)
static consteval std::array<double, n> left_weights() {
    if constexpr (n == 1) {
        return std::array<double, n>{2.000000000000000000000000};
    } else if constexpr (n == 2) {
        return std::array<double, n>{0.500000000000000000000000, 1.500000000000000000000000};
    } else if constexpr (n == 3) {
        return std::array<double, n>{0.222222222222222222222222, 1.024971652376843227770823,
                                     0.752806125400934550007955};
    } else if constexpr (n == 4) {
        return std::array<double, n>{0.125000000000000000000000, 0.657688639960119487888999,
                                     0.776386937686343761560489, 0.440924422353536750550512};
    } else if constexpr (n == 5) {
        return std::array<double, n>{0.080000000000000000000000, 0.446207802167141488805120,
                                     0.623653045951482508163709, 0.562712030298924120384345,
                                     0.287427121582451883759760};
    } else if constexpr (n == 6) {
        return std::array<double, n>{0.055555555555555555555556, 0.319640753220510966545780,
                                     0.485387188468969916159073, 0.520926783189574982570229,
                                     0.416901334311907738959406, 0.201588385253481840209956};
    } else if constexpr (n == 7) {
        return std::array<double, n>{0.040816326530612244897959, 0.239227489225312405787077,
                                     0.380949873644231153805386, 0.447109829014566469907074,
                                     0.424703779005955608398308, 0.318204231467301481744062,
                                     0.148988471112020635459571};
    } else if constexpr (n == 8) {
        return std::array<double, n>{0.031250000000000000000000, 0.185358154802979278540728,
                                     0.304130620646785128975744, 0.376517545389118556572129,
                                     0.391572167452493533969767, 0.347014795634501286672499,
                                     0.249647901329864963257869, 0.114508814744257252011264};
    } else if constexpr (n == 9) {
        return std::array<double, n>{0.024691358024691358024691, 0.147654019046315936300379,
                                     0.247189378204593052361362, 0.316843775670437978338000,
                                     0.348273002772966565633394, 0.337693966975929687977374,
                                     0.286386696357231687174394, 0.200553298024551655039738,
                                     0.090714504923282660650670};
    } else if constexpr (n == 10) {
        return std::array<double, n>{0.020000000000000000000000, 0.120296670557481632535152,
                                     0.204270131879000675555789, 0.268194837841178587051838,
                                     0.305859287724422665888023, 0.313582457226938393490701,
                                     0.290610164832918242534987, 0.239193431714379704861481,
                                     0.164376012736921457954567, 0.073617005486760640972286};
    } else {
        static_assert(unsupported_gauss_radau_weight_order<n>,
                      "Gauss-Radau weights are implemented only for orders 1 through 10.");
    }
}

template<std::size_t n, Endpoint endpoint = Endpoint::Left> requires (n > 0)
static consteval std::array<double, n> weights() {
    if constexpr (endpoint == Endpoint::Left) {
        return left_weights<n>();
    } else {
        auto left = left_weights<n>();
        std::array<double, n> right{};
        for (std::size_t i = 0; i < n; ++i) {
            right[i] = left[n - 1 - i];
        }
        return right;
    }
}

} // namespace GaussRadau

#endif // FALL_N_GAUSS_RADAU_WEIGHTS_HH
