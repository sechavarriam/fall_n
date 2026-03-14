#ifndef FALL_N_CONSTITUTIVE_PROTOCOL_HH
#define FALL_N_CONSTITUTIVE_PROTOCOL_HH

#include <concepts>
#include <cstddef>
#include <functional>
#include <iterator>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

// =============================================================================
//  ConstitutiveProtocol utilities
// =============================================================================
//
//  Lightweight material-level sampling helpers used to exercise constitutive
//  laws and constitutive sites under prescribed loading histories.
//
//  The design goal is to stay solver-independent and zero-overhead on the hot
//  path:
//    - the protocol is just a span/range of kinematic states;
//    - the response is sampled through the existing constitutive interface;
//    - commit is dispatched at compile time to either update(k) or commit(k).
//
//  This is intentionally strain/displacement-driven.  Force-controlled local
//  protocols would require an additional inverse solve and are left as a higher
//  level concern.
//
// =============================================================================

struct ConstitutiveCurveSample {
    double abscissa{0.0};
    double ordinate{0.0};
    double path_parameter{0.0};
    std::size_t step{0};
};

template <typename SiteT, typename KinematicT>
inline void commit_protocol_step(SiteT& site, const KinematicT& k) {
    if constexpr (requires { site.commit(k); }) {
        site.commit(k);
    } else if constexpr (requires { site.update(k); }) {
        site.update(k);
    } else {
        static_assert(std::is_void_v<SiteT>,
                      "Constitutive protocol requires either commit(k) or update(k).");
    }
}

template <typename SiteT,
          typename HistoryRange,
          typename AbscissaProjector,
          typename OrdinateProjector>
requires requires(const HistoryRange& history) {
    std::begin(history);
    std::end(history);
}
inline std::vector<ConstitutiveCurveSample> sample_constitutive_protocol(
    SiteT& site,
    const HistoryRange& history,
    AbscissaProjector&& abscissa_of,
    OrdinateProjector&& ordinate_of)
{
    std::vector<ConstitutiveCurveSample> out;

    std::size_t step = 0;
    for (const auto& k : history) {
        const auto response = site.compute_response(k);
        out.push_back(ConstitutiveCurveSample{
            .abscissa = static_cast<double>(std::invoke(abscissa_of, k)),
            .ordinate = static_cast<double>(std::invoke(ordinate_of, response)),
            .path_parameter = static_cast<double>(step),
            .step = step,
        });
        commit_protocol_step(site, k);
        ++step;
    }

    return out;
}

#endif // FALL_N_CONSTITUTIVE_PROTOCOL_HH
