#ifndef FALL_N_SRC_ANALYSIS_BEAM_MACRO_BRIDGE_HH
#define FALL_N_SRC_ANALYSIS_BEAM_MACRO_BRIDGE_HH

#include <array>
#include <cstddef>
#include <limits>
#include <stdexcept>

#include "MultiscaleCoordinator.hh"
#include "MultiscaleTypes.hh"
#include "../reconstruction/FieldTransfer.hh"

namespace fall_n {

template <typename MacroModelT, typename BeamElementT>
class BeamMacroBridge {
    MacroModelT* model_{nullptr};

    [[nodiscard]] BeamElementT& beam_for_(std::size_t element_id) const {
        auto* beam = model_->elements()[element_id].template as<BeamElementT>();
        if (!beam) {
            throw std::runtime_error(
                "BeamMacroBridge: requested element is not a compatible beam");
        }
        return *beam;
    }

public:
    explicit BeamMacroBridge(MacroModelT& model) : model_{&model} {}

    [[nodiscard]] MacroModelT& model() const noexcept { return *model_; }

    [[nodiscard]] CouplingSite default_site(
        std::size_t element_id,
        double preferred_xi = 0.0) const
    {
        auto& beam = beam_for_(element_id);
        std::size_t best_gp = 0;
        double best_dist = std::numeric_limits<double>::infinity();

        for (std::size_t gp = 0; gp < beam.num_integration_points(); ++gp) {
            const double xi = beam.section_gp_xi(gp);
            const double dist = std::abs(xi - preferred_xi);
            if (dist < best_dist) {
                best_dist = dist;
                best_gp = gp;
            }
        }

        CouplingSite site;
        site.macro_element_id = element_id;
        site.section_gp = best_gp;
        site.xi = beam.section_gp_xi(best_gp);
        site.local_frame = beam.rotation_matrix();
        return site;
    }

    [[nodiscard]] ElementKinematics
    extract_element_kinematics(std::size_t element_id) const
    {
        auto& beam = beam_for_(element_id);
        const auto u_loc = beam.local_state_vector(model_->state_vector());

        ElementKinematics ek;
        ek.element_id = element_id;
        ek.kin_A = extract_section_kinematics(beam, u_loc, -1.0);
        ek.kin_B = extract_section_kinematics(beam, u_loc, +1.0);
        ek.endpoint_A = beam.geometry().map_local_point(std::array{-1.0});
        ek.endpoint_B = beam.geometry().map_local_point(std::array{+1.0});

        const auto up = beam.rotation_matrix().transpose().col(1);
        ek.up_direction = {up[0], up[1], up[2]};
        return ek;
    }

    [[nodiscard]] MacroSectionState
    extract_section_state(const CouplingSite& site) const
    {
        auto& beam = beam_for_(site.macro_element_id);
        const auto u_loc = beam.local_state_vector(model_->state_vector());

        MacroSectionState state;
        state.site = site;
        state.site.local_frame = beam.rotation_matrix();
        auto strain = beam.sample_generalized_strain_local(site.xi, u_loc);
        auto forces = beam.sample_resultants_at_gp(site.section_gp, u_loc);
        state.strain = strain.components();
        state.forces = forces.components();
        return state;
    }

    void inject_response(const SectionHomogenizedResponse& response)
    {
        auto& beam = beam_for_(response.site.macro_element_id);
        beam.set_homogenized_tangent_at_gp(response.site.section_gp,
                                           response.tangent);
        beam.set_homogenized_forces_at_gp(response.site.section_gp,
                                          response.forces,
                                          response.strain_ref);
    }

    void clear_response(const CouplingSite& site)
    {
        auto& beam = beam_for_(site.macro_element_id);
        beam.clear_homogenized_override_at_gp(site.section_gp);
    }
};

}  // namespace fall_n

#endif // FALL_N_SRC_ANALYSIS_BEAM_MACRO_BRIDGE_HH
