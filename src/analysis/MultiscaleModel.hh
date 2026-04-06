#ifndef FALL_N_SRC_ANALYSIS_MULTISCALE_MODEL_HH
#define FALL_N_SRC_ANALYSIS_MULTISCALE_MODEL_HH

#include <cstddef>
#include <utility>
#include <vector>

#include "MultiscaleTypes.hh"
#include "../reconstruction/LocalModelAdapter.hh"

namespace fall_n {

template <typename MacroBridgeT, LocalModelAdapter LocalModelT>
class MultiscaleModel {
    MacroBridgeT macro_bridge_;
    std::vector<CouplingSite> sites_{};
    std::vector<LocalModelT>  local_models_{};

public:
    using macro_bridge_type = MacroBridgeT;
    using local_model_type  = LocalModelT;

    explicit MultiscaleModel(MacroBridgeT macro_bridge)
        : macro_bridge_{std::move(macro_bridge)} {}

    void register_local_model(CouplingSite site, LocalModelT model)
    {
        sites_.push_back(std::move(site));
        local_models_.push_back(std::move(model));
    }

    [[nodiscard]] std::size_t num_local_models() const noexcept {
        return local_models_.size();
    }

    [[nodiscard]] auto& macro_bridge() noexcept { return macro_bridge_; }
    [[nodiscard]] const auto& macro_bridge() const noexcept {
        return macro_bridge_;
    }

    [[nodiscard]] auto& local_models() noexcept { return local_models_; }
    [[nodiscard]] const auto& local_models() const noexcept {
        return local_models_;
    }

    [[nodiscard]] auto& sites() noexcept { return sites_; }
    [[nodiscard]] const auto& sites() const noexcept { return sites_; }

    [[nodiscard]] CouplingSite& site(std::size_t i) noexcept {
        return sites_.at(i);
    }

    [[nodiscard]] const CouplingSite& site(std::size_t i) const noexcept {
        return sites_.at(i);
    }
};

}  // namespace fall_n

#endif // FALL_N_SRC_ANALYSIS_MULTISCALE_MODEL_HH
