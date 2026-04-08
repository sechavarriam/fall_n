#ifndef FALL_N_SRC_MODEL_MODEL_CHECKPOINT_HH
#define FALL_N_SRC_MODEL_MODEL_CHECKPOINT_HH

#include <cstddef>
#include <optional>

#include "../petsc/PetscRaii.hh"

template <std::size_t dim, typename ElementContainerT>
struct ModelCheckpoint {
    petsc::OwnedVec state_vector{};
    petsc::OwnedVec imposed_solution{};
    petsc::OwnedVec force_vector{};
    std::optional<ElementContainerT> elements{};

    [[nodiscard]] bool valid() const noexcept {
        return static_cast<bool>(state_vector)
            && static_cast<bool>(imposed_solution);
    }
};

#endif // FALL_N_SRC_MODEL_MODEL_CHECKPOINT_HH
