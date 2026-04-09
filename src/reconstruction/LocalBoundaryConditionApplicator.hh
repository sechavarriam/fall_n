#ifndef FALL_N_SRC_RECONSTRUCTION_LOCAL_BOUNDARY_CONDITION_APPLICATOR_HH
#define FALL_N_SRC_RECONSTRUCTION_LOCAL_BOUNDARY_CONDITION_APPLICATOR_HH

#include <stdexcept>

#include <petsc.h>

#include "FieldTransfer.hh"

namespace fall_n {

template <typename ModelT, typename SubModelT>
class LocalBoundaryConditionApplicator {
    ModelT* model_{nullptr};
    SubModelT* sub_{nullptr};

    void rebuild_face_boundary_conditions_() const
    {
        if (!sub_) {
            return;
        }
        if (sub_->face_min_z_ids.empty() || sub_->face_max_z_ids.empty()) {
            throw std::runtime_error(
                "LocalBoundaryConditionApplicator: empty face-node cache. "
                "The sub-model must cache boundary face nodes before "
                "kinematics are updated.");
        }

        sub_->bc_min_z = compute_boundary_displacements(
            sub_->kin_A, sub_->domain, sub_->face_min_z_ids);
        sub_->bc_max_z = compute_boundary_displacements(
            sub_->kin_B, sub_->domain, sub_->face_max_z_ids);
    }

public:
    LocalBoundaryConditionApplicator() = default;

    LocalBoundaryConditionApplicator(ModelT* model, SubModelT* sub) noexcept
        : model_{model}
        , sub_{sub}
    {}

    void update_kinematics(const SectionKinematics& kin_A,
                           const SectionKinematics& kin_B) const
    {
        if (!sub_) {
            return;
        }

        sub_->kin_A = kin_A;
        sub_->kin_B = kin_B;
        rebuild_face_boundary_conditions_();
    }

    void rebuild_current_boundary_conditions() const
    {
        rebuild_face_boundary_conditions_();
    }

    void write_imposed_values() const
    {
        if (!model_ || !sub_) {
            return;
        }

        Vec imposed = model_->imposed_solution();
        DM  dm = model_->get_plex();

        PetscSection section;
        DMGetLocalSection(dm, &section);

        VecSet(imposed, 0.0);

        PetscScalar* arr;
        VecGetArray(imposed, &arr);

        auto write_bc = [&](const auto& bc_list) {
            for (const auto& [nid, u] : bc_list) {
                const PetscInt plex_pt = sub_->domain.node(nid).sieve_id.value();
                PetscInt offset;
                PetscSectionGetOffset(section, plex_pt, &offset);
                arr[offset] = u[0];
                arr[offset + 1] = u[1];
                arr[offset + 2] = u[2];
            }
        };

        write_bc(sub_->bc_min_z);
        write_bc(sub_->bc_max_z);

        VecRestoreArray(imposed, &arr);
    }
};

}  // namespace fall_n

#endif // FALL_N_SRC_RECONSTRUCTION_LOCAL_BOUNDARY_CONDITION_APPLICATOR_HH
