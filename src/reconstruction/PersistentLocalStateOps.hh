#ifndef FALL_N_SRC_RECONSTRUCTION_PERSISTENT_LOCAL_STATE_OPS_HH
#define FALL_N_SRC_RECONSTRUCTION_PERSISTENT_LOCAL_STATE_OPS_HH

#include <utility>

#include <petsc.h>

namespace fall_n {

template <typename ModelT>
class PersistentLocalStateOps {
    ModelT* model_{nullptr};
    Vec displacement_{nullptr};

    template <typename Fn>
    void with_total_local_displacement_(Fn&& fn) const
    {
        if (!model_ || !displacement_) {
            return;
        }

        DM dm = model_->get_plex();

        Vec u_local;
        DMGetLocalVector(dm, &u_local);
        VecSet(u_local, 0.0);
        DMGlobalToLocal(dm, displacement_, INSERT_VALUES, u_local);
        VecAXPY(u_local, 1.0, model_->imposed_solution());

        std::forward<Fn>(fn)(u_local);

        DMRestoreLocalVector(dm, &u_local);
    }

public:
    PersistentLocalStateOps() = default;

    PersistentLocalStateOps(ModelT* model, Vec displacement) noexcept
        : model_{model}
        , displacement_{displacement}
    {}

    void sync_state_vector() const
    {
        with_total_local_displacement_([&](Vec u_local) {
            VecCopy(u_local, model_->state_vector());
        });
    }

    void commit_state() const
    {
        with_total_local_displacement_([&](Vec u_local) {
            for (auto& elem : model_->elements()) {
                elem.commit_material_state(u_local);
            }
            VecCopy(u_local, model_->state_vector());
        });
    }

    void revert_state() const
    {
        if (!model_) {
            return;
        }

        for (auto& elem : model_->elements()) {
            elem.revert_material_state();
        }
    }
};

}  // namespace fall_n

#endif // FALL_N_SRC_RECONSTRUCTION_PERSISTENT_LOCAL_STATE_OPS_HH
