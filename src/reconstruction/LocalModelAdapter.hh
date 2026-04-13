#ifndef FALL_N_SRC_RECONSTRUCTION_LOCAL_MODEL_ADAPTER_HH
#define FALL_N_SRC_RECONSTRUCTION_LOCAL_MODEL_ADAPTER_HH

// =============================================================================
//  LocalModelAdapter — concept and type-erased handle for FE² sub-models
// =============================================================================
//
//  Defines the interface that any persistent sub-model solver must satisfy
//  in order to participate in a MultiscaleAnalysis coupling loop.
//
//  Concrete models (NonlinearSubModelEvolver, future LinearSubModel, etc.)
//  satisfy the concept directly.  The typed path is the normative public
//  surface used by the publication-ready multiscale API.
//
//  When heterogeneous storage is needed (std::vector<LocalModelHandle>),
//  the type-erased handle below is available as an experimental utility.
//  It is intentionally kept out of the normative public API until exact
//  checkpoint/restore semantics are completed for the erased path.
//
//  Pattern follows FEM_Element (Concept/Model/Handle) — see FEM_Element.hh.
//
// =============================================================================

#include <concepts>
#include <cstddef>
#include <memory>
#include <type_traits>

#include <Eigen/Dense>

#include "../analysis/SubscaleModelConcepts.hh"
#include "../analysis/MultiscaleTypes.hh"
#include "FieldTransfer.hh"   // SectionKinematics


namespace fall_n {


// =============================================================================
//  FE2-specialized section contract
// =============================================================================
//
//  The generic subscale-model concepts now live one level higher in
//  analysis/SubscaleModelConcepts.hh.  The current FE2 production path is a
//  stricter specialization of that generic family:
//
//    - the driving data are beam-section kinematics on two end faces;
//    - the effective operator is a homogenized section law;
//    - and the local solve remains fully checkpointable and observable.
//
//  This preserves the current typed FE2 hot path while making it explicit that
//  "section-local model" is only one realization of the broader subscale-model
//  architecture.

struct SectionSubproblemDrivingState {
    SectionKinematics face_a{};
    SectionKinematics face_b{};
};

struct SectionEffectiveOperatorRequest {
    double width{0.0};
    double height{0.0};
    double tangent_perturbation{1.0e-6};
};

template <typename T>
concept SectionDrivenSubscaleModel = requires(
    T& t,
    const SectionSubproblemDrivingState& driving)
{
    { t.update_kinematics(driving.face_a, driving.face_b) };
};

template <typename T>
concept SectionEffectiveOperatorProvider = requires(
    T& t,
    const SectionEffectiveOperatorRequest& request)
{
    { t.section_tangent(
          request.width, request.height, request.tangent_perturbation) }
        -> std::convertible_to<Eigen::Matrix<double, 6, 6>>;
    { t.section_forces(request.width, request.height) }
        -> std::convertible_to<Eigen::Vector<double, 6>>;
    { t.section_response(
          request.width, request.height, request.tangent_perturbation) }
        -> std::convertible_to<SectionHomogenizedResponse>;
};

template <typename T>
concept LocalModelAdapter =
    StepSolvableSubscaleModel<T> &&
    CheckpointableSubscaleModel<T> &&
    ObservableSubscaleModel<T> &&
    SectionDrivenSubscaleModel<T> &&
    SectionEffectiveOperatorProvider<T>;

template <SectionDrivenSubscaleModel T>
inline void apply_driving_state(
    T& model, const SectionSubproblemDrivingState& driving)
{
    model.update_kinematics(driving.face_a, driving.face_b);
}

template <SectionEffectiveOperatorProvider T>
[[nodiscard]] inline SectionHomogenizedResponse
effective_operator(
    T& model, const SectionEffectiveOperatorRequest& request)
{
    return model.section_response(
        request.width, request.height, request.tangent_perturbation);
}


// =============================================================================
//  LocalModelHandle — experimental type-erased wrapper (value semantics,
//  move-only)
// =============================================================================

class LocalModelHandle {

    // ── Inner concept (virtual interface) ─────────────────────────

    struct Concept {
        virtual ~Concept() = default;

        virtual void update_kinematics(const SectionKinematics& kin_A,
                                       const SectionKinematics& kin_B) = 0;
        virtual void solve_step(double time) = 0;

        virtual Eigen::Matrix<double,6,6>
            section_tangent(double w, double h, double pert) = 0;
        virtual Eigen::Vector<double,6>
            section_forces(double w, double h) = 0;
        virtual SectionHomogenizedResponse
            section_response(double w, double h, double pert) = 0;

        virtual void commit_state() = 0;
        virtual void revert_state() = 0;
        virtual void commit_trial_state() = 0;
        virtual void end_of_step(double time) = 0;
        virtual void set_auto_commit(bool enabled) = 0;
        virtual std::unique_ptr<Concept> clone_checkpointed_model() const = 0;

        virtual std::size_t parent_element_id() const = 0;
    };

    // ── Inner model (type-specific bridge) ────────────────────────

    template <LocalModelAdapter T>
    struct Model final : Concept {
        T adapter_;

        explicit Model(T a) : adapter_(std::move(a)) {}

        void update_kinematics(const SectionKinematics& kA,
                               const SectionKinematics& kB) override
        { adapter_.update_kinematics(kA, kB); }

        void solve_step(double time) override
        { adapter_.solve_step(time); }

        Eigen::Matrix<double,6,6>
        section_tangent(double w, double h, double pert) override
        { return adapter_.section_tangent(w, h, pert); }

        Eigen::Vector<double,6>
        section_forces(double w, double h) override
        { return adapter_.section_forces(w, h); }
        SectionHomogenizedResponse
        section_response(double w, double h, double pert) override
        { return adapter_.section_response(w, h, pert); }

        void commit_state() override { adapter_.commit_state(); }
        void revert_state() override { adapter_.revert_state(); }
        void commit_trial_state() override { adapter_.commit_trial_state(); }
        void end_of_step(double time) override { adapter_.end_of_step(time); }
        void set_auto_commit(bool enabled) override
        { adapter_.set_auto_commit(enabled); }
        std::unique_ptr<Concept> clone_checkpointed_model() const override
        { return nullptr; }

        std::size_t parent_element_id() const override
        { return adapter_.parent_element_id(); }
    };

    // ── Pimpl ─────────────────────────────────────────────────────

    std::unique_ptr<Concept> pimpl_;

public:

    // ── Construct from any LocalModelAdapter ─────────────────────

    template <LocalModelAdapter T>
        requires (!std::same_as<std::remove_cvref_t<T>, LocalModelHandle>)
    LocalModelHandle(T adapter)                                          // NOLINT
        : pimpl_(std::make_unique<Model<std::remove_cvref_t<T>>>(
              std::move(adapter))) {}

    // ── Move-only (sub-models own PETSc resources → non-copyable) ─

    LocalModelHandle(LocalModelHandle&&) noexcept = default;
    LocalModelHandle& operator=(LocalModelHandle&&) noexcept = default;
    ~LocalModelHandle() = default;

    LocalModelHandle(const LocalModelHandle&) = delete;
    LocalModelHandle& operator=(const LocalModelHandle&) = delete;

    // ── Forwarding interface ─────────────────────────────────────

    void update_kinematics(const SectionKinematics& kA,
                           const SectionKinematics& kB)
    { pimpl_->update_kinematics(kA, kB); }

    void solve_step(double time)
    { pimpl_->solve_step(time); }

    auto section_tangent(double w, double h, double pert = 1e-6)
        -> Eigen::Matrix<double,6,6>
    { return pimpl_->section_tangent(w, h, pert); }

    auto section_forces(double w, double h)
        -> Eigen::Vector<double,6>
    { return pimpl_->section_forces(w, h); }
    auto section_response(double w, double h, double pert = 1e-6)
        -> SectionHomogenizedResponse
    { return pimpl_->section_response(w, h, pert); }

    void commit_state()       { pimpl_->commit_state(); }
    void revert_state()       { pimpl_->revert_state(); }
    void commit_trial_state() { pimpl_->commit_trial_state(); }
    void end_of_step(double time) { pimpl_->end_of_step(time); }
    void set_auto_commit(bool enabled) { pimpl_->set_auto_commit(enabled); }

    auto parent_element_id() const -> std::size_t
    { return pimpl_->parent_element_id(); }
};


}  // namespace fall_n

#endif // FALL_N_SRC_RECONSTRUCTION_LOCAL_MODEL_ADAPTER_HH
