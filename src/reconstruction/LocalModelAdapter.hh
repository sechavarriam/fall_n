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
//  satisfy the concept directly.  When heterogeneous storage is needed
//  (std::vector<LocalModelHandle>), use the type-erased LocalModelHandle.
//
//  Pattern follows FEM_Element (Concept/Model/Handle) — see FEM_Element.hh.
//
// =============================================================================

#include <concepts>
#include <cstddef>
#include <memory>
#include <type_traits>

#include <Eigen/Dense>

#include "FieldTransfer.hh"   // SectionKinematics


namespace fall_n {


// =============================================================================
//  LocalModelAdapter concept
// =============================================================================

template <typename T>
concept LocalModelAdapter = requires(
    T& t,
    const SectionKinematics& kin,
    double width, double height, double h_pert, double time)
{
    // Apply new beam kinematics at both ends
    { t.update_kinematics(kin, kin) };

    // Solve from current converged state to new BC target
    { t.solve_step(time) };

    // Upscaled section tangent  D(6×6) = ∂s/∂e
    { t.section_tangent(width, height, h_pert) }
        -> std::convertible_to<Eigen::Matrix<double, 6, 6>>;

    // Upscaled section forces  s = [N, My, Mz, Vy, Vz, Mt]
    { t.section_forces(width, height) }
        -> std::convertible_to<Eigen::Vector<double, 6>>;

    // Commit / revert material state
    { t.commit_state() };
    { t.revert_state() };

    // Owning beam element id in the global model
    { t.parent_element_id() } -> std::convertible_to<std::size_t>;
};


// =============================================================================
//  LocalModelHandle — type-erased wrapper (value semantics, move-only)
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

        virtual void commit_state() = 0;
        virtual void revert_state() = 0;

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

        void commit_state() override { adapter_.commit_state(); }
        void revert_state() override { adapter_.revert_state(); }

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

    void commit_state()       { pimpl_->commit_state(); }
    void revert_state()       { pimpl_->revert_state(); }

    auto parent_element_id() const -> std::size_t
    { return pimpl_->parent_element_id(); }
};


}  // namespace fall_n

#endif // FALL_N_SRC_RECONSTRUCTION_LOCAL_MODEL_ADAPTER_HH
