#ifndef FALL_N_MATERIAL_POINT_HH
#define FALL_N_MATERIAL_POINT_HH

#include <optional> 
#include <memory>

#include "../materials/Material.hh"
#include "../geometry/IntegrationPoint.hh"
#include "../continuum/ConstitutiveKinematics.hh"

template<class MaterialPolicy> 
class MaterialPoint{// Legacy name for a continuum constitutive site.
        static constexpr std::size_t counter{0};

    private:
        using ConstitutiveHandleT = Material<MaterialPolicy>;
        using MaterialT           = ConstitutiveHandleT; // legacy local alias
        using StateVariableT      = MaterialPolicy::StateVariableT;
        using StressT             = MaterialPolicy::StressT;

        std::size_t id_; // Unique identifier for the material point.

        IntegrationPoint<MaterialPolicy::dim>* gauss_point_;
        // Non-owning link to the numerical sampling site.  MaterialPoint is a
        // constitutive role layered on top of an integration site, not a
        // geometric/topological entity of the mesh.

        MaterialT material_;
        // Owning erased constitutive handle attached to this continuum site.

    public:

        static constexpr std::size_t dim = MaterialPolicy::dim;
        using ConstitutiveSpace = MaterialPolicy;

        [[nodiscard]] const std::array<double, dim>& coord() const noexcept { return gauss_point_->coord(); }
        [[nodiscard]] double coord(std::size_t i) const noexcept { return gauss_point_->coord(i); }
        [[nodiscard]] const std::array<double, dim>& coord_ref() const noexcept { return gauss_point_->coord_ref(); }
        [[nodiscard]] const double* data() const noexcept { return gauss_point_->data(); }
        [[nodiscard]] double* data() noexcept { return gauss_point_->data(); }
        [[nodiscard]] double weight() const noexcept { return gauss_point_->weight(); }

        void update_state(const StateVariableT& state) noexcept {material_.update_state(state);};
        void update_state(StateVariableT&& state) noexcept {material_.update_state(std::forward<StateVariableT>(state));};
        
        decltype(auto) current_state() const noexcept { return material_.current_state(); };

        // ─── Constitutive interface (Strategy-mediated) ─────────────
        //  These methods delegate through the type-erased Material<>,
        //  which routes through the injected UpdateStrategy.

        [[nodiscard]] auto compute_response(const StateVariableT& k) const {
            return material_.compute_response(k);
        }

        [[nodiscard]] auto compute_response(
            const continuum::ConstitutiveKinematics<dim>& kin) const {
            return material_.compute_response(kin);
        }

        [[nodiscard]] auto tangent(const StateVariableT& k) const {
            return material_.tangent(k);
        }

        [[nodiscard]] auto tangent(
            const continuum::ConstitutiveKinematics<dim>& kin) const {
            return material_.tangent(kin);
        }

        void commit(const StateVariableT& k) {
            material_.commit(k);
        }

        void commit(const continuum::ConstitutiveKinematics<dim>& kin) {
            material_.commit(kin);
        }

        // ─── Internal state export (post-processing) ────────────
        [[nodiscard]] InternalFieldSnapshot internal_field_snapshot() const {
            return material_.internal_field_snapshot();
        }

        [[nodiscard]] MaterialConstRef<MaterialPolicy> material_cref() const {
            return constitutive_cref();
        }

        [[nodiscard]] MaterialRef<MaterialPolicy> material_ref() {
            return constitutive_ref();
        }

        [[nodiscard]] MaterialConstRef<MaterialPolicy> constitutive_cref() const {
            return material_.cref();
        }

        [[nodiscard]] MaterialRef<MaterialPolicy> constitutive_ref() {
            return material_.ref();
        }

        void bind_integration_point(IntegrationPoint<dim>& p){
            gauss_point_ = &p;
        };

        [[nodiscard]] IntegrationPoint<dim>* integration_point() noexcept { return gauss_point_; }
        [[nodiscard]] const IntegrationPoint<dim>* integration_point() const noexcept { return gauss_point_; }


        Eigen::Matrix<double, StateVariableT::num_components, StateVariableT::num_components> C() {return material_.C();}; 
        //C() const {return material_.C();}; 

        MaterialPoint(MaterialT material)   : material_{material} {};
        MaterialPoint(MaterialT&& material) : material_{std::move(material)} {};

        MaterialPoint() = default;
        ~MaterialPoint() = default;
        
};

template<class ConstitutiveSpace>
using ContinuumConstitutiveSite = MaterialPoint<ConstitutiveSpace>;

#endif // FALL_N_MATERIAL_POINT_HH
