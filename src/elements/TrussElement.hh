#ifndef FALL_N_TRUSS_ELEMENT_HH
#define FALL_N_TRUSS_ELEMENT_HH

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <petsc.h>

#include "FiniteElementConcept.hh"
#include "element_geometry/ElementGeometry.hh"

#include "../materials/InternalFieldSnapshot.hh"
#include "../materials/Material.hh"
#include "../materials/MaterialPolicy.hh"
#include "../materials/Strain.hh"

template <std::size_t Dim = 3, std::size_t NNodes = 2>
class TrussElement {
    static_assert(Dim == 2 || Dim == 3, "TrussElement supports 2D or 3D space");
    static_assert(
        NNodes == 2 || NNodes == 3,
        "TrussElement currently supports 2-node and 3-node interpolation");

    static constexpr std::size_t num_nodes_ = NNodes;
    static constexpr std::size_t total_dofs_ = num_nodes_ * Dim;
    static constexpr int TD = static_cast<int>(total_dofs_);

    using BMatrixT = Eigen::Matrix<double, 1, TD>;
    using KMatrixT = Eigen::Matrix<double, TD, TD>;
    using FVectorT = Eigen::Matrix<double, TD, 1>;

    struct AxialQuadratureState {
        BMatrixT B = BMatrixT::Zero();
        double strain = 0.0;
        double measure = 0.0;
        double weight = 0.0;
        double weight_measure = 0.0;
    };

    ElementGeometry<Dim>* geometry_{};
    std::vector<Material<UniaxialMaterial>> materials_{};
    double area_{0.0};
    double reference_length_{0.0};
    double density_{0.0};

    std::vector<PetscInt> dof_indices_{};
    bool dofs_cached_{false};

    void ensure_geometry_compatibility_() const
    {
        if (geometry_ == nullptr) {
            throw std::invalid_argument("TrussElement requires a valid geometry.");
        }
        if (geometry_->topological_dimension() != 1) {
            throw std::invalid_argument(
                "TrussElement requires a one-dimensional geometry.");
        }
        if (geometry_->num_nodes() != num_nodes_) {
            throw std::invalid_argument(
                "TrussElement geometry/node-count mismatch.");
        }
        if (geometry_->num_integration_points() == 0) {
            throw std::invalid_argument(
                "TrussElement requires at least one integration point.");
        }
    }

    void initialize_material_sites_(Material<UniaxialMaterial> prototype)
    {
        materials_.clear();
        materials_.reserve(geometry_->num_integration_points());
        for (std::size_t gp = 0; gp < geometry_->num_integration_points(); ++gp) {
            materials_.push_back(prototype);
        }
    }

    void compute_reference_length_()
    {
        reference_length_ = 0.0;
        for (std::size_t gp = 0; gp < geometry_->num_integration_points(); ++gp) {
            const auto xi = geometry_->reference_integration_point(gp);
            reference_length_ +=
                geometry_->weight(gp) * geometry_->differential_measure(xi);
        }
    }

    void ensure_dof_cache()
    {
        if (dofs_cached_) {
            return;
        }
        dof_indices_.clear();
        dof_indices_.reserve(total_dofs_);
        for (std::size_t node = 0; node < num_nodes_; ++node) {
            const auto node_dofs = geometry_->node_p(node).dof_index();
            if (node_dofs.size() < Dim) {
                throw std::runtime_error(
                    "TrussElement found a node with fewer translational DOFs than required.");
            }

            // Mixed continuum/XFEM models may append enriched solid DOFs to
            // an otherwise ordinary displacement node.  A truss bar owns only
            // the translational block; any bond/slip/enrichment coupling must
            // be assembled by a dedicated coupling policy.
            for (std::size_t dim = 0; dim < Dim; ++dim) {
                dof_indices_.push_back(node_dofs[dim]);
            }
        }
        dofs_cached_ = true;
    }

    [[nodiscard]] FVectorT extract_element_dofs_fixed_(Vec u_local)
    {
        ensure_dof_cache();
        FVectorT u_e = FVectorT::Zero();
        VecGetValues(
            u_local,
            static_cast<PetscInt>(total_dofs_),
            dof_indices_.data(),
            u_e.data());
        return u_e;
    }

    [[nodiscard]] AxialQuadratureState evaluate_quadrature_state_(
        std::size_t gp,
        const FVectorT& u_e) const
    {
        AxialQuadratureState state{};
        const auto xi = geometry_->reference_integration_point(gp);
        state.weight = geometry_->weight(gp);
        state.measure = geometry_->differential_measure(xi);
        state.weight_measure = state.weight * state.measure;

        const auto jacobian = geometry_->evaluate_jacobian(xi);
        if (jacobian.cols() < 1) {
            throw std::runtime_error(
                "TrussElement Jacobian does not expose a line tangent.");
        }

        Eigen::Matrix<double, static_cast<int>(Dim), 1> tangent =
            jacobian.col(0);
        const double tangent_norm = tangent.norm();
        if (tangent_norm <= 1.0e-14) {
            throw std::runtime_error(
                "TrussElement encountered a degenerate line geometry.");
        }
        tangent /= tangent_norm;

        for (std::size_t node = 0; node < num_nodes_; ++node) {
            const double dN_ds = geometry_->dH_dx(node, 0, xi) / tangent_norm;
            for (std::size_t dim = 0; dim < Dim; ++dim) {
                state.B[static_cast<Eigen::Index>(node * Dim + dim)] =
                    dN_ds * tangent[static_cast<Eigen::Index>(dim)];
            }
        }

        state.strain = state.B.dot(u_e);
        return state;
    }

    void commit_material_state_(const FVectorT& u_e)
    {
        for (std::size_t gp = 0; gp < materials_.size(); ++gp) {
            const auto state = evaluate_quadrature_state_(gp, u_e);
            Strain<1> strain(state.strain);
            materials_[gp].commit(strain);
            materials_[gp].update_state(std::move(strain));
        }
    }

    [[nodiscard]] std::vector<GaussFieldRecord> collect_gauss_fields_(
        const FVectorT& u_e) const
    {
        std::vector<GaussFieldRecord> records;
        records.reserve(materials_.size());
        for (std::size_t gp = 0; gp < materials_.size(); ++gp) {
            const auto state = evaluate_quadrature_state_(gp, u_e);
            const auto sigma =
                materials_[gp].compute_response(Strain<1>{state.strain});

            GaussFieldRecord rec;
            rec.strain = {state.strain, 0.0, 0.0, 0.0, 0.0, 0.0};
            rec.stress = {sigma.components(), 0.0, 0.0, 0.0, 0.0, 0.0};
            rec.snapshot = materials_[gp].internal_field_snapshot();
            records.push_back(std::move(rec));
        }
        return records;
    }

public:
    TrussElement(
        ElementGeometry<Dim>* geometry,
        Material<UniaxialMaterial> material,
        double area)
        : geometry_{geometry}
        , area_{area}
    {
        ensure_geometry_compatibility_();
        initialize_material_sites_(std::move(material));
        compute_reference_length_();
    }

    TrussElement() = delete;
    ~TrussElement() = default;

    [[nodiscard]] constexpr std::size_t num_nodes() const noexcept
    {
        return num_nodes_;
    }

    [[nodiscard]] std::size_t num_integration_points() const noexcept
    {
        return geometry_->num_integration_points();
    }

    [[nodiscard]] PetscInt sieve_id() const noexcept { return geometry_->sieve_id(); }

    [[nodiscard]] const auto& geometry() const noexcept { return *geometry_; }
    [[nodiscard]] double area() const noexcept { return area_; }
    [[nodiscard]] double length() const noexcept { return reference_length_; }

    [[nodiscard]] const Material<UniaxialMaterial>& material() const noexcept
    {
        return materials_.front();
    }

    [[nodiscard]] Material<UniaxialMaterial>& material() noexcept
    {
        return materials_.front();
    }

    [[nodiscard]] const std::vector<Material<UniaxialMaterial>>& materials() const
        noexcept
    {
        return materials_;
    }

    [[nodiscard]] std::vector<Material<UniaxialMaterial>>& materials() noexcept
    {
        return materials_;
    }

    [[nodiscard]] const std::string& physical_group() const noexcept
    {
        return geometry_->physical_group();
    }

    [[nodiscard]] bool has_physical_group() const noexcept
    {
        return geometry_->has_physical_group();
    }

    [[nodiscard]] Eigen::VectorXd extract_element_dofs(Vec u_local)
    {
        const auto u_e_fixed = extract_element_dofs_fixed_(u_local);
        Eigen::VectorXd u_e(total_dofs_);
        u_e = u_e_fixed;
        return u_e;
    }

    void set_num_dof_in_nodes() noexcept
    {
        for (std::size_t node = 0; node < num_nodes_; ++node) {
            geometry_->node_p(node).set_num_dof(Dim);
        }
    }

    void inject_K(Mat K)
    {
        ensure_dof_cache();
        KMatrixT K_e = KMatrixT::Zero();

        for (std::size_t gp = 0; gp < materials_.size(); ++gp) {
            const auto state =
                evaluate_quadrature_state_(gp, FVectorT::Zero());
            const double E = materials_[gp].C()(0, 0);
            K_e += (area_ * state.weight_measure * E) *
                   (state.B.transpose() * state.B);
        }

        const auto n = static_cast<PetscInt>(total_dofs_);
        MatSetValuesLocal(
            K,
            n,
            dof_indices_.data(),
            n,
            dof_indices_.data(),
            K_e.data(),
            ADD_VALUES);
    }

    void compute_internal_forces(Vec u_local, Vec f_local)
    {
        const FVectorT u_e = extract_element_dofs_fixed_(u_local);
        FVectorT f_e = FVectorT::Zero();

        for (std::size_t gp = 0; gp < materials_.size(); ++gp) {
            const auto state = evaluate_quadrature_state_(gp, u_e);
            const auto sigma =
                materials_[gp].compute_response(Strain<1>{state.strain});
            f_e += (area_ * state.weight_measure * sigma.components()) *
                   state.B.transpose();
        }

        ensure_dof_cache();
        VecSetValues(
            f_local,
            static_cast<PetscInt>(total_dofs_),
            dof_indices_.data(),
            f_e.data(),
            ADD_VALUES);
    }

    [[nodiscard]] Eigen::VectorXd
    compute_internal_force_vector(const Eigen::VectorXd& u_e_dyn)
    {
        if (u_e_dyn.size() != TD) {
            return {};
        }

        const FVectorT u_e = u_e_dyn;
        FVectorT f_e = FVectorT::Zero();
        for (std::size_t gp = 0; gp < materials_.size(); ++gp) {
            const auto state = evaluate_quadrature_state_(gp, u_e);
            const auto sigma =
                materials_[gp].compute_response(Strain<1>{state.strain});
            f_e += (area_ * state.weight_measure * sigma.components()) *
                   state.B.transpose();
        }

        Eigen::VectorXd out(total_dofs_);
        out = f_e;
        return out;
    }

    void inject_tangent_stiffness(Vec u_local, Mat K)
    {
        const FVectorT u_e = extract_element_dofs_fixed_(u_local);
        KMatrixT K_e = KMatrixT::Zero();

        for (std::size_t gp = 0; gp < materials_.size(); ++gp) {
            const auto state = evaluate_quadrature_state_(gp, u_e);
            const double Et =
                materials_[gp].tangent(Strain<1>{state.strain})(0, 0);
            K_e += (area_ * state.weight_measure * Et) *
                   (state.B.transpose() * state.B);
        }

        ensure_dof_cache();
        const auto n = static_cast<PetscInt>(total_dofs_);
        MatSetValuesLocal(
            K,
            n,
            dof_indices_.data(),
            n,
            dof_indices_.data(),
            K_e.data(),
            ADD_VALUES);
    }

    [[nodiscard]] Eigen::MatrixXd
    compute_tangent_stiffness_matrix(const Eigen::VectorXd& u_e_dyn)
    {
        if (u_e_dyn.size() != TD) {
            return {};
        }

        const FVectorT u_e = u_e_dyn;
        KMatrixT K_e = KMatrixT::Zero();
        for (std::size_t gp = 0; gp < materials_.size(); ++gp) {
            const auto state = evaluate_quadrature_state_(gp, u_e);
            const double Et =
                materials_[gp].tangent(Strain<1>{state.strain})(0, 0);
            K_e += (area_ * state.weight_measure * Et) *
                   (state.B.transpose() * state.B);
        }

        Eigen::MatrixXd out(total_dofs_, total_dofs_);
        out = K_e;
        return out;
    }

    [[nodiscard]] const std::vector<PetscInt>& get_dof_indices()
    {
        ensure_dof_cache();
        return dof_indices_;
    }

    void commit_material_state(Vec u_local)
    {
        const FVectorT u_e = extract_element_dofs_fixed_(u_local);
        commit_material_state_(u_e);
    }

    void commit_material_state(const Eigen::VectorXd& u_e_dyn)
    {
        if (u_e_dyn.size() != TD) {
            throw std::invalid_argument(
                "TrussElement local commit requires an element vector with the exact truss DOF count.");
        }
        const FVectorT u_e = u_e_dyn;
        commit_material_state_(u_e);
    }

    void revert_material_state()
    {
        for (auto& material : materials_) {
            material.revert();
        }
    }

    void inject_material_state(impl::StateRef state)
    {
        for (auto& material : materials_) {
            material.inject_internal_state(state);
        }
    }

    [[nodiscard]] bool supports_state_injection() const noexcept
    {
        return !materials_.empty() &&
               std::ranges::all_of(
                   materials_,
                   [](const auto& material) {
                       return material.supports_state_injection();
                   });
    }

    [[nodiscard]] std::vector<GaussFieldRecord> collect_gauss_fields(Vec u_local) const
    {
        auto& self = const_cast<TrussElement&>(*this);
        const FVectorT u_e = self.extract_element_dofs_fixed_(u_local);
        return collect_gauss_fields_(u_e);
    }

    [[nodiscard]] std::vector<GaussFieldRecord> collect_gauss_fields(
        const Eigen::VectorXd& u_e_dyn) const
    {
        if (u_e_dyn.size() != TD) {
            throw std::invalid_argument(
                "TrussElement local Gauss-field extraction requires an element vector with the exact truss DOF count.");
        }
        const FVectorT u_e = u_e_dyn;
        return collect_gauss_fields_(u_e);
    }

    void set_density(double rho) noexcept { density_ = rho; }
    [[nodiscard]] double density() const noexcept { return density_; }

    void inject_mass(Mat M)
    {
        if (density_ <= 0.0) {
            return;
        }
        ensure_dof_cache();

        KMatrixT M_e = KMatrixT::Zero();
        for (std::size_t gp = 0; gp < geometry_->num_integration_points(); ++gp) {
            const auto xi = geometry_->reference_integration_point(gp);
            const double weight_measure =
                geometry_->weight(gp) * geometry_->differential_measure(xi);
            for (std::size_t a = 0; a < num_nodes_; ++a) {
                const double Na = geometry_->H(a, xi);
                for (std::size_t b = 0; b < num_nodes_; ++b) {
                    const double Nb = geometry_->H(b, xi);
                    const double coeff = density_ * area_ * weight_measure * Na * Nb;
                    for (std::size_t dim = 0; dim < Dim; ++dim) {
                        const auto row =
                            static_cast<Eigen::Index>(a * Dim + dim);
                        const auto col =
                            static_cast<Eigen::Index>(b * Dim + dim);
                        M_e(row, col) += coeff;
                    }
                }
            }
        }

        const auto n = static_cast<PetscInt>(total_dofs_);
        MatSetValuesLocal(
            M,
            n,
            dof_indices_.data(),
            n,
            dof_indices_.data(),
            M_e.data(),
            ADD_VALUES);
    }
};

static_assert(
    FiniteElement<TrussElement<3>>,
    "TrussElement<3> must satisfy the FiniteElement concept");
static_assert(
    FiniteElement<TrussElement<2>>,
    "TrussElement<2> must satisfy the FiniteElement concept");
static_assert(
    FiniteElement<TrussElement<3, 3>>,
    "TrussElement<3,3> must satisfy the FiniteElement concept");

#endif // FALL_N_TRUSS_ELEMENT_HH
