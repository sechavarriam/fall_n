#include <cmath>
#include <cstddef>
#include <iostream>

#include "src/continuum/Continuum.hh"
#include "src/continuum/HyperelasticRelation.hh"
#include "src/materials/LinealElasticMaterial.hh"
#include "src/materials/Material.hh"

using namespace continuum;

namespace {

constexpr bool approx(double a, double b, double tol = 1e-10) {
    return std::abs(a - b) <= tol;
}

int passed = 0;
int failed = 0;

void report(const char* name, bool ok) {
    if (ok) {
        ++passed;
        std::cout << "  PASS  " << name << "\n";
    } else {
        ++failed;
        std::cout << "  FAIL  " << name << "\n";
    }
}

template <typename VecA, typename VecB>
double max_abs_diff(const VecA& a, const VecB& b) {
    return (a - b).cwiseAbs().maxCoeff();
}

template <typename MatA, typename MatB>
double max_abs_diff_m(const MatA& a, const MatB& b) {
    return (a - b).cwiseAbs().maxCoeff();
}

void test_constitutive_kinematics_small_strain() {
    GPKinematics<3> gp;
    gp.strain_voigt << 0.01, -0.02, 0.03, 0.04, -0.06, 0.02;
    gp.F = Tensor2<3>::identity();
    gp.detF = 1.0;

    auto kin = make_constitutive_kinematics<SmallStrain>(gp);

    report("ck_small_measure_kind",
        kin.active_strain_measure == StrainMeasureKind::infinitesimal);
    report("ck_small_stress_kind",
        kin.conjugate_stress_measure == StressMeasureKind::cauchy);
    report("ck_small_voigt_passthrough",
        max_abs_diff(kin.active_strain_voigt(), gp.strain_voigt) < 1e-14);
    report("ck_small_tensor_roundtrip",
        max_abs_diff(kin.infinitesimal_strain.voigt_engineering(), gp.strain_voigt) < 1e-14);
    report("ck_small_green_matches_linearized",
        max_abs_diff(kin.green_lagrange_strain.voigt_engineering(), gp.strain_voigt) < 1e-14);
}

void test_constitutive_kinematics_total_lagrangian() {
    Tensor2<3> F;
    F(0,0)=1.10; F(0,1)=0.02; F(0,2)=-0.01;
    F(1,0)=0.01; F(1,1)=0.97; F(1,2)=0.03;
    F(2,0)=-0.02; F(2,1)=0.01; F(2,2)=1.04;

    GPKinematics<3> gp;
    gp.F = F;
    gp.detF = F.determinant();
    gp.strain_voigt = strain::green_lagrange(F).voigt_engineering();

    auto kin = make_constitutive_kinematics<TotalLagrangian>(gp);
    auto E = strain::green_lagrange(F);

    report("ck_tl_measure_kind",
        kin.active_strain_measure == StrainMeasureKind::green_lagrange);
    report("ck_tl_stress_kind",
        kin.conjugate_stress_measure == StressMeasureKind::second_piola_kirchhoff);
    report("ck_tl_green_from_F",
        max_abs_diff(kin.green_lagrange_strain.voigt_engineering(), E.voigt_engineering()) < 1e-14);
    report("ck_tl_voigt_matches_green",
        max_abs_diff(kin.active_strain_voigt(), E.voigt_engineering()) < 1e-14);
    report("ck_tl_detF",
        approx(kin.detF, F.determinant(), 1e-14));
}

void test_constitutive_kinematics_updated_lagrangian() {
    Tensor2<3> F;
    F(0,0)=1.07; F(0,1)=0.04; F(0,2)=-0.01;
    F(1,0)=0.02; F(1,1)=0.96; F(1,2)=0.03;
    F(2,0)=-0.01; F(2,1)=0.01; F(2,2)=1.02;

    GPKinematics<3> gp;
    gp.F = F;
    gp.detF = F.determinant();
    gp.strain_voigt = strain::almansi(F).voigt_engineering();

    auto kin = make_constitutive_kinematics<UpdatedLagrangian>(gp);
    auto e = strain::almansi(F);
    auto E = strain::green_lagrange(F);

    report("ck_ul_measure_kind",
        kin.active_strain_measure == StrainMeasureKind::almansi);
    report("ck_ul_stress_kind",
        kin.conjugate_stress_measure == StressMeasureKind::cauchy);
    report("ck_ul_almansi_from_F",
        max_abs_diff(kin.almansi_strain.voigt_engineering(), e.voigt_engineering()) < 1e-14);
    report("ck_ul_active_voigt_matches_almansi",
        max_abs_diff(kin.active_strain_voigt(), e.voigt_engineering()) < 1e-14);
    report("ck_ul_green_from_F",
        max_abs_diff(kin.green_lagrange_strain.voigt_engineering(), E.voigt_engineering()) < 1e-14);
}

void test_hyperelastic_relation_direct_continuum_interface() {
    static_assert(ContinuumKinematicsAwareConstitutiveRelation<SVKRelation<3>>);
    static_assert(ContinuumKinematicsAwareConstitutiveRelation<NeoHookeanRelation<3>>);

    auto svk_model = SaintVenantKirchhoff<3>::from_E_nu(21000.0, 0.25);
    auto nh_model  = CompressibleNeoHookean<3>::from_E_nu(21000.0, 0.25);

    SVKRelation<3> svk{svk_model};
    NeoHookeanRelation<3> nh{nh_model};

    Tensor2<3> F;
    F(0,0)=1.08; F(0,1)=0.03; F(0,2)=-0.01;
    F(1,0)=0.02; F(1,1)=0.98; F(1,2)=0.015;
    F(2,0)=-0.01; F(2,1)=0.00; F(2,2)=1.05;

    GPKinematics<3> gp;
    gp.F = F;
    gp.detF = F.determinant();
    gp.strain_voigt = strain::green_lagrange(F).voigt_engineering();
    auto kin = make_constitutive_kinematics<TotalLagrangian>(gp);

    Strain<6> legacy_strain;
    legacy_strain.set_strain(kin.active_strain_voigt());

    auto svk_old = svk.compute_response(legacy_strain);
    auto svk_new = svk.compute_response(kin);
    auto nh_old = nh.compute_response(legacy_strain);
    auto nh_new = nh.compute_response(kin);

    auto C_svk_old = svk.tangent(legacy_strain);
    auto C_svk_new = svk.tangent(kin);
    auto C_nh_old = nh.tangent(legacy_strain);
    auto C_nh_new = nh.tangent(kin);

    report("svk_direct_carrier_matches_legacy",
        max_abs_diff(svk_old.components(), svk_new.components()) < 1e-12);
    report("nh_direct_carrier_matches_legacy",
        max_abs_diff(nh_old.components(), nh_new.components()) < 1e-10);
    report("svk_direct_tangent_matches_legacy",
        max_abs_diff_m(C_svk_old, C_svk_new) < 1e-12);
    report("nh_direct_tangent_matches_legacy",
        max_abs_diff_m(C_nh_old, C_nh_new) < 1e-10);
}

void test_hyperelastic_relation_direct_spatial_interface() {
    auto svk_model = SaintVenantKirchhoff<3>::from_E_nu(21000.0, 0.25);
    auto nh_model  = CompressibleNeoHookean<3>::from_E_nu(21000.0, 0.25);

    SVKRelation<3> svk{svk_model};
    NeoHookeanRelation<3> nh{nh_model};

    Tensor2<3> F;
    F(0,0)=1.06; F(0,1)=0.02; F(0,2)=-0.01;
    F(1,0)=0.01; F(1,1)=0.99; F(1,2)=0.025;
    F(2,0)=-0.02; F(2,1)=0.00; F(2,2)=1.03;

    GPKinematics<3> gp;
    gp.F = F;
    gp.detF = F.determinant();
    gp.strain_voigt = strain::almansi(F).voigt_engineering();
    auto kin = make_constitutive_kinematics<UpdatedLagrangian>(gp);
    auto E = strain::green_lagrange(F);

    auto S_svk = svk_model.second_piola_kirchhoff(E);
    auto sigma_svk = stress::cauchy_from_2pk(S_svk, F);
    auto c_svk = ops::push_forward_tangent(svk_model.material_tangent(E), F);

    auto S_nh = nh_model.second_piola_kirchhoff(E);
    auto sigma_nh = stress::cauchy_from_2pk(S_nh, F);
    auto c_nh = ops::push_forward_tangent(nh_model.material_tangent(E), F);

    auto svk_spatial = svk.compute_response(kin);
    auto nh_spatial = nh.compute_response(kin);

    auto C_svk_spatial = svk.tangent(kin);
    auto C_nh_spatial = nh.tangent(kin);

    report("svk_direct_spatial_stress",
        max_abs_diff(svk_spatial.components(), sigma_svk.voigt()) < 1e-12);
    report("nh_direct_spatial_stress",
        max_abs_diff(nh_spatial.components(), sigma_nh.voigt()) < 1e-10);
    report("svk_direct_spatial_tangent",
        max_abs_diff_m(C_svk_spatial, c_svk.voigt_matrix()) < 1e-12);
    report("nh_direct_spatial_tangent",
        max_abs_diff_m(C_nh_spatial, c_nh.voigt_matrix()) < 1e-10);
}

void test_erased_continuum_handle_path() {
    auto svk_model = SaintVenantKirchhoff<3>::from_E_nu(21000.0, 0.25);
    MaterialInstance<SVKRelation<3>> site{svk_model};
    Material<ThreeDimensionalMaterial> mat{site, ElasticUpdate{}};

    Tensor2<3> F;
    F(0,0)=1.03; F(0,1)=0.01; F(0,2)=0.00;
    F(1,0)=0.00; F(1,1)=0.99; F(1,2)=0.02;
    F(2,0)=0.00; F(2,1)=0.00; F(2,2)=1.01;

    GPKinematics<3> gp;
    gp.F = F;
    gp.detF = F.determinant();
    gp.strain_voigt = strain::green_lagrange(F).voigt_engineering();
    auto kin = make_constitutive_kinematics<TotalLagrangian>(gp);

    Strain<6> legacy_strain;
    legacy_strain.set_strain(kin.active_strain_voigt());

    auto s_mat = mat.compute_response(kin);
    auto s_old = mat.compute_response(legacy_strain);

    report("erased_material_continuum_overload",
        max_abs_diff(s_mat.components(), s_old.components()) < 1e-12);
    report("erased_material_tangent_continuum_overload",
        max_abs_diff_m(mat.tangent(kin), mat.tangent(legacy_strain)) < 1e-12);
}

void test_erased_material_spatial_overload() {
    auto nh_model = CompressibleNeoHookean<3>::from_E_nu(21000.0, 0.25);
    MaterialInstance<NeoHookeanRelation<3>> site{nh_model};
    Material<ThreeDimensionalMaterial> mat{site, ElasticUpdate{}};

    Tensor2<3> F;
    F(0,0)=1.05; F(0,1)=0.02; F(0,2)=0.00;
    F(1,0)=0.01; F(1,1)=0.97; F(1,2)=0.03;
    F(2,0)=0.00; F(2,1)=0.00; F(2,2)=1.02;

    GPKinematics<3> gp;
    gp.F = F;
    gp.detF = F.determinant();
    gp.strain_voigt = strain::almansi(F).voigt_engineering();
    auto kin = make_constitutive_kinematics<UpdatedLagrangian>(gp);
    auto E = strain::green_lagrange(F);

    auto sigma_ref = stress::cauchy_from_2pk(nh_model.second_piola_kirchhoff(E), F);
    auto c_ref = ops::push_forward_tangent(nh_model.material_tangent(E), F);
    auto sigma_mat = mat.compute_response(kin);

    report("erased_material_spatial_stress",
        max_abs_diff(sigma_mat.components(), sigma_ref.voigt()) < 1e-10);
    report("erased_material_spatial_tangent",
        max_abs_diff_m(mat.tangent(kin), c_ref.voigt_matrix()) < 1e-10);
}

} // namespace

int main() {
    std::cout << "=== Continuum Constitutive Interface Tests ===\n";

    test_constitutive_kinematics_small_strain();
    test_constitutive_kinematics_total_lagrangian();
    test_constitutive_kinematics_updated_lagrangian();
    test_hyperelastic_relation_direct_continuum_interface();
    test_hyperelastic_relation_direct_spatial_interface();
    test_erased_continuum_handle_path();
    test_erased_material_spatial_overload();

    std::cout << "\nPassed: " << passed << "  Failed: " << failed << "\n";
    return failed == 0 ? 0 : 1;
}
