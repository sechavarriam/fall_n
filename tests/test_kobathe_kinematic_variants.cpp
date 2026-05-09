#include <cassert>
#include <iostream>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>

#include <Eigen/Dense>
#include <petsc.h>

#include "src/analysis/MultiscaleCoordinator.hh"
#include "src/reconstruction/LocalModelAdapter.hh"
#include "src/validation/SeismicFE2LocalModelVariant.hh"

using namespace fall_n;

namespace {

int g_pass = 0;
int g_fail = 0;

void check(bool condition, const std::string& message)
{
    if (condition) {
        ++g_pass;
        std::cout << "  [PASS] " << message << "\n";
    } else {
        ++g_fail;
        std::cout << "  [FAIL] " << message << "\n";
    }
}

[[nodiscard]] ElementKinematics make_kinematics()
{
    ElementKinematics ek;
    ek.element_id = 6;
    ek.endpoint_A = {0.0, 0.0, 0.0};
    ek.endpoint_B = {3.2, 0.0, 0.0};
    ek.up_direction = {0.0, 1.0, 0.0};

    const Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    ek.kin_A.centroid = Eigen::Vector3d{0.0, 0.0, 0.0};
    ek.kin_A.R = R;
    ek.kin_A.u_local = Eigen::Vector3d::Zero();
    ek.kin_A.theta_local = Eigen::Vector3d::Zero();
    ek.kin_A.E = 25000.0;
    ek.kin_A.G = 10000.0;
    ek.kin_A.nu = 0.2;

    ek.kin_B.centroid = Eigen::Vector3d{3.2, 0.0, 0.0};
    ek.kin_B.R = R;
    ek.kin_B.u_local = Eigen::Vector3d{1.0e-5, 2.0e-4, -1.0e-4};
    ek.kin_B.theta_local = Eigen::Vector3d{0.0, 1.0e-4, 2.0e-4};
    ek.kin_B.E = 25000.0;
    ek.kin_B.G = 10000.0;
    ek.kin_B.nu = 0.2;
    return ek;
}

[[nodiscard]] MultiscaleCoordinator make_single_site_coordinator()
{
    MultiscaleCoordinator coordinator;
    coordinator.add_critical_element(make_kinematics());
    coordinator.build_sub_models(SubModelSpec{
        .section_width = 0.50,
        .section_height = 0.50,
        .nx = 1,
        .ny = 1,
        .nz = 2,
        .hex_order = HexOrder::Quadratic,
    });
    return coordinator;
}

template <typename EvolverT,
          SeismicFE2ContinuumKinematics ExpectedKinematics>
void test_variant(std::string_view label)
{
    static_assert(LocalModelAdapter<EvolverT>);

    auto coordinator = make_single_site_coordinator();
    EvolverT evolver{
        coordinator.sub_models()[0],
        30.0,
        ".",
        9999};
    SeismicFE2LocalModel local{
        std::move(evolver),
        SeismicFE2LocalFamily::continuum_kobathe_hex27};

    check(local.family() == SeismicFE2LocalFamily::continuum_kobathe_hex27,
          std::string(label) + " keeps continuum family");
    check(local.continuum_kinematics() == ExpectedKinematics,
          std::string(label) + " records continuum kinematics");

    const auto checkpoint = local.capture_checkpoint();
    check(std::holds_alternative<typename EvolverT::checkpoint_type>(
              checkpoint.local),
          std::string(label) + " stores typed checkpoint");
}

} // namespace

int main(int argc, char** argv)
{
    PetscInitialize(&argc, &argv, nullptr, nullptr);

    std::cout << "Ko-Bathe continuum kinematic variant smoke\n";

    check(to_string(SeismicFE2ContinuumKinematics::small_strain) == "small",
          "small-strain CLI label");
    check(to_string(SeismicFE2ContinuumKinematics::total_lagrangian) == "tl",
          "TL CLI label");
    check(to_string(SeismicFE2ContinuumKinematics::updated_lagrangian) == "ul",
          "UL CLI label");
    check(to_string(SeismicFE2ContinuumKinematics::corotational) ==
              "corotational",
          "corotational CLI label");

    test_variant<NonlinearSubModelEvolver,
                 SeismicFE2ContinuumKinematics::small_strain>(
        "SmallStrain");
    test_variant<TotalLagrangianNonlinearSubModelEvolver,
                 SeismicFE2ContinuumKinematics::total_lagrangian>(
        "TotalLagrangian");
    test_variant<UpdatedLagrangianNonlinearSubModelEvolver,
                 SeismicFE2ContinuumKinematics::updated_lagrangian>(
        "UpdatedLagrangian");
    test_variant<CorotationalNonlinearSubModelEvolver,
                 SeismicFE2ContinuumKinematics::corotational>(
        "Corotational");

    std::cout << "Summary: " << g_pass << " passed, "
              << g_fail << " failed\n";

    PetscFinalize();
    return g_fail == 0 ? 0 : 1;
}
