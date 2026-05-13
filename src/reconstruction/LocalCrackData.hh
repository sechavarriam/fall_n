#ifndef FALL_N_SRC_RECONSTRUCTION_LOCAL_CRACK_DATA_HH
#define FALL_N_SRC_RECONSTRUCTION_LOCAL_CRACK_DATA_HH

#include <limits>
#include <vector>

#include <Eigen/Dense>

namespace fall_n {

struct CrackRecord {
    Eigen::Vector3d position;
    Eigen::Vector3d displacement{Eigen::Vector3d::Zero()};
    int             plane_id{0};
    int             sequence_id{0};
    int             activation_step{0};
    double          activation_time{0.0};
    int             source_id{0};
    int             num_cracks{0};
    Eigen::Vector3d normal_1{Eigen::Vector3d::Zero()};
    Eigen::Vector3d normal_2{Eigen::Vector3d::Zero()};
    Eigen::Vector3d normal_3{Eigen::Vector3d::Zero()};
    double          opening_1{0};
    double          opening_2{0};
    double          opening_3{0};
    double          opening_max_1{0};
    double          opening_max_2{0};
    double          opening_max_3{0};
    bool            closed_1{true};
    bool            closed_2{true};
    bool            closed_3{true};
    double          damage{0};
    bool            damage_scalar_available{false};
    bool            fracture_history_available{false};
    double          sigma_o_max{0.0};
    double          tau_o_max{0.0};
};

struct CrackSummary {
    int    num_cracked_gps{0};
    int    total_cracks{0};
    bool   damage_scalar_available{false};
    double max_damage_scalar{std::numeric_limits<double>::quiet_NaN()};
    bool   fracture_history_available{false};
    double most_compressive_sigma_o_max{0.0};
    double max_tau_o_max{0.0};
    double max_opening{0.0};
    double max_historical_opening{0.0};
};

struct LocalCrackState {
    std::vector<CrackRecord> cracks{};
    CrackSummary summary{};
};

} // namespace fall_n

#endif // FALL_N_SRC_RECONSTRUCTION_LOCAL_CRACK_DATA_HH
