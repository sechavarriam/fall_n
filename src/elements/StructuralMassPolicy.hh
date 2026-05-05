#ifndef FALL_N_STRUCTURAL_MASS_POLICY_HH
#define FALL_N_STRUCTURAL_MASS_POLICY_HH

#include <string_view>

namespace fall_n {

/// Mass representation used by structural elements in dynamic analyses.
///
/// The policy is deliberately separated from the element kinematics: the same
/// Timoshenko interpolation may be audited with a consistent mass, a classical
/// row-sum lumping, or a positive nodal lumping.  This is essential when
/// comparing against external solvers whose default inertial carriers are
/// often nodal rather than high-order element-consistent.
enum class StructuralMassPolicy {
    consistent,
    row_sum_lumped,
    positive_nodal_lumped,
};

[[nodiscard]] inline constexpr std::string_view
to_string(StructuralMassPolicy policy) noexcept
{
    switch (policy) {
    case StructuralMassPolicy::consistent:
        return "consistent";
    case StructuralMassPolicy::row_sum_lumped:
        return "row_sum_lumped";
    case StructuralMassPolicy::positive_nodal_lumped:
        return "positive_nodal_lumped";
    }
    return "unknown";
}

[[nodiscard]] inline constexpr bool
is_lumped(StructuralMassPolicy policy) noexcept
{
    return policy == StructuralMassPolicy::row_sum_lumped ||
           policy == StructuralMassPolicy::positive_nodal_lumped;
}

} // namespace fall_n

#endif // FALL_N_STRUCTURAL_MASS_POLICY_HH
