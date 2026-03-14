#ifndef FN_INTEGRATION_STRATEGY_HH
#define FN_INTEGRATION_STRATEGY_HH

// Compatibility header.
//
// The semantic home of the local constitutive algorithm is now
// `materials/ConstitutiveIntegrator.hh`.  The old name is kept because the
// existing codebase still includes `update_strategy/IntegrationStrategy.hh`.

#include "../ConstitutiveIntegrator.hh"

template <typename Strategy, typename MaterialType>
concept IntegrationStrategyConcept =
    ConstitutiveIntegratorConcept<Strategy, MaterialType>;

#endif // FN_INTEGRATION_STRATEGY_HH
