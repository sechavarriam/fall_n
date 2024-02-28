
#ifndef FALL_N_MATERIAL_POLICY_HH
#define FALL_N_MATERIAL_POLICY_HH




template <typename MaterialPolicy>
class MaterialContainer {
public:
    // Constructor
    MaterialContainer() {
        // Initialize the material container based on the material policy
        MaterialPolicy::Initialize();
    }

    // Other member functions and data members...

private:
    // Data members specific to the material container
};

// Example material policies

// Uniaxial material policy
struct UniaxialMaterialPolicy {
    static void Initialize() {
        // Initialize the uniaxial material container
    }

    
    double stress() const {
        // Return the stress
    }

    // Other member functions specific to uniaxial material
};

// Continuum material policy
struct ContinuumMaterialPolicy {
    static void Initialize() {
        // Initialize the continuum material container
    }

    // Other member functions specific to continuum material
};

// Other material policies...


#endif // FALL_N_MATERIAL_POLICY_HH