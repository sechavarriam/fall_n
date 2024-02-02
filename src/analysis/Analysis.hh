#ifndef FALL_N_ANALYSIS_INTERFACE
#define FALL_N_ANALYSIS_INTERFACE

template<typename T> // Concept? CRTP? Policy?
class Analysis {

    // Mesh
    // Model
    // Solver
    // Domain
    // Loads
    // Boundary conditions

protected:
    void solve() = 0;

public:

    Analysis() = default;
    ~Analysis() = default;
    
};



#endif // FALL_N_ANALYSIS_INTERFACE