#ifndef FALL_N_MESH_INTERFACE
#define FALL_N_MESH_INTERFACE

#include <cstddef>
#include <string_view>
#include <petsc.h>

//template <std::size_t dim>
class Mesh { // Thin DMPlex wrapper. The DM is created lazily so a geometry-only
             // Domain can exist without immediately requiring PetscInitialize().

public:

    DM dm{nullptr}; // DMPLEX (created lazily)

    PetscInt sieve_size{0}; // Number of points in the sieve

    void ensure_created() {
        if (dm != nullptr) return;
        DMCreate(PETSC_COMM_WORLD, &dm);
        DMSetType(dm, DMPLEX);
        DMSetBasicAdjacency(dm, PETSC_FALSE, PETSC_TRUE);
    }

    void set_size(PetscInt N){ // Number of DAG points = nodes + edges + faces + cells
        ensure_created();
        DMPlexSetChart(dm, 0, N); // Set the interval for all mesh points [pStart, pEnd) {0, 1, 2, ..., N-1}
    }

    void set_dimension(PetscInt dim) {
        ensure_created();
        DMSetDimension(dm, dim);
    }

    void setup(PetscInt n){
        ensure_created();
        sieve_size = n;
        DMSetUp(dm);
        } // Setup the mesh   

    void set_sieve_cone_size(PetscInt point_idx, PetscInt N){
        ensure_created();
        DMPlexSetConeSize(dm, point_idx, N); // Set the number of points in the cone of a point
    }

    void set_sieve_cone(PetscInt point_idx, PetscInt *cone_idx){ 
        ensure_created();
        DMPlexSetCone(dm, point_idx, cone_idx); // Set the cone of a point
    }

    void symmetrize_sieve(){
        ensure_created();
        DMPlexSymmetrize(dm); // Symmetrize the sieve
        DMPlexStratify(dm); // Stratify the mesh
    }

    DMLabel ensure_label(std::string_view name) {
        ensure_created();

        DMLabel label{nullptr};
        DMGetLabel(dm, name.data(), &label);
        if (label == nullptr) {
            DMCreateLabel(dm, name.data());
            DMGetLabel(dm, name.data(), &label);
        }
        return label;
    }

    void set_label_value(std::string_view label_name, PetscInt point, PetscInt value) {
        auto label = ensure_label(label_name);
        DMLabelSetValue(label, point, value);
    }

    Mesh() = default;

    // Move constructor: transfer DM ownership, nullify source.
    Mesh(Mesh&& other) noexcept
        : dm(other.dm), sieve_size(other.sieve_size) {
        other.dm = nullptr;
        other.sieve_size = 0;
    }

    // Move assignment: destroy own DM, then take over source's.
    Mesh& operator=(Mesh&& other) noexcept {
        if (this != &other) {
            if (dm != nullptr) DMDestroy(&dm);
            dm = other.dm;
            sieve_size = other.sieve_size;
            other.dm = nullptr;
            other.sieve_size = 0;
        }
        return *this;
    }

    // Copying a DM handle is unsafe (double-destroy).
    Mesh(const Mesh&) = delete;
    Mesh& operator=(const Mesh&) = delete;

    ~Mesh(){ if (dm != nullptr) DMDestroy(&dm); }


};


#endif // FALL_N_MESH_INTERFACE
