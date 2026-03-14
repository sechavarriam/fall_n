#include <cassert>
#include <iostream>

#include <petsc.h>

#include "header_files.hh"

namespace {

using BeamElem = BeamElement<TimoshenkoBeam3D, 3>;
using ShellElem = ShellElement<MindlinReissnerShell3D>;
using StructuralPolicy = SingleElementPolicy<StructuralElement>;
using StructuralModel =
    Model<TimoshenkoBeam3D, continuum::SmallStrain, 6, StructuralPolicy>;

Domain<3> make_mixed_domain() {
    Domain<3> domain;
    domain.add_node(0, 0.0, 0.0, 0.0);
    domain.add_node(1, 5.0, 0.0, 0.0);
    domain.add_node(2, 0.0, 5.0, 0.0);
    domain.add_node(3, 5.0, 5.0, 0.0);

    {
        PetscInt conn[2] = {0, 1};
        auto& beam = domain.make_element<LagrangeElement3D<2>>(
            GaussLegendreCellIntegrator<2>{}, 0, conn);
        beam.set_physical_group("beam");
    }

    {
        PetscInt conn[4] = {0, 1, 2, 3};
        auto& shell = domain.make_element<LagrangeElement3D<2, 2>>(
            GaussLegendreCellIntegrator<2, 2>{}, 1, conn);
        shell.set_physical_group("shell");
    }

    domain.assemble_sieve();
    return domain;
}

void test_dmplex_labels_capture_mixed_topology() {
    auto domain = make_mixed_domain();

    PetscInt plex_dim = -1;
    DMGetDimension(domain.mesh.dm, &plex_dim);
    assert(plex_dim == 2);
    assert(domain.plex_dimension() == 2);

    PetscInt cStart = -1, cEnd = -1;
    PetscInt vStart = -1, vEnd = -1;
    DMPlexGetHeightStratum(domain.mesh.dm, 0, &cStart, &cEnd);
    DMPlexGetDepthStratum(domain.mesh.dm, 0, &vStart, &vEnd);

    assert((cEnd - cStart) == 2);
    assert((vEnd - vStart) == 4);

    DMLabel point_role{nullptr};
    DMLabel topo_dim{nullptr};
    DMLabel phys_group{nullptr};

    DMGetLabel(domain.mesh.dm, Domain<3>::point_role_label_name().data(), &point_role);
    DMGetLabel(domain.mesh.dm, Domain<3>::topological_dimension_label_name().data(), &topo_dim);
    DMGetLabel(domain.mesh.dm, Domain<3>::physical_group_label_name().data(), &phys_group);

    assert(point_role != nullptr);
    assert(topo_dim != nullptr);
    assert(phys_group != nullptr);

    PetscInt value = -1;
    DMLabelGetValue(point_role, domain.element(0).sieve_id(), &value);
    assert(value == 1);
    DMLabelGetValue(point_role, domain.node(0).sieve_id.value(), &value);
    assert(value == 0);

    DMLabelGetValue(topo_dim, domain.element(0).sieve_id(), &value);
    assert(value == 1);
    DMLabelGetValue(topo_dim, domain.element(1).sieve_id(), &value);
    assert(value == 2);
    DMLabelGetValue(topo_dim, domain.node(0).sieve_id.value(), &value);
    assert(value == 0);

    const auto beam_group = domain.physical_group_id("beam");
    const auto shell_group = domain.physical_group_id("shell");
    assert(beam_group.has_value());
    assert(shell_group.has_value());
    assert(beam_group != shell_group);

    DMLabelGetValue(phys_group, domain.element(0).sieve_id(), &value);
    assert(value == *beam_group);
    DMLabelGetValue(phys_group, domain.element(1).sieve_id(), &value);
    assert(value == *shell_group);
}

void test_structural_model_layout_uses_depth_zero_vertices() {
    auto domain = make_mixed_domain();

    TimoshenkoBeamMaterial3D beam_relation{
        21000.0, 8000.0, 0.40 * 0.40,
        0.40 * 0.40 * 0.40 * 0.40 / 12.0,
        0.40 * 0.40 * 0.40 * 0.40 / 12.0,
        0.02, 5.0 / 6.0, 5.0 / 6.0
    };
    Material<TimoshenkoBeam3D> beam_material{beam_relation, ElasticUpdate{}};

    MindlinShellMaterial shell_relation{26000.0, 0.20, 0.18};
    Material<MindlinReissnerShell3D> shell_material{shell_relation, ElasticUpdate{}};

    StructuralPolicy::container_type elements;
    elements.emplace_back(BeamElem{std::addressof(domain.element(0)), beam_material});
    elements.emplace_back(ShellElem{std::addressof(domain.element(1)), shell_material});

    StructuralModel model(domain, std::move(elements));
    model.setup();

    PetscSection local_section{nullptr};
    DMGetLocalSection(domain.mesh.dm, &local_section);
    assert(local_section != nullptr);

    PetscInt cStart = -1, cEnd = -1;
    PetscInt vStart = -1, vEnd = -1;
    DMPlexGetHeightStratum(domain.mesh.dm, 0, &cStart, &cEnd);
    DMPlexGetDepthStratum(domain.mesh.dm, 0, &vStart, &vEnd);
    assert((cEnd - cStart) == 2);
    assert((vEnd - vStart) == 4);

    for (const auto& node : domain.nodes()) {
        PetscInt ndof = -1;
        PetscSectionGetDof(local_section, node.sieve_id.value(), &ndof);
        assert(ndof == 6);
    }

    for (std::size_t e = 0; e < domain.num_elements(); ++e) {
        PetscInt ndof = -1;
        PetscSectionGetDof(local_section, domain.element(e).sieve_id(), &ndof);
        assert(ndof == 0);
    }
}

} // namespace

int main(int argc, char** argv) {
    PetscInitialize(&argc, &argv, nullptr, nullptr);

    test_dmplex_labels_capture_mixed_topology();
    test_structural_model_layout_uses_depth_zero_vertices();

    PetscFinalize();

    std::cout << "dmplex_mixed_topology: PASS\n";
    return 0;
}
