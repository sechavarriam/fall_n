#ifndef STRUCTURAL_MODEL_BUILDER_HH
#define STRUCTURAL_MODEL_BUILDER_HH

// ═══════════════════════════════════════════════════════════════════════
//  StructuralModelBuilder — physical-group → element material mapping
// ═══════════════════════════════════════════════════════════════════════
//
//  Maps physical group names (set on ElementGeometry during domain
//  construction) to typed structural elements (frame or shell) with
//  their assigned materials.
//
//  Usage:
//    auto elements = fall_n::StructuralModelBuilder{}
//        .set_frame_material("Columns", col_mat)
//        .set_frame_material("Beams",   beam_mat)
//        .set_shell_material("Slabs",   slab_mat)
//        .build_elements(domain);
//
//    StructuralModel model{domain, std::move(elements)};
//
//  Design notes:
//  - Default element types: Corotational Timoshenko beams (3D) and
//    Mindlin-Reissner shells.  Override via template parameters.
//  - Fluent (method-chaining) API with string-based group names.
//  - No global state; all configuration is explicit.
//
// ═══════════════════════════════════════════════════════════════════════

#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "../domain/Domain.hh"
#include "../elements/BeamElement.hh"
#include "../elements/ElementPolicy.hh"
#include "../elements/ShellElement.hh"
#include "../elements/StructuralElement.hh"
#include "../materials/Material.hh"

namespace fall_n {

template <
    typename FrameElemT   = BeamElement<TimoshenkoBeam3D, 3, beam::Corotational>,
    typename ShellElemT   = ShellElement<MindlinReissnerShell3D>,
    typename FrameMatPol  = TimoshenkoBeam3D,
    typename ShellMatPol  = MindlinReissnerShell3D>
class StructuralModelBuilder {
public:
    using frame_material_type = Material<FrameMatPol>;
    using shell_material_type = Material<ShellMatPol>;
    using policy_type         = SingleElementPolicy<StructuralElement>;
    using container_type      = typename policy_type::container_type;

    // ── Fluent material assignment ──────────────────────────────────

    StructuralModelBuilder& set_frame_material(
        std::string_view group,
        const frame_material_type& mat)
    {
        frame_materials_.insert_or_assign(std::string(group), mat);
        return *this;
    }

    StructuralModelBuilder& set_shell_material(
        std::string_view group,
        const shell_material_type& mat)
    {
        shell_materials_.insert_or_assign(std::string(group), mat);
        return *this;
    }

    // ── Build element container from an assembled domain ────────────
    //
    //  Iterates all domain elements, matches physical_group() against
    //  frame and shell material maps, and creates typed elements.
    //  Throws if a group is not found in either map.
    //
    //  The optional shell_geometries output collects pointers to shell
    //  element geometries (useful for surface-load or VTK queries).
    //
    container_type build_elements(
        Domain<3>& domain,
        std::vector<const ElementGeometry<3>*>* shell_geometries = nullptr) const
    {
        container_type elements;
        elements.reserve(domain.num_elements());

        for (auto& geom : domain.elements()) {
            const auto& group = geom.physical_group();

            if (auto it = frame_materials_.find(group);
                it != frame_materials_.end())
            {
                elements.emplace_back(FrameElemT{&geom, it->second});
            }
            else if (auto jt = shell_materials_.find(group);
                     jt != shell_materials_.end())
            {
                elements.emplace_back(ShellElemT{&geom, jt->second});
                if (shell_geometries)
                    shell_geometries->push_back(&geom);
            }
            else {
                throw std::runtime_error(
                    "StructuralModelBuilder: no material registered for "
                    "physical group '" + group + "'.");
            }
        }

        return elements;
    }

private:
    // Transparent comparator (std::less<>) enables string_view lookups.
    std::map<std::string, frame_material_type, std::less<>> frame_materials_;
    std::map<std::string, shell_material_type, std::less<>> shell_materials_;
};

} // namespace fall_n

#endif // STRUCTURAL_MODEL_BUILDER_HH
