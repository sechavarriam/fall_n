#ifndef FALL_N_MATERIAL_POINT_HH
#define FALL_N_MATERIAL_POINT_HH

#include <optional> 
#include <memory>

#include "../materials/Material.hh"


template<class MaterialPolicy> 
class MaterialPoint{// : public IntegrationPoint<dim>{ or Point

        static constexpr std::size_t counter{0};

    private:

        using Material       = Material<MaterialPolicy>;
        using StateVariableT = MaterialPolicy::StateVariableT;
        using StressT        = MaterialPolicy::StressType;

        static constexpr std::size_t dim = MaterialPolicy::dim;

        std::size_t id_; // Unique identifier for the material point.

        Material material_; // material instance. 
                            // It encapsulates the constitutive relation, the state variable and the update strategy.

        std::array<double, dim> coord_{0.0}; // Coordinates of the material point in the mesh.
        
        bool is_coordinated_{false}; // Flag to check if the material point has the coordinates set.
        bool is_weighted_   {false}; // Flag to check if the material point has weights set.

    public:

        inline constexpr auto id() const noexcept {return id_;};
        inline constexpr auto set_id(const std::size_t id) noexcept {id_ = id;};

        inline constexpr auto coord() const noexcept {return coord_;};
        inline constexpr auto coord(const std::size_t i) const noexcept {return coord_[i];};

        inline constexpr void set_coord(std::floating_point auto&... coord) noexcept 
        requires (sizeof...(coord) == dim){
            coord_ = {coord...};
            is_coordinated_ = true;
        }

        inline constexpr void set_coord(const std::array<double, dim>& coord) noexcept {
            coord_ = coord;
            is_coordinated_ = true;
        }

        inline constexpr void set_coord(const double* coord) noexcept {
            std::copy(coord, coord + dim, coord_.begin());
            is_coordinated_ = true;
        }


        auto& C() const {return material_.C();}; //The Compliance Matrix (Stiffness Matrix)

        MaterialPoint(Material material)   : material_{material} {};
        MaterialPoint(Material&& material) : material_{std::move(material)} {};

        MaterialPoint() = default;
        ~MaterialPoint() = default;
        
};

#endif // FALL_N_MATERIAL_POINT_HH