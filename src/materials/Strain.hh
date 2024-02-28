#ifndef FN_ABSTRACT_STRAIN
#define FN_ABSTRACT_STRAIN




template<typename StrainType>
concept Strain = requires(StrainType e){
    {e.tensor};
    {e.get_strain()};
};


template<typename StrainPolicy>
class CauchyStrain{
    private:
        StrainPolicy strain{};
    public:

        
    CauchyStrain(){};
    ~CauchyStrain(){};
};

template<std::size_t N> requires (N > 0)
class VoigtStrain{
    private:
        std::array<double, N> component_;
    
    public:
        static constexpr std::size_t num_components = N;

        Vector tensor{component_};

        constexpr std::span<const double, N> get_strain() const{return component_;};
        constexpr std::floating_point auto get_stain(std::size_t i) const{return component_[i];};

        template<typename... S> requires (sizeof...(S) == N)
        constexpr void set_strain(S... s){
            std::size_t i{0};
            ((component_[++i]=s), ...);
        };

        template<typename... S> requires (sizeof...(S) == N)
        VoigtStrain(S... s) : component_{s...}{};

        VoigtStrain() = default;
        ~VoigtStrain() = default;
};

//Specialization for 1D stress (Uniaxial Stress) avoiding array overhead
template<>
class VoigtStrain<1>{
    private:
        double component_{0.0};
    
    public:
        static constexpr std::size_t num_components = 1;

        double tensor{component_};

        constexpr void set_strain(double e){component_ = e;};

        constexpr std::floating_point auto get_strain() const{return component_;};

        VoigtStrain() = default;
        ~VoigtStrain() = default;
};



/*
class UniaxialStrain{
    private:
        double component_{0.0};

    public:

        double tensor{component_};

        constexpr void set_strain(double s){component_ = s;};

    UniaxialStrain(){};
    ~UniaxialStrain(){};
};

class PlaneStrain{

    private:
        std::array<double, 3> component_{0.0, 0.0, 0.0};
    public:

        Vector tensor{component_}; 

        constexpr std::span<const double, 3> get_strain() const{return component_;};
        constexpr std::floating_point auto   get_strain(std::size_t i) const{return component_[i];};

        constexpr void set_strain(double e11, double e22, double e12){
            component_[0] = e11; component_[1] = e22; // Normal stresses
            component_[2] = e12;                      // Shear Strain
        };

    PlaneStrain(double e11, double e22, double e12) : component_{e11, e22, e12}{};

    PlaneStrain(){};
    ~PlaneStrain(){};
};


class ContinuumStrain{
    private:
        std::array<double, 6> component_{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    public:

        Vector tensor{component_};

        constexpr std::span<const double, 6> get_strain() const{return component_;};
        constexpr std::floating_point auto   get_strain(std::size_t i) const{return component_[i];};

        constexpr void set_strain(double e11, double e22, double e33, double e12, double e23, double e13){
            component_[0] = e11; component_[1] = e22; component_[2] = e33; // Normal stresses
            component_[3] = e12; component_[4] = e23; component_[5] = e13; // Shear  stresses
        };

    ContinuumStrain(double e11, double e22, double e33, double e12, double e23, double e13) : component_{e11, e22, e33, e12, e23, e13}{};

    ContinuumStrain(){};
    ~ContinuumStrain(){};
};

*/








#endif