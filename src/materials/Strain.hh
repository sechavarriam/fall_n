#ifndef FN_ABSTRACT_STRAIN
#define FN_ABSTRACT_STRAIN


#include <concepts>
#include <ranges>

template<typename StrainType>
concept Strain = requires(StrainType e){
    {e.num_components}->std::convertible_to<std::size_t>;
    {e.tensor};
    {e.get_strain()};
};

template<typename T>
concept StrainRange = std::ranges::range<T> && requires(T s){
    {s.begin().num_components}->std::convertible_to<std::size_t>;
    {s.begin().tensor};
    {s.begin().get_strain()};
};

// https://stackoverflow.com/questions/64228197/range-concept-for-a-specific-type
template <typename R, typename V>
concept RangeOf = std::ranges::range<R> && std::same_as<std::ranges::range_value_t<R>, V>;



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
        }

        template<typename... S> requires (sizeof...(S) == N)
        VoigtStrain(S... s) : component_{s...}{}

        VoigtStrain() = default;
        ~VoigtStrain() = default;
};

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






#endif