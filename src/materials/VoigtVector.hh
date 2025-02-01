#ifndef VOIGTVECTOR_HH
#define VOIGTVECTOR_HH


#include <array>
#include <cstddef>


#include <Eigen/Dense>


template<std::size_t N> requires (N > 0)
class VoigtVector
{
    Eigen::Vector<double, N> components_;


  
  public: 
    
    //getter
    constexpr std::size_t num_components() const { return N; };

    constexpr double& operator[](std::size_t i) { return components_[i]; };

    constexpr Eigen::Ref<const Eigen::Vector<double, N>> vector() const { return components_; };
    


    template<typename... S> requires (sizeof...(S) == N)
    VoigtVector(S... s) : components_{s...}{}
    
    VoigtVector() = default;
    ~VoigtVector() = default;

};





#endif // VOIGTVECTOR_HH