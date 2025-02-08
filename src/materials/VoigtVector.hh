#ifndef VOIGTVECTOR_HH
#define VOIGTVECTOR_HH


#include <array>
#include <cstddef>


#include <Eigen/Dense>


template<std::size_t N> requires (N > 0)
class VoigtVector
{
    Eigen::Vector<double, N> components_ = Eigen::Vector<double, N>::Zero();

  public: 
    
    //getter
    constexpr std::size_t num_components() const { return N; };

    constexpr double& operator[](std::size_t i) { return components_[i]; };

    constexpr Eigen::Ref<const Eigen::Vector<double, N>> vector() const { return components_; };

    //constexpr void set_vector(const Eigen::Ref<const Eigen::Vector<double, N>> &v) { components_ = v; };
    
    template <typename Derived> //requires std::same_as<Derived, Eigen::Matrix<double, N, 1>>
    constexpr void set_vector(const Eigen::MatrixBase<Derived> &v) { components_ = v; };
    
    
    // Constructors

    //Copy constructor
    constexpr VoigtVector(const VoigtVector<N> &e) : components_{e.components_} {};

    //Move constructor
    constexpr VoigtVector(VoigtVector<N> &&e) : components_{std::move(e.components_)} {};

    //Copy assignment
    constexpr VoigtVector &operator=(const VoigtVector<N> &e){
        components_ = e.components_;
        return *this;
    }

    //Move assignment
    constexpr VoigtVector &operator=(VoigtVector<N> &&e){
        components_ = std::move(e.components_);
        return *this;
    }   
    
    template<typename... S> requires (sizeof...(S) == N)
    VoigtVector(S... s) : components_{s...}{}
    
    VoigtVector() = default;
    ~VoigtVector() = default;

};





#endif // VOIGTVECTOR_HH