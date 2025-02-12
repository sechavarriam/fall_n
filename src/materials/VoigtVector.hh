#ifndef VOIGTVECTOR_HH
#define VOIGTVECTOR_HH


#include <array>
#include <cstddef>


#include <Eigen/Dense>


template<std::size_t N> requires (N == 1 || N == 3 || N == 6) // Si N==1 no se requeriria Vector
class VoigtVector{
  public: 
    static constexpr std::size_t num_components{N};
    static constexpr std::size_t dim{[](){if      constexpr (N == 1) return 1;
                                          else if constexpr (N == 3) return 2;
                                          else if constexpr (N == 6) return 3;
                                          else                       return std::unreachable();  
                                          }()};
  private:
  
    using MatrixT = Eigen::Matrix<double, dim, dim>;
    using VectorT = Eigen::Vector<double, N>;

    VectorT components_ = VectorT::Zero();

  public:
    
    constexpr double& operator[](std::size_t i) { return components_[i]; };

    constexpr Eigen::Ref<const VectorT> vector() const { return components_; };

    constexpr MatrixT matrix() const {
        MatrixT m = MatrixT::Zero();
        if constexpr (dim == 1){
            return components_;
            }
        else if constexpr (dim == 2){
            m << components_[0], components_[2],
                 components_[2], components_[1];
            return m;
            }
        else if constexpr (dim == 3){ 
            m << components_[0], components_[5], components_[4],
                 components_[5], components_[1], components_[3],
                 components_[4], components_[3], components_[2];
            return m;          
         }
         };



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