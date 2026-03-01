#ifndef VOIGTVECTOR_HH
#define VOIGTVECTOR_HH

#include <array>
#include <cstddef>

#include <Eigen/Dense>

template<std::size_t N> requires (N == 1 || N == 3 || N == 6) 
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
    using VectorT = Eigen::Matrix<double, N, 1>;

    VectorT components_ = VectorT::Zero();

  public:
    
    constexpr double& operator[](std::size_t i) { return components_[i]; }
    constexpr double  operator[](std::size_t i) const { return components_[i]; }

    // Accessors are only valid on lvalues. This prevents dangling references
    // when called on temporaries (e.g. current_state() returning by value).
    Eigen::Ref<const VectorT> components() const & { return components_; }
    //Eigen::Ref<const VectorT> components() const && = delete;

    const double* data() const & { return components_.data(); }
    const double* data() const && = delete;

    constexpr MatrixT matrix() const {
        MatrixT m = MatrixT::Zero();
        if constexpr (dim == 1){
            m(0,0) = components_[0];
            return m;
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

    template <typename Derived> //requires std::same_as<Derived, Eigen::Matrix<double, N, 1>>

    // =======================================================================/
    // To deprecate ======================================================== //
    //vector()                                                             ////
    Eigen::Ref<const VectorT> vector() const & { return components_; }     ////
    Eigen::Ref<const VectorT> vector() const && = delete;                  ////
    // ===================================================================== //
    // =======================================================================/

    template <typename Derived> //requires std::same_as<Derived, Eigen::Matrix<double, N, 1>>
    constexpr void set_components(const Eigen::MatrixBase<Derived> &v) { components_ = v; }
    
    constexpr VoigtVector(const VoigtVector<N>  &e) : components_{          e.components_ } {}; //Copy constructor
    constexpr VoigtVector(      VoigtVector<N> &&e) : components_{std::move(e.components_)} {}; //Move constructor

    constexpr VoigtVector &operator=(const VoigtVector<N> &e){ //Copy assignment
        components_ = e.components_;
        return *this;
    }

    constexpr VoigtVector &operator=(VoigtVector<N> &&e){ //Move assignment
        components_ = std::move(e.components_);
        return *this;
    }   
    
    template<typename... S> requires (sizeof...(S) == N)
    VoigtVector(S... s) : components_{s...}{}
    
    VoigtVector()  = default;
    ~VoigtVector() = default;
};

// Specialization for N=1, where the Voigt vector is just a scalar
template<>
class VoigtVector<1>{
  public:
    static constexpr std::size_t num_components{1};
    static constexpr std::size_t dim{1};
  
  private:
    double component_ = 0.0;

  public:
    constexpr double& operator[](std::size_t i [[maybe_unused]]) { return component_; }; // No se necesita índice, pero se proporciona para mantener la interfaz consistente
    constexpr double  components() const { return component_; };
    const double* data() const & { return &component_; }
    const double* data() const && = delete;
    constexpr void set_components(double v) { component_ = v; };


    // Constructors    
    constexpr VoigtVector(const VoigtVector<1>  &e) : component_{          e.component_ } {}; //Copy constructor  
    constexpr VoigtVector(      VoigtVector<1> &&e) : component_{std::move(e.component_)} {}; //Move constructor
    
    constexpr VoigtVector &operator=(const VoigtVector<1> &e){ //Copy assignment
        component_ = e.component_;
        return *this;
    }
    
    constexpr VoigtVector &operator=(VoigtVector<1> &&e){ //Move assignment
        component_ = std::move(e.component_);
        return *this;
    }

    VoigtVector(double s) : component_{s} {};
    VoigtVector()  = default;
    ~VoigtVector() = default;
};

#endif // VOIGTVECTOR_HH