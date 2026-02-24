#ifndef FALL_N_NODAL_SECTION
#define FALL_N_NODAL_SECTION

template<std::size_t Dim> 
class NodalSection : Node<Dim>{ 
    using Node<Dim>::Node; // Inherit constructors


    public:
    static constexpr std::size_t dim = Dim;

    // Constructor
    constexpr NodalSection() = default;
    constexpr ~NodalSection() = default;
};  


#endif // FALL_N_NODAL_SECTION