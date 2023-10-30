#ifndef FALL_N_MISC_CONCEPTS
#define FALL_N_MISC_CONCEPTS

#include <concepts>

namespace fall_n {

template<typename Entity>
concept Taggable = requires(Entity e){
    {e.id()} -> std::convertible_to<unsigned int> ;
    {e.id}   -> std::convertible_to<unsigned int> ;
};

} // namespace fall_n




#endif // FALL_N_MISC_CONCEPTS