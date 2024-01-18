#ifndef FALL_N_TUPLE_UTILITIES
#define FALL_N_TUPLE_UTILITIES



#include <tuple>

namespace utils {


template <class Tuple, class F> // From
       // https://www.fluentcpp.com/2021/03/05/stdindex_sequence-and-its-improvement-in-c20/

constexpr decltype(auto) for_each(Tuple &&tuple, F &&f) 
{
    return []<std::size_t... I>(Tuple &&tuple, F &&f, std::index_sequence<I...>) 
    {
        (f(std::get<I>(tuple)), ...);
        return f;
    } // End of lambda
    ( // Inmediate invocation (use std::invoke to clarify code?)
        std::forward<Tuple>(tuple), std::forward<F>(f),
        std::make_index_sequence<
        std::tuple_size<std::remove_reference_t<Tuple>>::value>{}
    );
};

// TODO: Generalize to N-puples with variadic templates. Consider diferent sizes of tuples.
template <class Tuple1, class Tuple2, class F> // TODO: requires size of tuple = size of array
constexpr decltype(auto) for_each_tuple_pair(Tuple1 &&tuple, Tuple2 &&array,
                                             F &&f) {
  return []<std::size_t... I>(Tuple1 &&tuple, Tuple2 &&array, F &&f,
                              std::index_sequence<I...>) {
    (f(std::get<I>(tuple), std::get<I>(array)), ...);
    return f;
  }(std::forward<Tuple1>(tuple), std::forward<Tuple2>(array), std::forward<F>(f),
         std::make_index_sequence<std::tuple_size<std::remove_reference_t<Tuple1>>::value>{});
}


}


#endif // FALL_N_TUPLE_UTILITIES