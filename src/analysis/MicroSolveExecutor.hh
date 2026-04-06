#ifndef FALL_N_SRC_ANALYSIS_MICRO_SOLVE_EXECUTOR_HH
#define FALL_N_SRC_ANALYSIS_MICRO_SOLVE_EXECUTOR_HH

#include <cstddef>

#include <mpi.h>

namespace fall_n {

struct CouplingCommunicators {
    MPI_Comm macro_comm{MPI_COMM_WORLD};
    MPI_Comm micro_comm{MPI_COMM_WORLD};
    bool     shared_memory_execution{true};
};

class SerialExecutor {
public:
    template <typename Fn>
    void for_each(std::size_t count, Fn&& fn) const
    {
        for (std::size_t i = 0; i < count; ++i)
            fn(i);
    }
};

class OpenMPExecutor {
    int num_threads_{0};

public:
    explicit OpenMPExecutor(int num_threads = 0) : num_threads_{num_threads} {}

    [[nodiscard]] int num_threads() const noexcept { return num_threads_; }

    template <typename Fn>
    void for_each(std::size_t count, Fn&& fn) const
    {
#ifdef _OPENMP
        if (num_threads_ > 0) {
#pragma omp parallel for schedule(static) num_threads(num_threads_)
            for (int i = 0; i < static_cast<int>(count); ++i)
                fn(static_cast<std::size_t>(i));
            return;
        }

#pragma omp parallel for schedule(static)
        for (int i = 0; i < static_cast<int>(count); ++i)
            fn(static_cast<std::size_t>(i));
#else
        for (std::size_t i = 0; i < count; ++i)
            fn(i);
#endif
    }
};

}  // namespace fall_n

#endif // FALL_N_SRC_ANALYSIS_MICRO_SOLVE_EXECUTOR_HH
