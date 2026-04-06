#include <atomic>
#include <iostream>
#include <vector>

#include "src/analysis/MicroSolveExecutor.hh"

namespace {

using namespace fall_n;

int g_pass = 0;
int g_fail = 0;

#define CHECK_TRUE(cond, msg)                                                   \
    do {                                                                        \
        if (cond) {                                                             \
            std::cout << "  [PASS] " << msg << "\n";                            \
            ++g_pass;                                                           \
        } else {                                                                \
            std::cout << "  [FAIL] " << msg << "\n";                            \
            ++g_fail;                                                           \
        }                                                                       \
    } while (0)

template <typename ExecutorT>
std::vector<int> visit_counts_with(const ExecutorT& executor, std::size_t n)
{
    std::vector<std::atomic<int>> counts(n);
    for (auto& count : counts) {
        count.store(0);
    }

    executor.for_each(n, [&](std::size_t i) {
        counts.at(i).fetch_add(1, std::memory_order_relaxed);
    });

    std::vector<int> out(n, 0);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = counts[i].load(std::memory_order_relaxed);
    }
    return out;
}

void test_serial_executor_visits_each_index_once()
{
    const auto counts = visit_counts_with(SerialExecutor{}, 32);
    bool all_once = true;
    for (const int count : counts) {
        all_once = all_once && (count == 1);
    }

    CHECK_TRUE(all_once, "SerialExecutor visits each micro-problem exactly once");
}

void test_openmp_executor_matches_serial_schedule()
{
    const auto serial_counts = visit_counts_with(SerialExecutor{}, 128);
    const auto omp_counts = visit_counts_with(OpenMPExecutor{4}, 128);

    CHECK_TRUE(omp_counts == serial_counts,
               "OpenMPExecutor matches SerialExecutor visit counts");
    CHECK_TRUE(OpenMPExecutor{4}.num_threads() == 4,
               "OpenMPExecutor preserves the configured thread count");
}

void test_coupling_communicators_have_explicit_defaults()
{
    CouplingCommunicators comms;
    CHECK_TRUE(comms.macro_comm == MPI_COMM_WORLD,
               "CouplingCommunicators defaults macro communicator to MPI_COMM_WORLD");
    CHECK_TRUE(comms.micro_comm == MPI_COMM_WORLD,
               "CouplingCommunicators defaults micro communicator to MPI_COMM_WORLD");
    CHECK_TRUE(comms.shared_memory_execution,
               "CouplingCommunicators defaults to shared-memory execution");
}

} // namespace

int main()
{
    std::cout << "Running micro solve executor tests...\n";

    test_serial_executor_visits_each_index_once();
    test_openmp_executor_matches_serial_schedule();
    test_coupling_communicators_have_explicit_defaults();

    std::cout << "\nSummary: " << g_pass << " passed, "
              << g_fail << " failed.\n";
    return g_fail == 0 ? 0 : 1;
}
