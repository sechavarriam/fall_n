// Placeholder for the historical fall_n entry point.
//
// The CMakeLists.txt still wires `add_executable(fall_n main.cpp)` for the
// legacy umbrella target, but the production drivers now live in the dedicated
// main_*.cpp files (one per benchmark). This stub keeps the legacy target
// linkable until the CMakeLists is refactored.
int main() { return 0; }
