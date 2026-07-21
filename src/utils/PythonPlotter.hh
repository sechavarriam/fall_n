// =============================================================================
//  PythonPlotter.hh — Thin C++ bridge for invoking Python postprocessing
// =============================================================================
#pragma once

#include <cstdlib>
#include <print>
#include <string>

namespace fall_n {

/// Invokes a Python plotting script from C++, forwarding data/figure paths.
class PythonPlotter {
    std::string python_exe_  = "py -3.12";
    std::string script_path_;

public:
    explicit PythonPlotter(std::string script)
        : script_path_(std::move(script)) {}

    void set_python(std::string exe) { python_exe_ = std::move(exe); }

    /// Run the script: python <script> --data <data_dir> --figures <fig_dir>
    /// Returns the shell exit code, or -1 if an argument is rejected.
    int plot(const std::string& data_dir,
             const std::string& figures_dir) const
    {
        // These paths are interpolated into a double-quoted shell command, so
        // a double quote or newline in any of them would break out of the
        // quoting and inject commands. Reject such inputs instead of running a
        // malformed command line (this bridge is not a general shell escape).
        for (const std::string* p : {&script_path_, &data_dir, &figures_dir}) {
            if (p->find('"') != std::string::npos ||
                p->find('\n') != std::string::npos) {
                std::println("  PythonPlotter: refusing unsafe path: {}", *p);
                return -1;
            }
        }
        const std::string cmd =
            python_exe_ + " \"" + script_path_ + "\""
            + " --data \""    + data_dir    + "\""
            + " --figures \"" + figures_dir  + "\"";
        std::println("  PythonPlotter: {}", cmd);
        return std::system(cmd.c_str());
    }
};

} // namespace fall_n
