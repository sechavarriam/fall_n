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
    int plot(const std::string& data_dir,
             const std::string& figures_dir) const
    {
        const std::string cmd =
            python_exe_ + " \"" + script_path_ + "\""
            + " --data \""    + data_dir    + "\""
            + " --figures \"" + figures_dir  + "\"";
        std::println("  PythonPlotter: {}", cmd);
        return std::system(cmd.c_str());
    }
};

} // namespace fall_n
