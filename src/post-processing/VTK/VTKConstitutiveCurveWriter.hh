#ifndef FALL_N_VTK_CONSTITUTIVE_CURVE_WRITER_HH
#define FALL_N_VTK_CONSTITUTIVE_CURVE_WRITER_HH

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <vtkCellArray.h>
#include <vtkDoubleArray.h>
#include <vtkNew.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <vtkTable.h>
#include <vtkXMLPolyDataWriter.h>
#include <vtkXMLTableWriter.h>

#include "../../materials/ConstitutiveProtocol.hh"

namespace fall_n::vtk {

struct ConstitutiveCurveSeries {
    std::string name;
    std::string abscissa_name{"strain"};
    std::string ordinate_name{"stress"};
    std::vector<ConstitutiveCurveSample> samples;
};

namespace detail {

inline vtkSmartPointer<vtkDoubleArray> make_curve_series_array(std::string_view name) {
    auto arr = vtkSmartPointer<vtkDoubleArray>::New();
    arr->SetName(std::string(name).c_str());
    arr->SetNumberOfComponents(1);
    return arr;
}

inline std::string sanitize_curve_series_name(std::string_view name) {
    std::string out;
    out.reserve(name.size());
    for (const char ch : name) {
        if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z')
            || (ch >= '0' && ch <= '9') || ch == '_' || ch == '-') {
            out.push_back(ch);
        } else {
            out.push_back('_');
        }
    }
    return out;
}

inline std::string escape_curve_xml(std::string_view value) {
    std::string out;
    out.reserve(value.size());
    for (const char ch : value) {
        switch (ch) {
        case '&': out += "&amp;"; break;
        case '"': out += "&quot;"; break;
        case '\'': out += "&apos;"; break;
        case '<': out += "&lt;"; break;
        case '>': out += "&gt;"; break;
        default: out.push_back(ch); break;
        }
    }
    return out;
}

inline std::string escape_python_string(std::string_view value) {
    std::string out;
    out.reserve(value.size());
    for (const char ch : value) {
        switch (ch) {
        case '\\': out += "\\\\"; break;
        case '"': out += "\\\""; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default: out.push_back(ch); break;
        }
    }
    return out;
}

inline vtkSmartPointer<vtkPolyData> build_curve_polydata(
    const ConstitutiveCurveSeries& series)
{
    auto poly = vtkSmartPointer<vtkPolyData>::New();
    auto points = vtkSmartPointer<vtkPoints>::New();
    auto lines = vtkSmartPointer<vtkCellArray>::New();

    auto abscissa = make_curve_series_array(series.abscissa_name);
    auto ordinate = make_curve_series_array(series.ordinate_name);
    auto step = make_curve_series_array("step");
    auto path_parameter = make_curve_series_array("path_parameter");

    for (const auto& sample : series.samples) {
        points->InsertNextPoint(sample.abscissa, sample.ordinate, 0.0);
        abscissa->InsertNextValue(sample.abscissa);
        ordinate->InsertNextValue(sample.ordinate);
        step->InsertNextValue(static_cast<double>(sample.step));
        path_parameter->InsertNextValue(sample.path_parameter);
    }

    if (!series.samples.empty()) {
        lines->InsertNextCell(static_cast<vtkIdType>(series.samples.size()));
        for (vtkIdType i = 0; i < static_cast<vtkIdType>(series.samples.size()); ++i) {
            lines->InsertCellPoint(i);
        }
    }

    poly->SetPoints(points);
    poly->SetLines(lines);
    poly->GetPointData()->AddArray(abscissa);
    poly->GetPointData()->AddArray(ordinate);
    poly->GetPointData()->AddArray(step);
    poly->GetPointData()->AddArray(path_parameter);
    poly->GetPointData()->SetActiveScalars(series.ordinate_name.c_str());
    return poly;
}

inline vtkSmartPointer<vtkTable> build_curve_table(
    const ConstitutiveCurveSeries& series)
{
    auto table = vtkSmartPointer<vtkTable>::New();
    auto abscissa = make_curve_series_array(series.abscissa_name);
    auto ordinate = make_curve_series_array(series.ordinate_name);
    auto step = make_curve_series_array("step");
    auto path_parameter = make_curve_series_array("path_parameter");

    for (const auto& sample : series.samples) {
        abscissa->InsertNextValue(sample.abscissa);
        ordinate->InsertNextValue(sample.ordinate);
        step->InsertNextValue(static_cast<double>(sample.step));
        path_parameter->InsertNextValue(sample.path_parameter);
    }

    table->AddColumn(abscissa);
    table->AddColumn(ordinate);
    table->AddColumn(step);
    table->AddColumn(path_parameter);
    return table;
}

inline void write_curve_multiblock_index(
    const std::string& filename,
    std::span<const std::pair<std::string, vtkSmartPointer<vtkPolyData>>> blocks)
{
    namespace fs = std::filesystem;

    const fs::path vtm_path{filename};
    if (!vtm_path.parent_path().empty()) {
        fs::create_directories(vtm_path.parent_path());
    }

    const std::string stem = vtm_path.stem().string();

    for (const auto& [name, poly] : blocks) {
        const fs::path sidecar =
            vtm_path.parent_path() / (stem + "_" + sanitize_curve_series_name(name) + ".vtp");
        vtkNew<vtkXMLPolyDataWriter> writer;
        writer->SetFileName(sidecar.string().c_str());
        writer->SetInputData(poly);
        writer->SetDataModeToAscii();
        writer->Write();
    }

    std::ofstream vtm(filename);
    if (!vtm) {
        throw std::runtime_error(
            "VTKConstitutiveCurveWriter: failed to open '" + filename + "' for writing.");
    }

    vtm << "<?xml version=\"1.0\"?>\n";
    vtm << "<VTKFile type=\"vtkMultiBlockDataSet\" version=\"1.0\" "
           "byte_order=\"LittleEndian\" header_type=\"UInt64\">\n";
    vtm << "  <vtkMultiBlockDataSet>\n";
    for (std::size_t i = 0; i < blocks.size(); ++i) {
        const auto& [name, _poly] = blocks[i];
        const std::string sidecar = stem + "_" + sanitize_curve_series_name(name) + ".vtp";
        vtm << "    <DataSet index=\"" << i << "\" name=\""
            << escape_curve_xml(name) << "\" file=\"" << escape_curve_xml(sidecar) << "\"/>\n";
    }
    vtm << "  </vtkMultiBlockDataSet>\n";
    vtm << "</VTKFile>\n";
}

inline void write_curve_csv(
    const ConstitutiveCurveSeries& series,
    const std::filesystem::path& filename)
{
    if (!filename.parent_path().empty()) {
        std::filesystem::create_directories(filename.parent_path());
    }

    std::ofstream out(filename);
    if (!out) {
        throw std::runtime_error(
            "VTKConstitutiveCurveWriter: failed to open '" + filename.string() + "' for writing.");
    }

    out << series.abscissa_name << ","
        << series.ordinate_name << ",step,path_parameter\n";
    for (const auto& sample : series.samples) {
        out << sample.abscissa << ","
            << sample.ordinate << ","
            << sample.step << ","
            << sample.path_parameter << "\n";
    }
}

inline void write_curve_manifest_csv(
    std::span<const ConstitutiveCurveSeries> series,
    const std::filesystem::path& filename)
{
    if (!filename.parent_path().empty()) {
        std::filesystem::create_directories(filename.parent_path());
    }

    std::ofstream out(filename);
    if (!out) {
        throw std::runtime_error(
            "VTKConstitutiveCurveWriter: failed to open '" + filename.string() + "' for writing.");
    }

    out << "name,file,abscissa,ordinate,num_samples\n";
    const auto stem = filename.stem().string();
    for (const auto& entry : series) {
        const auto sidecar =
            stem + "_" + sanitize_curve_series_name(entry.name) + ".vtt";
        out << entry.name << ","
            << sidecar << ","
            << entry.abscissa_name << ","
            << entry.ordinate_name << ","
            << entry.samples.size() << "\n";
    }
}

inline void write_paraview_chart_script(
    std::span<const ConstitutiveCurveSeries> series,
    const std::filesystem::path& filename,
    std::string_view stem)
{
    if (!filename.parent_path().empty()) {
        std::filesystem::create_directories(filename.parent_path());
    }

    std::ofstream out(filename);
    if (!out) {
        throw std::runtime_error(
            "VTKConstitutiveCurveWriter: failed to open '" + filename.string() + "' for writing.");
    }

    out << "# Auto-generated ParaView helper for constitutive curves.\n";
    out << "# Run from ParaView with Tools -> Python Shell -> Run Script,\n";
    out << "# or with pvpython " << filename.filename().string() << "\n";
    out << "from pathlib import Path\n";
    out << "from paraview.simple import *\n\n";
    out << "base_dir = Path(__file__).resolve().parent\n";
    out << "curves = [\n";
    for (const auto& entry : series) {
        const auto sidecar =
            std::string(stem) + "_" + sanitize_curve_series_name(entry.name) + ".vtt";
        out << "    {\n";
        out << "        \"name\": \"" << escape_python_string(entry.name) << "\",\n";
        out << "        \"file\": \"" << escape_python_string(sidecar) << "\",\n";
        out << "        \"x\": \"" << escape_python_string(entry.abscissa_name) << "\",\n";
        out << "        \"y\": \"" << escape_python_string(entry.ordinate_name) << "\",\n";
        out << "    },\n";
    }
    out << "]\n\n";
    out << "layout = GetLayout()\n";
    out << "if layout is None:\n";
    out << "    layout = CreateLayout(name='Constitutive Curves')\n\n";
    out << "view = CreateView('XYChartView')\n";
    out << "AssignViewToLayout(view=view, layout=layout, hint=0)\n";
    out << "SetActiveView(view)\n";
    out << "view.ViewSize = [1200, 800]\n\n";
    out << "for curve in curves:\n";
    out << "    reader = OpenDataFile(str(base_dir / curve['file']))\n";
    out << "    RenameSource(curve['name'], reader)\n";
    out << "    display = Show(reader, view, 'XYChartRepresentation')\n";
    out << "    if hasattr(display, 'UseIndexForXAxis'):\n";
    out << "        display.UseIndexForXAxis = 0\n";
    out << "    if hasattr(display, 'XArrayName'):\n";
    out << "        display.XArrayName = curve['x']\n";
    out << "    if hasattr(display, 'SeriesVisibility'):\n";
    out << "        display.SeriesVisibility = [curve['y']]\n";
    out << "    if hasattr(display, 'SeriesLabel'):\n";
    out << "        display.SeriesLabel = [curve['y'], curve['name']]\n\n";
    out << "RenderAllViews()\n";
}

} // namespace detail

class VTKConstitutiveCurveWriter {
public:
    void write_curve(const ConstitutiveCurveSeries& series,
                     const std::string& filename) const
    {
        namespace fs = std::filesystem;
        const fs::path path{filename};
        if (!path.parent_path().empty()) {
            fs::create_directories(path.parent_path());
        }
        const auto ext = path.extension().string();

        if (ext == ".vtp") {
            auto poly = detail::build_curve_polydata(series);
            vtkNew<vtkXMLPolyDataWriter> writer;
            writer->SetFileName(filename.c_str());
            writer->SetInputData(poly);
            writer->SetDataModeToAscii();
            writer->Write();
            return;
        }

        if (ext == ".vtt") {
            auto table = detail::build_curve_table(series);
            vtkNew<vtkXMLTableWriter> writer;
            writer->SetFileName(filename.c_str());
            writer->SetInputData(table);
            writer->SetDataModeToAscii();
            writer->Write();
            return;
        }

        if (ext == ".csv") {
            detail::write_curve_csv(series, path);
            return;
        }

        throw std::runtime_error(
            "VTKConstitutiveCurveWriter: unsupported curve extension '" + ext +
            "'. Use .vtt, .vtp or .csv.");
    }

    void write_multiblock(
        const std::vector<ConstitutiveCurveSeries>& series,
        const std::string& filename) const
    {
        // Auxiliary geometric export: useful when the protocol itself should be
        // rendered as a polyline in 3D space. For constitutive calibration and
        // hysteresis plots, prefer write_table_bundle(), which produces chart-
        // ready vtkTable outputs for ParaView.
        std::vector<std::pair<std::string, vtkSmartPointer<vtkPolyData>>> blocks;
        blocks.reserve(series.size());
        for (const auto& entry : series) {
            blocks.emplace_back(entry.name, detail::build_curve_polydata(entry));
        }
        detail::write_curve_multiblock_index(filename, blocks);
    }

    // Preferred export for constitutive protocols: one vtkTable (.vtt) per
    // series plus a lightweight CSV manifest and a ParaView Python helper.
    // vtkTable is the right data carrier for a 2D chart, while the companion
    // script reconstructs an XYChartView directly inside ParaView.
    void write_table_bundle(
        const std::vector<ConstitutiveCurveSeries>& series,
        const std::string& filename_prefix) const
    {
        namespace fs = std::filesystem;
        fs::path prefix{filename_prefix};
        const fs::path directory = prefix.parent_path();
        const std::string stem =
            prefix.has_extension() ? prefix.stem().string() : prefix.filename().string();

        if (!directory.empty()) {
            fs::create_directories(directory);
        }

        for (const auto& entry : series) {
            const fs::path sidecar =
                directory / (stem + "_" + detail::sanitize_curve_series_name(entry.name) + ".vtt");
            write_curve(entry, sidecar.string());
        }

        detail::write_curve_manifest_csv(
            series,
            directory / (stem + "_manifest.csv"));
        detail::write_paraview_chart_script(
            series,
            directory / (stem + "_paraview_chart.py"),
            stem);
    }
};

} // namespace fall_n::vtk

#endif // FALL_N_VTK_CONSTITUTIVE_CURVE_WRITER_HH
