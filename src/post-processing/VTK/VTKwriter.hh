#ifndef VTKWRITER_HH
#define VTKWRITER_HH

#include "VTKheaders.hh"

// https://discourse.vtk.org/t/vtkhdf-roadmap/13257



















class VTKwriterHDF5
{
    std::string filename;

    vtkNew<vtkUnstructuredGrid> vtk_grid;
    vtkNew<vtkPoints>           vtk_points; //https://vtk.org/doc/nightly/html/classvtkPoints.html
    vtkNew<vtkCellArray>        vtk_cells;

public:

    void load_domain(auto& domain) const {

        vtk_points->SetNumberOfPoints(domain.num_nodes());

        for (std::size_t i = 0; i < domain.num_nodes(); i++){
            auto node = domain.node_p(i);

            [&]<std::size_t... Is>(std::index_sequence<Is...>){
                vtk_points->SetPoint(i, node->coord(Is)...);
            }(std::make_index_sequence<std::remove_pointer_t<decltype(node)>::dim>{});
   
            //vtk_points->SetPoint(i, node->coord(0), node->coord(1), node->coord(2));
        }

        vtk_grid->SetPoints(vtk_points);

        //VTK cells
    }

public:
    VTKwriterHDF5(std::string output_file) : filename{output_file}
    {
        //create file
    }
};


#endif // VTKWRITER_HH
