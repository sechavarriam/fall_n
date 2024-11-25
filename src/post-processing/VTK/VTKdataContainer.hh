#ifndef FALL_N_VTKDATACONTAINER_HH
#define FALL_N_VTKDATACONTAINER_HH

#include "VTKheaders.hh"

#include <vtkXMLUnstructuredGridWriter.h>
#include <vtkXMLUnstructuredGridReader.h>

class VTKDataContainer
{
    vtkNew<vtkUnstructuredGrid> vtk_grid;

    vtkNew<vtkPoints> vtk_points; // https://vtk.org/doc/nightly/html/classvtkPoints.html
    // vtkNew<vtkCellArray>        vtk_cells;

public:
    void load_domain(auto &domain) const
    {
        vtk_points->SetNumberOfPoints(domain.num_nodes());
        vtk_grid->Allocate(domain.num_elements());

        for (std::size_t i = 0; i < domain.num_nodes(); i++)
        {
            auto node = domain.node_p(i);

            //[&]<std::size_t... Is>(std::index_sequence<Is...>){
            //    vtk_points->SetPoint(i, node->coord(Is)...);
            //}(std::make_index_sequence<std::remove_pointer_t<decltype(node)>::dim>{});

            vtk_points->SetPoint(node->id(), node->coord(0), node->coord(1), node->coord(2));
        }
        vtk_grid->SetPoints(vtk_points);

        for (auto &element : domain.elements())
        {
            auto ids = element.VTK_ordered_node_ids().data();
            vtk_grid->InsertNextCell(element.VTK_cell_type(), static_cast<vtkIdType>(element.num_nodes()), ids);
        }


        vtkNew<vtkXMLUnstructuredGridWriter> writer;
        writer->SetFileName("/home/sechavarriam/MyLibs/fall_n/data/output/structure.vtu");
        writer->SetInputData(vtk_grid);
        
        //writer->SetDataModeToAscii();
        //writer->Update();
        
        writer->Write(); 

        /*
        vtkNew<vtkXMLUnstructuredGridReader> reader;
        reader->SetFileName("/home/sechavarriam/MyLibs/fall_n/data/output/structure.vtu");
        reader->Update();

        vtkNew<vtkDataSetMapper> mapper;
        mapper->SetInputConnection(reader->GetOutputPort());

        vtkNew<vtkActor> actor;
        actor->SetMapper(mapper);
        actor->GetProperty()->SetColor(1.0, 0.0, 0.0);

        vtkNew<vtkRenderer> renderer;
        vtkNew<vtkRenderWindow> renderWindow;
        renderWindow->AddRenderer(renderer);
        renderWindow->SetWindowName("WriteVTU");

        vtkNew<vtkRenderWindowInteractor> renderWindowInteractor;
        renderWindowInteractor->SetRenderWindow(renderWindow);

        renderer->AddActor(actor);
        renderer->SetBackground(0.0, 0.0, 0.0);

        renderWindow->Render();
        renderWindowInteractor->Start();
    */

    }

public:
    VTKDataContainer() = default;
    ~VTKDataContainer() = default;
};

#endif // FALL_N_VTKDATACONTAINER_HH