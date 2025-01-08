#ifndef FALL_N_VTKDATACONTAINER_HH
#define FALL_N_VTKDATACONTAINER_HH



#include "VTKheaders.hh"

#include <vtkDoubleArray.h>
#include <vtkUnstructuredGrid.h>
#include <vtkPoints.h>
#include <vtkPointData.h>

#include <vtkXMLUnstructuredGridWriter.h>
#include <vtkXMLUnstructuredGridReader.h>

class VTKDataContainer
{

    vtkNew<vtkXMLUnstructuredGridWriter> writer;

    vtkNew<vtkPoints> vtk_points; // https://vtk.org/doc/nightly/html/classvtkPoints.html
    vtkNew<vtkUnstructuredGrid> vtk_grid;

    std::vector<vtkSmartPointer<vtkDoubleArray>> scalar_field;
    std::vector<vtkSmartPointer<vtkDoubleArray>> vector_field;
    std::vector<vtkSmartPointer<vtkDoubleArray>> tensor_field;
    
//https://stackoverflow.com/questions/38937139/how-to-store-a-vector-field-with-vtk-c-vtkwriter


// vtkPointData requires arrays that have a tuple for each point
// vtkCellData requires arrays that have a tuple for each cell
// vtkFieldData has no constraints on the number of tuples (but also cannot be used directly to color geometry, etc. since there is no context that specifies how the arrays map to any geometry).
// Just put your one big array in the field data (inherited from vtkDataObject) and your offset array in the cell-data.

public:

    template<std::size_t N>
    void load_vector_field(std::string_view name, double* data_array, std::size_t num_points)
    {
        //check array size

        //https://vtk.org/doc/nightly/html/classvtkDataArray.html#a32eefb4180f5c18455995bb0315bc356

        auto vtk_field = vtkSmartPointer<vtkDoubleArray>::New();

        vtk_field->SetNumberOfComponents(N);
        vtk_field->SetNumberOfTuples(num_points);
        vtk_field->SetName(name.data());

        for (std::size_t i = 0; i < num_points; i++){
            vtk_field->SetTuple(i, data_array + i*N);            
        }

        vtk_grid->GetPointData()->AddArray(vtk_field);

        vector_field.push_back(vtk_field);
    }



    void load_domain(auto &domain) const
    {
        //vtk_points->Initialize();
        //vtk_points->Reset();
        
        //vtk_points->SetDataTypeToDouble();  //Esto es segmentation fault inmediato. Por que? (Solo da segfault si se usa SetPoint en ves de InsertNextPoint)
        //vtk_points->SetNumberOfPoints(domain.num_nodes());
        
        for (const auto& node : domain.nodes())
        {
            //std::cout << "Node: " << node.id() << std::endl;
            //std::cout <<  node.coord(0) << " " << node.coord(1) << " " << node.coord(2) << std::endl;

            //vtk_points->SetPoint(static_cast<vtkIdType>(node.id()), node.coord(0), node.coord(1), node.coord(2));
            //vtk_points->InsertNextPoint(node.coord(0), node.coord(1), node.coord(2)); // Esto resuelve el segfault. Pero es mas lento... Por ahora usar.
                                                                                      // Desconfigura la numeración de los nodos en el VTK.
            vtk_points->InsertPoint(static_cast<vtkIdType>(node.id()), node.coord(0), node.coord(1), node.coord(2));
        }

        vtk_points->Modified();


        vtk_grid->Allocate(domain.num_elements());
        vtk_grid->SetPoints(vtk_points);

        for (auto &element : domain.elements()){
            auto ids = element.VTK_ordered_node_ids().data();
            vtk_grid->InsertNextCell(element.VTK_cell_type(), static_cast<vtkIdType>(element.num_nodes()), ids); 
        }
        //write_vtu("/home/sechavarriam/MyLibs/fall_n/data/output/structure.vtu");
    }

    void write_vtu(std::string file_name) const
    {
        //writer->SetFileName(file_name.data());
        writer->SetFileName(file_name.c_str());
        writer->SetInputData(vtk_grid);
        writer->SetDataModeToAscii(); // Remove tyo keep binary 
        writer->Update();             // Remove tyo keep binary
        writer->Write(); 


        //vtkNew<vtkXMLUnstructuredGridReader> reader;
        //reader->SetFileName("/home/sechavarriam/MyLibs/fall_n/data/output/structure.vtu");
        //reader->Update();
        //
        //vtkNew<vtkDataSetMapper> mapper;
        //mapper->SetInputConnection(reader->GetOutputPort());
//
        //vtkNew<vtkActor> actor;
        //actor->SetMapper(mapper);
        //actor->GetProperty()->SetColor(1.0, 0.0, 0.0);
//
        //vtkNew<vtkRenderer> renderer;
        //vtkNew<vtkRenderWindow> renderWindow;
        //renderWindow->AddRenderer(renderer);
        //renderWindow->SetWindowName("WriteVTU");
//
        //vtkNew<vtkRenderWindowInteractor> renderWindowInteractor;
        //renderWindowInteractor->SetRenderWindow(renderWindow);
//
        //renderer->AddActor(actor);
        //renderer->SetBackground(0.0, 0.0, 0.0);
//
        //renderWindow->Render();
        //renderWindowInteractor->Start();
    }

public:
    VTKDataContainer() = default;
    ~VTKDataContainer() = default;
};

#endif // FALL_N_VTKDATACONTAINER_HH