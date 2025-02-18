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

    vtkNew<vtkPoints>           vtk_points; // https://vtk.org/doc/nightly/html/classvtkPoints.html
    vtkNew<vtkUnstructuredGrid> vtk_grid;

    std::vector<vtkSmartPointer<vtkDoubleArray>> scalar_field;
    std::vector<vtkSmartPointer<vtkDoubleArray>> vector_field;
    std::vector<vtkSmartPointer<vtkDoubleArray>> tensor_field;


    vtkNew<vtkPoints>           vtk_gauss_points; //SEPARAR EN OTRA CLASE! un GAUSS RECORDER o algo asi
    vtkNew<vtkUnstructuredGrid> vtk_gauss_cells;

    std::vector<vtkSmartPointer<vtkDoubleArray>> gauss_scalar_field;
    std::vector<vtkSmartPointer<vtkDoubleArray>> gauss_vector_field;
    std::vector<vtkSmartPointer<vtkDoubleArray>> gauss_tensor_field;

    //bool gauss_data{false};
    
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
            vtk_field->SetTuple(i, data_array + i*N); // Afectar por un local index???            
        }

        vtk_grid->GetPointData()->AddArray(vtk_field);

        vector_field.push_back(vtk_field);
    }

    



    void load_gauss_points(auto &domain){
        //TODO: if domain hasn't gauss points seted up. setup them. DESDE AC[A SZE PODr[iA DAR ESA ORDEN...]]

        // Preallocate
        vtk_gauss_points->SetNumberOfPoints(domain.num_integration_points());
        
        for (const auto& element : domain.elements()){
            for (const auto& gauss_point : element.integration_point_){
                vtk_gauss_points->SetPoint(gauss_point.id(), gauss_point.coord(0), gauss_point.coord(1), gauss_point.coord(2));
            }
        }

        vtk_gauss_points->Modified();
        vtk_gauss_cells->Allocate(domain.num_integration_points());
        vtk_gauss_cells->SetPoints(vtk_gauss_points);

        for (auto &element : domain.elements()){
            for (auto &gauss_point : element.integration_point_){
                vtk_gauss_cells->InsertNextCell(VTK_VERTEX,1, gauss_point.id_p());
            }
        }                   

    }

    //https://vtk.org/doc/nightly/html/classvtkDataSetAttributes.html
    void load_gauss_tensor_field(std::string_view name, double* data_array, std::size_t num_points)
    {
        //check array size //https://vtk.org/doc/nightly/html/classvtkDataArray.html#a32eefb4180f5c18455995bb0315bc356
        auto vtk_tensor_field = vtkSmartPointer<vtkDoubleArray>::New();
        
        vtk_tensor_field->SetNumberOfComponents(6); // 3x3 tensor (symmetric)
        vtk_tensor_field->SetNumberOfTuples(num_points);
        vtk_tensor_field->SetName(name.data());

        for (std::size_t i = 0; i < num_points; i++){
            vtk_tensor_field->SetTuple(i, data_array+i*6);             
        }

        vtk_gauss_cells->GetPointData()->AddArray(vtk_tensor_field);

        gauss_tensor_field.push_back(vtk_tensor_field);
    }
                   

    void load_domain(auto &domain) const {
        
        vtk_points->SetNumberOfPoints(domain.num_nodes());
        
        for (const auto& node : domain.nodes()){
            //vtk_points->InsertPoint(static_cast<vtkIdType>(node.id()), node.coord(0), node.coord(1), node.coord(2));
            vtk_points->SetPoint(static_cast<vtkIdType>(node.id()), node.coord(0), node.coord(1), node.coord(2));
        }

        vtk_points->Modified();

        vtk_grid->Allocate(domain.num_elements());
        vtk_grid->SetPoints(vtk_points);

        for (auto &element : domain.elements()){
            auto ids = element.VTK_ordered_node_ids().data();
            vtk_grid->InsertNextCell(element.VTK_cell_type(), static_cast<vtkIdType>(element.num_nodes()), ids); 
        }
    }

    void write_gauss_vtu(std::string file_name) const
    {
        //if (!gauss_data) throw std::runtime_error("Gauss points not loaded.");
        vtkNew<vtkXMLUnstructuredGridWriter> writer;

        writer->SetFileName(file_name.c_str());
        writer->SetInputData(vtk_gauss_cells);
        writer->SetDataModeToAscii(); // Remove to keep binary 
        writer->Update();             // Remove to keep binary
        writer->Write(); 
    }


    void write_vtu(std::string file_name) const
    {
        //writer->SetFileName(file_name.data());
        writer->SetFileName(file_name.c_str());
        writer->SetInputData(vtk_grid);
        writer->SetDataModeToAscii(); // Remove to keep binary 
        writer->Update();             // Remove to keep binary
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