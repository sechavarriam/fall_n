#ifndef FALL_N_VTKDATACONTAINER_HH
#define FALL_N_VTKDATACONTAINER_HH



#include "VTKheaders.hh"






class VTKDataContainer
{
    vtkNew<vtkUnstructuredGrid> vtk_grid;

    vtkNew<vtkPoints>           vtk_points; //https://vtk.org/doc/nightly/html/classvtkPoints.html
    //vtkNew<vtkCellArray>        vtk_cells;

public:
    
    void load_domain(auto& domain) const {

        vtkNew<vtkIdList> aux_cell_point_ids;
    
        vtk_points->SetNumberOfPoints(domain.num_nodes());
        vtk_grid->Allocate(domain.num_elements());

        for (std::size_t i = 0; i < domain.num_nodes(); i++){
            auto node = domain.node_p(i);

            [&]<std::size_t... Is>(std::index_sequence<Is...>){
                vtk_points->SetPoint(i, node->coord(Is)...);
            }(std::make_index_sequence<std::remove_pointer_t<decltype(node)>::dim>{});
    
            //vtk_points->SetPoint(i, node->coord(0), node->coord(1), node->coord(2));
        }    
        
        vtk_grid->SetPoints(vtk_points);

        for(auto& element : domain.elements()){

            aux_cell_point_ids->SetNumberOfIds(element.num_nodes());
            aux_cell_point_ids->SetArray(element.VTK_ordered_node_ids().data(), element.num_nodes());

            vtk_grid->InsertNextCell(element.VTK_cell_type(), aux_cell_point_ids;);

            aux_cell_point_ids->Reset();
            
        }



            //VTK cells
    }

public:
    VTKDataContainer() = default;
    ~VTKDataContainer() = default;
};




#endif // FALL_N_VTKDATACONTAINER_HH