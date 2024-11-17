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

            // print nodes id
            std::cout << "Element nodes: -------------------------------------------" << std::endl;
            for (std::size_t i = 0; i < element.num_nodes(); i++){
                std::cout << element.node(i) << " ";
            }
            std::cout << std::endl;

            // print VTK ordered nodes id
            for (std::size_t i = 0; i < element.VTK_ordered_node_ids().size(); i++){
                std::cout << element.VTK_ordered_node_ids()[i] << " ";
            }
            std::cout << std::endl;
            std::cout << "-------------------------------------------" << std::endl;

            //vtk_grid->InsertNextCell(element.VTK_cell_type(), /*vtkNodesMapID*/)
        }



            //VTK cells
    }

public:
    VTKDataContainer() = default;
    ~VTKDataContainer() = default;
};




#endif // FALL_N_VTKDATACONTAINER_HH