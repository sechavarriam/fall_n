#ifndef VTKWRITER_HH
#define VTKWRITER_HH

#include "VTKheaders.hh"

#include "VTKdataContainer.hh"
// https://discourse.vtk.org/t/vtkhdf-roadmap/13257



class VTKwriterHDF5
{
    std::string filename;
    VTKDataContainer* data;

public:

    void bind_data(VTKDataContainer* data){
        this->data = data;
    }

public:
    VTKwriterHDF5(std::string output_file) : filename{output_file}
    {
        //create file
    }
};


#endif // VTKWRITER_HH
