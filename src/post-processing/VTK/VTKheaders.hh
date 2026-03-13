#ifndef VTKHEADERS_HH
#define VTKHEADERS_HH

// Deprecated compatibility umbrella.
//
// Historically this header pulled a broad VTK rendering stack into every
// translation unit that touched post-processing. The active exporter no longer
// depends on that API surface. Keep only the minimal data-model and IO pieces
// required by the legacy VTKDataContainer.

#include <vtkCellType.h>
#include <vtkDoubleArray.h>
#include <vtkNew.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLUnstructuredGridReader.h>
#include <vtkXMLUnstructuredGridWriter.h>

#endif // VTKHEADERS_HH
