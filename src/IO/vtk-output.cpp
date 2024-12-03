/**
 * @file vtk_outputs.cpp
 * @author Miguel Molinos ([migmolper](https://github.com/migmolper))
 * @brief
 * @version 0.1
 * @date 2022-06-22
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifdef USE_VTK

#include <cstdlib>
#include <math.h>
#include <iostream> //std::cout//std::cin
#include <fstream>
#include <sstream> //std::istringstream
#include <cstdio>
#include <string> //std::string
#include <cstring>
#include <vector>
#include <vtkCellArray.h>
#include <vtkNew.h>
#include <vtkCellData.h>
#include <vtkDoubleArray.h>
#include <vtkIntArray.h>
#include <vtkLine.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <vtkTensorGlyph.h>
#include <vtkVersion.h>
#include <vtkVertexGlyphFilter.h>
#include <vtkXMLPolyDataWriter.h>
#include <vtkZLibDataCompressor.h>
#include <Eigen/Dense>
#include "Atoms/Atom.hpp"
#include "Macros.hpp"
#include "ADP/MgHx-V.hpp"


using namespace std;
extern char OutputFolder[MAXC];

/********************************************************************************/

int write_vtk(int step, atom* atoms, adpPotential* adp[2][2]) {

  unsigned int dim = 3;
  int n_atoms = atoms->n_atoms;

  /* File name */
  char Name_file_t[10000];
  snprintf(Name_file_t, sizeof(Name_file_t), "%s/atomic_sites_%i.vtp",
           OutputFolder, step);

  auto points = vtkSmartPointer<vtkPoints>::New();

  auto Type = vtkSmartPointer<vtkIntArray>::New();
  Type->SetNumberOfComponents(1);
  Type->SetName("Type");

  auto atomic_number = vtkSmartPointer<vtkDoubleArray>::New();
  atomic_number->SetNumberOfComponents(1);
  atomic_number->SetName("Atomic number");

  auto stddedv_q = vtkSmartPointer<vtkDoubleArray>::New();
  stddedv_q->SetNumberOfComponents(1);
  stddedv_q->SetName("Standard-desviation-q");

  auto xi = vtkSmartPointer<vtkDoubleArray>::New();
  xi->SetNumberOfComponents(1);
  xi->SetName("Molar fraction");

  auto mass = vtkSmartPointer<vtkDoubleArray>::New();
  mass->SetNumberOfComponents(1);
  mass->SetName("Mass");

  for (unsigned int i_site = 0; i_site < n_atoms; ++i_site) {
    points->InsertNextPoint(atoms->mean_q[i_site * dim + 0],
                            atoms->mean_q[i_site * dim + 1],
                            atoms->mean_q[i_site * dim + 2]);
    Type->InsertNextValue(atoms->specie[i_site]);
    if (atoms->specie[i_site] == 0) {
      atomic_number->InsertNextValue(12);
    } else if (atoms->specie[i_site] == 1) {
      atomic_number->InsertNextValue(1);
    }
    int spc_i = atoms->specie[i_site];
    mass->InsertNextValue(adp[spc_i][spc_i]->mass);
    xi->InsertNextValue(atoms->xi[i_site]);
    stddedv_q->InsertNextValue(atoms->stdv_q[i_site]);
  }

  // Create a polydata object and add the points to it.
  vtkSmartPointer<vtkPolyData> polydata =
    vtkSmartPointer<vtkPolyData>::New();
  polydata->SetPoints(points);
  polydata->GetPointData()->AddArray(Type);
  polydata->GetPointData()->AddArray(stddedv_q);
  polydata->GetPointData()->AddArray(atomic_number);
  polydata->GetPointData()->AddArray(xi);
  polydata->GetPointData()->AddArray(mass);

  // Write the file
  vtkSmartPointer<vtkXMLPolyDataWriter> writer = vtkSmartPointer<vtkXMLPolyDataWriter>::New();
  writer->SetFileName(Name_file_t);

#if VTK_MAJOR_VERSION <= 5
  writer->SetInput(polydata);
#else
  writer->SetInputData(polydata);
#endif

  // Optional - set the mode. The default is binary.
  writer->SetDataModeToBinary();
  // writer->SetDataModeToAscii();
  writer->SetCompressor(vtkZLibDataCompressor::New());
  writer->Write();

  return EXIT_SUCCESS;
}

/********************************************************************************/

int write_neighborhood_vtk(int step, atom* atoms, adpPotential* adp[2][2],
                           unsigned int i_site, const int* neigh_i,
                           unsigned int numneigh_site_i) {
  unsigned int dim = 3;

  /* File name */
  char Name_file_t[10000];
  snprintf(Name_file_t, sizeof(Name_file_t), "%s/neighbours_%i.vtp",
           OutputFolder, step);

  auto points = vtkSmartPointer<vtkPoints>::New();

  auto Type = vtkSmartPointer<vtkIntArray>::New();
  Type->SetNumberOfComponents(1);
  Type->SetName("Type");

  auto atomic_number = vtkSmartPointer<vtkDoubleArray>::New();
  atomic_number->SetNumberOfComponents(1);
  atomic_number->SetName("Atomic number");

  auto frequency = vtkSmartPointer<vtkDoubleArray>::New();
  frequency->SetNumberOfComponents(1);
  frequency->SetName("Frequency");

  auto occupancy = vtkSmartPointer<vtkDoubleArray>::New();
  occupancy->SetNumberOfComponents(1);
  occupancy->SetName("Occupancy");

  auto chemical_potential = vtkSmartPointer<vtkDoubleArray>::New();
  chemical_potential->SetNumberOfComponents(1);
  chemical_potential->SetName("Chemical potential");

  points->InsertNextPoint(atoms->mean_q[i_site * dim + 0],
                          atoms->mean_q[i_site * dim + 1],
                          atoms->mean_q[i_site * dim + 2]);
  Type->InsertNextValue(atoms->specie[i_site]);
  if (atoms->specie[i_site] == 0) {
    atomic_number->InsertNextValue(12);
  } else if (atoms->specie[i_site] == 1) {
    atomic_number->InsertNextValue(1);
  }

  int spc_i = atoms->specie[i_site];
  double beta_i = atoms->beta[i_site];
  double m_i = adp[spc_i][spc_i]->mass;
  frequency->InsertNextValue(unit_change_w /
                             (atoms->stdv_q[i_site] * sqrt(beta_i * m_i)));
  occupancy->InsertNextValue(atoms->occupancy[i_site]);
  chemical_potential->InsertNextValue(atoms->gamma[i_site]);

  for (unsigned int idx_neigh = 0; idx_neigh < numneigh_site_i; ++idx_neigh) {

    unsigned int j_site = neigh_i[idx_neigh];

    points->InsertNextPoint(atoms->mean_q[j_site * dim + 0],
                            atoms->mean_q[j_site * dim + 1],
                            atoms->mean_q[j_site * dim + 2]);
    Type->InsertNextValue(atoms->specie[j_site]);
    if (atoms->specie[j_site] == 0) {
      atomic_number->InsertNextValue(12);
    } else if (atoms->specie[j_site] == 1) {
      atomic_number->InsertNextValue(1);
    }
    int spc_i = atoms->specie[i_site];
    double beta_i = atoms->beta[i_site];
    double m_i = adp[spc_i][spc_i]->mass;
    frequency->InsertNextValue(
        unit_change_w / (atoms->stdv_q[i_site] * sqrt(beta_i * m_i)));
    occupancy->InsertNextValue(atoms->occupancy[j_site]);
    chemical_potential->InsertNextValue(atoms->gamma[j_site]);
  }

  // Create a polydata object and add the points to it.
  vtkSmartPointer<vtkPolyData> polydata =
    vtkSmartPointer<vtkPolyData>::New();
  polydata->SetPoints(points);
  polydata->GetPointData()->AddArray(Type);
  polydata->GetPointData()->AddArray(frequency);
  polydata->GetPointData()->AddArray(atomic_number);
  polydata->GetPointData()->AddArray(occupancy);
  polydata->GetPointData()->AddArray(chemical_potential);

  // Write the file
  vtkSmartPointer<vtkXMLPolyDataWriter> writer = vtkSmartPointer<vtkXMLPolyDataWriter>::New();
  writer->SetFileName(Name_file_t);

#if VTK_MAJOR_VERSION <= 5
  writer->SetInput(polydata);
#else
  writer->SetInputData(polydata);
#endif

  // Optional - set the mode. The default is binary.
  writer->SetDataModeToBinary();
  // writer->SetDataModeToAscii();
  writer->SetCompressor(vtkZLibDataCompressor::New());
  writer->Write();

  return EXIT_SUCCESS;
}

/********************************************************************************/

int write_potential_vtk(int step, atom* atoms, adpPotential* adp[2][2],
                        double* potential) {
  unsigned int dim = 3;
  int n_atoms = atoms->n_atoms;

  /* File name */
  char Name_file_t[10000];
  snprintf(Name_file_t, sizeof(Name_file_t), "%s/potential_%i.vtp",
           OutputFolder, step);

  auto points = vtkSmartPointer<vtkPoints>::New();

  auto Type = vtkSmartPointer<vtkIntArray>::New();
  Type->SetNumberOfComponents(1);
  Type->SetName("Type");

  auto atomic_number = vtkSmartPointer<vtkDoubleArray>::New();
  atomic_number->SetNumberOfComponents(1);
  atomic_number->SetName("Atomic number");

  auto frequency = vtkSmartPointer<vtkDoubleArray>::New();
  frequency->SetNumberOfComponents(1);
  frequency->SetName("Frequency");

  auto occupancy = vtkSmartPointer<vtkDoubleArray>::New();
  occupancy->SetNumberOfComponents(1);
  occupancy->SetName("Occupancy");

  auto chemical_potential = vtkSmartPointer<vtkDoubleArray>::New();
  chemical_potential->SetNumberOfComponents(1);
  chemical_potential->SetName("Chemical potential");

  auto potential_value = vtkSmartPointer<vtkDoubleArray>::New();
  potential_value->SetNumberOfComponents(1);
  potential_value->SetName("Potential");

  for (unsigned int i_site = 0; i_site < n_atoms; i_site++) {
    points->InsertNextPoint(atoms->mean_q[i_site * dim + 0],
                            atoms->mean_q[i_site * dim + 1],
                            atoms->mean_q[i_site * dim + 2]);
    Type->InsertNextValue(atoms->specie[i_site]);
    if (atoms->specie[i_site] == 0) {
      atomic_number->InsertNextValue(12);
    } else if (atoms->specie[i_site] == 1) {
      atomic_number->InsertNextValue(1);
    }
    int spc_i = atoms->specie[i_site];
    double beta_i = atoms->beta[i_site];
    double m_i = adp[spc_i][spc_i]->mass;
    frequency->InsertNextValue(unit_change_w /
                               (atoms->stdv_q[i_site] * sqrt(beta_i * m_i)));
    occupancy->InsertNextValue(atoms->occupancy[i_site]);
    chemical_potential->InsertNextValue(atoms->gamma[i_site]);
    potential_value->InsertNextValue(potential[i_site]);
  }

  // Create a polydata object and add the points to it.
  vtkSmartPointer<vtkPolyData> polydata =
    vtkSmartPointer<vtkPolyData>::New();
  polydata->SetPoints(points);
  polydata->GetPointData()->AddArray(Type);
  polydata->GetPointData()->AddArray(frequency);
  polydata->GetPointData()->AddArray(atomic_number);
  polydata->GetPointData()->AddArray(occupancy);
  polydata->GetPointData()->AddArray(chemical_potential);
  polydata->GetPointData()->AddArray(potential_value);

  // Write the file
  vtkSmartPointer<vtkXMLPolyDataWriter> writer = vtkSmartPointer<vtkXMLPolyDataWriter>::New();
  writer->SetFileName(Name_file_t);

#if VTK_MAJOR_VERSION <= 5
  writer->SetInput(polydata);
#else
  writer->SetInputData(polydata);
#endif

  // Optional - set the mode. The default is binary.
  writer->SetDataModeToBinary();
  // writer->SetDataModeToAscii();
  writer->SetCompressor(vtkZLibDataCompressor::New());
  writer->Write();

  return EXIT_SUCCESS;
}

/********************************************************************************/

int write_forces_vtk(int step, atom* atoms, const Eigen::MatrixXd& mean_forces,
                     const Eigen::VectorXd& stdv_forces) {
  unsigned int dim = 3;
  int n_atoms = atoms->n_atoms;

  /* File name */
  char Name_file_t[10000];
  snprintf(Name_file_t, sizeof(Name_file_t), "%s/forces_%i.vtp", OutputFolder,
           step);

  auto points = vtkSmartPointer<vtkPoints>::New();

  auto Type = vtkSmartPointer<vtkIntArray>::New();
  Type->SetNumberOfComponents(1);
  Type->SetName("Type");

  auto atomic_number = vtkSmartPointer<vtkDoubleArray>::New();
  atomic_number->SetNumberOfComponents(1);
  atomic_number->SetName("Atomic number");

  auto stddedv_q = vtkSmartPointer<vtkDoubleArray>::New();
  stddedv_q->SetNumberOfComponents(1);
  stddedv_q->SetName("q-Stdv");

  auto xi = vtkSmartPointer<vtkDoubleArray>::New();
  xi->SetNumberOfComponents(1);
  xi->SetName("Molar fraction");

  auto stdv_forces_vtk = vtkSmartPointer<vtkDoubleArray>::New();
  stdv_forces_vtk->SetNumberOfComponents(1);
  stdv_forces_vtk->SetName("Forces-Stdv");

  auto mean_forces_vtk = vtkSmartPointer<vtkDoubleArray>::New();
  mean_forces_vtk->SetNumberOfComponents(3);
  mean_forces_vtk->SetName("Forces-Mean");

  for (unsigned int i_site = 0; i_site < n_atoms; ++i_site) {
    points->InsertNextPoint(atoms->mean_q[i_site * dim + 0],
                            atoms->mean_q[i_site * dim + 1],
                            atoms->mean_q[i_site * dim + 2]);
    Type->InsertNextValue(atoms->specie[i_site]);
    if (atoms->specie[i_site] == 0) {
      atomic_number->InsertNextValue(12);
    } else if (atoms->specie[i_site] == 1) {
      atomic_number->InsertNextValue(1);
    }
    int spc_i = atoms->specie[i_site];
    xi->InsertNextValue(atoms->xi[i_site]);
    stddedv_q->InsertNextValue(atoms->stdv_q[i_site]);
    stdv_forces_vtk->InsertNextValue(stdv_forces(i_site));
    mean_forces_vtk->InsertNextValue(mean_forces(i_site, 0));
    mean_forces_vtk->InsertNextValue(mean_forces(i_site, 1));
    mean_forces_vtk->InsertNextValue(mean_forces(i_site, 2));
  }

  // Create a polydata object and add the points to it.
  vtkSmartPointer<vtkPolyData> polydata =
    vtkSmartPointer<vtkPolyData>::New();
  polydata->SetPoints(points);
  polydata->GetPointData()->AddArray(Type);
  polydata->GetPointData()->AddArray(stddedv_q);
  polydata->GetPointData()->AddArray(atomic_number);
  polydata->GetPointData()->AddArray(xi);
  polydata->GetPointData()->AddArray(stdv_forces_vtk);
  polydata->GetPointData()->AddArray(mean_forces_vtk);

  // Write the file
  vtkSmartPointer<vtkXMLPolyDataWriter> writer = vtkSmartPointer<vtkXMLPolyDataWriter>::New();
  writer->SetFileName(Name_file_t);

#if VTK_MAJOR_VERSION <= 5
  writer->SetInput(polydata);
#else
  writer->SetInputData(polydata);
#endif

  // Optional - set the mode. The default is binary.
  writer->SetDataModeToBinary();
  // writer->SetDataModeToAscii();
  writer->SetCompressor(vtkZLibDataCompressor::New());
  writer->Write();

  return EXIT_SUCCESS;
}

/********************************************************************************/

int write_residual_stdvq_vtk(int step, atom* atoms, adpPotential* adp[2][2],
                             const double* residual) {
  unsigned int dim = 3;
  int n_atoms = atoms->n_atoms;

  /* File name */
  char Name_file_t[10000];
  snprintf(Name_file_t, sizeof(Name_file_t), "%s/residual_stdvq_%i.vtp",
           OutputFolder, step);

  auto points = vtkSmartPointer<vtkPoints>::New();

  auto Type = vtkSmartPointer<vtkIntArray>::New();
  Type->SetNumberOfComponents(1);
  Type->SetName("Type");

  auto atomic_number = vtkSmartPointer<vtkDoubleArray>::New();
  atomic_number->SetNumberOfComponents(1);
  atomic_number->SetName("Atomic number");

  auto stddedv_q = vtkSmartPointer<vtkDoubleArray>::New();
  stddedv_q->SetNumberOfComponents(1);
  stddedv_q->SetName("Standard-desviation-q");

  auto xi = vtkSmartPointer<vtkDoubleArray>::New();
  xi->SetNumberOfComponents(1);
  xi->SetName("Molar fraction");

  auto mass = vtkSmartPointer<vtkDoubleArray>::New();
  mass->SetNumberOfComponents(1);
  mass->SetName("Mass");

  auto Residual = vtkSmartPointer<vtkDoubleArray>::New();
  Residual->SetNumberOfComponents(1);
  Residual->SetName("Residual");

  for (unsigned int i_site = 0; i_site < n_atoms; ++i_site) {
    points->InsertNextPoint(atoms->mean_q[i_site * dim + 0],
                            atoms->mean_q[i_site * dim + 1],
                            atoms->mean_q[i_site * dim + 2]);
    Type->InsertNextValue(atoms->specie[i_site]);
    if (atoms->specie[i_site] == 0) {
      atomic_number->InsertNextValue(12);
    } else if (atoms->specie[i_site] == 1) {
      atomic_number->InsertNextValue(1);
    }
    int spc_i = atoms->specie[i_site];
    mass->InsertNextValue(adp[spc_i][spc_i]->mass);
    xi->InsertNextValue(atoms->xi[i_site]);
    stddedv_q->InsertNextValue(atoms->stdv_q[i_site]);
    Residual->InsertNextValue(residual[i_site]);
  }

  // Create a polydata object and add the points to it.
  vtkSmartPointer<vtkPolyData> polydata =
    vtkSmartPointer<vtkPolyData>::New();
  polydata->SetPoints(points);
  polydata->GetPointData()->AddArray(Type);
  polydata->GetPointData()->AddArray(stddedv_q);
  polydata->GetPointData()->AddArray(atomic_number);
  polydata->GetPointData()->AddArray(xi);
  polydata->GetPointData()->AddArray(mass);
  polydata->GetPointData()->AddArray(Residual);

  // Write the file
  vtkSmartPointer<vtkXMLPolyDataWriter> writer = vtkSmartPointer<vtkXMLPolyDataWriter>::New();
  writer->SetFileName(Name_file_t);

#if VTK_MAJOR_VERSION <= 5
  writer->SetInput(polydata);
#else
  writer->SetInputData(polydata);
#endif

  // Optional - set the mode. The default is binary.
  writer->SetDataModeToBinary();
  // writer->SetDataModeToAscii();
  writer->SetCompressor(vtkZLibDataCompressor::New());
  writer->Write();

  return EXIT_SUCCESS;
}

/********************************************************************************/

#endif
