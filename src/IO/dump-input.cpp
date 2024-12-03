/**
 * @file IO/dump-input.cpp
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2023-12-23
 *
 * @copyright Copyright (c) 2023
 *
 */

#if __APPLE__
#include <malloc/_malloc.h>
#endif
#ifdef USE_MPI
#include <mpi.h>
#endif
#include "Atoms/Atom.hpp"
#include "Macros.hpp"
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using namespace std;
extern int diffusion;

extern PetscMPIInt size_MPI;
extern PetscMPIInt rank_MPI;

extern Eigen::Vector3d lattice_x_B0;
extern Eigen::Vector3d lattice_y_B0;
extern Eigen::Vector3d lattice_z_B0;

extern Eigen::Vector3d lattice_x_Bn;
extern Eigen::Vector3d lattice_y_Bn;
extern Eigen::Vector3d lattice_z_Bn;

extern double box_x_min;
extern double box_x_max;
extern double box_y_min;
extern double box_y_max;
extern double box_z_min;
extern double box_z_max;

extern Eigen::Vector3d box_origin_0;

/*******************************************************/

dump_file read_dump_information(const char* SimulationFile) {

  unsigned int dim = NumberDimensions;
  dump_file Simulation;

  //! Variables to read atomistic data
  FILE* simulation_data = fopen(SimulationFile, "r");
  char aux_line[10000];
  int error;

  //! @brief Read the time step of the file
  int time_step;
  fgets(aux_line, sizeof(aux_line), simulation_data);
  error = fscanf(simulation_data, "%d\n", &time_step);

  //! @brief Read number of atomic position
  int n_atoms = 0;
  fgets(aux_line, sizeof(aux_line), simulation_data);
  error = fscanf(simulation_data, "%d\n", &n_atoms);
  Simulation.n_atoms = n_atoms;

  //! @brief Read box bcc
  char bc_x[10000];
  char bc_y[10000];
  char bc_z[10000];
  error = fscanf(simulation_data, "ITEM: BOX BOUNDS %s %s %s\n",  //!
                 bc_x, bc_y, bc_z);                               //!

  if (strcmp(bc_x, "pp") == 0) {
    Simulation.bx = DM_BOUNDARY_PERIODIC;
  } else if (strcmp(bc_x, "ff") == 0) {
    Simulation.bx = DM_BOUNDARY_NONE;
  } else {
    std::cout << "Unrecognised type of bc" << std::endl;
  }

  if (strcmp(bc_y, "pp") == 0) {
    Simulation.by = DM_BOUNDARY_PERIODIC;
  } else if (strcmp(bc_y, "ff") == 0) {
    Simulation.by = DM_BOUNDARY_NONE;
  } else {
    std::cout << "Unrecognised type of bc" << std::endl;
  }

  if (strcmp(bc_z, "pp") == 0) {
    Simulation.bz = DM_BOUNDARY_PERIODIC;
  } else if (strcmp(bc_z, "ff") == 0) {
    Simulation.bz = DM_BOUNDARY_NONE;
  } else {
    std::cout << "Unrecognised type of bc" << std::endl;
  }

  //! @brief Read box bounds
  error = fscanf(simulation_data, "%lf %lf\n", &box_x_min, &box_x_max);
  error = fscanf(simulation_data, "%lf %lf\n", &box_y_min, &box_y_max);
  error = fscanf(simulation_data, "%lf %lf\n", &box_z_min, &box_z_max);

  //! @brief Create box lattice vectors
  lattice_x_B0 << 1.0, 0.0, 0.0;
  lattice_y_B0 << 0.0, 1.0, 0.0;
  lattice_z_B0 << 0.0, 0.0, 1.0;

  //! @brief Compute box-parameters
  box_origin_0(0) = box_x_min;
  box_origin_0(1) = box_y_min;
  box_origin_0(2) = box_z_min;

  lattice_x_B0(0) = (box_x_max - box_x_min);
  lattice_y_B0(1) = (box_y_max - box_y_min);
  lattice_z_B0(2) = (box_z_max - box_z_min);

  lattice_x_Bn = lattice_x_B0;
  lattice_y_Bn = lattice_y_B0;
  lattice_z_Bn = lattice_z_B0;

  //! @brief Create arrays
  Simulation.specie = (AtomicSpecie*)calloc(n_atoms, sizeof(AtomicSpecie));

  Simulation.beta = (double*)calloc(n_atoms, sizeof(double));

  Simulation.beta_bcc = (int*)calloc(n_atoms, sizeof(int));

  Simulation.gamma = (double*)calloc(n_atoms, sizeof(double));

  Simulation.gamma_bcc = (int*)calloc(n_atoms, sizeof(int));

  Simulation.mean_q = (double*)calloc(n_atoms * dim, sizeof(double));

  Simulation.stdv_q = (double*)calloc(n_atoms, sizeof(double));

  Simulation.xi = (double*)calloc(n_atoms, sizeof(double));

  Simulation.diffusive_idx = (int*)calloc(n_atoms, sizeof(int));

  //! @brief Read properties from dump file
  fgets(aux_line, sizeof(aux_line), simulation_data);

  for (unsigned int i_site = 0; i_site < n_atoms; i_site++) {

    unsigned int Particle_Type;
    double mean_q_x_value;
    double mean_q_y_value;
    double mean_q_z_value;
    double stdv_q_value;
    double xi_value;
    double gamma_value;
    double beta_value;
    double gamma_bcc_value;
    double beta_bcc_value;

    error =
        fscanf(simulation_data, "%i %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",
               &(Particle_Type), &(mean_q_x_value), &(mean_q_y_value),
               &(mean_q_z_value), &(stdv_q_value), &(xi_value), &(gamma_value),
               &(beta_value), &(gamma_bcc_value), &(beta_bcc_value));

    if (mean_q_x_value > box_x_max || mean_q_x_value < box_x_min) {
      std::cout << "FAIL mean_q_x_value " << mean_q_x_value << " at site "
                << i_site << std::endl;
    }

    if (mean_q_y_value > box_y_max || mean_q_y_value < box_y_min) {
      std::cout << "FAIL mean_q_y_value " << mean_q_y_value << " at site "
                << i_site << std::endl;
    }

    if (mean_q_z_value > box_z_max || mean_q_z_value < box_z_min) {
      std::cout << "FAIL mean_q_z_value " << mean_q_z_value << " at site "
                << i_site << std::endl;
    }

    Simulation.mean_q[3 * i_site + 0] = mean_q_x_value;
    Simulation.mean_q[3 * i_site + 1] = mean_q_y_value;
    Simulation.mean_q[3 * i_site + 2] = mean_q_z_value;
    Simulation.stdv_q[i_site] = stdv_q_value;
    Simulation.xi[i_site] = xi_value;
    Simulation.gamma[i_site] = gamma_value;
    Simulation.beta[i_site] = beta_value;
    Simulation.gamma_bcc[i_site] = (int)gamma_bcc_value;
    Simulation.beta_bcc[i_site] = (int)beta_bcc_value;

    // Initialize particle type.
    if ((Particle_Type < 1) || (Particle_Type > 111)) {
      std::cout << "FAIL Particle_Type " << Particle_Type << " at site "
                << i_site << std::endl;
    } else {
      Simulation.specie[i_site] = (AtomicSpecie)Particle_Type;
    }
  }

  //! Identify H atoms as diffusive sites
  for (unsigned int i_site = 0; i_site < n_atoms; i_site++) {
    if (Simulation.specie[i_site] == H) {
      Simulation.diffusive_idx[i_site] = true;
    } else {
      Simulation.diffusive_idx[i_site] = false;
    }
  }

  //  Close file
  fclose(simulation_data);

  return Simulation;
}

/*******************************************************/

void free_dump_information(dump_file* Simulation_data) {

  free(Simulation_data->specie);

  free(Simulation_data->beta);

  free(Simulation_data->beta_bcc);

  free(Simulation_data->gamma);

  free(Simulation_data->gamma_bcc);

  free(Simulation_data->mean_q);

  free(Simulation_data->stdv_q);

  free(Simulation_data->xi);

  free(Simulation_data->diffusive_idx);
}

/*******************************************************/