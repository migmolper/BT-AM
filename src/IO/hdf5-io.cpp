/**
 * @file IO/hdf5-io.cpp
 * @author your name (@migmolper)
 * @brief
 * @version 0.1
 * @date 2024-03-28
 *
 * @copyright Copyright (c) 2024
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
#include "petscviewerhdf5.h"
#include <fstream>   // Para ofstream
#include <iostream>  // Para cout
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern Eigen::Vector3d box_origin_0;

extern Eigen::Vector3d lattice_x_Bn;
extern Eigen::Vector3d lattice_y_Bn;
extern Eigen::Vector3d lattice_z_Bn;

extern Eigen::Vector3d lattice_x_B0;
extern Eigen::Vector3d lattice_y_B0;
extern Eigen::Vector3d lattice_z_B0;

/********************************************************/

PetscErrorCode DMSwarmWriteHDF5(DMD* Simulation, const char filename[],
                                PetscFileMode type) {

  PetscFunctionBegin;

  unsigned int dim = NumberDimensions;

  //! Get mesh information from the DMD simulation
  PetscInt n_sites_global;
  PetscCall(DMSwarmGetSize(Simulation->atomistic_data, &n_sites_global));

  PetscInt n_sites_local;
  PetscCall(DMSwarmGetLocalSize(Simulation->atomistic_data, &n_sites_local));

  //! Load Hdf5 viewer
  PetscViewer viewer_hdf5;
  PetscCall(PetscViewerHDF5Open(MPI_COMM_WORLD, filename, type, &viewer_hdf5));
  PetscCall(PetscViewerHDF5PushTimestepping(viewer_hdf5));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Write general atomistic atributes
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(PetscViewerHDF5WriteAttribute(viewer_hdf5, NULL, "Temperature-env",
                                          PETSC_REAL,
                                          &(Simulation->Temperature_env)));
  PetscCall(PetscViewerHDF5WriteAttribute(
      viewer_hdf5, NULL, "ChemicalPotential-env", PETSC_REAL,
      &(Simulation->ChemicalPotential_env)));

  PetscCall(PetscViewerHDF5WriteAttribute(viewer_hdf5, NULL, "Fxx", PETSC_REAL,
                                          &(Simulation->F(0, 0))));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer_hdf5, NULL, "Fyy", PETSC_REAL,
                                          &(Simulation->F(1, 1))));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer_hdf5, NULL, "Fzz", PETSC_REAL,
                                          &(Simulation->F(2, 2))));

  PetscCall(PetscViewerHDF5WriteAttribute(viewer_hdf5, NULL, "Xmin", PETSC_REAL,
                                          &(box_origin_0(0))));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer_hdf5, NULL, "Ymin", PETSC_REAL,
                                          &(box_origin_0(1))));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer_hdf5, NULL, "Zmin", PETSC_REAL,
                                          &(box_origin_0(2))));

  Eigen::Vector3d box_end_0 =
      box_origin_0 + lattice_x_B0 + lattice_y_B0 + lattice_z_B0;

  PetscCall(PetscViewerHDF5WriteAttribute(viewer_hdf5, NULL, "Xmax", PETSC_REAL,
                                          &(box_end_0(0))));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer_hdf5, NULL, "Ymax", PETSC_REAL,
                                          &(box_end_0(1))));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer_hdf5, NULL, "Zmax", PETSC_REAL,
                                          &(box_end_0(2))));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Mean position of each site
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  Vec mean_q;
  PetscCall(DMSwarmCreateGlobalVectorFromField(Simulation->atomistic_data,
                                               DMSwarmPICField_coor, &mean_q));

  PetscCall(VecView(mean_q, viewer_hdf5));

  PetscCall(DMSwarmDestroyGlobalVectorFromField(Simulation->atomistic_data,
                                                DMSwarmPICField_coor, &mean_q));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Standard desviation of the position
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  Vec stdv_q;
  PetscCall(DMSwarmCreateGlobalVectorFromField(Simulation->atomistic_data,
                                               "stdv-q", &stdv_q));

  PetscCall(VecView(stdv_q, viewer_hdf5));

  PetscCall(DMSwarmDestroyGlobalVectorFromField(Simulation->atomistic_data,
                                                "stdv-q", &stdv_q));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Molar fraction of each site
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  Vec xi;
  PetscCall(DMSwarmCreateGlobalVectorFromField(Simulation->atomistic_data,
                                               "molar-fraction", &xi));

  PetscCall(VecView(xi, viewer_hdf5));

  PetscCall(DMSwarmDestroyGlobalVectorFromField(Simulation->atomistic_data,
                                                "molar-fraction", &xi));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Molar fraction rate of each site
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  Vec dxidt;
  PetscCall(DMSwarmCreateGlobalVectorFromField(Simulation->atomistic_data,
                                               "molar-fraction-dt", &dxidt));

  PetscCall(VecView(dxidt, viewer_hdf5));

  PetscCall(DMSwarmDestroyGlobalVectorFromField(Simulation->atomistic_data,
                                                "molar-fraction-dt", &dxidt));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Write Thermal Lagrange multiplier
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  Vec beta;
  PetscCall(DMSwarmCreateGlobalVectorFromField(Simulation->atomistic_data,
                                               "beta", &beta));

  PetscCall(VecView(beta, viewer_hdf5));

  PetscCall(DMSwarmDestroyGlobalVectorFromField(Simulation->atomistic_data,
                                                "beta", &beta));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Write Chemical Lagrange multiplier
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  Vec gamma;
  PetscCall(DMSwarmCreateGlobalVectorFromField(Simulation->atomistic_data,
                                               "gamma", &gamma));

  PetscCall(VecView(gamma, viewer_hdf5));

  PetscCall(DMSwarmDestroyGlobalVectorFromField(Simulation->atomistic_data,
                                                "gamma", &gamma));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Write Particle index
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  IS idx;
  const PetscInt* idx_ptr;

  PetscCall(DMSwarmGetField(Simulation->atomistic_data, "idx", NULL, NULL,
                            (void**)&idx_ptr));

  PetscCall(ISCreateGeneral(MPI_COMM_WORLD, n_sites_local, idx_ptr,
                            PETSC_USE_POINTER, &idx));
  PetscCall(PetscObjectSetName((PetscObject)idx, "idx"));

  PetscCall(ISView(idx, viewer_hdf5));

  PetscCall(ISDestroy(&idx));

  PetscCall(DMSwarmRestoreField(Simulation->atomistic_data, "idx", NULL, NULL,
                                (void**)&idx_ptr));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Write Read atomic specie
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  IS specie;
  const PetscInt* specie_ptr;

  PetscCall(DMSwarmGetField(Simulation->atomistic_data, "specie", NULL, NULL,
                            (void**)&specie_ptr));

  PetscCall(ISCreateGeneral(MPI_COMM_WORLD, n_sites_local, specie_ptr,
                            PETSC_USE_POINTER, &specie));
  PetscCall(PetscObjectSetName((PetscObject)specie, "specie"));

  PetscCall(ISView(specie, viewer_hdf5));

  PetscCall(ISDestroy(&specie));

  PetscCall(DMSwarmRestoreField(Simulation->atomistic_data, "specie", NULL,
                                NULL, (void**)&specie_ptr));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Write Read MPI-rank
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  IS MPI_rank;
  const PetscInt* MPI_rank_ptr;

  PetscCall(DMSwarmGetField(Simulation->atomistic_data, "MPI-rank", NULL, NULL,
                            (void**)&MPI_rank_ptr));

  PetscCall(ISCreateGeneral(MPI_COMM_WORLD, n_sites_local, MPI_rank_ptr,
                            PETSC_USE_POINTER, &MPI_rank));
  PetscCall(PetscObjectSetName((PetscObject)MPI_rank, "MPI-rank"));

  PetscCall(ISView(MPI_rank, viewer_hdf5));

  PetscCall(ISDestroy(&MPI_rank));

  PetscCall(DMSwarmRestoreField(Simulation->atomistic_data, "MPI-rank", NULL,
                                NULL, (void**)&MPI_rank_ptr));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Write thermal bcc index
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  IS beta_bcc;
  const PetscInt* beta_bcc_ptr;

  PetscCall(DMSwarmGetField(Simulation->atomistic_data, "idx-bcc-beta", NULL, NULL,
                            (void**)&beta_bcc_ptr));

  PetscCall(ISCreateGeneral(MPI_COMM_WORLD, n_sites_local, beta_bcc_ptr,
                            PETSC_USE_POINTER, &beta_bcc));
  PetscCall(PetscObjectSetName((PetscObject)beta_bcc, "idx-bcc-beta"));

  PetscCall(ISView(beta_bcc, viewer_hdf5));

  PetscCall(ISDestroy(&beta_bcc));

  PetscCall(DMSwarmRestoreField(Simulation->atomistic_data, "idx-bcc-beta", NULL,
                                NULL, (void**)&beta_bcc_ptr));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Write chemical bcc index
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  IS gamma_bcc;
  const PetscInt* gamma_bcc_ptr;

  PetscCall(DMSwarmGetField(Simulation->atomistic_data, "idx-bcc-gamma", NULL, NULL,
                            (void**)&gamma_bcc_ptr));

  PetscCall(ISCreateGeneral(MPI_COMM_WORLD, n_sites_local, gamma_bcc_ptr,
                            PETSC_USE_POINTER, &gamma_bcc));
  PetscCall(PetscObjectSetName((PetscObject)gamma_bcc, "idx-bcc-gamma"));

  PetscCall(ISView(gamma_bcc, viewer_hdf5));

  PetscCall(ISDestroy(&gamma_bcc));

  PetscCall(DMSwarmRestoreField(Simulation->atomistic_data, "idx-bcc-gamma", NULL,
                                NULL, (void**)&gamma_bcc_ptr));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Destroy HDF5 viewer
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(PetscViewerDestroy(&viewer_hdf5));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/********************************************************/

PetscErrorCode DMSwarmReadHDF5(DMD* Simulation, const char hdf5_filename[]) {

  PetscFunctionBegin;

  unsigned int dim = NumberDimensions;

  //! Get mesh information from the DMD simulation
  PetscInt n_sites_global;
  PetscCall(DMSwarmGetSize(Simulation->atomistic_data, &n_sites_global));

  PetscInt n_sites_local;
  PetscCall(DMSwarmGetLocalSize(Simulation->atomistic_data, &n_sites_local));

  //! Load Hdf5 viewer
  PetscViewer viewer_hdf5;
  PetscCall(PetscViewerHDF5Open(MPI_COMM_WORLD, hdf5_filename, FILE_MODE_READ,
                                &viewer_hdf5));
  PetscCall(PetscViewerHDF5PushTimestepping(viewer_hdf5));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Read general atomistic atributes
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(PetscViewerHDF5ReadAttribute(
      viewer_hdf5, NULL, "Temperature-env", PETSC_REAL,
      &(Simulation->Temperature_env), &(Simulation->Temperature_env)));
  PetscCall(PetscViewerHDF5ReadAttribute(viewer_hdf5, NULL,
                                         "ChemicalPotential-env", PETSC_REAL,
                                         &(Simulation->ChemicalPotential_env),
                                         &(Simulation->ChemicalPotential_env)));

  PetscCall(PetscViewerHDF5ReadAttribute(viewer_hdf5, NULL, "Fxx", PETSC_REAL,
                                         &(Simulation->F(0, 0)),
                                         &(Simulation->F(0, 0))));
  PetscCall(PetscViewerHDF5ReadAttribute(viewer_hdf5, NULL, "Fyy", PETSC_REAL,
                                         &(Simulation->F(1, 1)),
                                         &(Simulation->F(1, 1))));
  PetscCall(PetscViewerHDF5ReadAttribute(viewer_hdf5, NULL, "Fzz", PETSC_REAL,
                                         &(Simulation->F(2, 2)),
                                         &(Simulation->F(2, 2))));

  lattice_x_B0 = Simulation->F * lattice_x_B0;
  lattice_y_B0 = Simulation->F * lattice_y_B0;
  lattice_z_B0 = Simulation->F * lattice_z_B0;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Read idx
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  IS idx;
  PetscInt* idx_ptr;

  PetscCall(DMSwarmGetField(Simulation->atomistic_data, "idx", NULL, NULL,
                            (void**)&idx_ptr));
  PetscCall(ISCreateGeneral(MPI_COMM_WORLD, n_sites_local, idx_ptr,
                            PETSC_USE_POINTER, &idx));
  PetscCall(PetscObjectSetName((PetscObject)idx, "idx"));

  PetscCall(ISLoad(idx, viewer_hdf5));

  PetscCall(DMSwarmRestoreField(Simulation->atomistic_data, "idx", NULL, NULL,
                                (void**)&idx_ptr));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Read atomic specie
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  IS specie;
  PetscInt* specie_ptr;

  PetscCall(DMSwarmGetField(Simulation->atomistic_data, "specie", NULL, NULL,
                            (void**)&specie_ptr));
  PetscCall(ISCreateGeneral(MPI_COMM_WORLD, n_sites_local, specie_ptr,
                            PETSC_USE_POINTER, &specie));
  PetscCall(PetscObjectSetName((PetscObject)specie, "specie"));

  PetscCall(ISLoad(specie, viewer_hdf5));

  PetscCall(DMSwarmRestoreField(Simulation->atomistic_data, "specie", NULL,
                                NULL, (void**)&specie_ptr));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Read MPI-rank
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  IS MPI_rank;
  PetscInt* MPI_rank_ptr;

  PetscCall(DMSwarmGetField(Simulation->atomistic_data, "MPI-rank", NULL, NULL,
                            (void**)&MPI_rank_ptr));
  PetscCall(ISCreateGeneral(MPI_COMM_WORLD, n_sites_local, MPI_rank_ptr,
                            PETSC_USE_POINTER, &MPI_rank));
  PetscCall(PetscObjectSetName((PetscObject)MPI_rank, "MPI-rank"));

  PetscCall(ISLoad(MPI_rank, viewer_hdf5));

  PetscCall(DMSwarmRestoreField(Simulation->atomistic_data, "MPI-rank", NULL,
                                NULL, (void**)&MPI_rank_ptr));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Push "/particle_fields" group
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(PetscViewerHDF5PushGroup(viewer_hdf5, "/particle_fields"));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Read mean-q
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  Vec mean_q;
  PetscCall(DMSwarmCreateGlobalVectorFromField(Simulation->atomistic_data,
                                               DMSwarmPICField_coor, &mean_q));

  Vec mean_q_hdf5;
  PetscCall(VecDuplicate(mean_q, &mean_q_hdf5));
  PetscCall(PetscObjectSetName((PetscObject)mean_q_hdf5,
                               "DMSwarmSharedField_DMSwarmPIC_coor"));

  PetscCall(VecLoad(mean_q_hdf5, viewer_hdf5));

  PetscCall(VecSwap(mean_q, mean_q_hdf5));
  PetscCall(VecDestroy(&mean_q_hdf5));

  PetscCall(DMSwarmDestroyGlobalVectorFromField(Simulation->atomistic_data,
                                                DMSwarmPICField_coor, &mean_q));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Read stdv-q
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  Vec stdv_q;
  PetscCall(DMSwarmCreateGlobalVectorFromField(Simulation->atomistic_data,
                                               "stdv-q", &stdv_q));

  Vec stdv_q_hdf5;
  PetscCall(VecDuplicate(stdv_q, &stdv_q_hdf5));
  PetscCall(PetscObjectSetName((PetscObject)stdv_q_hdf5,
                               "DMSwarmSharedField_stdv-q"));

  PetscCall(VecLoad(stdv_q_hdf5, viewer_hdf5));

  PetscCall(VecSwap(stdv_q, stdv_q_hdf5));
  PetscCall(VecDestroy(&stdv_q_hdf5));

  PetscCall(DMSwarmDestroyGlobalVectorFromField(Simulation->atomistic_data,
                                                "stdv-q", &stdv_q));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Read molar-fraction
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  Vec xi;
  PetscCall(DMSwarmCreateGlobalVectorFromField(Simulation->atomistic_data,
                                               "molar-fraction", &xi));

  Vec xi_hdf5;
  PetscCall(VecDuplicate(xi, &xi_hdf5));
  PetscCall(PetscObjectSetName((PetscObject)xi_hdf5,
                               "DMSwarmSharedField_molar-fraction"));

  PetscCall(VecLoad(xi_hdf5, viewer_hdf5));

  PetscCall(VecSwap(xi, xi_hdf5));
  PetscCall(VecDestroy(&xi_hdf5));

  PetscCall(DMSwarmDestroyGlobalVectorFromField(Simulation->atomistic_data,
                                                "molar-fraction", &xi));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Read molar-fraction rate
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*   Vec dxidt;
    PetscCall(DMSwarmCreateGlobalVectorFromField(Simulation->atomistic_data,
                                                 "molar-fraction-dt", &dxidt));

    Vec dxidt_hdf5;
    PetscCall(VecDuplicate(dxidt, &dxidt_hdf5));
    PetscCall(PetscObjectSetName((PetscObject)dxidt_hdf5,
                                 "DMSwarmSharedField_molar-fraction"));

    PetscCall(VecLoad(dxidt_hdf5, viewer_hdf5));

    PetscCall(VecSwap(dxidt, dxidt_hdf5));
    PetscCall(VecDestroy(&dxidt_hdf5));

    PetscCall(DMSwarmDestroyGlobalVectorFromField(Simulation->atomistic_data,
                                                  "molar-fraction-dt", &dxidt));
  */

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Read beta (Thermal Lagrange multiplier)
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  Vec beta;
  PetscCall(DMSwarmCreateGlobalVectorFromField(Simulation->atomistic_data,
                                               "beta", &beta));

  Vec beta_hdf5;
  PetscCall(VecDuplicate(beta, &beta_hdf5));
  PetscCall(
      PetscObjectSetName((PetscObject)beta_hdf5, "DMSwarmSharedField_beta"));

  PetscCall(VecLoad(beta_hdf5, viewer_hdf5));

  PetscCall(VecSwap(beta, beta_hdf5));
  PetscCall(VecDestroy(&beta_hdf5));

  PetscCall(DMSwarmDestroyGlobalVectorFromField(Simulation->atomistic_data,
                                                "beta", &beta));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Read gamma (Chemical Lagrange multiplier)
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  Vec gamma;
  PetscCall(DMSwarmCreateGlobalVectorFromField(Simulation->atomistic_data,
                                               "gamma", &gamma));

  Vec gamma_hdf5;
  PetscCall(VecDuplicate(gamma, &gamma_hdf5));
  PetscCall(
      PetscObjectSetName((PetscObject)gamma_hdf5, "DMSwarmSharedField_gamma"));

  PetscCall(VecLoad(gamma_hdf5, viewer_hdf5));

  PetscCall(VecSwap(gamma, gamma_hdf5));
  PetscCall(VecDestroy(&gamma_hdf5));

  PetscCall(DMSwarmDestroyGlobalVectorFromField(Simulation->atomistic_data,
                                                "gamma", &gamma));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Destroy Hdf5 viewer
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(PetscViewerHDF5PopGroup(viewer_hdf5));
  PetscCall(PetscViewerDestroy(&viewer_hdf5));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/********************************************************/