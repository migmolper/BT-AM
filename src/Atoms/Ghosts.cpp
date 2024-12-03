/**
 * @file Atoms/Ghosts.cpp
 * @author Miguel Molinos (@migmolper)
 * @brief
 * @version 0.1
 * @date 2024-08-22
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "Macros.hpp"
#include "Periodic-Boundary/boundary_conditions.hpp"
#include "petscdmswarm.h"
#include <petscsystypes.h>

extern PetscMPIInt size_MPI;
extern PetscMPIInt rank_MPI;

extern PetscInt ndiv_mesh_X;
extern PetscInt ndiv_mesh_Y;
extern PetscInt ndiv_mesh_Z;

extern double box_x_min;
extern double box_x_max;
extern double box_y_min;
extern double box_y_max;
extern double box_z_min;
extern double box_z_max;

extern bool crystal_directions[27];

static PetscErrorCode GetElementCoords(const PetscScalar _coords[],
                                       const PetscInt e2n[],
                                       PetscScalar el_coords[]);

static bool In_Out_Cell(const Eigen::Vector3d mean_q,
                        const PetscScalar el_coords[]);

/********************************************************************************/

PetscErrorCode DMSwarmCreateGhostAtoms(DMD* Simulation, double buffer_width) {

  unsigned int dim = NumberDimensions;

  PetscFunctionBegin;

  PetscInt* idx_ptr;
  PetscInt* idx_diff_ptr;
  PetscInt* swarm_rank_ptr;
  PetscInt* ghost_ptr;
  PetscInt* rank_ptr;
  AtomicSpecie* specie_ptr;
  PetscScalar* mean_q_ptr;
  PetscScalar* stdv_q_ptr;
  PetscScalar* xi_ptr;
  PetscScalar* beta_ptr;
  PetscScalar* gamma_ptr;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Check we have the correct migration algorithm
      - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  DMSwarmMigrateType atoms_migrate_type;
  DMSwarmGetMigrateType(Simulation->atomistic_data, &atoms_migrate_type);

  if (atoms_migrate_type != DMSWARM_MIGRATE_BASIC) {

    PetscCall(
        PetscError(PETSC_COMM_WORLD, __LINE__, "DMSwarmCreateGhostAtoms",
                   __FILE__, PETSC_ERR_RETURN, PETSC_ERROR_INITIAL,
                   "Set DMSWARM_MIGRATE_BASIC using DMSwarmSetMigrateType"));

    PetscFunctionReturn(PETSC_ERR_RETURN);
  }

  //! Get global size
  PetscInt n_global_size = 0;
  PetscCall(DMSwarmGetSize(Simulation->atomistic_data, &n_global_size));

  //! Get local size
  PetscInt n_local_size = 0;
  PetscCall(DMSwarmGetLocalSize(Simulation->atomistic_data, &n_local_size));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Get the global (x,y,z) indices of the lower left corner and size of the
    local region, excluding ghost points.
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscReal lmin[3], lmax[3];
  PetscCall(DMGetLocalBoundingBox(Simulation->background_mesh, lmin, lmax));

  //! Lower left coordinates of the brick
  PetscReal X_ll, Y_ll, Z_ll;
  X_ll = lmin[0];
  Y_ll = lmin[1];
  Z_ll = lmin[2];

  //! Upper right coordinates of the brick
  PetscReal X_ur, Y_ur, Z_ur;
  X_ur = lmax[0];
  Y_ur = lmax[1];
  Z_ur = lmax[2];

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Get the list of processes neighboring this one.
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  const PetscMPIInt* neig_ranks_ptr;
  PetscInt nranks;
  PetscCall(
      DMGetNeighbors(Simulation->background_mesh, &nranks, &neig_ranks_ptr));
  Eigen::Map<const List1D> ranks_list_1D(neig_ranks_ptr, nranks);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Mark ghost particles in the buffer region of the domain.
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMSwarmGetField(Simulation->atomistic_data, DMSwarmPICField_coor,
                            NULL, NULL, (void**)&mean_q_ptr));
  Eigen::Map<MatrixType> mean_q(mean_q_ptr, n_local_size, dim);

  int num_ghost = 0;
  int* idx_ghost = (int*)malloc(n_global_size * sizeof(int));
  int* rank_ghost = (int*)malloc(n_global_size * sizeof(int));

  for (int site_i = 0; site_i < n_local_size; site_i++) {

    Eigen::Vector3d mean_q_i = mean_q.row(site_i);

    // Direction (-1 -1 -1): n0
    int rank_m1_m1_m1 = ranks_list_1D(0);
    if (rank_m1_m1_m1 >= 0) {

      PetscScalar buffer_coords[6];
      buffer_coords[0] = X_ll;
      buffer_coords[1] = Y_ll;
      buffer_coords[2] = Z_ll;

      buffer_coords[3] = X_ll + buffer_width;
      buffer_coords[4] = Y_ll + buffer_width;
      buffer_coords[5] = Z_ll + buffer_width;

      if (In_Out_Cell(mean_q_i, buffer_coords)) {  //!
        idx_ghost[num_ghost] = site_i;
        rank_ghost[num_ghost] = rank_m1_m1_m1;
        num_ghost++;
      }
    }

    // Direction (0 -1 -1): n1
    int rank_0_m1_m1 = ranks_list_1D(1);
    if (rank_0_m1_m1 >= 0) {

      PetscScalar buffer_coords[6];
      buffer_coords[0] = X_ll;
      buffer_coords[1] = Y_ll;
      buffer_coords[2] = Z_ll;

      buffer_coords[3] = X_ur;
      buffer_coords[4] = Y_ll + buffer_width;
      buffer_coords[5] = Z_ll + buffer_width;

      if (In_Out_Cell(mean_q_i, buffer_coords)) {  //!
        idx_ghost[num_ghost] = site_i;
        rank_ghost[num_ghost] = rank_0_m1_m1;
        num_ghost++;
      }
    }

    // Direction (1 -1 -1): n2
    int rank_p1_m1_m1 = ranks_list_1D(2);
    if (rank_p1_m1_m1 >= 0) {

      PetscScalar buffer_coords[6];
      buffer_coords[0] = X_ur - buffer_width;
      buffer_coords[1] = Y_ll;
      buffer_coords[2] = Z_ll;

      buffer_coords[3] = X_ur;
      buffer_coords[4] = Y_ll + buffer_width;
      buffer_coords[5] = Z_ll + buffer_width;

      if (In_Out_Cell(mean_q_i, buffer_coords)) {  //!
        idx_ghost[num_ghost] = site_i;
        rank_ghost[num_ghost] = rank_p1_m1_m1;
        num_ghost++;
      }
    }

    // Direction (-1 0 -1): n3
    int rank_m1_0_m1 = ranks_list_1D(3);
    if (rank_m1_0_m1 >= 0) {

      PetscScalar buffer_coords[6];
      buffer_coords[0] = X_ll;
      buffer_coords[1] = Y_ll;
      buffer_coords[2] = Z_ll;

      buffer_coords[3] = X_ll + buffer_width;
      buffer_coords[4] = Y_ur;
      buffer_coords[5] = Z_ll + buffer_width;

      if (In_Out_Cell(mean_q_i, buffer_coords)) {  //!
        idx_ghost[num_ghost] = site_i;
        rank_ghost[num_ghost] = rank_m1_0_m1;
        num_ghost++;
      }
    }

    // Direction (0 0 -1): n4
    int rank_0_0_m1 = ranks_list_1D(4);
    if (rank_0_0_m1 >= 0) {

      PetscScalar buffer_coords[6];
      buffer_coords[0] = X_ll;
      buffer_coords[1] = Y_ll;
      buffer_coords[2] = Z_ll;

      buffer_coords[3] = X_ur;
      buffer_coords[4] = Y_ur;
      buffer_coords[5] = Z_ll + buffer_width;

      if (In_Out_Cell(mean_q_i, buffer_coords)) {  //!
        idx_ghost[num_ghost] = site_i;
        rank_ghost[num_ghost] = rank_0_0_m1;
        num_ghost++;
      }
    }

    // Direction (1 0 -1): n5
    int rank_p1_0_m1 = ranks_list_1D(5);
    if (rank_p1_0_m1 >= 0) {

      PetscScalar buffer_coords[6];
      buffer_coords[0] = X_ur - buffer_width;
      buffer_coords[1] = Y_ll;
      buffer_coords[2] = Z_ll;

      buffer_coords[3] = X_ur;
      buffer_coords[4] = Y_ur;
      buffer_coords[5] = Z_ll + buffer_width;

      if (In_Out_Cell(mean_q_i, buffer_coords)) {  //!
        idx_ghost[num_ghost] = site_i;
        rank_ghost[num_ghost] = rank_p1_0_m1;
        num_ghost++;
      }
    }

    // Direction (-1 1 -1): n6
    int rank_m1_p1_m1 = ranks_list_1D(6);
    if (rank_m1_p1_m1 >= 0) {

      PetscScalar buffer_coords[6];
      buffer_coords[0] = X_ll;
      buffer_coords[1] = Y_ur - buffer_width;
      buffer_coords[2] = Z_ll;

      buffer_coords[3] = X_ll + buffer_width;
      buffer_coords[4] = Y_ur;
      buffer_coords[5] = Z_ll + buffer_width;

      if (In_Out_Cell(mean_q_i, buffer_coords)) {  //!
        idx_ghost[num_ghost] = site_i;
        rank_ghost[num_ghost] = rank_m1_p1_m1;
        num_ghost++;
      }
    }

    // Direction (0 1 -1): n7
    int rank_0_p1_m1 = ranks_list_1D(7);
    if (rank_0_p1_m1 >= 0) {

      PetscScalar buffer_coords[6];
      buffer_coords[0] = X_ll;
      buffer_coords[1] = Y_ur - buffer_width;
      buffer_coords[2] = Z_ll;

      buffer_coords[3] = X_ur;
      buffer_coords[4] = Y_ur;
      buffer_coords[5] = Z_ll + buffer_width;

      if (In_Out_Cell(mean_q_i, buffer_coords)) {  //!
        idx_ghost[num_ghost] = site_i;
        rank_ghost[num_ghost] = rank_0_p1_m1;
        num_ghost++;
      }
    }

    // Direction (1 1 -1): n8
    int rank_p1_p1_m1 = ranks_list_1D(8);
    if (rank_p1_p1_m1 >= 0) {

      PetscScalar buffer_coords[6];
      buffer_coords[0] = X_ur - buffer_width;
      buffer_coords[1] = Y_ur - buffer_width;
      buffer_coords[2] = Z_ll;

      buffer_coords[3] = X_ur;
      buffer_coords[4] = Y_ur;
      buffer_coords[5] = Z_ll + buffer_width;

      if (In_Out_Cell(mean_q_i, buffer_coords)) {  //!
        idx_ghost[num_ghost] = site_i;
        rank_ghost[num_ghost] = rank_p1_p1_m1;
        num_ghost++;
      }
    }

    // Direction (-1 -1 0): n9
    int rank_m1_m1_0 = ranks_list_1D(9);
    if (rank_m1_m1_0 >= 0) {

      PetscScalar buffer_coords[6];
      buffer_coords[0] = X_ll;
      buffer_coords[1] = Y_ll;
      buffer_coords[2] = Z_ll;

      buffer_coords[3] = X_ll + buffer_width;
      buffer_coords[4] = Y_ll + buffer_width;
      buffer_coords[5] = Z_ur;

      if (In_Out_Cell(mean_q_i, buffer_coords)) {  //!
        idx_ghost[num_ghost] = site_i;
        rank_ghost[num_ghost] = rank_m1_m1_0;
        num_ghost++;
      }
    }

    // Direction (0 -1 0): n10
    int rank_0_m1_0 = ranks_list_1D(10);
    if (rank_0_m1_0 >= 0) {

      PetscScalar buffer_coords[6];
      buffer_coords[0] = X_ll;
      buffer_coords[1] = Y_ll;
      buffer_coords[2] = Z_ll;

      buffer_coords[3] = X_ur;
      buffer_coords[4] = Y_ll + buffer_width;
      buffer_coords[5] = Z_ur;

      if (In_Out_Cell(mean_q_i, buffer_coords)) {  //!
        idx_ghost[num_ghost] = site_i;
        rank_ghost[num_ghost] = rank_0_m1_0;
        num_ghost++;
      }
    }

    // Direction (1 -1 0): n11
    int rank_p1_m1_0 = ranks_list_1D(11);
    if (rank_p1_m1_0 >= 0) {

      PetscScalar buffer_coords[6];
      buffer_coords[0] = X_ur - buffer_width;
      buffer_coords[1] = Y_ll;
      buffer_coords[2] = Z_ll;

      buffer_coords[3] = X_ur;
      buffer_coords[4] = Y_ll + buffer_width;
      buffer_coords[5] = Z_ur;

      if (In_Out_Cell(mean_q_i, buffer_coords)) {  //!
        idx_ghost[num_ghost] = site_i;
        rank_ghost[num_ghost] = rank_p1_m1_0;
        num_ghost++;
      }
    }

    // Direction (-1 0 0): n12
    int rank_m1_0_0 = ranks_list_1D(12);
    if (rank_m1_0_0 >= 0) {

      PetscScalar buffer_coords[6];
      buffer_coords[0] = X_ll;
      buffer_coords[1] = Y_ll;
      buffer_coords[2] = Z_ll;

      buffer_coords[3] = X_ll + buffer_width;
      buffer_coords[4] = Y_ur;
      buffer_coords[5] = Z_ur;

      if (In_Out_Cell(mean_q_i, buffer_coords)) {  //!
        idx_ghost[num_ghost] = site_i;
        rank_ghost[num_ghost] = rank_m1_0_0;
        num_ghost++;
      }
    }

    // Direction (0 0 0) <- current rank (we skip it): n13

    // Direction (1 0 0): n14
    int rank_p1_0_0 = ranks_list_1D(14);
    if (rank_p1_0_0 >= 0) {

      PetscScalar buffer_coords[6];
      buffer_coords[0] = X_ur - buffer_width;
      buffer_coords[1] = Y_ll;
      buffer_coords[2] = Z_ll;

      buffer_coords[3] = X_ur;
      buffer_coords[4] = Y_ur;
      buffer_coords[5] = Z_ur;

      if (In_Out_Cell(mean_q_i, buffer_coords)) {  //!
        idx_ghost[num_ghost] = site_i;
        rank_ghost[num_ghost] = rank_p1_0_0;
        num_ghost++;
      }
    }

    // Direction (-1 1 0): n15
    int rank_m1_p1_0 = ranks_list_1D(15);
    if (rank_m1_p1_0 >= 0) {

      PetscScalar buffer_coords[6];
      buffer_coords[0] = X_ll;
      buffer_coords[1] = Y_ur - buffer_width;
      buffer_coords[2] = Z_ll;

      buffer_coords[3] = X_ll + buffer_width;
      buffer_coords[4] = Y_ur;
      buffer_coords[5] = Z_ur;

      if (In_Out_Cell(mean_q_i, buffer_coords)) {  //!
        idx_ghost[num_ghost] = site_i;
        rank_ghost[num_ghost] = rank_m1_p1_0;
        num_ghost++;
      }
    }

    // Direction (0 1 0): n16
    int rank_0_p1_0 = ranks_list_1D(16);
    if (rank_0_p1_0 >= 0) {

      PetscScalar buffer_coords[6];
      buffer_coords[0] = X_ll;
      buffer_coords[1] = Y_ur - buffer_width;
      buffer_coords[2] = Z_ll;

      buffer_coords[3] = X_ur;
      buffer_coords[4] = Y_ur;
      buffer_coords[5] = Z_ur;

      if (In_Out_Cell(mean_q_i, buffer_coords)) {  //!
        idx_ghost[num_ghost] = site_i;
        rank_ghost[num_ghost] = rank_0_p1_0;
        num_ghost++;
      }
    }

    // Direction (1 1 0): n17
    int rank_p1_p1_0 = ranks_list_1D(17);
    if (rank_p1_p1_0 >= 0) {

      PetscScalar buffer_coords[6];
      buffer_coords[0] = X_ur - buffer_width;
      buffer_coords[1] = Y_ur - buffer_width;
      buffer_coords[2] = Z_ll;

      buffer_coords[3] = X_ur;
      buffer_coords[4] = Y_ur;
      buffer_coords[5] = Z_ur;

      if (In_Out_Cell(mean_q_i, buffer_coords)) {  //!
        idx_ghost[num_ghost] = site_i;
        rank_ghost[num_ghost] = rank_p1_p1_0;
        num_ghost++;
      }
    }

    // Direction (-1 -1 1): n18
    int rank_m1_m1_p1 = ranks_list_1D(18);
    if (rank_m1_m1_p1 >= 0) {

      PetscScalar buffer_coords[6];
      buffer_coords[0] = X_ll;
      buffer_coords[1] = Y_ll;
      buffer_coords[2] = Z_ur - buffer_width;

      buffer_coords[3] = X_ll + buffer_width;
      buffer_coords[4] = Y_ll + buffer_width;
      buffer_coords[5] = Z_ur;

      if (In_Out_Cell(mean_q_i, buffer_coords)) {  //!
        idx_ghost[num_ghost] = site_i;
        rank_ghost[num_ghost] = rank_m1_m1_p1;
        num_ghost++;
      }
    }

    // Direction (0 -1 1): n19
    int rank_0_m1_p1 = ranks_list_1D(19);
    if (rank_0_m1_p1 >= 0) {

      PetscScalar buffer_coords[6];
      buffer_coords[0] = X_ll;
      buffer_coords[1] = Y_ll;
      buffer_coords[2] = Z_ur - buffer_width;

      buffer_coords[3] = X_ur;
      buffer_coords[4] = Y_ll + buffer_width;
      buffer_coords[5] = Z_ur;

      if (In_Out_Cell(mean_q_i, buffer_coords)) {  //!
        idx_ghost[num_ghost] = site_i;
        rank_ghost[num_ghost] = rank_0_m1_p1;
        num_ghost++;
      }
    }

    // Direction (1 -1 1): n20
    int rank_p1_m1_p1 = ranks_list_1D(20);
    if (rank_p1_m1_p1 >= 0) {

      PetscScalar buffer_coords[6];
      buffer_coords[0] = X_ur - buffer_width;
      buffer_coords[1] = Y_ll;
      buffer_coords[2] = Z_ur - buffer_width;

      buffer_coords[3] = X_ur;
      buffer_coords[4] = Y_ll + buffer_width;
      buffer_coords[5] = Z_ur;

      if (In_Out_Cell(mean_q_i, buffer_coords)) {  //!
        idx_ghost[num_ghost] = site_i;
        rank_ghost[num_ghost] = rank_p1_m1_p1;
        num_ghost++;
      }
    }

    // Direction (-1 0 1): n21
    int rank_m1_0_p1 = ranks_list_1D(21);
    if (rank_m1_0_p1 >= 0) {

      PetscScalar buffer_coords[6];
      buffer_coords[0] = X_ll;
      buffer_coords[1] = Y_ll;
      buffer_coords[2] = Z_ur - buffer_width;

      buffer_coords[3] = X_ll + buffer_width;
      buffer_coords[4] = Y_ur;
      buffer_coords[5] = Z_ur;

      if (In_Out_Cell(mean_q_i, buffer_coords)) {  //!
        idx_ghost[num_ghost] = site_i;
        rank_ghost[num_ghost] = rank_m1_0_p1;
        num_ghost++;
      }
    }

    // Direction (0 0 1): n22
    int rank_0_0_p1 = ranks_list_1D(22);
    if (rank_0_0_p1 >= 0) {

      PetscScalar buffer_coords[6];
      buffer_coords[0] = X_ll;
      buffer_coords[1] = Y_ll;
      buffer_coords[2] = Z_ur - buffer_width;

      buffer_coords[3] = X_ur;
      buffer_coords[4] = Y_ur;
      buffer_coords[5] = Z_ur;

      if (In_Out_Cell(mean_q_i, buffer_coords)) {  //!
        idx_ghost[num_ghost] = site_i;
        rank_ghost[num_ghost] = rank_0_0_p1;
        num_ghost++;
      }
    }

    // Direction (1 0 1): n23
    int rank_p1_0_p1 = ranks_list_1D(23);
    if (rank_p1_0_p1 >= 0) {

      PetscScalar buffer_coords[6];
      buffer_coords[0] = X_ur - buffer_width;
      buffer_coords[1] = Y_ll;
      buffer_coords[2] = Z_ur - buffer_width;

      buffer_coords[3] = X_ur;
      buffer_coords[4] = Y_ur;
      buffer_coords[5] = Z_ur;

      if (In_Out_Cell(mean_q_i, buffer_coords)) {  //!
        idx_ghost[num_ghost] = site_i;
        rank_ghost[num_ghost] = rank_p1_0_p1;
        num_ghost++;
      }
    }

    // Direction (-1 1 1): n24
    int rank_m1_p1_p1 = ranks_list_1D(24);
    if (rank_m1_p1_p1 >= 0) {

      PetscScalar buffer_coords[6];
      buffer_coords[0] = X_ll;
      buffer_coords[1] = Y_ur - buffer_width;
      buffer_coords[2] = Z_ur - buffer_width;

      buffer_coords[3] = X_ll + buffer_width;
      buffer_coords[4] = Y_ur;
      buffer_coords[5] = Z_ur;

      if (In_Out_Cell(mean_q_i, buffer_coords)) {  //!
        idx_ghost[num_ghost] = site_i;
        rank_ghost[num_ghost] = rank_m1_p1_p1;
        num_ghost++;
      }
    }

    // Direction (0 1 1): n25
    int rank_0_p1_p1 = ranks_list_1D(25);
    if (rank_0_p1_p1 >= 0) {

      PetscScalar buffer_coords[6];
      buffer_coords[0] = X_ll;
      buffer_coords[1] = Y_ur - buffer_width;
      buffer_coords[2] = Z_ur - buffer_width;

      buffer_coords[3] = X_ur;
      buffer_coords[4] = Y_ur;
      buffer_coords[5] = Z_ur;

      if (In_Out_Cell(mean_q_i, buffer_coords)) {  //!
        idx_ghost[num_ghost] = site_i;
        rank_ghost[num_ghost] = rank_0_p1_p1;
        num_ghost++;
      }
    }

    // Direction (1 1 1): n26
    int rank_p1_p1_p1 = ranks_list_1D(26);
    if (rank_p1_p1_p1 >= 0) {

      PetscScalar buffer_coords[6];
      buffer_coords[0] = X_ur - buffer_width;
      buffer_coords[1] = Y_ur - buffer_width;
      buffer_coords[2] = Z_ur - buffer_width;

      buffer_coords[3] = X_ur;
      buffer_coords[4] = Y_ur;
      buffer_coords[5] = Z_ur;

      if (In_Out_Cell(mean_q_i, buffer_coords)) {  //!
        idx_ghost[num_ghost] = site_i;
        rank_ghost[num_ghost] = rank_p1_p1_p1;
        num_ghost++;
      }
    }
  }

  PetscCall(DMSwarmRestoreField(Simulation->atomistic_data,
                                DMSwarmPICField_coor, NULL, NULL,
                                (void**)&mean_q_ptr));

#ifdef DEBUG_MODE
  std::cout << "Number of ghost atoms: " << num_ghost
            << " from rank: " << rank_MPI << std::endl;
#endif

  PetscCall(DMSwarmAddNPoints(Simulation->atomistic_data, num_ghost));

  //! Initialize new points
  PetscCall(DMSwarmGetField(Simulation->atomistic_data, "idx", NULL, NULL,
                            (void**)&idx_ptr));

  PetscCall(DMSwarmGetField(Simulation->atomistic_data, "specie", NULL, NULL,
                            (void**)&specie_ptr));

  PetscCall(DMSwarmGetField(Simulation->atomistic_data, DMSwarmField_rank, NULL,
                            NULL, (void**)&swarm_rank_ptr));

  PetscCall(DMSwarmGetField(Simulation->atomistic_data, "ghost", NULL, NULL,
                            (void**)&ghost_ptr));

  PetscCall(DMSwarmGetField(Simulation->atomistic_data, "MPI-rank", NULL, NULL,
                            (void**)&rank_ptr));

  PetscCall(DMSwarmGetField(Simulation->atomistic_data, "idx-diff", NULL, NULL,
                            (void**)&idx_diff_ptr));

  PetscCall(DMSwarmGetField(Simulation->atomistic_data, DMSwarmPICField_coor,
                            NULL, NULL, (void**)&mean_q_ptr));

  PetscCall(DMSwarmGetField(Simulation->atomistic_data, "stdv-q", NULL, NULL,
                            (void**)&stdv_q_ptr));

  PetscCall(DMSwarmGetField(Simulation->atomistic_data, "molar-fraction", NULL,
                            NULL, (void**)&xi_ptr));

  PetscCall(DMSwarmGetField(Simulation->atomistic_data, "beta", NULL, NULL,
                            (void**)&beta_ptr));

  PetscCall(DMSwarmGetField(Simulation->atomistic_data, "gamma", NULL, NULL,
                            (void**)&gamma_ptr));

  for (int idx = 0; idx < num_ghost; idx++) {

    int loc_site_i = idx_ghost[idx];
    int ghost_i = n_local_size + idx;

    swarm_rank_ptr[ghost_i] = rank_ghost[idx];
    ghost_ptr[ghost_i] = 1;
    rank_ptr[ghost_i] = rank_MPI;
    idx_ptr[ghost_i] = idx_ptr[loc_site_i];
    idx_diff_ptr[ghost_i] = idx_diff_ptr[loc_site_i];
    specie_ptr[ghost_i] = specie_ptr[loc_site_i];

    for (int alpha = 0; alpha < dim; alpha++) {
      mean_q_ptr[ghost_i * dim + alpha] = mean_q_ptr[loc_site_i * dim + alpha];
    }
    stdv_q_ptr[ghost_i] = stdv_q_ptr[loc_site_i];
    xi_ptr[ghost_i] = xi_ptr[loc_site_i];
    beta_ptr[ghost_i] = beta_ptr[loc_site_i];
    gamma_ptr[ghost_i] = gamma_ptr[loc_site_i];
  }

  PetscCall(DMSwarmRestoreField(Simulation->atomistic_data, "idx", NULL, NULL,
                                (void**)&idx_ptr));

  PetscCall(DMSwarmRestoreField(Simulation->atomistic_data, "idx-diff", NULL,
                                NULL, (void**)&idx_diff_ptr));

  PetscCall(DMSwarmRestoreField(Simulation->atomistic_data,
                                DMSwarmPICField_coor, NULL, NULL,
                                (void**)&mean_q_ptr));

  PetscCall(DMSwarmRestoreField(Simulation->atomistic_data, "ghost", NULL, NULL,
                                (void**)&ghost_ptr));

  PetscCall(DMSwarmRestoreField(Simulation->atomistic_data, "MPI-rank", NULL,
                                NULL, (void**)&rank_ptr));

  PetscCall(DMSwarmRestoreField(Simulation->atomistic_data, "specie", NULL,
                                NULL, (void**)&specie_ptr));

  PetscCall(DMSwarmRestoreField(Simulation->atomistic_data, DMSwarmField_rank,
                                NULL, NULL, (void**)&swarm_rank_ptr));

  PetscCall(DMSwarmRestoreField(Simulation->atomistic_data, "stdv-q", NULL,
                                NULL, (void**)&stdv_q_ptr));

  PetscCall(DMSwarmRestoreField(Simulation->atomistic_data, "molar-fraction",
                                NULL, NULL, (void**)&xi_ptr));

  PetscCall(DMSwarmRestoreField(Simulation->atomistic_data, "beta", NULL, NULL,
                                (void**)&beta_ptr));

  PetscCall(DMSwarmRestoreField(Simulation->atomistic_data, "gamma", NULL, NULL,
                                (void**)&gamma_ptr));

  //! Migrate new points
  PetscCall(DMSwarmMigrate(Simulation->atomistic_data, PETSC_TRUE));

  //! Destroy auxiliar variables
  free(idx_ghost);
  free(rank_ghost);

  //! Get new local size
  PetscInt n_sites_local_ghosted;
  PetscCall(
      DMSwarmGetLocalSize(Simulation->atomistic_data, &n_sites_local_ghosted));

  //! Get atomistic coordinates
  PetscCall(DMSwarmGetField(Simulation->atomistic_data, DMSwarmPICField_coor,
                            NULL, NULL, (void**)&mean_q_ptr));
  Eigen::Map<MatrixType> mean_q_new(mean_q_ptr, n_sites_local_ghosted, dim);

  //!  Apply periodic boundary conditions over the ghost atoms
  for (int site_i = n_local_size; site_i < n_sites_local_ghosted; site_i++) {

    //! Search neighbourhs in the periodic cells
    unsigned int num_sym_box = 27;
    for (unsigned box_idx = 0; box_idx < num_sym_box; box_idx++) {

      //! Update site position with the crystal direction
      Eigen::Vector3d mean_q_i = mean_q_new.block<1, 3>(site_i, 0);
      apply_periodic_kinematic_restrictions_Bn(mean_q_i, (Miller_Index)box_idx);

      //! Check if the site is inside the domain
      if ((crystal_directions[box_idx] == true) &&  //!
          (mean_q_i(0) >= X_ll - buffer_width) &&   //!
          (mean_q_i(0) < X_ur + buffer_width) &&    //!
          (mean_q_i(1) >= Y_ll - buffer_width) &&   //!
          (mean_q_i(1) < Y_ur + buffer_width) &&    //!
          (mean_q_i(2) >= Z_ll - buffer_width) &&   //!
          (mean_q_i(2) < Z_ur + buffer_width)) {

        mean_q_ptr[site_i * dim + 0] = mean_q_i(0);
        mean_q_ptr[site_i * dim + 1] = mean_q_i(1);
        mean_q_ptr[site_i * dim + 2] = mean_q_i(2);

        break;
      }
    }
  }

  //! Restore atomistic coordinates
  PetscCall(DMSwarmRestoreField(Simulation->atomistic_data,
                                DMSwarmPICField_coor, NULL, NULL,
                                (void**)&mean_q_ptr));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/********************************************************************************/

PetscErrorCode DMSwarmDestroyGhostAtoms(DMD* Simulation) {

  PetscFunctionBeginUser;

  //! Get global size
  PetscInt n_global_size = 0;
  PetscCall(DMSwarmGetSize(Simulation->atomistic_data, &n_global_size));

  //! Get local size
  PetscInt n_local_size = 0;
  PetscCall(DMSwarmGetLocalSize(Simulation->atomistic_data, &n_local_size));

  //!
  PetscInt num_remove_point = 0;

  PetscInt* swarm_rank_ptr;
  PetscCall(DMSwarmGetField(Simulation->atomistic_data, DMSwarmField_rank, NULL,
                            NULL, (void**)&swarm_rank_ptr));

  PetscInt* rank_ptr;
  PetscCall(DMSwarmGetField(Simulation->atomistic_data, "MPI-rank", NULL, NULL,
                            (void**)&rank_ptr));

  for (int site_i = 0; site_i < n_local_size; site_i++) {

    if ((swarm_rank_ptr[site_i] != rank_ptr[site_i]) &&
        (swarm_rank_ptr[site_i] == rank_MPI)) {
      num_remove_point += 1;
    }

    if ((swarm_rank_ptr[site_i] != rank_ptr[site_i]) &&
        (rank_ptr[site_i] != rank_MPI)) {
      swarm_rank_ptr[site_i] = rank_ptr[site_i];
    }
  }

  PetscCall(DMSwarmRestoreField(Simulation->atomistic_data, DMSwarmField_rank,
                                NULL, NULL, (void**)&swarm_rank_ptr));
  PetscCall(DMSwarmRestoreField(Simulation->atomistic_data, "MPI-rank", NULL,
                                NULL, (void**)&rank_ptr));

  for (int site_i = 0; site_i < num_remove_point; site_i++) {
    PetscCall(DMSwarmRemovePoint(Simulation->atomistic_data));
  }

#ifdef DEBUG_MODE
  std::cout << "Number of removed ghost atoms: " << num_remove_point
            << " at rank: " << rank_MPI << std::endl;
#endif

  PetscFunctionReturn(PETSC_SUCCESS);
}

/********************************************************************************/

PetscErrorCode DMSwarmMigrateGhostField(PetscInt n_local,           //!
                                        PetscInt n_ghost,           //!
                                        PetscInt dim,               //!
                                        const PetscInt* idx_ghost,  //!
                                        PetscScalar* X_ptr)         //!
{

  PetscFunctionBegin;

  PetscInt n_dof_local = dim * n_local;
  PetscInt n_dof_ghost = dim * n_ghost;

  Vec X;
  PetscInt* idx_dof_ghost = (PetscInt*)malloc(n_dof_ghost * sizeof(PetscInt));

  for (int i = 0; i < n_ghost; i++) {
    for (int j = 0; j < dim; j++) {
      idx_dof_ghost[i * dim + j] = idx_ghost[i] * dim + j;
    }
  }

  PetscCall(VecCreateGhostWithArray(PETSC_COMM_WORLD, n_dof_local,
                                    PETSC_DETERMINE, n_dof_ghost, idx_dof_ghost,
                                    X_ptr, &X));
  PetscCall(VecGhostUpdateBegin(X, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecGhostUpdateEnd(X, INSERT_VALUES, SCATTER_FORWARD));

  PetscCall(VecDestroy(&X));
  free(idx_dof_ghost);

  PetscFunctionReturn(PETSC_SUCCESS);
}

/********************************************************************************/

static PetscErrorCode GetElementCoords(const PetscScalar _coords[],
                                       const PetscInt e2n[],
                                       PetscScalar el_coords[]) {
  PetscInt dim = NumberDimensions;
  PetscInt n_nodes = 8;

  PetscFunctionBeginUser;
  /* get coords for the element */
  for (PetscInt i = 0; i < n_nodes; i++) {
    for (PetscInt d = 0; d < dim; d++)
      el_coords[dim * i + d] = _coords[dim * e2n[i] + d];
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/********************************************************************************/

static bool In_Out_Cell(const Eigen::Vector3d mean_q,
                        const PetscScalar el_coords[]) {

  unsigned int dim = NumberDimensions;

  //! Lower left coordinates of the brick
  PetscReal xI_lw, yI_lw, zI_lw;
  xI_lw = el_coords[0];
  yI_lw = el_coords[1];
  zI_lw = el_coords[2];

  //! Upper right coordinates of the brick
  PetscReal xI_up, yI_up, zI_up;
  xI_up = el_coords[3];
  yI_up = el_coords[4];
  zI_up = el_coords[5];

  if ((mean_q(0) >= xI_lw) &&  //!
      (mean_q(0) < xI_up) &&   //!
      (mean_q(1) >= yI_lw) &&  //!
      (mean_q(1) < yI_up) &&   //!
      (mean_q(2) >= zI_lw) &&  //!
      (mean_q(2) < zI_up)) {
    return true;
  } else {
    return false;
  }
}

/********************************************************************************/