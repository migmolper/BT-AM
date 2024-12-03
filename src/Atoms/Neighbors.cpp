/**
 * @file Atoms/Neighbors.cpp
 * @author Miguel Molinos ([migmolper](https://github.com/migmolper))
 * @brief
 * @version 0.1
 * @date 2022-07-23
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <cstdlib>
#if __APPLE__
#include <malloc/_malloc.h>
#endif
#include <petscsystypes.h>
#ifdef USE_MPI
#include <mpi.h>
#endif
#include "Atoms/Atom.hpp"
#include "Atoms/Neighbors.hpp"
#include "Atoms/Topology.hpp"
#include "Macros.hpp"
#include "Periodic-Boundary/boundary_conditions.hpp"
#include "petscdmswarm.h"
#include <Eigen/Dense>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

using namespace std;
extern PetscMPIInt size_MPI;
extern PetscMPIInt rank_MPI;
extern bool crystal_directions[27];

/************************************************************************/

PetscErrorCode neighbors(DMD* Simulation) {

  PetscFunctionBegin;

  unsigned int dim = NumberDimensions;

  //! Get the local size
  PetscInt n_sites_local = Simulation->n_sites_local;

  //! Get the local size (ghosted)
  PetscInt n_sites_local_ghosted;
  PetscCall(
      DMSwarmGetLocalSize(Simulation->atomistic_data, &n_sites_local_ghosted));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Get mean position vector
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscScalar* mean_q_ptr;
  PetscCall(DMSwarmGetField(Simulation->atomistic_data, DMSwarmPICField_coor,
                            NULL, NULL, (void**)&mean_q_ptr));
  Eigen::Map<MatrixType> mean_q(mean_q_ptr, n_sites_local_ghosted, 3);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Find neighbors
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  IS* mechanical_neighs_idx;
  PetscCall(PetscMalloc1(n_sites_local_ghosted, &mechanical_neighs_idx));

  for (PetscInt site_i = 0; site_i < n_sites_local_ghosted; site_i++) {

    //! Get mean position of site i
    Eigen::Vector3d mean_q_i = mean_q.block<1, 3>(site_i, 0);

    //! Create auxiliar list to store the diffusive neighbors of the site i
    unsigned int num_neigh_i = 0;
    PetscInt* mech_neighs_idx_i_ptr =
        (PetscInt*)calloc(maxneigh, sizeof(PetscInt));

    //! Search neighbourhs
    for (unsigned site_j = 0; site_j < n_sites_local_ghosted; site_j++) {

      if (site_i == site_j) {
        continue;
      }

      //! Get mean position and specie of site j in the periodic box
      Eigen::Vector3d mean_q_j = mean_q.block<1, 3>(site_j, 0);

      //! Check if site j is the neibourhood of the site i
      double norm_r_ij = (mean_q_i - mean_q_j).norm();

      //!
      if ((norm_r_ij <= r_cutoff_ADP_MgHx) && (num_neigh_i < maxneigh)) {

        mech_neighs_idx_i_ptr[num_neigh_i] = site_j;
        num_neigh_i += 1;

      } else if ((norm_r_ij <= r_cutoff_ADP_MgHx &&
                  (num_neigh_i == maxneigh))) {

        PetscCall(PetscError(PETSC_COMM_WORLD, __LINE__, "neighbors", __FILE__,
                             PETSC_ERR_RETURN, PETSC_ERROR_INITIAL,
                             "Max number of neighbors reached for particle %i",
                             site_i));
        PetscFunctionReturn(PETSC_ERR_RETURN);
      }
    }

    if (num_neigh_i <= 0) {
      PetscCall(PetscError(PETSC_COMM_WORLD, __LINE__, "neighbors", __FILE__,
                           PETSC_ERR_RETURN, PETSC_ERROR_INITIAL,
                           "No neighbors where finded for particle %i",
                           site_i));
      PetscFunctionReturn(PETSC_ERR_RETURN);
    }

    //! Create list of neighbors for the site i
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, num_neigh_i,
                              mech_neighs_idx_i_ptr, PETSC_COPY_VALUES,
                              &mechanical_neighs_idx[site_i]));

    free(mech_neighs_idx_i_ptr);
  }

  //! Set list of diffusive neighs
  Simulation->mechanical_neighs_idx = mechanical_neighs_idx;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Restore data
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMSwarmRestoreField(Simulation->atomistic_data,
                                DMSwarmPICField_coor, NULL, NULL,
                                (void**)&mean_q_ptr));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/********************************************************************************/


PetscErrorCode list_of_diffusive_neighs(DMD* Simulation) {

  PetscFunctionBegin;

  unsigned int dim = NumberDimensions;

  //! Get the local size
  PetscInt n_sites_local = Simulation->n_sites_local;

  //! Get the local size (including ghosts)
  PetscInt n_sites_local_ghosted;
  PetscCall(
      DMSwarmGetLocalSize(Simulation->atomistic_data, &n_sites_local_ghosted));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Get mean position vector
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscScalar* mean_q_ptr;
  PetscCall(DMSwarmGetField(Simulation->atomistic_data, DMSwarmPICField_coor,
                            NULL, NULL, (void**)&mean_q_ptr));
  Eigen::Map<MatrixType> mean_q(mean_q_ptr, n_sites_local_ghosted, dim);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Get index of the particles
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscInt* idx_ptr;
  PetscCall(DMSwarmGetField(Simulation->atomistic_data, "idx", NULL, NULL,
                            (void**)&idx_ptr));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Get the indx of the diffusive atoms
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscInt* idx_diff_ptr;
  PetscCall(DMSwarmGetField(Simulation->atomistic_data, "idx-diff", NULL, NULL,
                            (void**)&idx_diff_ptr));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create a table with the list of chemical
    neighbors of each diffusive site
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  IS* diffusive_neighs_idx;
  PetscCall(PetscMalloc1(n_sites_local_ghosted, &diffusive_neighs_idx));

  for (PetscInt site_i = 0; site_i < n_sites_local_ghosted; site_i++) {

    //! Get site_i information
    PetscInt idx_diff_i = idx_diff_ptr[site_i];
    Eigen::Vector3d mean_q_i = mean_q.block<1, 3>(site_i, 0);

    //! This is not a diffusive site
    if (idx_diff_i == false) {
      continue;
    }

    //! Create auxiliar list to store the diffusive neighbors of the site i
    unsigned int num_neigh_i = 0;
    PetscInt* diff_neighs_idx_i_ptr =
        (PetscInt*)calloc(maxneigh, sizeof(PetscInt));

    //! Search neighbourhs
    for (unsigned site_j = 0; site_j < n_sites_local_ghosted; site_j++) {

      //! @brief Get atomistic information of site j
      PetscInt idx_diff_j = idx_diff_ptr[site_j];
      Eigen::Vector3d mean_q_j = mean_q.block<1, 3>(site_j, 0);

      //! This is the same site
      if (site_i == site_j) {
        continue;
      }

      //! This is not a diffusive site
      if (idx_diff_j == false) {
        continue;
      }

      //! Check if site j is the neibourhood of the site i
      double norm_r_ij = (mean_q_i - mean_q_j).norm();

      //!
      if ((norm_r_ij <= r_cutoff_Eb) && (num_neigh_i < max_chemical_neighs)) {

        diff_neighs_idx_i_ptr[num_neigh_i] = site_j;
        num_neigh_i += 1;

      } else if ((norm_r_ij <= r_cutoff_Eb &&
                  (num_neigh_i == max_chemical_neighs))) {

        PetscCall(PetscError(
            PETSC_COMM_WORLD, __LINE__, "list_of_diffusive_neighs", __FILE__,
            PETSC_ERR_RETURN, PETSC_ERROR_INITIAL,
            "Max number of diffusive neighbors reached for particle %i",
            site_i));
        PetscFunctionReturn(PETSC_ERR_RETURN);
      }
    }

    if (num_neigh_i <= 0) {
      PetscCall(PetscError(
          PETSC_COMM_WORLD, __LINE__, "list_of_diffusive_neighs", __FILE__,
          PETSC_ERR_RETURN, PETSC_ERROR_INITIAL,
          "No diffusive neighbors where finded for particle %i", site_i));
      PetscFunctionReturn(PETSC_ERR_RETURN);
    }

    //! Create list of neighbors for the site i
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, num_neigh_i,
                              diff_neighs_idx_i_ptr, PETSC_COPY_VALUES,
                              &diffusive_neighs_idx[site_i]));

    free(diff_neighs_idx_i_ptr);
  }

  //! Set list of diffusive neighs
  Simulation->diffusive_neighs_idx = diffusive_neighs_idx;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Restore mean-q data
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMSwarmRestoreField(Simulation->atomistic_data,
                                DMSwarmPICField_coor, NULL, NULL,
                                (void**)&mean_q_ptr));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Restore idx data
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMSwarmRestoreField(Simulation->atomistic_data, "idx", NULL, NULL,
                                (void**)&idx_ptr));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Restore list of diffusive sites
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMSwarmRestoreField(Simulation->atomistic_data, "idx-diff", NULL,
                                NULL, (void**)&idx_diff_ptr));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/********************************************************************************/

PetscErrorCode destroy_diffusive_topology(DMD* Simulation) {

  PetscFunctionBegin;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Get the indx of the diffusive atoms
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscInt* idx_diff_ptr;
  PetscCall(DMSwarmGetField(Simulation->atomistic_data, "idx-diff", NULL, NULL,
                            (void**)&idx_diff_ptr));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Delete the list of diffusive neighs
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscInt n_atoms_local = 0;
  PetscCall(DMSwarmGetLocalSize(Simulation->atomistic_data, &n_atoms_local));

  for (PetscInt site_i = 0; site_i < n_atoms_local; site_i++) {
    if (idx_diff_ptr[site_i] == true) {
      PetscCall(ISDestroy(&(Simulation->diffusive_neighs_idx[site_i])));
    }
  }

  PetscCall(PetscFree(Simulation->diffusive_neighs_idx));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Restore list of diffusive sites
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMSwarmRestoreField(Simulation->atomistic_data, "idx-diff", NULL,
                                NULL, (void**)&idx_diff_ptr));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/********************************************************************************/

PetscErrorCode destroy_mechanical_topology(DMD* Simulation) {
  PetscFunctionBegin;

  PetscInt n_atoms_local = 0;
  PetscCall(DMSwarmGetLocalSize(Simulation->atomistic_data, &n_atoms_local));

  //! @brief Topological variables for the thermo-mechanical equations
  for (PetscInt site_i = 0; site_i < n_atoms_local; site_i++) {

    PetscCall(ISDestroy(&(Simulation->mechanical_neighs_idx[site_i])));
  }

  PetscCall(PetscFree(Simulation->mechanical_neighs_idx));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/********************************************************************************/