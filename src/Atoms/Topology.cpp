/**
 * @file Topology.cpp
 * @author Miguel Molinos (@migmolper)
 * @brief
 * @version 0.1
 * @date 2024-08-02
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "Macros.hpp"
#include "petscis.h"
#include <petscerror.h>
#include <petscsystypes.h>

/********************************************************************************/

PetscErrorCode read_atom_topology(AtomTopology* atom_topology,
                                  IS mechanical_neighs_idx) {

  PetscFunctionBegin;

  PetscCall(ISGetSize(mechanical_neighs_idx, &atom_topology->numneigh));

  if (atom_topology->numneigh <= 0) {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "No mechanical neighbors");
    PetscFunctionReturn(PETSC_ERR_ARG_OUTOFRANGE);
  }

  //! Get the indices of the mechanical neighbors
  PetscCall(
      ISGetIndices(mechanical_neighs_idx, &atom_topology->mech_neighs_ptr));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/********************************************************************************/

PetscErrorCode restore_atom_topology(AtomTopology* atom_topology,
                                     IS mechanical_neighs_idx) {

  PetscFunctionBegin;

  PetscCall(
      ISRestoreIndices(mechanical_neighs_idx, &atom_topology->mech_neighs_ptr));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/********************************************************************************/

PetscErrorCode list_of_active_mechanical_sites_MgHx(DMD* Simulation) {

  PetscFunctionBegin;

  unsigned int dim = NumberDimensions;

  //! Get the local size
  PetscInt n_sites_local = Simulation->n_sites_local;

  //! Get the local size (ghosted)
  PetscInt n_sites_local_ghosted;
  PetscCall(
      DMSwarmGetLocalSize(Simulation->atomistic_data, &n_sites_local_ghosted));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Get copies of the molar fraction vector all over the ranks
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscScalar* xi_ptr;
  PetscCall(DMSwarmGetField(Simulation->atomistic_data, "molar-fraction", NULL,
                            NULL, (void**)&xi_ptr));
  Eigen::Map<VectorType> xi(xi_ptr, n_sites_local_ghosted);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Create the list of chemically active sites.
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscInt counter_mechanical_sites = 0;
  PetscInt n_mechanical_sites_local = 0;
  PetscInt n_mechanical_sites_ghost = 0;

  PetscInt* mechanical_sites_ptr =
      (PetscInt*)calloc(n_sites_local_ghosted, sizeof(PetscInt));

  //! Search chemical neighbors in the main cell
  for (PetscInt site_i = 0; site_i < n_sites_local_ghosted; site_i++) {

    //! If the mass is larger than a threshold, the site i is active
    if (xi(site_i) > min_occupancy) {
      mechanical_sites_ptr[counter_mechanical_sites] = site_i;
      counter_mechanical_sites++;

      //! Count the number of mechanical sites inside the local domain
      //! (excluding ghost sites)
      if (site_i < n_sites_local) {
        n_mechanical_sites_local++;
      }

      //! Count the number of mechanical sites inside the local domain
      //! (excluding ghost sites)
      if (site_i >= n_sites_local) {
        n_mechanical_sites_ghost++;
      }
    }
  }

  //! Number of mechanical sites in the local region
  Simulation->n_mechanical_sites_local = n_mechanical_sites_local;
  Simulation->n_mechanical_sites_ghost = n_mechanical_sites_ghost;

  //! Create list of mechanical sites (local + ghost)
  IS mechanical_sites;
  PetscCall(ISCreateGeneral(PETSC_COMM_WORLD, counter_mechanical_sites,
                            mechanical_sites_ptr, PETSC_COPY_VALUES,
                            &mechanical_sites));

  Simulation->active_mech_sites = mechanical_sites;

  free(mechanical_sites_ptr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Restore molar fraction data
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMSwarmRestoreField(Simulation->atomistic_data, "molar-fraction",
                                NULL, NULL, (void**)&xi_ptr));

#ifdef DEBUG_MODE
  ISView(mechanical_sites, PETSC_VIEWER_STDOUT_WORLD);
#endif

  PetscFunctionReturn(PETSC_SUCCESS);
}

/********************************************************************************/

PetscErrorCode list_of_active_mechanical_sites_AlCu(DMD* Simulation) {

  PetscFunctionBegin;

  unsigned int dim = NumberDimensions;

  //! Get the local size
  PetscInt n_sites_local = Simulation->n_sites_local;

  //! Get the local size (ghosted)
  PetscInt n_sites_local_ghosted;
  PetscCall(
      DMSwarmGetLocalSize(Simulation->atomistic_data, &n_sites_local_ghosted));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Create the list of chemically active sites.
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscInt counter_mechanical_sites = 0;
  PetscInt n_mechanical_sites_local = 0;
  PetscInt n_mechanical_sites_ghost = 0;

  PetscInt* mechanical_sites_ptr =
      (PetscInt*)calloc(n_sites_local_ghosted, sizeof(PetscInt));

  //! Search chemical neighbors in the main cell
  for (PetscInt site_i = 0; site_i < n_sites_local_ghosted; site_i++) {

    mechanical_sites_ptr[counter_mechanical_sites] = site_i;
    counter_mechanical_sites++;

    //! Count the number of mechanical sites inside the local domain
    //! (excluding ghost sites)
    if (site_i < n_sites_local) {
      n_mechanical_sites_local++;
    }

    //! Count the number of mechanical sites inside the local domain
    //! (excluding ghost sites)
    if (site_i >= n_sites_local) {
      n_mechanical_sites_ghost++;
    }
  }

  //! Number of mechanical sites in the local region
  Simulation->n_mechanical_sites_local = n_mechanical_sites_local;
  Simulation->n_mechanical_sites_ghost = n_mechanical_sites_ghost;

  //! Create list of mechanical sites (local + ghost)
  IS mechanical_sites;
  PetscCall(ISCreateGeneral(PETSC_COMM_WORLD, counter_mechanical_sites,
                            mechanical_sites_ptr, PETSC_COPY_VALUES,
                            &mechanical_sites));

  Simulation->active_mech_sites = mechanical_sites;

  free(mechanical_sites_ptr);

#ifdef DEBUG_MODE
  ISView(mechanical_sites, PETSC_VIEWER_STDOUT_WORLD);
#endif

  PetscFunctionReturn(PETSC_SUCCESS);
}

/********************************************************************************/

PetscErrorCode list_of_active_diffusive_sites_MgHx(DMD* Simulation) {

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
  Eigen::Map<MatrixType> mean_q(mean_q_ptr, n_sites_local_ghosted, dim);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Get copies of the molar fraction vector all over the ranks
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscScalar* xi_ptr;
  PetscCall(DMSwarmGetField(Simulation->atomistic_data, "molar-fraction", NULL,
                            NULL, (void**)&xi_ptr));
  Eigen::Map<VectorType> xi(xi_ptr, n_sites_local_ghosted);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Get the indx of the diffusive atoms
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscInt* idx_diff_ptr;
  PetscCall(DMSwarmGetField(Simulation->atomistic_data, "idx-diff", NULL, NULL,
                            (void**)&idx_diff_ptr));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Get index of the particles
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscInt* idx_ptr;
  PetscCall(DMSwarmGetField(Simulation->atomistic_data, "idx", NULL, NULL,
                            (void**)&idx_ptr));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Create the list of chemically active sites.
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscInt counter_diffusive_sites = 0;
  PetscInt n_diffusive_sites_local = 0;
  PetscInt n_diffusive_sites_ghost = 0;

  PetscInt* diffusive_sites_ptr =
      (PetscInt*)calloc(n_sites_local_ghosted, sizeof(PetscInt));

  //! Search chemical neighbors in the main cell
  for (PetscInt site_i = 0; site_i < n_sites_local_ghosted; site_i++) {

    //! Get site_i information
    double xi_i = xi(site_i);
    PetscInt idx_diff_i = idx_diff_ptr[site_i];
    Eigen::Vector3d mean_q_i = mean_q.block<1, 3>(site_i, 0);

    //! This is not a diffusive site
    if (idx_diff_i == false) {
      continue;
    }

    //! @brief Get topologic information of site i
    AtomTopology atom_i_topology;
    PetscCall(read_atom_topology(&atom_i_topology,
                                 Simulation->mechanical_neighs_idx[site_i]));

    //! Initialize mass counter for the neighs of i
    double mass_neigh_i = xi_i;

    //! Loop neighs
    for (unsigned int idx_j = 0; idx_j < atom_i_topology.numneigh; idx_j++) {

      //! @brief Get atomistic information of site j
      PetscInt site_j = atom_i_topology.mech_neighs_ptr[idx_j];
      PetscInt idx_diff_j = idx_diff_ptr[site_j];
      double xi_j = xi(site_j);
      Eigen::Vector3d mean_q_j = mean_q.block<1, 3>(site_j, 0);

      //! This is not a diffusive site
      if (idx_diff_j == false) {
        continue;
      }

      //! Get distance between site i and site j
      double norm_r_ij = (mean_q_j - mean_q_i).norm();

      if ((norm_r_ij <= r_cutoff_Eb) &&  //!
          (xi_j >= min_occupancy))       //!
      {
        mass_neigh_i += xi_j;
      }
    }

    //! If the mass is larger than a threshold, the site i is diffusive
    if (mass_neigh_i >= 0.01) {
      diffusive_sites_ptr[counter_diffusive_sites] = site_i;
      counter_diffusive_sites++;

      //! Count the number of diffusive sites inside the local domain
      //! (excluding ghost sites)
      if (site_i < n_sites_local) {
        n_diffusive_sites_local++;
      }

      //! Count the number of diffusive sites inside the local domain
      //! (excluding ghost sites)
      if (site_i >= n_sites_local) {
        n_diffusive_sites_ghost++;
      }
    }

    //! @brief Restore atom topology
    PetscCall(restore_atom_topology(&atom_i_topology,
                                    Simulation->mechanical_neighs_idx[site_i]));
  }

  //! Number of diffusive sites in the local region
  Simulation->n_diffusive_sites_local = n_diffusive_sites_local;
  Simulation->n_diffusive_sites_ghost = n_diffusive_sites_ghost;

  //! Create list of diffusive sites (local + ghost)
  IS diffusive_sites;
  PetscCall(ISCreateGeneral(PETSC_COMM_WORLD, counter_diffusive_sites,
                            diffusive_sites_ptr, PETSC_COPY_VALUES,
                            &diffusive_sites));

  Simulation->active_diff_sites = diffusive_sites;

  free(diffusive_sites_ptr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Restore mean-q data
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMSwarmRestoreField(Simulation->atomistic_data,
                                DMSwarmPICField_coor, NULL, NULL,
                                (void**)&mean_q_ptr));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Restore molar fraction data
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMSwarmRestoreField(Simulation->atomistic_data, "molar-fraction",
                                NULL, NULL, (void**)&xi_ptr));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Restore atomic specie data
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMSwarmRestoreField(Simulation->atomistic_data, "idx-diff", NULL,
                                NULL, (void**)&idx_diff_ptr));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Restore idx data
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMSwarmRestoreField(Simulation->atomistic_data, "idx", NULL, NULL,
                                (void**)&idx_ptr));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/********************************************************************************/

/********************************************************************************/