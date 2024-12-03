/**
 * @file Topology.hpp
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

/*******************************************************/

/**
 * @brief
 *
 * @param atom_topology
 * @param mechanical_neighs_idx
 * @return PetscErrorCode
 */
PetscErrorCode read_atom_topology(AtomTopology *atom_topology,
                                  IS mechanical_neighs_idx);

/**
 * @brief
 *
 * @param atom_topology
 * @param mechanical_neighs_idx
 * @return PetscErrorCode
 */
PetscErrorCode restore_atom_topology(AtomTopology *atom_topology,
                                     IS mechanical_neighs_idx);

/**
 * @brief
 *
 * @param Simulation
 * @return PetscErrorCode
 */
PetscErrorCode list_of_active_mechanical_sites_MgHx(DMD *Simulation);

/**
 * @brief
 *
 * @param Simulation
 * @return PetscErrorCode
 */
PetscErrorCode list_of_active_mechanical_sites_AlCu(DMD *Simulation);

/**
 * @brief
 *
 * @param Simulation
 * @return PetscErrorCode
 */
PetscErrorCode list_of_active_diffusive_sites_MgHx(DMD *Simulation);

/*******************************************************/