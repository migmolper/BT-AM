/**
 * @file Atoms/Ghosts.hpp
 * @author Miguel Molinos (@migmolper)
 * @brief
 * @version 0.1
 * @date 2024-08-22
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "Macros.hpp"

#ifndef GHOST_HPP
#define GHOST_HPP

/**
 * @brief Add ghost atoms to the simulation
 *
 * @param Simulation
 * @param buffer_width
 * @return PetscErrorCode
 */
PetscErrorCode DMSwarmCreateGhostAtoms(DMD *Simulation, double buffer_width);

/**
 * @brief Remove ghost points
 *
 * @param Simulation
 * @return PetscErrorCode
 */
PetscErrorCode DMSwarmDestroyGhostAtoms(DMD *Simulation);

/**
 * @brief
 *
 * @param n_local
 * @param n_ghost
 * @param dim
 * @param idx_ghost
 * @param X_ptr
 * @return PetscErrorCode
 */
PetscErrorCode DMSwarmMigrateGhostField(PetscInt n_local,          //!
                                        PetscInt n_ghost,          //!
                                        PetscInt dim,              //!
                                        const PetscInt *idx_ghost, //!
                                        PetscScalar *X_ptr);

#endif /* GHOST_HPP */