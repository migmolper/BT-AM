/**
 * @file Atoms/Neighbors.hpp
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2022-07-23
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef NEIGHBORS_HPP
#define NEIGHBORS_HPP

#include "Macros.hpp"
#include <Eigen/Dense>

/**
 * @brief
 *
 * @param atoms Structure with the information of the lattice
 * @return ERROR/SUCCESS
 */

/**
 * @brief Function to compute the list of neighbors for the all simulation
 *
 * @param Simulation
 * @return PetscErrorCode
 */
PetscErrorCode neighbors(DMD *Simulation);

/**
 * @brief
 *
 * @param Simulation
 * @return PetscErrorCode
 */
PetscErrorCode list_of_diffusive_neighs(DMD *Simulation);

/**
 * @brief
 *
 * @param Simulation
 * @return PetscErrorCode
 */
PetscErrorCode destroy_diffusive_topology(DMD *Simulation);

/**
 * @brief
 *
 * @param Simulation
 * @return PetscErrorCode
 */
PetscErrorCode destroy_mechanical_topology(DMD *Simulation);

#endif /* NEIGHBORS_HPP */
