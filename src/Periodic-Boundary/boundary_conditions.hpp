/**
 * @file boundary_conditions.hpp
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2022-07-23
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef BOUNDARY_CONDITIONS_HPP
#define BOUNDARY_CONDITIONS_HPP

#include "Macros.hpp"
#include <Eigen/Dense>

/**
 * @brief Set the periodic boundary conditions object
 *
 * @param bx: Boundary condition in the x direction
 * @param by: Boundary condition in the y direction
 * @param bz: Boundary condition in the z direction
 * @return PetscErrorCode
 */
PetscErrorCode set_periodic_boundary_conditions(DMBoundaryType bx,
                                                DMBoundaryType by,
                                                DMBoundaryType bz);

/**
 * @brief Modify the value of q to match periodic bcc in the current
 * configuration
 *
 * @param q_n Mean position (central box) in the current configuration
 * @param box_idx Box index of the site
 */
void apply_periodic_kinematic_restrictions_Bn(Eigen::Vector3d &q_n,
                                              Miller_Index box_idx);

#endif /* BOUNDARY_CONDITIONS_HPP */
