/**
 * @file boundary_conditions.cpp
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2022-07-23
 *
 * @copyright Copyright (c) 2022
 *
 */

#include "Periodic-Boundary/boundary_conditions.hpp"
#include "Atoms/Atom.hpp"
#include "Atoms/Neighbors.hpp"
#include "Macros.hpp"
#include "petscsys.h"
#include <Eigen/Dense>
#include <iostream>
#include <math.h>
#include <petscsystypes.h>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

extern bool crystal_directions[27];

extern Eigen::Vector3d lattice_x_B0;
extern Eigen::Vector3d lattice_y_B0;
extern Eigen::Vector3d lattice_z_B0;

extern Eigen::Vector3d lattice_x_Bn;
extern Eigen::Vector3d lattice_y_Bn;
extern Eigen::Vector3d lattice_z_Bn;

extern Eigen::Vector3d box_origin_0;

const double matrix_translation_XYZ[81] = {0,  0,  0,   //! 0
                                           -1, -1, -1,  //! 1
                                           -1, -1, 0,   //! 2
                                           -1, -1, 1,   //! 3
                                           -1, 0,  -1,  //! 4
                                           -1, 0,  0,   //! 5
                                           -1, 0,  1,   //! 6
                                           -1, 1,  -1,  //! 7
                                           -1, 1,  0,   //! 8
                                           -1, 1,  1,   //! 9
                                           0,  -1, -1,  //! 10
                                           0,  -1, 0,   //! 11
                                           0,  -1, 1,   //! 12
                                           0,  0,  -1,  //! 13
                                           0,  0,  1,   //! 14
                                           0,  1,  -1,  //! 15
                                           0,  1,  0,   //! 16
                                           0,  1,  1,   //! 17
                                           1,  -1, -1,  //! 18
                                           1,  -1, 0,   //! 19
                                           1,  -1, 1,   //! 20
                                           1,  0,  -1,  //! 21
                                           1,  0,  0,   //! 22
                                           1,  0,  1,   //! 23
                                           1,  1,  -1,  //! 24
                                           1,  1,  0,   //! 25
                                           1,  1,  1};  //! 26

/********************************************************************************/

PetscErrorCode set_periodic_boundary_conditions(DMBoundaryType bx,
                                                DMBoundaryType by,
                                                DMBoundaryType bz) {

  PetscFunctionBegin;

  //! Initialize the crystal directions
  unsigned int num_sym_box = 27;
  for (unsigned int idx = 0; idx < num_sym_box; idx++) {
    crystal_directions[idx] = false;
  }

  //! Set the central box
  crystal_directions[idx_0_0_0] = true;

  //! Apply periodic boundary conditions in the x direction
  if (bx == DM_BOUNDARY_PERIODIC) {

    crystal_directions[idx_m1_0_0] = true;
    crystal_directions[idx_p1_0_0] = true;
  }

  //! Apply periodic boundary conditions in the y direction
  if (by == DM_BOUNDARY_PERIODIC) {

    crystal_directions[idx_0_m1_0] = true;
    crystal_directions[idx_0_p1_0] = true;
  }

  //! Apply periodic boundary conditions in the z direction
  if (bz == DM_BOUNDARY_PERIODIC) {

    crystal_directions[idx_0_0_m1] = true;
    crystal_directions[idx_0_0_p1] = true;
  }

  //! Apply periodic boundary conditions in the xy direction
  if ((bx == DM_BOUNDARY_PERIODIC) &&  //!
      (by == DM_BOUNDARY_PERIODIC)) {

    crystal_directions[idx_m1_m1_0] = true;
    crystal_directions[idx_m1_p1_0] = true;

    crystal_directions[idx_p1_m1_0] = true;
    crystal_directions[idx_p1_p1_0] = true;
  }

  //! Apply periodic boundary conditions in the xz direction
  if ((bx == DM_BOUNDARY_PERIODIC) &&  //!
      (bz == DM_BOUNDARY_PERIODIC)) {

    crystal_directions[idx_m1_0_m1] = true;
    crystal_directions[idx_m1_0_p1] = true;

    crystal_directions[idx_p1_0_m1] = true;
    crystal_directions[idx_p1_0_p1] = true;
  }

  //! Apply periodic boundary conditions in the yz direction
  if ((by == DM_BOUNDARY_PERIODIC) &&  //!
      (bz == DM_BOUNDARY_PERIODIC)) {

    crystal_directions[idx_0_m1_m1] = true;
    crystal_directions[idx_0_m1_p1] = true;

    crystal_directions[idx_0_p1_m1] = true;
    crystal_directions[idx_0_p1_p1] = true;
  }

  //! Apply periodic boundary conditions in the xyz direction
  if ((bx == DM_BOUNDARY_PERIODIC) &&  //!
      (by == DM_BOUNDARY_PERIODIC) &&  //!
      (bz == DM_BOUNDARY_PERIODIC)) {

    crystal_directions[idx_m1_m1_m1] = true;
    crystal_directions[idx_m1_m1_p1] = true;

    crystal_directions[idx_m1_p1_m1] = true;
    crystal_directions[idx_m1_p1_p1] = true;

    crystal_directions[idx_p1_m1_m1] = true;
    crystal_directions[idx_p1_m1_p1] = true;

    crystal_directions[idx_p1_p1_m1] = true;
    crystal_directions[idx_p1_p1_p1] = true;
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/********************************************************************************/

void apply_periodic_kinematic_restrictions_Bn(Eigen::Vector3d& q_n,
                                              Miller_Index box_idx) {

  unsigned int dim = NumberDimensions;

  Eigen::Matrix3d periodic_matrix_translation = Eigen::Matrix3d::Zero();

  periodic_matrix_translation(0, 0) = matrix_translation_XYZ[box_idx * dim + 0];
  periodic_matrix_translation(1, 1) = matrix_translation_XYZ[box_idx * dim + 1];
  periodic_matrix_translation(2, 2) = matrix_translation_XYZ[box_idx * dim + 2];

  Eigen::Vector3d dq = periodic_matrix_translation *
                       (lattice_x_Bn + lattice_y_Bn + lattice_z_Bn);

  q_n = q_n + dq;
}

/********************************************************************************/
