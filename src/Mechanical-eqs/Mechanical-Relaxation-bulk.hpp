/**
 * @file Mechanical-Relaxation-bulk.hpp
 * @author Miguel Molinos ([migmolper](https://github.com/migmolper))
 * @brief This function performs the mechanical relaxation of a periodic sistem
 * of atomic positions under zero temperature. This function provides a
 * PETSc-interface to solve the following set of Euler-Lagrange equations:
 * dV_dq_i = 0 Where q_i stands for the position i and V is the potential of the
 * system
 * @version 0.1
 * @date 2022-11-23
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef MECHANICAL_RELAXATION_BULK_HPP
#define MECHANICAL_RELAXATION_BULK_HPP

#include <petscsnes.h>

/**
 * @brief This function performs the mechanical relaxation of a periodic sistem
 * of atomic positions under zero temperature.
 *
 * @param Simulation Structure containing the information of the atomistic
 * lattice
 * @param system_equations Usefull equations
 * @return PetscErrorCode
 */
PetscErrorCode mechanical_relaxation_bulk(DMD *Simulation,
                                          dmd_equations system_equations);

#endif /* MECHANICAL_RELAXATION_BULK_HPP */