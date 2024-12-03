/**
 * @file Atoms/Atom.hpp
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2022-07-23
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef ATOM_HPP
#define ATOM_HPP

#include "Macros.hpp"
#include <vector>

using namespace std;

/**
 * @brief
 *
 * @param Simulation
 * @param Simulation_file
 * @return PetscErrorCode
 */
PetscErrorCode init_DMD_simulation(DMD *Simulation, dump_file Simulation_file);

/**
 * @brief
 *
 * @param Simulation
 */
PetscErrorCode destroy_DMD_simulation(DMD *Simulation);

#endif /* ATOM_HPP */