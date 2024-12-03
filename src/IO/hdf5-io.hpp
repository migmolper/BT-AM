/**
 * @file hdf5-io.hpp
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2024-03-28
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef OUTPUT_HDF5_HPP
#define OUTPUT_HDF5_HPP

#include "Macros.hpp"

/**
 * @brief
 *
 * @param Simulation
 * @param filename
 * @return PetscErrorCode
 */
PetscErrorCode DMSwarmWriteHDF5(DMD *Simulation, const char filename[],
                                PetscFileMode type);

/**
 * @brief
 *
 * @param Simulation
 * @param filename
 * @return PetscErrorCode
 */
PetscErrorCode DMSwarmReadHDF5(DMD *Simulation, const char filename[]);

#endif /* OUTPUT_HDF5_HPP */