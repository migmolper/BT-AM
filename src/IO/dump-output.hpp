/**
 * @file ovito-output.hpp
 * @author J.P. Mendez and M.Molinos ([migmolper](https://github.com/migmolper))
 * @brief
 * @version 0.1
 * @date 2022-07-23
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef OUTPUT_DUMP_HPP
#define OUTPUT_DUMP_HPP

#include "Macros.hpp"

/**
 * @brief Save usefull data to restart the simulation
 *
 * @param step Current time step
 * @param file_name Name of the dump file
 * @param atoms Atomic data
 * @return int
 */
PetscErrorCode write_dump_information(int step, const char *Name_file_t,
                                      DMD *Simulation);

/********************************************************/

/**
 * @brief  Write the file restart in parallel, using MPI.
 *
 * @param atoms
 */
PetscErrorCode Write_restart_mpi(int step, const char *Name_file_t,
                                 DMD *Simulation);

/********************************************************/

#ifdef USE_MPI
/**
 * @brief  Convert the binary file (q_w_x_binary) of output to a text file.
 *
 */
void ConvertBinaryFiletoTextFile();
#endif
/********************************************************/

#ifdef USE_MPI
/**
 * @brief Convert the binary file of output
 *
 */
void ConvertBinaryFiletoTextFile_Petsc();
#endif
/********************************************************/

#endif /* OUTPUT_DUMP_HPP */
