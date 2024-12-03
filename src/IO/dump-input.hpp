/**
 * @file dump-input.hpp
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2023-12-12
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef READ_DUMP_INFORMATION_HPP
#define READ_DUMP_INFORMATION_HPP

#include "Macros.hpp"

dump_file read_dump_information(const char *SimulationFile);

void free_dump_information(dump_file *Simulation_data);

#endif /* READ_DUMP_INFORMATION_HPP */