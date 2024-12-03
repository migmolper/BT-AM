/**
 * @file vtk_outputs.hpp
 * @author Miguel Molinos ([migmolper](https://github.com/migmolper))
 * @brief
 * @version 0.1
 * @date 2022-06-22
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifdef USE_VTK
#ifndef OUTPUT_VTK_HPP
#define OUTPUT_VTK_HPP

#include <Eigen/Dense>

/**
 * @brief
 *
 * @param step
 * @param adp
 * @return int
 */
int write_vtk(int step, atom *atoms, adpPotential *adp[2][2]);

/**
 * @brief Print in vtk format the sphere of neighbours of i_site
 *
 * @param step current step
 * @param atoms Atomic data
 * @param adp ADP potential
 * @param i_site Index of the current site
 * @param neigh_i List of particles in the neighborhood of i_site
 * @param numneigh_site_i Number of particles in the neighborhood of i_site
 * @return int
 */
int write_neighborhood_vtk(int step, atom *atoms, adpPotential *adp[2][2],
                           unsigned int i_site, const int *neigh_i,
                           unsigned int numneigh_site_i);

/**
 * @brief Write output data in the vtk format (+potential)
 *
 * @brief Write output data in the vtk format
 * @param atoms Atomic data
 * @param adp ADP potential
 * @param potential ADP potential field
 * @return int
 */
int write_potential_vtk(int step, atom *atoms, adpPotential *adp[2][2],
                        double *potential);

/**
 * @brief Write output data in the vtk format (+forces)
 *
 * @param step current time step
 * @param atoms Atomic data
 * @param mean_forces Mean value of the atomistic force field
 * @param stdv_forces Standard desviation value of the atomistic force field
 * @return int
 */
int write_forces_vtk(int step, atom *atoms, const Eigen::MatrixXd &mean_forces,
                     const Eigen::VectorXd &stdv_forces);

/**
 * @brief Write output data in the vtk format (and stdv-q)
 *
 * @param step current time step
 * @param atoms Atomic data
 * @param adp ADP potential
 * @param residual Forces vector
 * @return int
 */
int write_residual_stdvq_vtk(int step, atom *atoms, adpPotential *adp[2][2],
                             const double *residual);

#endif /* OUTPUT_VTK_HPP */
#endif