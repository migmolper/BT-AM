/**
 * @file Quadrature-Hermitian-9d.hpp
 * @author Miguel Molinos, Pilar Ariza, Miguel Ortiz
 * ([migmolper](https://github.com/migmolper),[mpariza](https://github.com/mpariza),[mortizcaltech](https://github.com/mortizcaltech))
 * @brief
 * @version 0.1
 * @date 2022-11-08
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef quadrature_hermite_5th_HPP
#define quadrature_hermite_5th_HPP

#include "Atoms/Atom.hpp"
#include "Atoms/Neighbors.hpp"
#include "Numerical/Quadrature-Measure.hpp"

/**
 * @brief This function computes phase average of a potential using a 5th order
 * integration rule.
 *
 * @param integral_f Value of the integral
 * @param function structure which contain function to pointer
 * @param ctx Integral auxiliar variables
 * @return Value of the integral
 */
void moment_1th_Vij_gaussian_measure_5th(double *integral_f,
                                         potential_function function,
                                         void *ctx);

/**
 * @brief This function computes phase average of a gradient potential using a
 * 5th order integration rule.
 *
 * @param direction Direction of the n-d function
 * @param integral_grad_f
 * @param function structure which contain function to pointer
 * @param ctx Integral auxiliar variables
 */
void moment_1th_grad_f_gaussian_measure_5th_6d(int direction,
                                               double *integral_grad_f,
                                               potential_function function,
                                               void *ctx);

/**
 * @brief This function computes phase average of a hessian potential using a
 * 5th order integration rule.
 *
 * @param direction Direction of the n-d function
 * @param integral_hess_f
 * @param hess_Function Pointer to the function that we want to integate
 * @param ctx Integral auxiliar variables
 */
void moment_1th_hess_f_gaussian_measure_5th_6d(int direction,
                                               double *integral_hess_f,
                                               potential_function function,
                                               void *ctx);

#endif // quadrature_hermite_5th_HPP