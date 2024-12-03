/**
 * @file Quadrature-Hermitian-3th.hpp
 * @author Miguel Molinos, Pilar Ariza, Miguel Ortiz
 * ([migmolper](https://github.com/migmolper),[mpariza](https://github.com/mpariza),[mortizcaltech](https://github.com/mortizcaltech))
 * @brief Hermitian quadrature of a 6d/9d function using a third order
 * quadrature
 * @version 0.1
 * @date 2022-11-08
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef quadrature_hermite_3th_HPP
#define quadrature_hermite_3th_HPP

#include "Atoms/Atom.hpp"
#include "Atoms/Neighbors.hpp"
#include "Numerical/Quadrature-Measure.hpp"

/**
 * @brief Hermitian quadrature of a n-d function using a third order quadrature
 *
 * @param integral_f Value of the integral
 * @param function structure which contain function to pointer
 * @param ctx_measure Integral auxiliar variables
 * @return Value of the integral
 */
void meanfield_integral_gh3th(double *integral_f, potential_function function,
                              void *ctx_measure);

/**
 * @brief Hermitian quadrature of a n-d function using a third order quadrature
 *
 * @param direction Direction of the n-d function
 * @param integral_f Value of the integral
 * @param function structure which contain function to pointer
 * @param ctx_measure Integral auxiliar variables
 * @return Value of the integral
 */
void meanfield_integral_gh3th_dsq(int direction, double *integral_f,
                                  potential_function function,
                                  void *ctx_measure);

/**
 * @brief Hermitian quadrature of a n-d function gradient using a third order
 * quadrature
 *
 * @param direction Direction of the n-d function
 * @param integral_grad_f Gradient of the function f to integrate
 * @param function structure which contain function to pointer
 * @param ctx_measure Integral auxiliar variables
 */
void meanfield_integral_gh_3th_dmq(int direction, double *integral_grad_f,
                                   potential_function function,
                                   void *ctx_measure);

/**
 * @brief Hermitian quadrature of a n-d function gradient using a third order
 * quadrature
 *
 * @param direction Direction of the n-d function
 * @param integral_grad_f Gradient of the function f to integrate
 * @param function structure which contain function to pointer
 * @param ctx_measure Integral auxiliar variables
 */
void meanfield_integral_gh_3th_dxi(int direction, double *integral_grad_f,
                                   potential_function function,
                                   void *ctx_measure);

/**
 * @brief Hermitian quadrature of a n-d function hessian using a third order
 * quadrature
 *
 * @param direction Direction of the n-d function
 * @param integral_hess_f Hessian of the function f to integrate
 * @param function structure which contain function to pointer
 * @param ctx Integral auxiliar variables
 */
void meanfield_integral_gh_3th_d2mq(int direction, double *integral_hess_f,
                                    potential_function function,
                                    void *ctx_measure);

#endif // quadrature_hermite_3th_HPP