/**
 * @file Quadrature-Hermitian-9d.cpp
 * @author Miguel Molinos, Pilar Ariza, Miguel Ortiz
 * ([migmolper](https://github.com/migmolper),[mpariza](https://github.com/mpariza),[mortizcaltech](https://github.com/mortizcaltech))
 * @brief Hermitian quadrature of a function using a third order quadrature
 * @version 0.1
 * @date 2022-11-08
 *
 * @copyright Copyright (c) 2022
 *
 */

// clang-format off
#include <cstdlib>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>  //std::cout//std::cin
#include <math.h>
#include "Macros.hpp"
#include "Atoms/Atom.hpp"
#include "Numerical/Quadrature-Measure.hpp"
#include "Numerical/Quadrature-Hermitian-5th.hpp"
#include <iostream>
#include <fstream>
// clang-format on
using namespace std;

/********************************************************************************/

void moment_1th_Vij_gaussian_measure_5th(double* integral_f,
                                         potential_function function,
                                         void* ctx_measure) {

  *integral_f = 0.0;

  unsigned int dim = NumberDimensions;
  unsigned int i = 0, j = 1;

  //! Read integral context
  const double* mean_q_ij = ((gaussian_measure_ctx*)ctx_measure)->mean_q_ij;
  double* sigma = ((gaussian_measure_ctx*)ctx_measure)->stddev_q_ij;
  double* xi_ij = ((gaussian_measure_ctx*)ctx_measure)->xi_ij;
  AtomicSpecie* spc = ((gaussian_measure_ctx*)ctx_measure)->spc;

  unsigned int nsites = 2;
  double dim_integral = nsites * dim;
  unsigned int N = 44;  // N = n2 + n + 2, when n=6 N=44.
  unsigned int half_N = 22;
  double c0 = pow(sqrt(PI), dim_integral / 2.0);

  //! Quadrature points and weights
  // clang-format off
  double zeta_l[22][6] = {
      {1.0, 1.0, 1.0, 1.0, 1.0, 1.0}, // eta (+)    
      {sqrt_2,0.0,0.0,0.0,0.0,0.0}, // lambda & chi (+)    
      {0.0,sqrt_2,0.0,0.0,0.0,0.0}, // lambda & chi (+)    
      {0.0,0.0,sqrt_2,0.0,0.0,0.0}, // lambda & chi (+)      
      {0.0,0.0,0.0,sqrt_2,0.0,0.0}, // lambda & chi (+)    
      {0.0,0.0,0.0,0.0,sqrt_2,0.0}, // lambda & chi (+)  
      {0.0,0.0,0.0,0.0,0.0,sqrt_2}, // lambda & chi (+)     
      {-1.0,-1.0,1.0,1.0,1.0,1.0}, // nu & gamma
      {-1.0,1.0,-1.0,1.0,1.0,1.0}, // nu & gamma
      {-1.0,1.0,1.0,-1.0,1.0,1.0}, // nu & gamma
      {-1.0,1.0,1.0,1.0,-1.0,1.0}, // nu & gamma
      {-1.0,1.0,1.0,1.0,1.0,-1.0}, // nu & gamma
      {1.0,-1.0,-1.0,1.0,1.0,1.0}, // nu & gamma
      {1.0,-1.0,1.0,-1.0,1.0,1.0}, // nu & gamma  
      {1.0,-1.0,1.0,1.0,-1.0,1.0}, // nu & gamma  
      {1.0,-1.0,1.0,1.0,1.0,-1.0}, // nu & gamma                  
      {1.0,1.0,-1.0,-1.0,1.0,1.0}, // nu & gamma
      {1.0,1.0,-1.0,1.0,-1.0,1.0}, // nu & gamma
      {1.0,1.0,-1.0,1.0,1.0,-1.0}, // nu & gamma            
      {1.0,1.0,1.0,-1.0,-1.0,1.0}, // nu & gamma
      {1.0,1.0,1.0,-1.0,1.0,-1.0}, // nu & gamma      
      {1.0,1.0,1.0,1.0,-1.0,-1.0}}; // nu & gamma       

  double W[22] = 
  {
   0.0078125, // W_A
   0.0625000, // W_B
   0.0625000, // W_B
   0.0625000, // W_B
   0.0625000, // W_B
   0.0625000, // W_B
   0.0625000, // W_B
   0.0078125, // W_C
   0.0078125, // W_C
   0.0078125, // W_C
   0.0078125, // W_C
   0.0078125, // W_C
   0.0078125, // W_C
   0.0078125, // W_C
   0.0078125, // W_C
   0.0078125, // W_C
   0.0078125, // W_C
   0.0078125, // W_C
   0.0078125, // W_C
   0.0078125, // W_C
   0.0078125, // W_C
   0.0078125}; // W_C
  // clang-format on

  //! In this integration rule we use 2 gp's for each dof
  double q_ij_l[2 * NumberDimensions];
  double f_gp;

  //! Positive contribution to the integral
  for (unsigned int gp = 0; gp < 22; gp++) {

    //! Compute the distance vector
    for (unsigned int alpha = 0; alpha < 3; alpha++) {

      //! Site i
      q_ij_l[i * dim + alpha] = mean_q_ij[i * dim + alpha] +
                                sqrt_2 * sigma[i] * zeta_l[gp][i * dim + alpha];

      //! Site j
      q_ij_l[j * dim + alpha] = mean_q_ij[j * dim + alpha] +
                                sqrt_2 * sigma[j] * zeta_l[gp][j * dim + alpha];
    }

    //! Evaluate the function using the modified position of the
    //! integration space
    f_gp = 0.0;
    function.F(&f_gp, xi_ij, q_ij_l, spc);

    //! Add contribution to the integral
    *integral_f += f_gp * W[gp] * c0;
  }

  //! Negative contribution to the integral
  for (unsigned int gp = 0; gp < 22; gp++) {

    //! Compute the distance vector
    for (unsigned int alpha = 0; alpha < 3; alpha++) {

      //! Site i
      q_ij_l[i * dim + alpha] = mean_q_ij[i * dim + alpha] -
                                sqrt_2 * sigma[i] * zeta_l[gp][i * dim + alpha];

      //! Site j
      q_ij_l[j * dim + alpha] = mean_q_ij[j * dim + alpha] -
                                sqrt_2 * sigma[j] * zeta_l[gp][j * dim + alpha];
    }

    //! Evaluate the function using the modified position of the
    //! integration space
    f_gp = 0.0;
    function.F(&f_gp, xi_ij, q_ij_l, spc);

    //! Add contribution to the integral
    *integral_f += f_gp * W[gp] * c0;
  }

  (*integral_f) = (*integral_f) * pow(1.0 / sqrt(PI), dim_integral / 2.0);
}

/********************************************************************************/

void moment_1th_grad_f_gaussian_measure_5th_6d(int direction,
                                               double* integral_grad_f,
                                               potential_function function,
                                               void* ctx_measure) {

  unsigned int dim = NumberDimensions;
  unsigned int i = 0, j = 1;

  //! Read integral context
  const double* mean_q_ij = ((gaussian_measure_ctx*)ctx_measure)->mean_q_ij;
  double* sigma = ((gaussian_measure_ctx*)ctx_measure)->stddev_q_ij;
  double* xi_ij = ((gaussian_measure_ctx*)ctx_measure)->xi_ij;
  AtomicSpecie* spc = ((gaussian_measure_ctx*)ctx_measure)->spc;

  unsigned int nsites = 2;
  double dim_integral = nsites * dim;
  unsigned int N = 44;  // N = n2 + n + 2, when n=6 N=44.
  unsigned int half_N = 22;
  double c0 = pow(sqrt(PI), dim_integral / 2.0);

  //! Quadrature points and weights
  // clang-format off
  double zeta_l[22][6] = {
      {1.0, 1.0, 1.0, 1.0, 1.0, 1.0}, // eta
      {sqrt_2,0.0,0.0,0.0,0.0,0.0}, // lambda & chi
      {0.0,sqrt_2,0.0,0.0,0.0,0.0}, // lambda & chi
      {0.0,0.0,sqrt_2,0.0,0.0,0.0}, // lambda & chi 
      {0.0,0.0,0.0,sqrt_2,0.0,0.0}, // lambda & chi
      {0.0,0.0,0.0,0.0,sqrt_2,0.0}, // lambda & chi 
      {0.0,0.0,0.0,0.0,0.0,sqrt_2}, // lambda & chi 
      {-1.0,-1.0,1.0,1.0,1.0,1.0}, // nu & gamma
      {-1.0,1.0,-1.0,1.0,1.0,1.0}, // nu & gamma
      {-1.0,1.0,1.0,-1.0,1.0,1.0}, // nu & gamma
      {-1.0,1.0,1.0,1.0,-1.0,1.0}, // nu & gamma
      {-1.0,1.0,1.0,1.0,1.0,-1.0}, // nu & gamma
      {1.0,-1.0,-1.0,1.0,1.0,1.0}, // nu & gamma
      {1.0,-1.0,1.0,-1.0,1.0,1.0}, // nu & gamma  
      {1.0,-1.0,1.0,1.0,-1.0,1.0}, // nu & gamma  
      {1.0,-1.0,1.0,1.0,1.0,-1.0}, // nu & gamma                  
      {1.0,1.0,-1.0,-1.0,1.0,1.0}, // nu & gamma
      {1.0,1.0,-1.0,1.0,-1.0,1.0}, // nu & gamma
      {1.0,1.0,-1.0,1.0,1.0,-1.0}, // nu & gamma            
      {1.0,1.0,1.0,-1.0,-1.0,1.0}, // nu & gamma
      {1.0,1.0,1.0,-1.0,1.0,-1.0}, // nu & gamma      
      {1.0,1.0,1.0,1.0,-1.0,-1.0}}; // nu & gamma       

  double W[22] = 
  {
   0.0078125, // W_A
   0.0625000, // W_B
   0.0625000, // W_B
   0.0625000, // W_B
   0.0625000, // W_B
   0.0625000, // W_B
   0.0625000, // W_B
   0.0078125, // W_C
   0.0078125, // W_C
   0.0078125, // W_C
   0.0078125, // W_C
   0.0078125, // W_C
   0.0078125, // W_C
   0.0078125, // W_C
   0.0078125, // W_C
   0.0078125, // W_C
   0.0078125, // W_C
   0.0078125, // W_C
   0.0078125, // W_C
   0.0078125, // W_C
   0.0078125, // W_C
   0.0078125}; // W_C
  // clang-format on

  //! In this integration rule we use 2 gp's for each dof
  double q_ij_l[2 * NumberDimensions];
  double grad_f_gp[NumberDimensions];
  integral_grad_f[0] = 0.0;
  integral_grad_f[1] = 0.0;
  integral_grad_f[2] = 0.0;

  //! Positive contribution to the integral
  for (unsigned int gp = 0; gp < 22; gp++) {

    //! Compute the distance vector
    for (unsigned int alpha = 0; alpha < 3; alpha++) {

      //! Site i
      q_ij_l[i * dim + alpha] = mean_q_ij[i * dim + alpha] +
                                sqrt_2 * sigma[i] * zeta_l[gp][i * dim + alpha];

      //! Site j
      q_ij_l[j * dim + alpha] = mean_q_ij[j * dim + alpha] +
                                sqrt_2 * sigma[j] * zeta_l[gp][j * dim + alpha];
    }

    //! Evaluate the function using the modified position of the
    //! integration space
    function.dF_dq(direction, grad_f_gp, xi_ij, q_ij_l, spc);
    for (unsigned int alpha = 0; alpha < dim; alpha++) {
      integral_grad_f[alpha] += grad_f_gp[alpha] * W[gp] * c0;
    }
  }

  //! Negative contribution to the integral
  for (unsigned int gp = 0; gp < 22; gp++) {

    //! Compute the distance vector
    for (unsigned int alpha = 0; alpha < 3; alpha++) {

      //! Site i
      q_ij_l[i * dim + alpha] = mean_q_ij[i * dim + alpha] -
                                sqrt_2 * sigma[i] * zeta_l[gp][i * dim + alpha];

      //! Site j
      q_ij_l[j * dim + alpha] = mean_q_ij[j * dim + alpha] -
                                sqrt_2 * sigma[j] * zeta_l[gp][j * dim + alpha];
    }

    //! Evaluate the function using the modified position of the
    //! integration space
    function.dF_dq(direction, grad_f_gp, xi_ij, q_ij_l, spc);
    for (unsigned int alpha = 0; alpha < dim; alpha++) {
      integral_grad_f[alpha] += grad_f_gp[alpha] * W[gp] * c0;
    }
  }

  //! Re-escale the integral
  for (unsigned int alpha = 0; alpha < dim; alpha++) {
    integral_grad_f[alpha] =
        integral_grad_f[alpha] * pow(1.0 / sqrt(PI), dim_integral / 2.0);
  }
}

/********************************************************************************/

void moment_1th_hess_f_gaussian_measure_5th_6d(int direction,
                                               double* integral_hess_f,
                                               potential_function function,
                                               void* ctx_measure) {

  unsigned int dim = NumberDimensions;
  unsigned int i = 0, j = 1;

  //! Read integral context
  const double* mean_q_ij = ((gaussian_measure_ctx*)ctx_measure)->mean_q_ij;
  double* sigma = ((gaussian_measure_ctx*)ctx_measure)->stddev_q_ij;
  double* xi_ij = ((gaussian_measure_ctx*)ctx_measure)->xi_ij;
  AtomicSpecie* spc = ((gaussian_measure_ctx*)ctx_measure)->spc;

  unsigned int nsites = 2;
  double dim_integral = nsites * dim;
  unsigned int N = 44;  // N = n2 + n + 2, when n=6 N=44.
  unsigned int half_N = 22;
  double c0 = pow(sqrt(PI), dim_integral / 2.0);

  //! Quadrature points and weights
  // clang-format off
  double zeta_l[22][6] = {
      {1.0, 1.0, 1.0, 1.0, 1.0, 1.0}, // eta
      {sqrt_2,0.0,0.0,0.0,0.0,0.0}, // lambda & chi
      {0.0,sqrt_2,0.0,0.0,0.0,0.0}, // lambda & chi
      {0.0,0.0,sqrt_2,0.0,0.0,0.0}, // lambda & chi 
      {0.0,0.0,0.0,sqrt_2,0.0,0.0}, // lambda & chi
      {0.0,0.0,0.0,0.0,sqrt_2,0.0}, // lambda & chi 
      {0.0,0.0,0.0,0.0,0.0,sqrt_2}, // lambda & chi 
      {-1.0,-1.0,1.0,1.0,1.0,1.0}, // nu & gamma
      {-1.0,1.0,-1.0,1.0,1.0,1.0}, // nu & gamma
      {-1.0,1.0,1.0,-1.0,1.0,1.0}, // nu & gamma
      {-1.0,1.0,1.0,1.0,-1.0,1.0}, // nu & gamma
      {-1.0,1.0,1.0,1.0,1.0,-1.0}, // nu & gamma
      {1.0,-1.0,-1.0,1.0,1.0,1.0}, // nu & gamma
      {1.0,-1.0,1.0,-1.0,1.0,1.0}, // nu & gamma  
      {1.0,-1.0,1.0,1.0,-1.0,1.0}, // nu & gamma  
      {1.0,-1.0,1.0,1.0,1.0,-1.0}, // nu & gamma                  
      {1.0,1.0,-1.0,-1.0,1.0,1.0}, // nu & gamma
      {1.0,1.0,-1.0,1.0,-1.0,1.0}, // nu & gamma
      {1.0,1.0,-1.0,1.0,1.0,-1.0}, // nu & gamma            
      {1.0,1.0,1.0,-1.0,-1.0,1.0}, // nu & gamma
      {1.0,1.0,1.0,-1.0,1.0,-1.0}, // nu & gamma      
      {1.0,1.0,1.0,1.0,-1.0,-1.0}}; // nu & gamma       

  double W[22] = 
  {
   0.0078125, // W_A
   0.0625000, // W_B
   0.0625000, // W_B
   0.0625000, // W_B
   0.0625000, // W_B
   0.0625000, // W_B
   0.0625000, // W_B
   0.0078125, // W_C
   0.0078125, // W_C
   0.0078125, // W_C
   0.0078125, // W_C
   0.0078125, // W_C
   0.0078125, // W_C
   0.0078125, // W_C
   0.0078125, // W_C
   0.0078125, // W_C
   0.0078125, // W_C
   0.0078125, // W_C
   0.0078125, // W_C
   0.0078125, // W_C
   0.0078125, // W_C
   0.0078125}; // W_C
  // clang-format on

  //! In this integration rule we use 2 gp's for each dof
  double q_ij_l[2 * NumberDimensions];
  double hess_f_gp[NumberDimensions * NumberDimensions];
  integral_hess_f[0] = 0.0;
  integral_hess_f[1] = 0.0;
  integral_hess_f[2] = 0.0;
  integral_hess_f[3] = 0.0;
  integral_hess_f[4] = 0.0;
  integral_hess_f[5] = 0.0;
  integral_hess_f[6] = 0.0;
  integral_hess_f[7] = 0.0;
  integral_hess_f[8] = 0.0;

  //! Positive contribution to the integral
  for (unsigned int gp = 0; gp < 22; gp++) {

    //! Compute the distance vector
    for (unsigned int alpha = 0; alpha < 3; alpha++) {

      //! Site i
      q_ij_l[i * dim + alpha] = mean_q_ij[i * dim + alpha] +
                                sqrt_2 * sigma[i] * zeta_l[gp][i * dim + alpha];

      //! Site j
      q_ij_l[j * dim + alpha] = mean_q_ij[j * dim + alpha] +
                                sqrt_2 * sigma[j] * zeta_l[gp][j * dim + alpha];
    }

    //! Evaluate the function using the modified position of the
    //! integration space
    function.d2F_dq2(direction, hess_f_gp, xi_ij, q_ij_l, spc);
    for (unsigned int alpha = 0; alpha < dim * dim; alpha++) {
      integral_hess_f[alpha] += hess_f_gp[alpha] * W[gp] * c0;
    }
  }

  //! Negative contribution to the integral
  for (unsigned int gp = 0; gp < 22; gp++) {

    //! Compute the distance vector
    for (unsigned int alpha = 0; alpha < 3; alpha++) {

      //! Site i
      q_ij_l[i * dim + alpha] = mean_q_ij[i * dim + alpha] -
                                sqrt_2 * sigma[i] * zeta_l[gp][i * dim + alpha];

      //! Site j
      q_ij_l[j * dim + alpha] = mean_q_ij[j * dim + alpha] -
                                sqrt_2 * sigma[j] * zeta_l[gp][j * dim + alpha];
    }

    //! Evaluate the function using the modified position of the
    //! integration space
    function.d2F_dq2(direction, hess_f_gp, xi_ij, q_ij_l, spc);
    for (unsigned int alpha = 0; alpha < dim * dim; alpha++) {
      integral_hess_f[alpha] += hess_f_gp[alpha] * W[gp] * c0;
    }
  }

  //! Re-escale the integral
  for (unsigned int alpha = 0; alpha < dim * dim; alpha++) {
    integral_hess_f[alpha] =
        integral_hess_f[alpha] * pow(1.0 / sqrt(PI), dim_integral / 2.0);
  }
}

/********************************************************************************/