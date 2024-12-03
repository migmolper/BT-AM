/**
 * @file Quadrature-Hermitian-3th.cpp
 * @author Miguel Molinos, Pilar Ariza, Miguel Ortiz
 * ([migmolper](https://github.com/migmolper),[mpariza](https://github.com/mpariza),[mortizcaltech](https://github.com/mortizcaltech))
 * @brief Hermitian quadrature of a function using a third order
 * quadrature
 * @version 0.1
 * @date 2022-11-08
 *
 * @copyright Copyright (c) 2022
 *
 */

// clang-format off
#include <cstdio>
#include <cstdlib>
#if __APPLE__
#include <malloc/_malloc.h>
#endif
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>  //std::cout//std::cin
#include <math.h>
#include "Macros.hpp"
#include "Atoms/Atom.hpp"
#include "Numerical/Quadrature-Measure.hpp"
#include "Numerical/Quadrature-Hermitian-3th.hpp"
#include <iostream>
#include <fstream>
// clang-format on
using namespace std;

/********************************************************************************/

void meanfield_integral_gh3th(double* integral_f, potential_function function,
                              void* ctx_measure) {

  //! Set to zero the value of the integral
  *integral_f = 0.0;

  unsigned int dim = NumberDimensions;

  //! Read integral context
  // clang-format off
  unsigned int num_sites = ((gaussian_measure_ctx*)ctx_measure)->num_sites;
  unsigned int intergal_dim = ((gaussian_measure_ctx*)ctx_measure)->intergal_dim;
  const int* gp_board = ((gaussian_measure_ctx*)ctx_measure)->gp_board;
  const double* mean_q_ij = ((gaussian_measure_ctx*)ctx_measure)->mean_q_ij;
  double* sigma = ((gaussian_measure_ctx*)ctx_measure)->stddev_q_ij;
  double* xi_ij = ((gaussian_measure_ctx*)ctx_measure)->xi_ij;
  AtomicSpecie* spc = ((gaussian_measure_ctx*)ctx_measure)->spc;
  // clang-format on

  //! Weight
  double W_gp = 1.0 / (2.0 * intergal_dim);

  //! Quadrature points
  //! zeta = (+,-) sqrt(intergal_dim/2)
  double zeta_l = sqrt((double)intergal_dim) / sqrt_2;

  //! In this integration rule we use 2 gp's for each dof
  double* q_l = (double*)calloc(dim * num_sites, sizeof(double));
  double f_gp;

  //! Positive contribution to the integral
  for (unsigned int l = 0; l < intergal_dim; l++) {

    for (unsigned int site_idx = 0; site_idx < num_sites; site_idx++) {

      for (unsigned int alpha = 0; alpha < dim; alpha++) {

        q_l[site_idx * dim + alpha] =
            mean_q_ij[site_idx * dim + alpha] +
            sqrt_2 * sigma[site_idx] * zeta_l *
                gp_board[l * num_sites * dim + site_idx * dim + alpha];
      }
    }

    //! Evaluate the function using the modified position of the
    //! integration space
    function.F(&f_gp, xi_ij, q_l, spc);

    *integral_f += f_gp * W_gp;
  }

  //! Negative contribution to the integral
  for (unsigned int l = 0; l < intergal_dim; l++) {

    for (unsigned int site_idx = 0; site_idx < num_sites; site_idx++) {

      for (unsigned int alpha = 0; alpha < dim; alpha++) {

        q_l[site_idx * dim + alpha] =
            mean_q_ij[site_idx * dim + alpha] -
            sqrt_2 * sigma[site_idx] * zeta_l *
                gp_board[l * num_sites * dim + site_idx * dim + alpha];
      }
    }

    //! Evaluate the function using the modified position of the
    //! integration space
    function.F(&f_gp, xi_ij, q_l, spc);

    *integral_f += f_gp * W_gp;
  }

  //! Free gauss-point auxiliar vector
  free(q_l);
}

/********************************************************************************/

void meanfield_integral_gh3th_dsq(int direction, double* integral_f,
                                  potential_function function,
                                  void* ctx_measure) {

  unsigned int dim = NumberDimensions;

  //! Read integral context
  // clang-format off
  unsigned int num_sites = ((gaussian_measure_ctx*)ctx_measure)->num_sites;
  unsigned int intergal_dim = ((gaussian_measure_ctx*)ctx_measure)->intergal_dim;
  const int* gp_board = ((gaussian_measure_ctx*)ctx_measure)->gp_board;
  const double* mean_q_ij = ((gaussian_measure_ctx*)ctx_measure)->mean_q_ij;
  double* sigma = ((gaussian_measure_ctx*)ctx_measure)->stddev_q_ij;
  double* xi_ij = ((gaussian_measure_ctx*)ctx_measure)->xi_ij;
  AtomicSpecie* spc = ((gaussian_measure_ctx*)ctx_measure)->spc;
  // clang-format on

  //! Function gradient
  double df_dq_gp[3];

  //! Weight
  double W_gp = 1.0 / (2.0 * intergal_dim);

  //! Quadrature points
  //! zeta = (+,-) sqrt(intergal_dim/2)
  double zeta_l = sqrt((double)intergal_dim) / sqrt_2;

  //! In this integration rule we use 2 gp's for each dof
  double* q_l = (double*)calloc(dim * num_sites, sizeof(double));
  double dq_ds_l[3];

  //! Set to zero the integral
  *integral_f = 0.0;

  //! Positive contribution to the integral
  for (unsigned int l = 0; l < intergal_dim; l++) {

    for (unsigned int site_idx = 0; site_idx < num_sites; site_idx++) {

      for (unsigned int alpha = 0; alpha < dim; alpha++) {

        int gp_board_l = gp_board[l * num_sites * dim + site_idx * dim + alpha];

        q_l[site_idx * dim + alpha] =
            mean_q_ij[site_idx * dim + alpha] +
            sqrt_2 * sigma[site_idx] * zeta_l * gp_board_l;
      }
    }

    for (unsigned int alpha = 0; alpha < dim; alpha++) {
      int gp_board_l = gp_board[l * num_sites * dim + direction * dim + alpha];
      dq_ds_l[alpha] = sqrt_2 * zeta_l * gp_board_l;
    }

    //! Evaluate the function using the modified position of the
    //! integration space
    function.dF_dq(direction, df_dq_gp, xi_ij, q_l, spc);

    for (unsigned int alpha = 0; alpha < dim; alpha++) {
      *integral_f += df_dq_gp[alpha] * dq_ds_l[alpha] * W_gp;
    }
  }

  //! Negative contribution to the integral
  for (unsigned int l = 0; l < intergal_dim; l++) {

    for (unsigned int site_idx = 0; site_idx < num_sites; site_idx++) {

      for (unsigned int alpha = 0; alpha < dim; alpha++) {

        int gp_board_l = gp_board[l * num_sites * dim + site_idx * dim + alpha];

        q_l[site_idx * dim + alpha] =
            mean_q_ij[site_idx * dim + alpha] -
            sqrt_2 * sigma[site_idx] * zeta_l * gp_board_l;
      }
    }

    for (unsigned int alpha = 0; alpha < dim; alpha++) {
      int gp_board_l = gp_board[l * num_sites * dim + direction * dim + alpha];
      dq_ds_l[alpha] = -sqrt_2 * zeta_l * gp_board_l;
    }

    //! Evaluate the function using the modified position of the
    //! integration space
    function.dF_dq(direction, df_dq_gp, xi_ij, q_l, spc);

    for (unsigned int alpha = 0; alpha < dim; alpha++) {
      *integral_f += df_dq_gp[alpha] * dq_ds_l[alpha] * W_gp;
    }
  }

  //! Free auxiliar vectors
  free(q_l);
}

/********************************************************************************/

void meanfield_integral_gh_3th_dmq(int direction, double* integral_grad_f,
                                   potential_function function,
                                   void* ctx_measure) {

  unsigned int dim = NumberDimensions;

  //! Read integral context
  // clang-format off
  unsigned int num_sites = ((gaussian_measure_ctx*)ctx_measure)->num_sites;
  unsigned int intergal_dim = ((gaussian_measure_ctx*)ctx_measure)->intergal_dim;
  const int* gp_board = ((gaussian_measure_ctx*)ctx_measure)->gp_board;
  const double* mean_q_ij = ((gaussian_measure_ctx*)ctx_measure)->mean_q_ij;
  double* sigma = ((gaussian_measure_ctx*)ctx_measure)->stddev_q_ij;
  double* xi_ij = ((gaussian_measure_ctx*)ctx_measure)->xi_ij;
  AtomicSpecie* spc = ((gaussian_measure_ctx*)ctx_measure)->spc;
  // clang-format on

  //! Function gradient
  double* df_dq_gp = (double*)calloc(dim, sizeof(double));

  //! Weight
  double W_gp = 1.0 / (2.0 * intergal_dim);

  //! Quadrature points
  //! zeta = (+,-) sqrt(intergal_dim/2)
  double zeta_l = sqrt((double)intergal_dim) / sqrt_2;

  //! In this integration rule we use 2 gp's for each dof
  double* q_l = (double*)calloc(dim * num_sites, sizeof(double));

  //! Set to zero the integral
  for (unsigned int alpha = 0; alpha < dim; alpha++) {
    integral_grad_f[alpha] = 0.0;
  }

  //! Positive contribution to the integral
  for (unsigned int l = 0; l < intergal_dim; l++) {

    for (unsigned int site_idx = 0; site_idx < num_sites; site_idx++) {

      for (unsigned int alpha = 0; alpha < dim; alpha++) {

        int gp_board_l = gp_board[l * num_sites * dim + site_idx * dim + alpha];

        q_l[site_idx * dim + alpha] =
            mean_q_ij[site_idx * dim + alpha] +
            sqrt_2 * sigma[site_idx] * zeta_l * gp_board_l;
      }
    }

    //! Evaluate the function using the modified position of the
    //! integration space
    function.dF_dq(direction, df_dq_gp, xi_ij, q_l, spc);
    for (unsigned int alpha = 0; alpha < dim; alpha++) {
      integral_grad_f[alpha] += df_dq_gp[alpha] * W_gp;
    }
  }

  //! Negative contribution to the integral
  for (unsigned int l = 0; l < intergal_dim; l++) {

    for (unsigned int site_idx = 0; site_idx < num_sites; site_idx++) {

      for (unsigned int alpha = 0; alpha < dim; alpha++) {

        int gp_board_l = gp_board[l * num_sites * dim + site_idx * dim + alpha];

        q_l[site_idx * dim + alpha] =
            mean_q_ij[site_idx * dim + alpha] -
            sqrt_2 * sigma[site_idx] * zeta_l * gp_board_l;
      }
    }

    //! Evaluate the function using the modified position of the
    //! integration space
    function.dF_dq(direction, df_dq_gp, xi_ij, q_l, spc);
    for (unsigned int alpha = 0; alpha < dim; alpha++) {
      integral_grad_f[alpha] += df_dq_gp[alpha] * W_gp;
    }
  }

  //! Free auxiliar vector
  free(q_l);
  free(df_dq_gp);
}

/********************************************************************************/

void meanfield_integral_gh_3th_dxi(int direction, double* integral_grad_f,
                                   potential_function function,
                                   void* ctx_measure) {

  unsigned int dim = NumberDimensions;

  //! Read integral context
  // clang-format off
  unsigned int num_sites = ((gaussian_measure_ctx*)ctx_measure)->num_sites;
  unsigned int intergal_dim = ((gaussian_measure_ctx*)ctx_measure)->intergal_dim;
  const int* gp_board = ((gaussian_measure_ctx*)ctx_measure)->gp_board;
  const double* mean_q_ij = ((gaussian_measure_ctx*)ctx_measure)->mean_q_ij;
  double* sigma = ((gaussian_measure_ctx*)ctx_measure)->stddev_q_ij;
  double* xi_ij = ((gaussian_measure_ctx*)ctx_measure)->xi_ij;
  AtomicSpecie* spc = ((gaussian_measure_ctx*)ctx_measure)->spc;
  // clang-format on

  //! Function gradient
  double df_dxi_gp = 0.0;

  //! Weight
  double W_gp = 1.0 / (2.0 * intergal_dim);

  //! Quadrature points
  //! zeta = (+,-) sqrt(intergal_dim/2)
  double zeta_l = sqrt((double)intergal_dim) / sqrt_2;

  //! In this integration rule we use 2 gp's for each dof
  double* q_l = (double*)calloc(dim * num_sites, sizeof(double));

  //! Set to zero the integral
  *integral_grad_f = 0.0;

  //! Positive contribution to the integral
  for (unsigned int l = 0; l < intergal_dim; l++) {

    for (unsigned int site_idx = 0; site_idx < num_sites; site_idx++) {

      for (unsigned int alpha = 0; alpha < dim; alpha++) {

        int gp_board_l = gp_board[l * num_sites * dim + site_idx * dim + alpha];

        q_l[site_idx * dim + alpha] =
            mean_q_ij[site_idx * dim + alpha] +
            sqrt_2 * sigma[site_idx] * zeta_l * gp_board_l;
      }
    }

    //! Evaluate the function using the modified position of the
    //! integration space
    function.dF_dn(direction, &df_dxi_gp, xi_ij, q_l, spc);
    *integral_grad_f += df_dxi_gp * W_gp;
  }

  //! Negative contribution to the integral
  for (unsigned int l = 0; l < intergal_dim; l++) {

    for (unsigned int site_idx = 0; site_idx < num_sites; site_idx++) {

      for (unsigned int alpha = 0; alpha < dim; alpha++) {

        int gp_board_l = gp_board[l * num_sites * dim + site_idx * dim + alpha];

        q_l[site_idx * dim + alpha] =
            mean_q_ij[site_idx * dim + alpha] -
            sqrt_2 * sigma[site_idx] * zeta_l * gp_board_l;
      }
    }

    //! Evaluate the function using the modified position of the
    //! integration space
    function.dF_dn(direction, &df_dxi_gp, xi_ij, q_l, spc);

    *integral_grad_f += df_dxi_gp * W_gp;
  }

  //! Free auxiliar vector
  free(q_l);
}

/********************************************************************************/

void meanfield_integral_gh_3th_d2mq(int direction, double* integral_hess_f,
                                    potential_function function,
                                    void* ctx_measure) {

  unsigned int dim = NumberDimensions;

  //! Read integral context
  // clang-format off
  unsigned int num_sites = ((gaussian_measure_ctx*)ctx_measure)->num_sites;
  unsigned int intergal_dim = ((gaussian_measure_ctx*)ctx_measure)->intergal_dim;
  const int* gp_board = ((gaussian_measure_ctx*)ctx_measure)->gp_board;
  const double* mean_q_ij = ((gaussian_measure_ctx*)ctx_measure)->mean_q_ij;
  double* sigma = ((gaussian_measure_ctx*)ctx_measure)->stddev_q_ij;
  double* xi_ij = ((gaussian_measure_ctx*)ctx_measure)->xi_ij;
  AtomicSpecie* spc = ((gaussian_measure_ctx*)ctx_measure)->spc;
  // clang-format on

  //! Weight
  double W_gp = 1.0 / (2.0 * intergal_dim);

  //! Quadrature points
  //! zeta = (+,-) sqrt(intergal_dim/2)
  double zeta_l = sqrt((double)intergal_dim) / sqrt_2;

  //! In this integration rule we use 2 gp's for each dof
  double* q_l = (double*)calloc(dim * num_sites, sizeof(double));

  //! Function hessian
  double* d2d_dq2_gp = (double*)calloc(dim * dim, sizeof(double));

  //! Set to zero the integral
  for (unsigned int alpha = 0; alpha < dim * dim; alpha++) {
    integral_hess_f[alpha] = 0.0;
  }

  //! Positive contribution to the integral
  for (unsigned int l = 0; l < intergal_dim; l++) {

    for (unsigned int site_idx = 0; site_idx < num_sites; site_idx++) {

      for (unsigned int alpha = 0; alpha < dim; alpha++) {

        q_l[site_idx * dim + alpha] =
            mean_q_ij[site_idx * dim + alpha] +
            sqrt_2 * sigma[site_idx] * zeta_l *
                gp_board[l * num_sites * dim + site_idx * dim + alpha];
      }
    }

    //! Evaluate the function using the modified position of the
    //! integration space
    function.d2F_dq2(direction, d2d_dq2_gp, xi_ij, q_l, spc);
    for (unsigned int alpha = 0; alpha < dim * dim; alpha++) {
      integral_hess_f[alpha] += d2d_dq2_gp[alpha] * W_gp;
    }
  }

  //! Negative contribution to the integral
  for (unsigned int l = 0; l < intergal_dim; l++) {

    for (unsigned int site_idx = 0; site_idx < num_sites; site_idx++) {

      for (unsigned int alpha = 0; alpha < dim; alpha++) {

        q_l[site_idx * dim + alpha] =
            mean_q_ij[site_idx * dim + alpha] -
            sqrt_2 * sigma[site_idx] * zeta_l *
                gp_board[l * num_sites * dim + site_idx * dim + alpha];
      }
    }

    //! Evaluate the function using the modified position of the
    //! integration space
    function.d2F_dq2(direction, d2d_dq2_gp, xi_ij, q_l, spc);
    for (unsigned int alpha = 0; alpha < dim * dim; alpha++) {
      integral_hess_f[alpha] += d2d_dq2_gp[alpha] * W_gp;
    }
  }

  //! Free memory
  free(d2d_dq2_gp);
  free(q_l);
}

/********************************************************************************/
