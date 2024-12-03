/**
 * @file MgHx-mf-V-bulk.cpp
 * @author Miguel Molinos ([migmolper](https://github.com/migmolper))
 * @brief Compute meanfield-variables for the Mg-Hx bulk
 * @version 0.1
 * @date 2023-05-19
 *
 * @copyright Copyright (c) 2023
 *
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#if __APPLE__
#include <malloc/_malloc.h>
#endif
#ifdef USE_MPI
#include <mpi.h>
#endif
#include "ADP/MgHx-mf-V-bulk.hpp"
#include "Atoms/Atom.hpp"
#include "Atoms/Topology.hpp"
#include "Macros.hpp"
#include "Numerical/Quadrature-Hermitian-3th.hpp"
#include "Numerical/Quadrature-Measure.hpp"
#include "Numerical/Quadrature-Multipole.hpp"
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

extern PetscMPIInt size_MPI;
extern PetscMPIInt rank_MPI;

extern adpPotential adp_MgMg;
extern adpPotential adp_HH;
extern adpPotential adp_MgH;

extern double element_mass[112];

/********************************************************************************/

dmd_equations DMD_MgHx_constructor() {

  dmd_equations MgHx_dmd_equations;

  //!
  MgHx_dmd_equations.evaluate_rho_i = evaluate_rho_i_adp_MgHx;

  MgHx_dmd_equations.evaluate_DV_i_Dq_j = evaluate_DV_i_Dq_u_adp_MgHx;

  MgHx_dmd_equations.evaluate_D2V_i_Dq2_j = evaluate_D2V_i_Dq2_u_MgHx;

  MgHx_dmd_equations.evaluate_DV_i_DF = evaluate_DV_i_dF_MgHx;

  //! <V>0 set of equation
  MgHx_dmd_equations.evaluate_mf_rho_i = evaluate_mf_rho_i_adp_MgHx;

  MgHx_dmd_equations.evaluate_V0_i = evaluate_V0_i_adp_MgHx;

  MgHx_dmd_equations.evaluate_S0_i = evaluate_S0_i_adp_MgHx;

  MgHx_dmd_equations.evaluate_DV0_i_Dmeanq_j = evaluate_DV0_i_Dmeanq_u_MgHx;

  MgHx_dmd_equations.evaluate_D2V0_i_Dmeanq2_j = evaluate_D2V0_i_Dmeanq2_u_MgHx;

  MgHx_dmd_equations.evaluate_DV0_i_Dstdvq_j = evaluate_DV0_i_Dstdvq_u_MgHx;

  MgHx_dmd_equations.evaluate_DV0_i_Dq_j = evaluate_DV0_i_Dq_u_MgHx;

  MgHx_dmd_equations.evaluate_DV0_i_Dxi_j = evaluate_DV0_i_Dxi_u_adp_MgHx;

  MgHx_dmd_equations.evaluate_DV0_i_DF = evaluate_DV0_i_dF_bulk_MgHx;

  return MgHx_dmd_equations;
}

/********************************************************************************/

double evaluate_rho_i_adp_MgHx(unsigned int site_i,            //!
                               const Eigen::MatrixXd& mean_q,  //! Mean q
                               const Eigen::VectorXd& xi,   //! Molar fraction
                               const AtomicSpecie* specie,  //! Atom
                               const AtomTopology atom_topology_i) {

  //! @brief Auxiliar variables
  double rho_i = 0.0;

  //! @brief Get topologic information of site i
  unsigned int numneigh_site_i = atom_topology_i.numneigh;
  const PetscInt* mech_neighs_i = atom_topology_i.mech_neighs_ptr;

  //! @brief Get atomistic information of site i
  AtomicSpecie spc_i = specie[site_i];
  double xi_i = xi(site_i);
  Eigen::Vector3d mean_q_i = mean_q.block<1, 3>(site_i, 0);

  //! If the site is empty, skip from the evaluation
  if (xi_i < min_occupancy) {
    return 0.0;
  }

  //! @brief Compute the gradient of the potential with respect the
  //! mean value of the position at site i
  for (unsigned int idx_j1 = 0; idx_j1 < numneigh_site_i; idx_j1++) {

    //! @brief Get atomistic information of site j
    unsigned int site_j1 = mech_neighs_i[idx_j1];
    AtomicSpecie spc_j1 = specie[site_j1];
    double xi_j1 = xi(site_j1);
    Eigen::Vector3d mean_q_j1 = mean_q.block<1, 3>(site_j1, 0);

    //! If the site is empty, skip from the evaluation
    if (xi_j1 < min_occupancy) {
      continue;
    }

    //! @brief Create dof table
    int dof_table_ij[4] = {1, 0, 0, 1};

    //! @brief Fill data for the measure (i,j1)
    Eigen::VectorXd mean_q_ij1(6);
    mean_q_ij1 << mean_q_i, mean_q_j1;
    Eigen::VectorXd xi_ij1(2);
    xi_ij1 << xi_i, xi_j1;
    Eigen::VectorXd sites_ij1(2);
    sites_ij1 << site_i, site_j1;
    AtomicSpecie spc_ij1[2] = {spc_i, spc_j1};

    //! Create functions
    potential_function functions_rho_ij = rho_ij_adp_MgHx_constructor();

    //! @brief Compute energy density
    double rho_ij = 0.0;
    functions_rho_ij.F(&rho_ij, xi_ij1.data(), mean_q_ij1.data(), spc_ij1);
    rho_i += rho_ij;
  }

  return rho_i;
}

/********************************************************************************/

Eigen::Vector3d evaluate_DV_i_Dq_u_adp_MgHx(
    unsigned int site_i_star,            //!
    unsigned int site_i,                 //!
    const Eigen::MatrixXd& mean_q,       //! Mean value of q
    const Eigen::VectorXd& xi,           //! Molar fraction
    const Eigen::VectorXd& rho,          //! Energy density
    const AtomicSpecie* specie,          //! Atom
    const AtomTopology atom_topology_i)  //!
{

  unsigned int dim = NumberDimensions;

  //! @brief If we are in a hydrogen site with a occupancy below a certain
  //! thereshold, skip the evaluation of this equation
  double xi_i_star = xi(site_i_star);
  AtomicSpecie spc_i_star = specie[site_i_star];
  if ((spc_i_star == H) && (xi_i_star < min_occupancy)) {
    return Eigen::Vector3d::Zero();
  }

  //! @brief Auxiliar variables
  double rho_i = rho(site_i);
  Eigen::Vector3d D_V_i_embed_Dq = Eigen::Vector3d::Zero();
  Eigen::Vector3d D_rho_i_Dq = Eigen::Vector3d::Zero();
  Eigen::Vector3d D_V_i_pair_Dq = Eigen::Vector3d::Zero();
  Eigen::Vector3d D_V_i_dipol_Dq = Eigen::Vector3d::Zero();
  Eigen::Vector3d D_V_i_quad_Dq = Eigen::Vector3d::Zero();

  //! @brief Get topologic information of site i
  unsigned int numneigh_site_i = atom_topology_i.numneigh;
  const PetscInt* mech_neighs_i = atom_topology_i.mech_neighs_ptr;

  //! @brief Get atomistic information of site i
  AtomicSpecie spc_i = specie[site_i];
  double xi_i = xi(site_i);
  Eigen::Vector3d mean_q_i = mean_q.block<1, 3>(site_i, 0);

  //! If the site is empty, skip from the evaluation
  if ((spc_i == H) && (xi_i < min_occupancy)) {
    return Eigen::Vector3d::Zero();
  }

  //! @brief Compute the gradient of the potential with respect the
  //! mean value of the position at site i
  for (unsigned int idx_j1 = 0; idx_j1 < numneigh_site_i; idx_j1++) {

    //! @brief Get atomistic information of site j
    unsigned int site_j1 = mech_neighs_i[idx_j1];
    AtomicSpecie spc_j1 = specie[site_j1];
    double xi_j1 = xi(site_j1);
    Eigen::Vector3d mean_q_j1 = mean_q.block<1, 3>(site_j1, 0);

    //! If the site is empty, skip from the evaluation
    if ((spc_j1 == H) && (xi_j1 < min_occupancy)) {
      continue;
    }

    //! @brief Create dof table
    int dof_table_ij[4] = {1, 0, 0, 1};

    //! @brief Fill data for the measure (i,j1)
    Eigen::VectorXd mean_q_ij1(6);
    mean_q_ij1 << mean_q_i, mean_q_j1;
    Eigen::VectorXd xi_ij1(2);
    xi_ij1 << xi_i, xi_j1;
    Eigen::VectorXd sites_ij1(2);
    sites_ij1 << site_i, site_j1;
    AtomicSpecie spc_ij1[2] = {spc_i, spc_j1};

    //! Create functions
    potential_function functions_rho_ij = rho_ij_adp_MgHx_constructor();
    potential_function functions_pair_ij = V_pair_ij_adp_MgHx_constructor();

    //! @brief Compute gradient terms
    Eigen::Vector3d d_rho_ij1_dq;
    Eigen::Vector3d d_V_pair_ij1_dq;

    for (unsigned int direction = 0; direction < 2; direction++) {
      if (sites_ij1(direction) == site_i_star) {

        //! @brief Compute q-grad energy density
        d_rho_ij1_dq.setZero();
        functions_rho_ij.dF_dq(direction, d_rho_ij1_dq.data(), xi_ij1.data(),
                               mean_q_ij1.data(), spc_ij1);

        //! @brief Compute q-grad pairing term
        d_V_pair_ij1_dq.setZero();
        functions_pair_ij.dF_dq(direction, d_V_pair_ij1_dq.data(),
                                xi_ij1.data(), mean_q_ij1.data(), spc_ij1);

        //! @brief Add up partial contributions
        D_rho_i_Dq += d_rho_ij1_dq;
        D_V_i_pair_Dq += d_V_pair_ij1_dq;
      }
    }

    //! @brief Loop in the neighborhood considering the simmetry of the
    //! opration a_ij1 * a_ij2 = a_ij2 * a_ij1
    for (unsigned int idx_j2 = idx_j1; idx_j2 < numneigh_site_i; idx_j2++) {

      //! @brief Compute atomistic information of site j2
      unsigned int site_j2 = mech_neighs_i[idx_j2];
      AtomicSpecie spc_j2 = specie[site_j2];
      double xi_j2 = xi(site_j2);
      Eigen::Vector3d mean_q_j2 = mean_q.block<1, 3>(site_j2, 0);

      //! If the site is empty, skip from the evaluation
      if ((spc_j2 == H) && (xi_j2 < min_occupancy)) {
        continue;
      }

      //! @brief Create dof table
      int dof_table_ij1j2[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

      //! Modify dof table for i, j1 and j2
      if (site_j1 == site_j2) {
        dof_table_ij1j2[5] = 1;
        dof_table_ij1j2[7] = 1;
      }

      //! @brief Factor to consider the simmetry of the opration
      //! a_ij1*a_ij2 = a_ij2*a_ij1
      double factor_j1j2 = idx_j1 == idx_j2 ? 1.0 : 2.0;

      //! @brief Fill data for the measure (i,j1,j2)
      Eigen::VectorXd mean_q_ij1j2(9);
      mean_q_ij1j2 << mean_q_i, mean_q_j1, mean_q_j2;
      Eigen::VectorXd xi_ij1j2(3);
      xi_ij1j2 << xi_i, xi_j1, xi_j2;
      Eigen::VectorXd sites_ij1j2(3);
      sites_ij1j2 << site_i, site_j1, site_j2;
      AtomicSpecie spc_ij1j2[3] = {spc_i, spc_j1, spc_j2};

      //! Create functions
      potential_function functions_dipole_ij1j2 =
          V_dipole_ij1j2_adp_MgHx_constructor();
      potential_function functions_quadrupole_ij1j2 =
          V_quadrupole_ij1j2_adp_MgHx_constructor();

      //! @brief Compute gradient terms
      Eigen::Vector3d dV_dipole_ij1j2_dq;
      Eigen::Vector3d dV_quadrupole_ij1j2_dq;

      for (unsigned int direction = 0; direction < 3; direction++) {
        if (sites_ij1j2(direction) == site_i_star) {

          //! @brief Compute q-grad dipole angular term
          dV_dipole_ij1j2_dq.setZero();
          functions_dipole_ij1j2.dF_dq(direction, dV_dipole_ij1j2_dq.data(),
                                       xi_ij1j2.data(), mean_q_ij1j2.data(),
                                       spc_ij1j2);

          //! @brief Compute q-grad quadrupole angular term
          dV_quadrupole_ij1j2_dq.setZero();
          functions_quadrupole_ij1j2.dF_dq(
              direction, dV_quadrupole_ij1j2_dq.data(), xi_ij1j2.data(),
              mean_q_ij1j2.data(), spc_ij1j2);

          //! @brief Add up partial contributions
          D_V_i_dipol_Dq += factor_j1j2 * dV_dipole_ij1j2_dq;
          D_V_i_quad_Dq += factor_j1j2 * dV_quadrupole_ij1j2_dq;
        }
      }
    }
  }

  //! @brief Compute embedding forces
  CubicSpline embed_ii;
  if (spc_i == Mg) {
    embed_ii = adp_MgMg.embed;
  } else if (spc_i == H) {
    embed_ii = adp_HH.embed;
  }
  double d_F_embed_i = d_cubic_spline(&embed_ii, rho_i);
  D_V_i_embed_Dq = xi_i * d_F_embed_i * D_rho_i_Dq;

  //! @brief Assembly the gradient of the potential V0 at the site i with
  //! respect the stretch tensor U
  Eigen::Vector3d D_V_i_Dq_i_star =
      D_V_i_embed_Dq + D_V_i_pair_Dq + D_V_i_dipol_Dq + D_V_i_quad_Dq;

  return D_V_i_Dq_i_star;
}

/********************************************************************************/

Eigen::Matrix3d evaluate_D2V_i_Dq2_u_MgHx(
    unsigned int site_i_star,            //!
    unsigned int site_i,                 //!
    const Eigen::MatrixXd& mean_q,       //! Mean value of q
    const Eigen::VectorXd& xi,           //! Molar fraction
    const Eigen::VectorXd& rho,          //! Energy density
    const AtomicSpecie* specie,          //! Atom
    const AtomTopology atom_topology_i)  //!
{

  //! @brief Auxiliar atomistic variables
  unsigned int dim = NumberDimensions;

  //! @brief If we are in a hydrogen site with a occupancy below a certain
  //! thereshold, skip the evaluation of this equation
  double xi_i_star = xi(site_i_star);
  AtomicSpecie spc_i_star = specie[site_i_star];
  if (xi_i_star < min_occupancy) {
    return Eigen::Matrix3d::Zero();
  }

  //! @brief Compute the hessian of the meanfield potential with respect
  //! the mean value of the position at site i
  double rho_i = rho(site_i);
  Eigen::Matrix3d D2_V_i_embed_Dq2 = Eigen::Matrix3d::Zero();
  Eigen::Vector3d D_rho_i_Dq = Eigen::Vector3d::Zero();
  Eigen::Matrix3d D2_rho_i_Dq2 = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d D2_V_i_pair_Dq2 = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d D2_V_i_dipol_Dq2 = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d D2_V_i_quad_Dq2 = Eigen::Matrix3d::Zero();

  //! @brief Get topologic information of site i
  unsigned int numneigh_site_i = atom_topology_i.numneigh;
  const PetscInt* mech_neighs_i = atom_topology_i.mech_neighs_ptr;

  //! @brief Get atomistic information of site i
  AtomicSpecie spc_i = specie[site_i];
  double xi_i = xi(site_i);
  Eigen::Vector3d mean_q_i = mean_q.block<1, 3>(site_i, 0);

  //! If the site is empty, skip from the evaluation
  if (xi_i < min_occupancy) {
    return Eigen::Matrix3d::Zero();
  }

  for (unsigned int idx_j1 = 0; idx_j1 < numneigh_site_i; idx_j1++) {

    //! @brief Get atomistic information of site j
    unsigned int site_j1 = mech_neighs_i[idx_j1];
    AtomicSpecie spc_j1 = specie[site_j1];
    double xi_j1 = xi(site_j1);
    Eigen::Vector3d mean_q_j1 = mean_q.block<1, 3>(site_j1, 0);

    //! If the site is empty, skip from the evaluation
    if (xi_j1 <= min_occupancy) {
      continue;
    }

    //! @brief Create dof table
    int dof_table_ij[4] = {1, 0, 0, 1};

    //! @brief Fill data for the measure (i,j1)
    Eigen::VectorXd mean_q_ij1(6);
    mean_q_ij1 << mean_q_i, mean_q_j1;
    Eigen::VectorXd xi_ij1(2);
    xi_ij1 << xi_i, xi_j1;
    Eigen::VectorXd sites_ij1(2);
    sites_ij1 << site_i, site_j1;
    AtomicSpecie spc_ij1[2] = {spc_i, spc_j1};

    //! Define function
    potential_function functions_rho_ij = rho_ij_adp_MgHx_constructor();
    potential_function functions_V_pair_ij = V_pair_ij_adp_MgHx_constructor();

    //! @brief Compute gradient/hessian terms
    Eigen::Vector3d d_rho_ij1_dq;
    Eigen::Matrix3d d2_rho_ij1_dq2;
    Eigen::Matrix3d d2_V0_pair_ij1_dq2;

    for (unsigned int direction = 0; direction < 2; direction++) {
      if (sites_ij1(direction) == site_i_star) {

        //! @brief Compute meanfield mean-q-grad energy density
        d_rho_ij1_dq.setZero();
        functions_rho_ij.dF_dq(direction, d_rho_ij1_dq.data(), xi_ij1.data(),
                               mean_q_ij1.data(), spc_ij1);

        //! @brief Compute meanfield mean-q-hess energy density
        d2_rho_ij1_dq2.setZero();
        functions_rho_ij.d2F_dq2(direction * 2 + direction,
                                 d2_rho_ij1_dq2.data(), xi_ij1.data(),
                                 mean_q_ij1.data(), spc_ij1);

        //! @brief Compute meanfield mean-q-hess pairing term
        d2_V0_pair_ij1_dq2.setZero();
        functions_V_pair_ij.d2F_dq2(direction * 2 + direction,
                                    d2_V0_pair_ij1_dq2.data(), xi_ij1.data(),
                                    mean_q_ij1.data(), spc_ij1);

        //! @brief Add up partial contributions
        D_rho_i_Dq += d_rho_ij1_dq;
        D2_rho_i_Dq2 += d2_rho_ij1_dq2;
        D2_V_i_pair_Dq2 += d2_V0_pair_ij1_dq2;
      }
    }

    //! @brief Loop in the neighborhood considering the simmetry of the
    //! opration a_ij1*a_ij2 = a_ij2*a_ij1
    for (unsigned int idx_j2 = idx_j1; idx_j2 < numneigh_site_i; idx_j2++) {

      //! @brief Get atomistic information of site j_2
      unsigned int site_j2 = mech_neighs_i[idx_j2];
      AtomicSpecie spc_j2 = specie[site_j2];
      double xi_j2 = xi(site_j2);
      Eigen::Vector3d mean_q_j2 = mean_q.block<1, 3>(site_j2, 0);

      //! If the site is empty, skip from the evaluation
      if (xi_j2 <= min_occupancy) {
        continue;
      }

      //! @brief Create dof table
      int dof_table_ij1j2[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

      //! Modify dof table for i, j1 and j2
      if (site_j1 == site_j2) {
        dof_table_ij1j2[5] = 1;
        dof_table_ij1j2[7] = 1;
      }

      //! @brief Factor to consider the simmetry of the operation
      //! a_ij*a_ik = a_ik*a_ij
      double factor_j1j2 = idx_j2 == idx_j1 ? 1.0 : 2.0;

      //! @brief Fill data for the measure (i,j1,j2)
      Eigen::VectorXd mean_q_ij1j2(9);
      mean_q_ij1j2 << mean_q_i, mean_q_j1, mean_q_j2;
      Eigen::VectorXd xi_ij1j2(3);
      xi_ij1j2 << xi_i, xi_j1, xi_j2;
      Eigen::VectorXd sites_ij1j2(3);
      sites_ij1j2 << site_i, site_j1, site_j2;
      AtomicSpecie spc_ij1j2[3] = {spc_i, spc_j1, spc_j2};

      //! Create functions
      potential_function V_dipole_ij1j2 = V_dipole_ij1j2_adp_MgHx_constructor();
      potential_function V_quadrupole_ij1j2 =
          V_quadrupole_ij1j2_adp_MgHx_constructor();

      //! @brief Compute hessian terms
      Eigen::Matrix3d d2V0_dipole_ij1j2_dq2;
      Eigen::Matrix3d d2V0_quadrupole_ij1j2_dq2;

      for (unsigned int direction = 0; direction < 3; direction++) {
        if (sites_ij1j2(direction) == site_i_star) {

          //! @brief Compute meanfield mean-q-hess dipole angular term
          d2V0_dipole_ij1j2_dq2.setZero();
          V_dipole_ij1j2.d2F_dq2(direction * 3 + direction,
                                 d2V0_dipole_ij1j2_dq2.data(), xi_ij1j2.data(),
                                 mean_q_ij1j2.data(), spc_ij1j2);

          //! @brief Compute meanfield mean-q-hess quadrupole angular
          //! term
          d2V0_quadrupole_ij1j2_dq2.setZero();
          V_quadrupole_ij1j2.d2F_dq2(
              direction * 3 + direction, d2V0_quadrupole_ij1j2_dq2.data(),
              xi_ij1j2.data(), mean_q_ij1j2.data(), spc_ij1j2);

          //! @brief Add up partial contributions
          D2_V_i_dipol_Dq2 += factor_j1j2 * d2V0_dipole_ij1j2_dq2;
          D2_V_i_quad_Dq2 += factor_j1j2 * d2V0_quadrupole_ij1j2_dq2;
        }
      }
    }
  }

  //! @brief Compute embedding hessian
  CubicSpline embed_ii;
  if (spc_i == Mg) {
    embed_ii = adp_MgMg.embed;
  } else if (spc_i == H) {
    embed_ii = adp_HH.embed;
  }
  double mf_d_F_i = d_cubic_spline(&embed_ii, rho_i);
  double mf_dd_F_i = d2_cubic_spline(&embed_ii, rho_i);
  D2_V_i_embed_Dq2 = xi_i * mf_dd_F_i * (D_rho_i_Dq * D_rho_i_Dq.transpose()) +
                     xi_i * mf_d_F_i * D2_rho_i_Dq2;

  //! @brief Assembly the hessian of the potential at each position
  Eigen::Matrix3d D2_V0_i_Dq2_i_star =
      D2_V_i_embed_Dq2 + D2_V_i_pair_Dq2 + D2_V_i_dipol_Dq2 + D2_V_i_quad_Dq2;

  return D2_V0_i_Dq2_i_star;
}

/********************************************************************************/

Eigen::Matrix3d evaluate_DV_i_dF_MgHx(
    unsigned int site_i,             //! Site to evalute the potential
    const Eigen::MatrixXd& mean_q,   //! Mean value of q
    const Eigen::MatrixXd& mean_q0,  //! Reference value of the mean position
    const Eigen::VectorXd& xi,       //! Molar fraction
    const Eigen::VectorXd& rho,      //!
    const AtomicSpecie* specie,      //! Atom
    const AtomTopology atom_topology_i)  //!
{

  unsigned int dim = NumberDimensions;

  //! @brief Get atomistic information of site i
  AtomicSpecie spc_i = specie[site_i];
  double xi_i = xi(site_i);
  Eigen::Vector3d q_i = mean_q.block<1, 3>(site_i, 0);
  Eigen::Vector3d q0_i = mean_q0.block<1, 3>(site_i, 0);

  //! If the site is empty, skip from the evaluation
  if (xi_i < min_occupancy) {
    return Eigen::Matrix3d::Zero();
  }

  //! @brief Get topologic information of site i
  unsigned int numneigh_site_i = atom_topology_i.numneigh;
  const PetscInt* mech_neighs_i = atom_topology_i.mech_neighs_ptr;

  //! @brief Initialize auxiliar variables
  double rho_i = rho(site_i);
  Eigen::Matrix3d D_V_i_embed_DF = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d D_rho_i_DF = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d D_V_i_pair_DF = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d D_V_i_dipol_DF = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d D_V_i_quad_DF = Eigen::Matrix3d::Zero();

  //! @brief Compute the gradient of the meanfield potential with respect the
  //! mean value of the position at site i
  for (unsigned int idx_j1 = 0; idx_j1 < numneigh_site_i; idx_j1++) {

    //! @brief Get atomistic information of site j
    unsigned int site_j1 = mech_neighs_i[idx_j1];
    AtomicSpecie spc_j1 = specie[site_j1];
    double xi_j1 = xi(site_j1);
    Eigen::Vector3d q_j1 = mean_q.block<1, 3>(site_j1, 0);
    Eigen::Vector3d q0_j1 = mean_q0.block<1, 3>(site_j1, 0);

    //! If the site is empty, skip from the evaluation
    if ((spc_j1 == H) && (xi_j1 < min_occupancy)) {
      continue;
    }

    //! @brief Fill data for the measure (i,j1)
    Eigen::VectorXd q_ij1(6);
    q_ij1 << q_i, q_j1;
    Eigen::VectorXd q0_ij1(6);
    q0_ij1 << q0_i, q0_j1;
    Eigen::VectorXd xi_ij1(2);
    xi_ij1 << xi_i, xi_j1;
    Eigen::VectorXd sites_ij1(2);
    sites_ij1 << site_i, site_j1;
    AtomicSpecie spc_ij1[2] = {spc_i, spc_j1};

    //! Create functions
    potential_function functions_rho_ij = rho_ij_adp_MgHx_constructor();
    potential_function functions_pair_ij = V_pair_ij_adp_MgHx_constructor();

    //! @brief Compute gradient terms
    Eigen::Vector3d d_rho_ij1_dq;
    Eigen::Vector3d d_V_pair_ij1_dq;
    Eigen::Vector3d d_q_ij1_dF;

    for (unsigned int direction = 0; direction < 2; direction++) {
      //! @brief Get direction
      d_q_ij1_dF << q0_ij1.segment<3>(direction * 3);

      //! @brief Compute q-grad energy density
      d_rho_ij1_dq.setZero();
      functions_rho_ij.dF_dq(direction, d_rho_ij1_dq.data(), xi_ij1.data(),
                             q_ij1.data(), spc_ij1);

      //! @brief Compute q-grad pairing term
      d_V_pair_ij1_dq.setZero();
      functions_pair_ij.dF_dq(direction, d_V_pair_ij1_dq.data(), xi_ij1.data(),
                              q_ij1.data(), spc_ij1);

      //! @brief Add up partial contributions
      D_rho_i_DF += d_rho_ij1_dq * d_q_ij1_dF.transpose();
      D_V_i_pair_DF += d_V_pair_ij1_dq * d_q_ij1_dF.transpose();
    }

    //! @brief Loop in the neighborhood considering the simmetry of the
    //! opration a_ij1 * a_ij2 = a_ij2 * a_ij1
    for (unsigned int idx_j2 = idx_j1; idx_j2 < numneigh_site_i; idx_j2++) {

      //! @brief Compute atomistic information of site j2
      unsigned int site_j2 = mech_neighs_i[idx_j2];
      AtomicSpecie spc_j2 = specie[site_j2];
      double xi_j2 = xi(site_j2);
      Eigen::Vector3d q_j2 = mean_q.block<1, 3>(site_j2, 0);
      Eigen::Vector3d q0_j2 = mean_q0.block<1, 3>(site_j2, 0);

      //! If the site is empty, skip from the evaluation
      if ((spc_j2 == H) && (xi_j2 < min_occupancy)) {
        continue;
      }

      //! @brief Factor to consider the simmetry of the opration
      //! a_ij1*a_ij2 = a_ij2*a_ij1
      double factor_j1j2 = idx_j1 == idx_j2 ? 1.0 : 2.0;

      //! @brief Fill data for the measure (i,j1,j2)
      Eigen::VectorXd q_ij1j2(9);
      q_ij1j2 << q_i, q_j1, q_j2;
      Eigen::VectorXd q0_ij1j2(9);
      q0_ij1j2 << q0_i, q0_j1, q0_j2;
      Eigen::VectorXd xi_ij1j2(3);
      xi_ij1j2 << xi_i, xi_j1, xi_j2;
      Eigen::VectorXd sites_ij1j2(3);
      sites_ij1j2 << site_i, site_j1, site_j2;
      AtomicSpecie spc_ij1j2[3] = {spc_i, spc_j1, spc_j2};

      //! Create functions
      potential_function functions_dipole_ij1j2 =
          V_dipole_ij1j2_adp_MgHx_constructor();
      potential_function functions_quadrupole_ij1j2 =
          V_quadrupole_ij1j2_adp_MgHx_constructor();

      //! @brief Compute gradient terms
      Eigen::Vector3d dV_dipole_ij1j2_dq;
      Eigen::Vector3d dV_quadrupole_ij1j2_dq;
      Eigen::Vector3d d_q_ij1j2_dF;

      for (unsigned int direction = 0; direction < 3; direction++) {

        //! @brief Get direction
        d_q_ij1j2_dF << q0_ij1j2.segment<3>(direction * 3);

        //! @brief Compute q-grad dipole angular term
        dV_dipole_ij1j2_dq.setZero();
        functions_dipole_ij1j2.dF_dq(direction, dV_dipole_ij1j2_dq.data(),
                                     xi_ij1j2.data(), q_ij1j2.data(),
                                     spc_ij1j2);

        //! @brief Compute q-grad quadrupole angular term
        dV_quadrupole_ij1j2_dq.setZero();
        functions_quadrupole_ij1j2.dF_dq(
            direction, dV_quadrupole_ij1j2_dq.data(), xi_ij1j2.data(),
            q_ij1j2.data(), spc_ij1j2);

        //! @brief Add up partial contributions
        D_V_i_dipol_DF +=
            factor_j1j2 * dV_dipole_ij1j2_dq * d_q_ij1j2_dF.transpose();
        D_V_i_quad_DF +=
            factor_j1j2 * dV_quadrupole_ij1j2_dq * d_q_ij1j2_dF.transpose();
      }
    }
  }

  //! @brief Compute embedding forces
  CubicSpline embed_ii;
  if (spc_i == Mg) {
    embed_ii = adp_MgMg.embed;
  } else if (spc_i == H) {
    embed_ii = adp_HH.embed;
  }
  double d_F_embed_i = d_cubic_spline(&embed_ii, rho_i);
  D_V_i_embed_DF = xi_i * d_F_embed_i * D_rho_i_DF;

  //! @brief Assembly the gradient of the potential V0 at the site i with
  //! respect the deformation gradient F
  Eigen::Matrix3d D_V_i_DF =
      D_V_i_embed_DF + D_V_i_pair_DF + D_V_i_dipol_DF + D_V_i_quad_DF;

  return D_V_i_DF;
}

/********************************************************************************/

double evaluate_mf_rho_i_adp_MgHx(unsigned int site_i,            //!
                                  const Eigen::MatrixXd& mean_q,  //!
                                  const Eigen::VectorXd& stdv_q,  //!
                                  const Eigen::VectorXd& xi,      //!
                                  const AtomicSpecie* specie,     //!
                                  const AtomTopology atom_topology_i) {

  //! Define Integration rule
  void (*meanfield_integral)(double* integral_f, potential_function function,
                             void* ctx_measure);
#if defined(MULTIPOLE_INTEGRAL)
  meanfield_integral = meanfield_integral_mp;
#elif defined(GH3TH_INTEGRAL)
  meanfield_integral = meanfield_integral_gh3th;
#else
#error "Define MULTIPOLE_INTEGRAL or GH3TH_INTEGRAL"
#endif

  //! Local variables
  double mf_rho_i = 0.0;  //! Meanfield Energy density term

  //! @brief Get topologic information of site i
  unsigned int numneigh_site_i = atom_topology_i.numneigh;
  const PetscInt* mech_neighs_i = atom_topology_i.mech_neighs_ptr;

  //! @brief Get atomistic information of site i
  AtomicSpecie spc_i = specie[site_i];
  double xi_i = xi(site_i);
  double stdv_q_i = stdv_q(site_i);
  Eigen::Vector3d mean_q_i = mean_q.block<1, 3>(site_i, 0);

  //! If the site is empty, skip from the evaluation
  if ((spc_i == H) && (xi_i < min_occupancy)) {
    return 0.0;
  }

  for (unsigned int idx_j1 = 0; idx_j1 < numneigh_site_i; idx_j1++) {

    //! @brief Get atomistic information of site j
    unsigned int site_j1 = mech_neighs_i[idx_j1];
    AtomicSpecie spc_j1 = specie[site_j1];
    double xi_j1 = xi(site_j1);
    double stdv_q_j1 = stdv_q(site_j1);
    Eigen::Vector3d mean_q_j1 = mean_q.block<1, 3>(site_j1, 0);

    //! If the site is empty, skip from the evaluation
    if ((spc_j1 == H) && (xi_j1 < min_occupancy)) {
      continue;
    }

    //! @brief Fill data for the measure
    Eigen::VectorXd mean_q_ij1(6);
    mean_q_ij1 << mean_q_i, mean_q_j1;
    Eigen::VectorXd stdv_q_ij1(2);
    stdv_q_ij1 << stdv_q_i, stdv_q_j1;
    Eigen::VectorXd xi_ij1(2);
    xi_ij1 << xi_i, xi_j1;
    AtomicSpecie spc_ij1[2] = {spc_i, spc_j1};

    //! @brief Create dof table for i-j par
    int dof_table_ij[4] = {1, 0, 0, 1};

    //! @brief Create measure/functions
    gaussian_measure_ctx measure_ij =
        fill_out_gaussian_measure(mean_q_ij1.data(), stdv_q_ij1.data(),
                                  xi_ij1.data(), spc_ij1, dof_table_ij, 2);

    potential_function rho_ij = rho_ij_adp_MgHx_constructor();

    //! @brief Compute meanfield energy density to evaluate the embeded energy
    double mf_rho_ij = 0.0;
    meanfield_integral(&mf_rho_ij, rho_ij, &measure_ij);
    mf_rho_i += mf_rho_ij;

    //! @brief Remove measure
    destroy_gaussian_measure(&measure_ij);
  }

  return mf_rho_i;
}

/********************************************************************************/

double evaluate_V0_i_adp_MgHx(unsigned int site_i,                 //!
                              const Eigen::MatrixXd& mean_q,       //!
                              const Eigen::VectorXd& stdv_q,       //!
                              const Eigen::VectorXd& xi,           //!
                              const Eigen::VectorXd& mf_rho,       //!
                              const AtomicSpecie* specie,          //!
                              const AtomTopology atom_topology_i)  //!
{

  unsigned int dim = NumberDimensions;

  //! Define Integration rule
  void (*meanfield_integral)(double* integral_f, potential_function function,
                             void* ctx_measure);
#if defined(MULTIPOLE_INTEGRAL)
  meanfield_integral = meanfield_integral_mp;
#elif defined(GH3TH_INTEGRAL)
  meanfield_integral = meanfield_integral_gh3th;
#else
#error "Define MULTIPOLE_INTEGRAL or GH3TH_INTEGRAL"
#endif

  //! Local variables
  double mf_rho_i = mf_rho(site_i);
  double V0_embed_i = 0.0;  //! Meanfield Embedded forces term
  double V0_pair_i = 0.0;   //! Meanfield Pairing forces term
  double V0_dip_i = 0.0;    //! Meanfield Dipole distortion term
  double V0_quad_i = 0.0;   //! Meanfield Quadrupole distortion term
  double V0_i = 0.0;        //! Total potential

  //! @brief Get topologic information of site i
  unsigned int numneigh_site_i = atom_topology_i.numneigh;
  const PetscInt* mech_neighs_i = atom_topology_i.mech_neighs_ptr;

  //! @brief Get atomistic information of site i
  AtomicSpecie spc_i = specie[site_i];
  double xi_i = xi(site_i);
  double stdv_q_i = stdv_q(site_i);
  Eigen::Vector3d mean_q_i = mean_q.block<1, 3>(site_i, 0);

  //! If the site is empty, skip from the evaluation
  if ((spc_i == H) && (xi_i < min_occupancy)) {
    return 0.0;
  }

  for (unsigned int idx_j1 = 0; idx_j1 < numneigh_site_i; idx_j1++) {

    //! @brief Get atomistic information of site j
    unsigned int site_j1 = mech_neighs_i[idx_j1];
    AtomicSpecie spc_j1 = specie[site_j1];
    double xi_j1 = xi(site_j1);
    double stdv_q_j1 = stdv_q(site_j1);
    Eigen::Vector3d mean_q_j1 = mean_q.block<1, 3>(site_j1, 0);

    //! If the site is empty, skip from the evaluation
    if ((spc_j1 == H) && (xi_j1 < min_occupancy)) {
      continue;
    }

    //! @brief Fill data for the measure
    Eigen::VectorXd mean_q_ij1(6);
    mean_q_ij1 << mean_q_i, mean_q_j1;
    Eigen::VectorXd stdv_q_ij1(2);
    stdv_q_ij1 << stdv_q_i, stdv_q_j1;
    Eigen::VectorXd xi_ij1(2);
    xi_ij1 << xi_i, xi_j1;
    AtomicSpecie spc_ij1[2] = {spc_i, spc_j1};

    //! @brief Create dof table for i-j par
    int dof_table_ij[4] = {1, 0, 0, 1};

    //! @brief Create measure/functions
    gaussian_measure_ctx measure_ij =
        fill_out_gaussian_measure(mean_q_ij1.data(), stdv_q_ij1.data(),
                                  xi_ij1.data(), spc_ij1, dof_table_ij, 2);

    potential_function V_pair_ij = V_pair_ij_adp_MgHx_constructor();

    //! @brief Compute meanfield pairing term
    double V0_pair_ij = 0.0;
    meanfield_integral(&V0_pair_ij, V_pair_ij, &measure_ij);
    V0_pair_i += V0_pair_ij;

    //! @brief Remove measure
    destroy_gaussian_measure(&measure_ij);

    //! @brief Loop in the neighborhood considering the simmetry of the
    //! opration a_ij1 * a_ij2 = a_ij2 * a_ij1
    for (unsigned int idx_j2 = idx_j1; idx_j2 < numneigh_site_i; idx_j2++) {

      //! @brief Compute atomistic information of site j2
      unsigned int site_j2 = mech_neighs_i[idx_j2];
      AtomicSpecie spc_j2 = specie[site_j2];
      double xi_j2 = xi(site_j2);
      double stdv_q_j2 = stdv_q(site_j2);
      Eigen::Vector3d mean_q_j2 = mean_q.block<1, 3>(site_j2, 0);

      //! If the site is empty, skip from the evaluation
      if (xi_j2 < min_occupancy) {
        continue;
      }

      //! @brief Create dof table for i-j1-j2
      int dof_table_ij1j2[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

      //! If the j1 match j2, avoid integration duplic
      if (site_j1 == site_j2) {
        dof_table_ij1j2[5] = 1;
        dof_table_ij1j2[7] = 1;
      }

      //! @brief Fill data for the measure
      Eigen::VectorXd mean_q_ij1j2(9);
      mean_q_ij1j2 << mean_q_i, mean_q_j1, mean_q_j2;
      Eigen::VectorXd stdv_q_ij1j2(3);
      stdv_q_ij1j2 << stdv_q_i, stdv_q_j1, stdv_q_j2;
      Eigen::VectorXd xi_ij1j2(3);
      xi_ij1j2 << xi_i, xi_j1, xi_j2;
      AtomicSpecie spc_ij1j2[3] = {spc_i, spc_j1, spc_j2};

      //! @brief Factor to consider the simmetry of the opration a_ij*a_ik =
      //! a_ik*a_ij
      double factor_j1j2 = (idx_j2 == idx_j1) ? 1.0 : 2.0;

      //! Create measure/functions
      gaussian_measure_ctx measure_ij1j2 = fill_out_gaussian_measure(
          mean_q_ij1j2.data(), stdv_q_ij1j2.data(), xi_ij1j2.data(), spc_ij1j2,
          dof_table_ij1j2, 3);

      potential_function V_dipole_ij1j2 = V_dipole_ij1j2_adp_MgHx_constructor();
      potential_function V_quadrupole_ij1j2 =
          V_quadrupole_ij1j2_adp_MgHx_constructor();

      //! Compute meanfield dipole angular term
      double V0_dip_ij1j2 = 0.0;
      meanfield_integral(&V0_dip_ij1j2, V_dipole_ij1j2, &measure_ij1j2);
      V0_dip_i += factor_j1j2 * V0_dip_ij1j2;

      //! Compute meanfield quadrupole angular term
      double V0_quad_ij1j2 = 0.0;
      meanfield_integral(&V0_quad_ij1j2, V_quadrupole_ij1j2, &measure_ij1j2);
      V0_quad_i += factor_j1j2 * V0_quad_ij1j2;

      //! Remove measure
      destroy_gaussian_measure(&measure_ij1j2);
    }
  }

  //! @brief Add the contribution of the embedded forces
  CubicSpline embed_ii;
  if (spc_i == Mg) {
    embed_ii = adp_MgMg.embed;
  } else if (spc_i == H) {
    embed_ii = adp_HH.embed;
  }
  double mf_F_i = cubic_spline(&embed_ii, mf_rho_i);
  V0_embed_i = xi_i * mf_F_i;

  //! @brief Add up each contribution to the meanfield potential
  V0_i = V0_embed_i + V0_pair_i + V0_dip_i + V0_quad_i;

  return V0_i;
}

/********************************************************************************/

double evaluate_S0_i_adp_MgHx(unsigned int site_i,                 //!
                              const Eigen::MatrixXd& mean_q,       //!
                              const Eigen::VectorXd& stdv_q,       //!
                              const Eigen::VectorXd& xi,           //!
                              const Eigen::VectorXd& mf_rho,       //!
                              const Eigen::VectorXd& beta,         //!
                              const Eigen::VectorXd& gamma,        //!
                              const AtomicSpecie* specie,          //!
                              const AtomTopology atom_topology_i)  //!
{

  unsigned int dim = NumberDimensions;

  //! Define Integration rule
  void (*meanfield_integral)(double* integral_f, potential_function function,
                             void* ctx_measure);
#if defined(MULTIPOLE_INTEGRAL)
  meanfield_integral = meanfield_integral_mp;
#elif defined(GH3TH_INTEGRAL)
  meanfield_integral = meanfield_integral_gh3th;
#else
#error "Define MULTIPOLE_INTEGRAL or GH3TH_INTEGRAL"
#endif

  //! Local variables
  double mf_rho_i = mf_rho(site_i);
  double V0_embed_i = 0.0;  //! Meanfield Embedded forces term
  double V0_pair_i = 0.0;   //! Meanfield Pairing forces term
  double V0_dip_i = 0.0;    //! Meanfield Dipole distortion term
  double V0_quad_i = 0.0;   //! Meanfield Quadrupole distortion term
  double V0_i = 0.0;        //! Total potential

  //! @brief Get topologic information of site i
  unsigned int numneigh_site_i = atom_topology_i.numneigh;
  const PetscInt* mech_neighs_i = atom_topology_i.mech_neighs_ptr;

  //! @brief Get atomistic information of site i
  Eigen::Vector3d mean_q_i = mean_q.block<1, 3>(site_i, 0);
  double stdv_q_i = stdv_q(site_i);
  double xi_i = xi(site_i);
  double beta_i = beta(site_i);
  double gamma_i = gamma(site_i);
  AtomicSpecie spc_i = specie[site_i];
  double m_i = unit_change_uma * xi_i * element_mass[spc_i];

  //! If the site is empty, skip from the evaluation
  if (xi_i < min_occupancy) {
    return 0.0;
  }

  for (unsigned int idx_j1 = 0; idx_j1 < numneigh_site_i; idx_j1++) {

    //! @brief Get atomistic information of site j
    unsigned int site_j1 = mech_neighs_i[idx_j1];
    AtomicSpecie spc_j1 = specie[site_j1];
    double xi_j1 = xi(site_j1);
    double stdv_q_j1 = stdv_q(site_j1);
    Eigen::Vector3d mean_q_j1 = mean_q.block<1, 3>(site_j1, 0);

    //! If the site is empty, skip from the evaluation
    if ((spc_j1 == H) && (xi_j1 < min_occupancy)) {
      continue;
    }

    //! @brief Fill data for the measure
    Eigen::VectorXd mean_q_ij1(6);
    mean_q_ij1 << mean_q_i, mean_q_j1;
    Eigen::VectorXd stdv_q_ij1(2);
    stdv_q_ij1 << stdv_q_i, stdv_q_j1;
    Eigen::VectorXd xi_ij1(2);
    xi_ij1 << xi_i, xi_j1;
    AtomicSpecie spc_ij1[2] = {spc_i, spc_j1};

    //! @brief Create dof table for i-j par
    int dof_table_ij[4] = {1, 0, 0, 1};

    //! @brief Create measure/functions
    gaussian_measure_ctx measure_ij =
        fill_out_gaussian_measure(mean_q_ij1.data(), stdv_q_ij1.data(),
                                  xi_ij1.data(), spc_ij1, dof_table_ij, 2);

    potential_function V_pair_ij = V_pair_ij_adp_MgHx_constructor();

    //! @brief Compute meanfield pairing term
    double V0_pair_ij = 0.0;
    meanfield_integral(&V0_pair_ij, V_pair_ij, &measure_ij);
    V0_pair_i += V0_pair_ij;

    //! @brief Remove measure
    destroy_gaussian_measure(&measure_ij);

    //! @brief Loop in the neighborhood considering the simmetry of the
    //! opration a_ij1 * a_ij2 = a_ij2 * a_ij1
    for (unsigned int idx_j2 = idx_j1; idx_j2 < numneigh_site_i; idx_j2++) {

      //! @brief Compute atomistic information of site j2
      unsigned int site_j2 = mech_neighs_i[idx_j2];
      AtomicSpecie spc_j2 = specie[site_j2];
      double xi_j2 = xi(site_j2);
      double stdv_q_j2 = stdv_q(site_j2);
      Eigen::Vector3d mean_q_j2 = mean_q.block<1, 3>(site_j2, 0);

      //! If the site is empty, skip from the evaluation
      if (xi_j2 < min_occupancy) {
        continue;
      }

      //! @brief Create dof table for i-j1-j2
      int dof_table_ij1j2[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

      //! If the j1 match j2, avoid integration duplic
      if (site_j1 == site_j2) {
        dof_table_ij1j2[5] = 1;
        dof_table_ij1j2[7] = 1;
      }

      //! @brief Fill data for the measure
      Eigen::VectorXd mean_q_ij1j2(9);
      mean_q_ij1j2 << mean_q_i, mean_q_j1, mean_q_j2;
      Eigen::VectorXd stdv_q_ij1j2(3);
      stdv_q_ij1j2 << stdv_q_i, stdv_q_j1, stdv_q_j2;
      Eigen::VectorXd xi_ij1j2(3);
      xi_ij1j2 << xi_i, xi_j1, xi_j2;
      AtomicSpecie spc_ij1j2[3] = {spc_i, spc_j1, spc_j2};

      //! @brief Factor to consider the simmetry of the opration a_ij*a_ik =
      //! a_ik*a_ij
      double factor_j1j2 = (idx_j2 == idx_j1) ? 1.0 : 2.0;

      //! Create measure/functions
      gaussian_measure_ctx measure_ij1j2 = fill_out_gaussian_measure(
          mean_q_ij1j2.data(), stdv_q_ij1j2.data(), xi_ij1j2.data(), spc_ij1j2,
          dof_table_ij1j2, 3);

      potential_function V_dipole_ij1j2 = V_dipole_ij1j2_adp_MgHx_constructor();
      potential_function V_quadrupole_ij1j2 =
          V_quadrupole_ij1j2_adp_MgHx_constructor();

      //! Compute meanfield dipole angular term
      double V0_dip_ij1j2 = 0.0;
      meanfield_integral(&V0_dip_ij1j2, V_dipole_ij1j2, &measure_ij1j2);
      V0_dip_i += factor_j1j2 * V0_dip_ij1j2;

      //! Compute meanfield quadrupole angular term
      double V0_quad_ij1j2 = 0.0;
      meanfield_integral(&V0_quad_ij1j2, V_quadrupole_ij1j2, &measure_ij1j2);
      V0_quad_i += factor_j1j2 * V0_quad_ij1j2;

      //! Remove measure
      destroy_gaussian_measure(&measure_ij1j2);
    }
  }

  //! @brief Add the contribution of the embedded forces
  CubicSpline embed_ii;
  if (spc_i == Mg) {
    embed_ii = adp_MgMg.embed;
  } else if (spc_i == H) {
    embed_ii = adp_HH.embed;
  }
  double mf_F_i = cubic_spline(&embed_ii, mf_rho_i);
  V0_embed_i = xi_i * mf_F_i;

  //! @brief Add up each contribution to the meanfield potential
  V0_i = V0_embed_i + V0_pair_i + V0_dip_i + V0_quad_i;

  //! @brief Compute mean meanfield Hamiltonian
  double H0_i = 1.0 / (2.0 * beta_i) + V0_i;

  //! @brief Compute log of the grand-cannonical partition function
  double log_Z0_i = 3.0 * log((stdv_q_i * sqrt(m_i / beta_i)) / h_planck);
  if (xi_i < 1.0) {
    log_Z0_i += log(1.0 / (1.0 - xi_i));
  }

  //! @brief Chemical multiplier
  double gamma_xi_i = 0.0;
  if (spc_i == H) {
    gamma_xi_i = gamma_i * (xi_i > 0.9999 ? 0.9999 : xi_i);
  }

  //! @brief Compute the meanfield Entropy
  double S0_i = -log_Z0_i + beta_i * H0_i - gamma_xi_i;

  return S0_i;
}

/********************************************************************************/

Eigen::Vector3d evaluate_DV0_i_Dmeanq_u_MgHx(
    unsigned int site_i_star,            //!
    unsigned int site_i,                 //!
    const Eigen::MatrixXd& mean_q,       //!
    const Eigen::VectorXd& stdv_q,       //!
    const Eigen::VectorXd& xi,           //!
    const Eigen::VectorXd& mf_rho,       //!
    const AtomicSpecie* specie,          //!
    const AtomTopology atom_topology_i)  //!
{

  //! @brief Auxiliar atomistic variables
  unsigned int dim = NumberDimensions;

  //! @brief Define Integration rule
  void (*meanfield_integral)(double* integral_f, potential_function function,
                             void* ctx_measure);
  void (*meanfield_integral_dmq)(int direction, double* integral_grad_f,
                                 potential_function function,
                                 void* ctx_measure);
#if defined(MULTIPOLE_INTEGRAL)
  meanfield_integral = meanfield_integral_mp;
  meanfield_integral_dmq = meanfield_integral_mp_dmq;
#elif defined(GH3TH_INTEGRAL)
  meanfield_integral = meanfield_integral_gh3th;
  meanfield_integral_dmq = meanfield_integral_gh_3th_dmq;
#else
#error "Define MULTIPOLE_INTEGRAL or GH3TH_INTEGRAL"
#endif

  //! @brief If we are in a hydrogen site with a occupancy below a certain
  //! thereshold, skip the evaluation of this equation
  double xi_i_star = xi(site_i_star);
  AtomicSpecie spc_i_star = specie[site_i_star];
  if ((spc_i_star == H) && (xi_i_star < min_occupancy)) {
    return Eigen::Vector3d::Zero();
  }

  //! @brief Auxiliar variables
  double mf_rho_i = mf_rho(site_i);
  Eigen::Vector3d D_V0_i_embed_Dq = Eigen::Vector3d::Zero();
  Eigen::Vector3d mf_D_rho_i_Dq = Eigen::Vector3d::Zero();
  Eigen::Vector3d D_V0_i_pair_Dq = Eigen::Vector3d::Zero();
  Eigen::Vector3d D_V0_i_dipol_Dq = Eigen::Vector3d::Zero();
  Eigen::Vector3d D_V0_i_quad_Dq = Eigen::Vector3d::Zero();

  //! @brief Get topologic information of site i
  unsigned int numneigh_site_i = atom_topology_i.numneigh;
  const PetscInt* mech_neighs_i = atom_topology_i.mech_neighs_ptr;

  //! @brief Get atomistic information of site i
  AtomicSpecie spc_i = specie[site_i];
  double xi_i = xi(site_i);
  double stdv_q_i = stdv_q(site_i);
  Eigen::Vector3d mean_q_i = mean_q.block<1, 3>(site_i, 0);

  //! If the site is empty, skip from the evaluation
  if ((spc_i == H) && (xi_i < min_occupancy)) {
    return Eigen::Vector3d::Zero();
  }

  //! @brief Compute the gradient of the potential with respect the
  //! mean value of the position at site i
  for (unsigned int idx_j1 = 0; idx_j1 < numneigh_site_i; idx_j1++) {

    //! @brief Get atomistic information of site j
    unsigned int site_j1 = mech_neighs_i[idx_j1];
    AtomicSpecie spc_j1 = specie[site_j1];
    double xi_j1 = xi(site_j1);
    double stdv_q_j1 = stdv_q(site_j1);
    Eigen::Vector3d mean_q_j1 = mean_q.block<1, 3>(site_j1, 0);

    //! If the site is empty, skip from the evaluation
    if ((spc_j1 == H) && (xi_j1 <= min_occupancy)) {
      continue;
    }

    //! @brief Create dof table
    int dof_table_ij[4] = {1, 0, 0, 1};

    //! @brief Fill data for the measure (i,j1)
    Eigen::VectorXd mean_q_ij1(6);
    mean_q_ij1 << mean_q_i, mean_q_j1;
    Eigen::VectorXd stdv_q_ij1(2);
    stdv_q_ij1 << stdv_q_i, stdv_q_j1;
    Eigen::VectorXd xi_ij1(2);
    xi_ij1 << xi_i, xi_j1;
    Eigen::VectorXd sites_ij1(2);
    sites_ij1 << site_i, site_j1;
    AtomicSpecie spc_ij1[2] = {spc_i, spc_j1};

    //! Create measure/functions
    gaussian_measure_ctx measure_ij1 =
        fill_out_gaussian_measure(mean_q_ij1.data(), stdv_q_ij1.data(),
                                  xi_ij1.data(), spc_ij1, dof_table_ij, 2);
    potential_function rho_ij = rho_ij_adp_MgHx_constructor();
    potential_function V_pair_ij = V_pair_ij_adp_MgHx_constructor();

    //! @brief Compute gradient terms
    Eigen::Vector3d mf_d_rho_ij1_dq;
    Eigen::Vector3d d_V0_pair_ij1_dq;

    for (unsigned int direction = 0; direction < 2; direction++) {
      if (sites_ij1(direction) == site_i_star) {

        //! @brief Compute meanfield mean-q-grad energy density
        mf_d_rho_ij1_dq.setZero();
        meanfield_integral_dmq(direction, mf_d_rho_ij1_dq.data(), rho_ij,
                               &measure_ij1);

        //! @brief Compute meanfield mean-q-grad pairing term
        d_V0_pair_ij1_dq.setZero();
        meanfield_integral_dmq(direction, d_V0_pair_ij1_dq.data(), V_pair_ij,
                               &measure_ij1);

        //! @brief Add up partial contributions
        mf_D_rho_i_Dq += mf_d_rho_ij1_dq;
        D_V0_i_pair_Dq += d_V0_pair_ij1_dq;
      }
    }

    //! Remove integral measure (i,j1)
    destroy_gaussian_measure(&measure_ij1);

    //! @brief Loop in the neighborhood considering the simmetry of the
    //! opration a_ij1 * a_ij2 = a_ij2 * a_ij1
    for (unsigned int idx_j2 = idx_j1; idx_j2 < numneigh_site_i; idx_j2++) {

      //! @brief Get atomistic information of site j2
      unsigned int site_j2 = mech_neighs_i[idx_j2];
      AtomicSpecie spc_j2 = specie[site_j2];
      double xi_j2 = xi(site_j2);
      double stdv_q_j2 = stdv_q(site_j2);
      Eigen::Vector3d mean_q_j2 = mean_q.block<1, 3>(site_j2, 0);

      //! If the site is empty, skip from the evaluation
      if ((spc_j2 == H) && (xi_j2 <= min_occupancy)) {
        continue;
      }

      //! @brief Create dof table
      int dof_table_ij1j2[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

      //! Modify dof table for i, j1 and j2
      if (site_j1 == site_j2) {
        dof_table_ij1j2[5] = 1;
        dof_table_ij1j2[7] = 1;
      }

      //! @brief Factor to consider the simmetry of the opration
      //! a_ij1*a_ij2 = a_ij2*a_ij1
      double factor_j1j2 = idx_j1 == idx_j2 ? 1.0 : 2.0;

      //! @brief Fill data for the measure (i,j1,j2)
      Eigen::VectorXd mean_q_ij1j2(9);
      mean_q_ij1j2 << mean_q_i, mean_q_j1, mean_q_j2;
      Eigen::VectorXd stdv_q_ij1j2(3);
      stdv_q_ij1j2 << stdv_q_i, stdv_q_j1, stdv_q_j2;
      Eigen::VectorXd xi_ij1j2(3);
      xi_ij1j2 << xi_i, xi_j1, xi_j2;
      Eigen::VectorXd sites_ij1j2(3);
      sites_ij1j2 << site_i, site_j1, site_j2;
      AtomicSpecie spc_ij1j2[3] = {spc_i, spc_j1, spc_j2};

      //! Create measure/functions
      gaussian_measure_ctx measure_ij1j2 = fill_out_gaussian_measure(
          mean_q_ij1j2.data(), stdv_q_ij1j2.data(), xi_ij1j2.data(), spc_ij1j2,
          dof_table_ij1j2, 3);
      potential_function V_dipole_ij1j2 = V_dipole_ij1j2_adp_MgHx_constructor();
      potential_function V_quadrupole_ij1j2 =
          V_quadrupole_ij1j2_adp_MgHx_constructor();

      //! @brief Compute gradient terms
      Eigen::Vector3d dV0_dipole_ij1j2_dq;
      Eigen::Vector3d dV0_quadrupole_ij1j2_dq;

      for (unsigned int direction = 0; direction < 3; direction++) {
        if (sites_ij1j2(direction) == site_i_star) {

          //! @brief Compute meanfield mean-q-grad dipole angular term
          dV0_dipole_ij1j2_dq.setZero();
          meanfield_integral_dmq(direction, dV0_dipole_ij1j2_dq.data(),
                                 V_dipole_ij1j2, &measure_ij1j2);

          //! @brief Compute meanfield mean-q-grad quadrupole angular
          //! term
          dV0_quadrupole_ij1j2_dq.setZero();
          meanfield_integral_dmq(direction, dV0_quadrupole_ij1j2_dq.data(),
                                 V_quadrupole_ij1j2, &measure_ij1j2);

          //! @brief Add up partial contributions
          D_V0_i_dipol_Dq += factor_j1j2 * dV0_dipole_ij1j2_dq;
          D_V0_i_quad_Dq += factor_j1j2 * dV0_quadrupole_ij1j2_dq;
        }
      }
      //! Remove integral measure (i,j1,j2)
      destroy_gaussian_measure(&measure_ij1j2);
    }
  }

  //! @brief Compute embedding forces
  CubicSpline embed_ii;
  if (spc_i == Mg) {
    embed_ii = adp_MgMg.embed;
  } else if (spc_i == H) {
    embed_ii = adp_HH.embed;
  }
  double d_F_embed_i = d_cubic_spline(&embed_ii, mf_rho_i);
  D_V0_i_embed_Dq = xi_i * d_F_embed_i * mf_D_rho_i_Dq;

  //! @brief Assembly the gradient of the potential V0 at the site i with
  //! respect the mean value of q
  Eigen::Vector3d D_V0_i_Dq_i_star =
      D_V0_i_embed_Dq + D_V0_i_pair_Dq + D_V0_i_dipol_Dq + D_V0_i_quad_Dq;

  return D_V0_i_Dq_i_star;
}

/********************************************************************************/

Eigen::Matrix3d evaluate_D2V0_i_Dmeanq2_u_MgHx(
    unsigned int site_i_star,            //!
    unsigned int site_i,                 //!
    const Eigen::MatrixXd& mean_q,       //!
    const Eigen::VectorXd& stdv_q,       //!
    const Eigen::VectorXd& xi,           //!
    const Eigen::VectorXd& mf_rho,       //!
    const AtomicSpecie* specie,          //!
    const AtomTopology atom_topology_i)  //!
{

  //! @brief Auxiliar atomistic variables
  unsigned int dim = NumberDimensions;

  //! @brief Define Integration rule
  void (*meanfield_integral)(double* integral_f, potential_function function,
                             void* ctx_measure);
  void (*meanfield_integral_dmq)(int direction, double* integral_grad_f,
                                 potential_function function,
                                 void* ctx_measure);
#if defined(MULTIPOLE_INTEGRAL)
  meanfield_integral = meanfield_integral_mp;
  meanfield_integral_dmq = meanfield_integral_mp_dmq;
#elif defined(GH3TH_INTEGRAL)
  meanfield_integral = meanfield_integral_gh3th;
  meanfield_integral_dmq = meanfield_integral_gh_3th_dmq;
#else
#error "Define MULTIPOLE_INTEGRAL or GH3TH_INTEGRAL"
#endif

  //! @brief If we are in a hydrogen site with a occupancy below a certain
  //! thereshold, skip the evaluation of this equation
  double xi_i_star = xi(site_i_star);
  AtomicSpecie spc_i_star = specie[site_i_star];
  if ((spc_i_star == H) && (xi_i_star < min_occupancy)) {
    return Eigen::Matrix3d::Zero();
  }

  //! @brief Compute the hessian of the meanfield potential with respect
  //! the mean value of the position at site i
  double mf_rho_i = mf_rho(site_i);
  Eigen::Matrix3d D2_V0_i_embed_Dq2 = Eigen::Matrix3d::Zero();
  Eigen::Vector3d mf_D_rho_i_Dq = Eigen::Vector3d::Zero();
  Eigen::Matrix3d mf_D2_rho_i_Dq2 = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d D2_V0_i_pair_Dq2 = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d D2_V0_i_dipol_Dq2 = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d D2_V0_i_quad_Dq2 = Eigen::Matrix3d::Zero();

  //! @brief Get topologic information of site i
  unsigned int numneigh_site_i = atom_topology_i.numneigh;
  const PetscInt* mech_neighs_i = atom_topology_i.mech_neighs_ptr;

  //! @brief Get atomistic information of site i
  AtomicSpecie spc_i = specie[site_i];
  double xi_i = xi(site_i);
  double stdv_q_i = stdv_q(site_i);
  Eigen::Vector3d mean_q_i = mean_q.block<1, 3>(site_i, 0);

  //! If the site is empty, skip from the evaluation
  if ((spc_i == H) && (xi_i < min_occupancy)) {
    return Eigen::Matrix3d::Zero();
  }

  for (unsigned int idx_j1 = 0; idx_j1 < numneigh_site_i; idx_j1++) {

    //! @brief Get atomistic information of site j
    unsigned int site_j1 = mech_neighs_i[idx_j1];
    AtomicSpecie spc_j1 = specie[site_j1];
    double xi_j1 = xi(site_j1);
    double stdv_q_j1 = stdv_q(site_j1);
    Eigen::Vector3d mean_q_j1 = mean_q.block<1, 3>(site_j1, 0);

    //! If the site is empty, skip from the evaluation
    if ((spc_j1 == H) && (xi_j1 <= min_occupancy)) {
      continue;
    }

    //! @brief Create dof table
    int dof_table_ij[4] = {1, 0, 0, 1};

    //! @brief Fill data for the measure (i,j1)
    Eigen::VectorXd mean_q_ij1(6);
    mean_q_ij1 << mean_q_i, mean_q_j1;
    Eigen::VectorXd stdv_q_ij1(2);
    stdv_q_ij1 << stdv_q_i, stdv_q_j1;
    Eigen::VectorXd xi_ij1(2);
    xi_ij1 << xi_i, xi_j1;
    Eigen::VectorXd sites_ij1(2);
    sites_ij1 << site_i, site_j1;
    AtomicSpecie spc_ij1[2] = {spc_i, spc_j1};

    //! Define function
    potential_function rho_ij = rho_ij_adp_MgHx_constructor();
    potential_function V_pair_ij = V_pair_ij_adp_MgHx_constructor();

    //! Create measure
    gaussian_measure_ctx measure_ij =
        fill_out_gaussian_measure(mean_q_ij1.data(), stdv_q_ij1.data(),
                                  xi_ij1.data(), spc_ij1, dof_table_ij, 2);

    //! @brief Compute gradient/hessian terms
    Eigen::Vector3d mf_d_rho_ij1_dq;
    Eigen::Matrix3d mf_d2_rho_ij1_dq2;
    Eigen::Matrix3d mf_d2_V0_pair_ij1_dq2;

    for (unsigned int direction = 0; direction < 2; direction++) {
      if (sites_ij1(direction) == site_i_star) {

        //! @brief Compute meanfield mean-q-grad energy density
        mf_d_rho_ij1_dq.setZero();
        meanfield_integral_dmq(direction, mf_d_rho_ij1_dq.data(), rho_ij,
                               &measure_ij);

        //! @brief Compute meanfield mean-q-hess energy density
        mf_d2_rho_ij1_dq2.setZero();
        meanfield_integral_gh_3th_d2mq(direction * 2 + direction,
                                       mf_d2_rho_ij1_dq2.data(), rho_ij,
                                       &measure_ij);

        //! @brief Compute meanfield mean-q-hess pairing term
        mf_d2_V0_pair_ij1_dq2.setZero();
        meanfield_integral_gh_3th_d2mq(direction * 2 + direction,
                                       mf_d2_V0_pair_ij1_dq2.data(), V_pair_ij,
                                       &measure_ij);

        //! @brief Add up partial contributions
        mf_D_rho_i_Dq += mf_d_rho_ij1_dq;
        mf_D2_rho_i_Dq2 += mf_d2_rho_ij1_dq2;
        D2_V0_i_pair_Dq2 += mf_d2_V0_pair_ij1_dq2;
      }
    }

    //! Remove measures
    destroy_gaussian_measure(&measure_ij);

    //! @brief Loop in the neighborhood considering the simmetry of the
    //! opration a_ij1*a_ij2 = a_ij2*a_ij1
    for (unsigned int idx_j2 = idx_j1; idx_j2 < numneigh_site_i; idx_j2++) {

      //! @brief Get atomistic information of site j_2
      unsigned int site_j2 = mech_neighs_i[idx_j2];
      AtomicSpecie spc_j2 = specie[site_j2];
      double xi_j2 = xi(site_j2);
      double stdv_q_j2 = stdv_q(site_j2);
      Eigen::Vector3d mean_q_j2 = mean_q.block<1, 3>(site_j2, 0);

      //! If the site is empty, skip from the evaluation
      if ((spc_j2 == H) && (xi_j2 <= min_occupancy)) {
        continue;
      }

      //! @brief Create dof table
      int dof_table_ij1j2[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

      //! Modify dof table for i, j1 and j2
      if (site_j1 == site_j2) {
        dof_table_ij1j2[5] = 1;
        dof_table_ij1j2[7] = 1;
      }

      //! @brief Factor to consider the simmetry of the operation
      //! a_ij*a_ik = a_ik*a_ij
      double factor_j1j2 = idx_j2 == idx_j1 ? 1.0 : 2.0;

      //! @brief Fill data for the measure (i,j1,j2)
      Eigen::VectorXd mean_q_ij1j2(9);
      mean_q_ij1j2 << mean_q_i, mean_q_j1, mean_q_j2;
      Eigen::VectorXd stdv_q_ij1j2(3);
      stdv_q_ij1j2 << stdv_q_i, stdv_q_j1, stdv_q_j2;
      Eigen::VectorXd xi_ij1j2(3);
      xi_ij1j2 << xi_i, xi_j1, xi_j2;
      Eigen::VectorXd sites_ij1j2(3);
      sites_ij1j2 << site_i, site_j1, site_j2;
      AtomicSpecie spc_ij1j2[3] = {spc_i, spc_j1, spc_j2};

      //! Create measure/functions
      gaussian_measure_ctx measure_ij1j2 = fill_out_gaussian_measure(
          mean_q_ij1j2.data(), stdv_q_ij1j2.data(), xi_ij1j2.data(), spc_ij1j2,
          dof_table_ij1j2, 3);

      potential_function V_dipole_ij1j2 = V_dipole_ij1j2_adp_MgHx_constructor();
      potential_function V_quadrupole_ij1j2 =
          V_quadrupole_ij1j2_adp_MgHx_constructor();

      //! @brief Compute hessian terms
      Eigen::Matrix3d d2V0_dipole_ij1j2_dq2;
      Eigen::Matrix3d d2V0_quadrupole_ij1j2_dq2;

      for (unsigned int direction = 0; direction < 3; direction++) {
        if (sites_ij1j2(direction) == site_i_star) {

          //! @brief Compute meanfield mean-q-hess dipole angular term
          d2V0_dipole_ij1j2_dq2.setZero();
          meanfield_integral_gh_3th_d2mq(direction * 3 + direction,
                                         d2V0_dipole_ij1j2_dq2.data(),
                                         V_dipole_ij1j2, &measure_ij1j2);

          //! @brief Compute meanfield mean-q-hess quadrupole angular
          //! term
          d2V0_quadrupole_ij1j2_dq2.setZero();
          meanfield_integral_gh_3th_d2mq(direction * 3 + direction,
                                         d2V0_quadrupole_ij1j2_dq2.data(),
                                         V_quadrupole_ij1j2, &measure_ij1j2);

          //! @brief Add up partial contributions
          D2_V0_i_dipol_Dq2 += factor_j1j2 * d2V0_dipole_ij1j2_dq2;
          D2_V0_i_quad_Dq2 += factor_j1j2 * d2V0_quadrupole_ij1j2_dq2;
        }
      }

      //! Remove measures
      destroy_gaussian_measure(&measure_ij1j2);
    }
  }

  //! @brief Compute embedding hessian
  CubicSpline embed_ii;
  if (spc_i == Mg) {
    embed_ii = adp_MgMg.embed;
  } else if (spc_i == H) {
    embed_ii = adp_HH.embed;
  }
  double mf_d_F_i = d_cubic_spline(&embed_ii, mf_rho_i);
  double mf_dd_F_i = d2_cubic_spline(&embed_ii, mf_rho_i);
  D2_V0_i_embed_Dq2 =
      xi_i * mf_dd_F_i * (mf_D_rho_i_Dq * mf_D_rho_i_Dq.transpose()) +
      xi_i * mf_d_F_i * mf_D2_rho_i_Dq2;

  //! @brief Assembly the hessian of the potential at each position
  Eigen::Matrix3d D2_V0_i_Dq2_i_star = D2_V0_i_embed_Dq2 + D2_V0_i_pair_Dq2 +
                                       D2_V0_i_dipol_Dq2 + D2_V0_i_quad_Dq2;

  return D2_V0_i_Dq2_i_star;
}

/********************************************************************************/

double evaluate_DV0_i_Dstdvq_u_MgHx(unsigned int site_u,                 //!
                                    unsigned int site_i,                 //!
                                    const Eigen::MatrixXd& mean_q,       //!
                                    const Eigen::VectorXd& stdv_q,       //!
                                    const Eigen::VectorXd& xi,           //!
                                    const Eigen::VectorXd& mf_rho,       //!
                                    const AtomicSpecie* specie,          //!
                                    const AtomTopology atom_topology_i)  //!
{

  //! @brief Auxiliar atomistic variables
  unsigned int dim = NumberDimensions;

  //! @brief Define Integration rule
  void (*meanfield_integral)(double* integral_f, potential_function function,
                             void* ctx_measure);
  void (*meanfield_integral_dsq)(int direction, double* integral_grad_f,
                                 potential_function function,
                                 void* ctx_measure);
#if defined(MULTIPOLE_INTEGRAL)
  meanfield_integral = meanfield_integral_mp;
  meanfield_integral_dsq = meanfield_integral_mp_dsq;
#elif defined(GH3TH_INTEGRAL)
  meanfield_integral = meanfield_integral_gh3th;
  meanfield_integral_dsq = meanfield_integral_gh3th_dsq;
#else
#error "Define MULTIPOLE_INTEGRAL or GH3TH_INTEGRAL"
#endif

  //! @brief If we are in a hydrogen site with a occupancy below a certain
  //! thereshold, skip the evaluation of this equation
  double xi_u = xi(site_u);
  if (xi_u < min_occupancy) {
    return 0.0;
  }

  //! @brief Compute the gradient of the meanfield potential with respect the
  //! standard desviation of the position
  double mf_rho_i = mf_rho(site_i);
  double mf_D_rho_i_Dsq = 0.0;
  double D_V0_i_embed_Dsq = 0.0;
  double D_V0_i_pair_Dsq = 0.0;
  double D_V0_i_dipol_Dsq = 0.0;
  double D_V0_i_quad_Dsq = 0.0;

  //! @brief Get topologic information of site i
  unsigned int numneigh_site_i = atom_topology_i.numneigh;
  const PetscInt* mech_neighs_i = atom_topology_i.mech_neighs_ptr;

  //! @brief Get atomistic information of site i
  AtomicSpecie spc_i = specie[site_i];
  double xi_i = xi(site_i);
  double stdv_q_i = stdv_q(site_i);
  Eigen::Vector3d mean_q_i = mean_q.block<1, 3>(site_i, 0);

  //! If the site is empty, skip from the evaluation
  if (xi_i < min_occupancy) {
    return 0.0;
  }

  for (unsigned int idx_j1 = 0; idx_j1 < numneigh_site_i; idx_j1++) {

    //! @brief Get atomistic information of site j
    unsigned int site_j1 = mech_neighs_i[idx_j1];
    AtomicSpecie spc_j1 = specie[site_j1];
    double xi_j1 = xi(site_j1);
    double stdv_q_j1 = stdv_q(site_j1);
    Eigen::Vector3d mean_q_j1 = mean_q.block<1, 3>(site_j1, 0);

    //! If the site is empty, skip from the evaluation
    if (xi_j1 < min_occupancy) {
      continue;
    }

    //! @brief Create dof table
    int dof_table_ij[4] = {1, 0, 0, 1};

    //! @brief Fill data for the measure
    Eigen::VectorXd mean_q_ij1(6);
    mean_q_ij1 << mean_q_i, mean_q_j1;
    Eigen::VectorXd stdv_q_ij1(2);
    stdv_q_ij1 << stdv_q_i, stdv_q_j1;
    Eigen::VectorXd xi_ij1(2);
    xi_ij1 << xi_i, xi_j1;
    Eigen::VectorXd sites_ij1(2);
    sites_ij1 << site_i, site_j1;
    AtomicSpecie spc_ij1[2] = {spc_i, spc_j1};

    //! Define function
    potential_function rho_ij = rho_ij_adp_MgHx_constructor();
    potential_function V_pair_ij = V_pair_ij_adp_MgHx_constructor();

    //! Create measure
    gaussian_measure_ctx measure_ij =
        fill_out_gaussian_measure(mean_q_ij1.data(), stdv_q_ij1.data(),
                                  xi_ij1.data(), spc_ij1, dof_table_ij, 2);

    //! @brief Compute gradient terms
    double mf_D_rho_i_Dsqj;
    double D_V0_i_pair_Dsqj;

    for (unsigned int direction = 0; direction < 2; direction++) {
      if (sites_ij1(direction) == site_u) {

        //! @brief Compute meanfield stdv-q-grad energy density
        mf_D_rho_i_Dsqj = 0.0;
        meanfield_integral_dsq(direction, &mf_D_rho_i_Dsqj, rho_ij,
                               &measure_ij);

        //! @brief Compute meanfield stdv-q-grad pairing term
        D_V0_i_pair_Dsqj = 0.0;
        meanfield_integral_dsq(direction, &D_V0_i_pair_Dsqj, V_pair_ij,
                               &measure_ij);

        //! @brief Add up partial contributions
        mf_D_rho_i_Dsq += mf_D_rho_i_Dsqj;
        D_V0_i_pair_Dsq += D_V0_i_pair_Dsqj;
      }
    }

    //! Remove measure
    destroy_gaussian_measure(&measure_ij);

    //! @brief Loop in the neighborhood considering the simmetry of the
    //! operation a_ij*a_ik = a_ik*a_ij
    for (unsigned int idx_j2 = idx_j1; idx_j2 < numneigh_site_i; idx_j2++) {

      //! @brief Get atomistic information of site j2
      unsigned int site_j2 = mech_neighs_i[idx_j2];
      AtomicSpecie spc_j2 = specie[site_j2];
      double xi_j2 = xi(site_j2);
      double stdv_q_j2 = stdv_q(site_j2);
      Eigen::Vector3d mean_q_j2 = mean_q.block<1, 3>(site_j2, 0);

      //! If the site is empty, skip from the evaluation
      if (xi_j2 < min_occupancy) {
        continue;
      }

      //! @brief Create dof table
      int dof_table_ij1j2[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

      //! Modify dof table for i, j1 and j2
      if (site_j1 == site_j2) {
        dof_table_ij1j2[5] = 1;
        dof_table_ij1j2[7] = 1;
      }

      //! @brief Factor to consider the simmetry of the opration a_ij*a_ik
      //! = a_ik*a_ij
      double factor_j1j2 = (idx_j2 == idx_j1) ? 1.0 : 2.0;

      //! @brief Fill data for the measure
      Eigen::VectorXd mean_q_ij1j2(9);
      mean_q_ij1j2 << mean_q_i, mean_q_j1, mean_q_j2;
      Eigen::VectorXd stdv_q_ij1j2(3);
      stdv_q_ij1j2 << stdv_q_i, stdv_q_j1, stdv_q_j2;
      Eigen::VectorXd xi_ij1j2(3);
      xi_ij1j2 << xi_i, xi_j1, xi_j2;
      Eigen::VectorXd sites_ij1j2(3);
      sites_ij1j2 << site_i, site_j1, site_j2;
      AtomicSpecie spc_ij1j2[3] = {spc_i, spc_j1, spc_j2};

      //! Create measure/functions
      gaussian_measure_ctx measure_ij1j2 = fill_out_gaussian_measure(
          mean_q_ij1j2.data(), stdv_q_ij1j2.data(), xi_ij1j2.data(), spc_ij1j2,
          dof_table_ij1j2, 3);
      potential_function V_dipole_ij1j2 = V_dipole_ij1j2_adp_MgHx_constructor();
      potential_function V_quadrupole_ij1j2 =
          V_quadrupole_ij1j2_adp_MgHx_constructor();

      //! @brief Compute gradient terms
      double D_V0_i_dipol_Dsqj1j2;
      double D_V0_i_quad_Dsqj1j2;

      for (unsigned int direction = 0; direction < 3; direction++) {
        if (sites_ij1j2(direction) == site_u) {

          //! @brief Compute meanfield stdv-q-grad dipole angular term
          D_V0_i_dipol_Dsqj1j2 = 0.0;
          meanfield_integral_dsq(direction, &D_V0_i_dipol_Dsqj1j2,
                                 V_dipole_ij1j2, &measure_ij1j2);

          //! @brief Compute meanfield stdv-q-grad quadrupole angular term
          D_V0_i_quad_Dsqj1j2 = 0.0;
          meanfield_integral_dsq(direction, &D_V0_i_quad_Dsqj1j2,
                                 V_quadrupole_ij1j2, &measure_ij1j2);

          //! @brief Add up the partial contribution
          D_V0_i_dipol_Dsq += factor_j1j2 * D_V0_i_dipol_Dsqj1j2;
          D_V0_i_quad_Dsq += factor_j1j2 * D_V0_i_quad_Dsqj1j2;
        }
      }

      //! Remove measures
      destroy_gaussian_measure(&measure_ij1j2);
    }
  }

  //! @brief Compute the gradient of the embedding term
  CubicSpline embed_ii;
  if (spc_i == Mg) {
    embed_ii = adp_MgMg.embed;
  } else if (spc_i == H) {
    embed_ii = adp_HH.embed;
  }
  D_V0_i_embed_Dsq =
      xi_i * d_cubic_spline(&embed_ii, mf_rho_i) * mf_D_rho_i_Dsq;

  //! @brief Compute the contribution of the site i
  double dV0_i_dsq_u = D_V0_i_embed_Dsq +  //! Embedded
                       D_V0_i_pair_Dsq +   //! Pairing
                       D_V0_i_dipol_Dsq +  //!  Dipole
                       D_V0_i_quad_Dsq;    //! Quadrupole

  return dV0_i_dsq_u;
}

/********************************************************************************/

Eigen::Vector4d evaluate_DV0_i_Dq_u_MgHx(
    unsigned int site_i_star,            //!
    unsigned int site_i,                 //!
    const Eigen::MatrixXd& mean_q,       //! Mean value of q
    const Eigen::VectorXd& stdv_q,       //! Standard desviation of q
    const Eigen::VectorXd& xi,           //! Molar fraction
    const Eigen::VectorXd& mf_rho,       //!
    const AtomicSpecie* specie,          //! Atom
    const AtomTopology atom_topology_i)  //!
{

  //! @brief Auxiliar atomistic variables
  unsigned int dim = NumberDimensions;

  //! @brief Define Integration rule
  void (*meanfield_integral)(double* integral_f, potential_function function,
                             void* ctx_measure);
  void (*meanfield_integral_dmq)(int direction, double* integral_grad_f,
                                 potential_function function,
                                 void* ctx_measure);
  void (*meanfield_integral_dsq)(int direction, double* integral_grad_f,
                                 potential_function function,
                                 void* ctx_measure);
#if defined(MULTIPOLE_INTEGRAL)
  meanfield_integral = meanfield_integral_mp;
  meanfield_integral_dmq = meanfield_integral_mp_dmq;
  meanfield_integral_dsq = meanfield_integral_mp_dsq;
#elif defined(GH3TH_INTEGRAL)
  meanfield_integral = meanfield_integral_gh3th;
  meanfield_integral_dmq = meanfield_integral_gh_3th_dmq;
  meanfield_integral_dsq = meanfield_integral_gh3th_dsq;
#else
#error "Define MULTIPOLE_INTEGRAL or GH3TH_INTEGRAL"
#endif

  //! @brief If we are in a hydrogen site with a occupancy below a certain
  //! thereshold, skip the evaluation of this equation
  double xi_i_star = xi(site_i_star);
  AtomicSpecie spc_i_star = specie[site_i_star];
  if (xi_i_star < min_occupancy) {
    return Eigen::Vector4d::Zero();
  }

  //! @brief Compute the gradient of the meanfield potential with respect the
  //! mean value of the position
  double mf_rho_i = mf_rho(site_i);
  Eigen::Vector3d D_V0_i_embed_Dmq_i_star = Eigen::Vector3d::Zero();
  Eigen::Vector3d mf_D_rho_i_Dmq_i_star = Eigen::Vector3d::Zero();
  Eigen::Vector3d D_V0_i_pair_Dmq_i_star = Eigen::Vector3d::Zero();
  Eigen::Vector3d D_V0_i_dipol_Dmq_i_star = Eigen::Vector3d::Zero();
  Eigen::Vector3d D_V0_i_quad_Dmq_i_star = Eigen::Vector3d::Zero();
  double D_V0_i_embed_Dsq_i_star = 0.0;
  double mf_D_rho_i_Dsq_i_star = 0.0;
  double D_V0_i_pair_Dsq_i_star = 0.0;
  double D_V0_i_dipol_Dsq_i_star = 0.0;
  double D_V0_i_quad_Dsq_i_star = 0.0;

  //! @brief Get topologic information of site i
  unsigned int numneigh_site_i = atom_topology_i.numneigh;
  const PetscInt* mech_neighs_i = atom_topology_i.mech_neighs_ptr;

  //! @brief Get atomistic information of site i
  AtomicSpecie spc_i = specie[site_i];
  double xi_i = xi(site_i);
  double stdv_q_i = stdv_q(site_i);
  Eigen::Vector3d mean_q_i = mean_q.block<1, 3>(site_i, 0);

  //! If the site is empty, skip from the evaluation
  if (xi_i < min_occupancy) {
    return Eigen::Vector4d::Zero();
  }

  for (unsigned int idx_j1 = 0; idx_j1 < numneigh_site_i; idx_j1++) {

    //! @brief Get atomistic information of site j
    unsigned int site_j1 = mech_neighs_i[idx_j1];
    AtomicSpecie spc_j1 = specie[site_j1];
    double xi_j1 = xi(site_j1);
    double stdv_q_j1 = stdv_q(site_j1);
    Eigen::Vector3d mean_q_j1 = mean_q.block<1, 3>(site_j1, 0);

    //! If the site is empty, skip from the evaluation
    if (xi_j1 < min_occupancy) {
      continue;
    }

    //! @brief Create dof table
    int dof_table_ij[4] = {1, 0, 0, 1};

    //! @brief Fill data for the measure (i,j1)
    Eigen::VectorXd mean_q_ij1(6);
    mean_q_ij1 << mean_q_i, mean_q_j1;
    Eigen::VectorXd stdv_q_ij1(2);
    stdv_q_ij1 << stdv_q_i, stdv_q_j1;
    Eigen::VectorXd xi_ij1(2);
    xi_ij1 << xi_i, xi_j1;
    Eigen::VectorXd sites_ij1(2);
    sites_ij1 << site_i, site_j1;
    AtomicSpecie spc_ij1[2] = {spc_i, spc_j1};

    //! Define function
    potential_function rho_ij = rho_ij_adp_MgHx_constructor();
    potential_function V_pair_ij = V_pair_ij_adp_MgHx_constructor();

    //! Create measure for rho
    gaussian_measure_ctx measure_ij1 =
        fill_out_gaussian_measure(mean_q_ij1.data(), stdv_q_ij1.data(),
                                  xi_ij1.data(), spc_ij1, dof_table_ij, 2);

    //! @brief Compute gradient terms
    Eigen::Vector3d mf_D_rho_i_Dsqj1;
    Eigen::Vector3d D_V0_i_pair_Dsqj1;

    double mf_D_rho_i_Dsqj;
    double D_V0_i_pair_Dsqj;

    for (unsigned int direction = 0; direction < 2; direction++) {
      if (sites_ij1(direction) == site_i_star) {

        //! @brief Compute meanfield mean-q-grad energy density
        mf_D_rho_i_Dsqj1.setZero();
        meanfield_integral_dmq(direction, mf_D_rho_i_Dsqj1.data(), rho_ij,
                               &measure_ij1);

        //! @brief Compute meanfield mean-q-grad pairing term
        D_V0_i_pair_Dsqj1.setZero();
        meanfield_integral_dmq(direction, D_V0_i_pair_Dsqj1.data(), V_pair_ij,
                               &measure_ij1);

        //! @brief Compute meanfield stdv-q-grad energy density
        mf_D_rho_i_Dsqj = 0.0;
        meanfield_integral_dsq(direction, &mf_D_rho_i_Dsqj, rho_ij,
                               &measure_ij1);

        //! @brief Compute meanfield stdv-q-grad pairing term
        D_V0_i_pair_Dsqj = 0.0;
        meanfield_integral_dsq(direction, &D_V0_i_pair_Dsqj, V_pair_ij,
                               &measure_ij1);

        //! @brief Add up partial contributions
        mf_D_rho_i_Dmq_i_star += mf_D_rho_i_Dsqj1;
        D_V0_i_pair_Dmq_i_star += D_V0_i_pair_Dsqj1;

        mf_D_rho_i_Dsq_i_star += mf_D_rho_i_Dsqj;
        D_V0_i_pair_Dsq_i_star += D_V0_i_pair_Dsqj;
      }
    }

    //! Remove integral measure (i,j1)
    destroy_gaussian_measure(&measure_ij1);

    //! @brief Loop in the neighborhood considering the simmetry of the
    //! opration a_ij1 * a_ij2 = a_ij2 * a_ij1
    for (unsigned int idx_j2 = idx_j1; idx_j2 < numneigh_site_i; idx_j2++) {

      //! @brief Get atomistic information of site j2
      unsigned int site_j2 = mech_neighs_i[idx_j2];
      AtomicSpecie spc_j2 = specie[site_j2];
      double xi_j2 = xi(site_j2);
      double stdv_q_j2 = stdv_q(site_j2);
      Eigen::Vector3d mean_q_j2 = mean_q.block<1, 3>(site_j2, 0);

      //! If the site is empty, skip from the evaluation
      if (xi_j2 <= min_occupancy) {
        continue;
      }

      //! @brief Create dof table
      int dof_table_ij1j2[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

      //! Modify dof table for i, j1 and j2
      if (site_j1 == site_j2) {
        dof_table_ij1j2[5] = 1;
        dof_table_ij1j2[7] = 1;
      }

      //! @brief Factor to consider the simmetry of the opration
      //! a_ij1*a_ij2 = a_ij2*a_ij1
      double factor_j1j2 = idx_j1 == idx_j2 ? 1.0 : 2.0;

      //! @brief Fill data for the measure (i,j1,j2)
      Eigen::VectorXd mean_q_ij1j2(9);
      mean_q_ij1j2 << mean_q_i, mean_q_j1, mean_q_j2;
      Eigen::VectorXd stdv_q_ij1j2(3);
      stdv_q_ij1j2 << stdv_q_i, stdv_q_j1, stdv_q_j2;
      Eigen::VectorXd xi_ij1j2(3);
      xi_ij1j2 << xi_i, xi_j1, xi_j2;
      Eigen::VectorXd sites_ij1j2(3);
      sites_ij1j2 << site_i, site_j1, site_j2;
      AtomicSpecie spc_ij1j2[3] = {spc_i, spc_j1, spc_j2};

      //! Functions for angular terms
      potential_function V_dipole_ij1j2 = V_dipole_ij1j2_adp_MgHx_constructor();
      potential_function V_quadrupole_ij1j2 =
          V_quadrupole_ij1j2_adp_MgHx_constructor();

      //! Create measure for angular terms
      gaussian_measure_ctx measure_ij1j2 = fill_out_gaussian_measure(
          mean_q_ij1j2.data(), stdv_q_ij1j2.data(), xi_ij1j2.data(), spc_ij1j2,
          dof_table_ij1j2, 3);

      //! @brief Compute gradient terms
      Eigen::Vector3d D_V0_i_dipol_Dmqj1j2;
      Eigen::Vector3d D_V0_i_quad_Dmqj1j2;

      double D_V0_i_dipol_Dsqj1j2;
      double D_V0_i_quad_Dsqj1j2;

      for (unsigned int direction = 0; direction < 3; direction++) {
        if (sites_ij1j2(direction) == site_i_star) {

          //! @brief Compute meanfield mean-q-grad dipole angular term
          D_V0_i_dipol_Dmqj1j2.setZero();
          meanfield_integral_dmq(direction, D_V0_i_dipol_Dmqj1j2.data(),
                                 V_dipole_ij1j2, &measure_ij1j2);

          //! @brief Compute meanfield mean-q-grad quadrupole angular
          //! term
          D_V0_i_quad_Dmqj1j2.setZero();
          meanfield_integral_dmq(direction, D_V0_i_quad_Dmqj1j2.data(),
                                 V_quadrupole_ij1j2, &measure_ij1j2);

          //! @brief Compute meanfield stdv-q-grad dipole angular term
          D_V0_i_dipol_Dsqj1j2 = 0.0;
          meanfield_integral_dsq(direction, &D_V0_i_dipol_Dsqj1j2,
                                 V_dipole_ij1j2, &measure_ij1j2);

          //! @brief Compute meanfield stdv-q-grad quadrupole angular term
          D_V0_i_quad_Dsqj1j2 = 0.0;
          meanfield_integral_dsq(direction, &D_V0_i_quad_Dsqj1j2,
                                 V_quadrupole_ij1j2, &measure_ij1j2);

          //! @brief Add up partial contributions
          D_V0_i_dipol_Dmq_i_star += factor_j1j2 * D_V0_i_dipol_Dmqj1j2;
          D_V0_i_quad_Dmq_i_star += factor_j1j2 * D_V0_i_quad_Dmqj1j2;

          D_V0_i_dipol_Dsq_i_star += factor_j1j2 * D_V0_i_dipol_Dsqj1j2;
          D_V0_i_quad_Dsq_i_star += factor_j1j2 * D_V0_i_quad_Dsqj1j2;
        }
      }
      //! Remove integral measure (i,j1,j2)
      destroy_gaussian_measure(&measure_ij1j2);
    }
  }

  //! @brief Compute embedding forces
  CubicSpline embed_ii;
  if (spc_i == Mg) {
    embed_ii = adp_MgMg.embed;
  } else if (spc_i == H) {
    embed_ii = adp_HH.embed;
  }
  D_V0_i_embed_Dmq_i_star =
      xi_i * d_cubic_spline(&embed_ii, mf_rho_i) * mf_D_rho_i_Dmq_i_star;
  D_V0_i_embed_Dsq_i_star =
      xi_i * d_cubic_spline(&embed_ii, mf_rho_i) * mf_D_rho_i_Dsq_i_star;

  //! @brief Assembly the gradient of the potential V0 at the site i with
  //! respect the mean value of q
  Eigen::Vector3d D_V0_i_Dmq_i_star = D_V0_i_embed_Dmq_i_star +  //! Embedded
                                      D_V0_i_pair_Dmq_i_star +   //! Pairing
                                      D_V0_i_dipol_Dmq_i_star +  //!  Dipole
                                      D_V0_i_quad_Dmq_i_star;    //! Quadrupole

  //! @brief Assembly the gradient of the potential V0 at the site i with
  //! respect the standard desviation q at site i_star
  double DV0_i_Dsq_i_star = D_V0_i_embed_Dsq_i_star +  //! Embedded
                            D_V0_i_pair_Dsq_i_star +   //! Pairing
                            D_V0_i_dipol_Dsq_i_star +  //!  Dipole
                            D_V0_i_quad_Dsq_i_star;    //! Quadrupole

  Eigen::Vector4d D_V0_i_Dq_i_star = Eigen::Vector4d::Zero();
  D_V0_i_Dq_i_star << D_V0_i_Dmq_i_star, DV0_i_Dsq_i_star;

  return D_V0_i_Dq_i_star;
}

/********************************************************************************/

double evaluate_DV0_i_Dxi_u_adp_MgHx(unsigned int site_u,                 //!
                                     unsigned int site_i,                 //!
                                     const Eigen::MatrixXd& mean_q,       //!
                                     const Eigen::VectorXd& stdv_q,       //!
                                     const Eigen::VectorXd& xi,           //!
                                     const Eigen::VectorXd& mf_rho,       //!
                                     const AtomicSpecie* specie,          //!
                                     const AtomTopology atom_topology_i)  //!
{

  //! @brief Auxiliar atomistic variables
  unsigned int dim = NumberDimensions;

  //! @brief Define Integration rule
  void (*meanfield_integral)(double* integral_f, potential_function function,
                             void* ctx_measure);
#if defined(MULTIPOLE_INTEGRAL)
  meanfield_integral = meanfield_integral_mp;
#elif defined(GH3TH_INTEGRAL)
  meanfield_integral = meanfield_integral_gh3th;
#else
#error "Define MULTIPOLE_INTEGRAL or GH3TH_INTEGRAL"
#endif

  //! @brief If we are in a Magnesium site or if we are in a hydrogen site
  //! with a occupancy below a certain thereshold, skip the evaluation of this
  //! equation
  double xi_u = xi(site_u);
  AtomicSpecie spc_i_star = specie[site_u];

  //! If the site is empty, skip from the evaluation
  if (xi_u < min_occupancy) {
    return 0.0;
  }

  //! @brief Compute the gradient of the meanfield potential with respect the
  //! mean value of the occupancy at site i
  double mf_rho_i = mf_rho(site_i);
  double mf_D_rho_i_Dxi_u = 0.0;
  double D_V0_i_embed_Dxi_u = 0.0;
  double D_V0_i_pair_Dxi_u = 0.0;
  double D_V0_i_dipol_Dxi_u = 0.0;
  double D_V0_i_quad_Dxi_u = 0.0;

  //! @brief Get topologic information of site i
  unsigned int numneigh_site_i = atom_topology_i.numneigh;
  const PetscInt* mech_neighs_i = atom_topology_i.mech_neighs_ptr;

  //! @brief Get atomistic information of site i
  AtomicSpecie spc_i = specie[site_i];
  double xi_i = xi(site_i);
  double stdv_q_i = stdv_q(site_i);
  Eigen::Vector3d mean_q_i = mean_q.block<1, 3>(site_i, 0);

  //! If the site is empty, skip from the evaluation
  if (xi_i < min_occupancy) {
    return 0.0;
  }

  for (unsigned int idx_j1 = 0; idx_j1 < numneigh_site_i; idx_j1++) {

    //! @brief Get atomistic information of site j
    unsigned int site_j1 = mech_neighs_i[idx_j1];
    AtomicSpecie spc_j1 = specie[site_j1];
    double xi_j1 = xi(site_j1);
    double stdv_q_j1 = stdv_q(site_j1);
    Eigen::Vector3d mean_q_j1 = mean_q.block<1, 3>(site_j1, 0);

    //! If the site is empty, skip from the evaluation
    if (xi_j1 < min_occupancy) {
      continue;
    }

    //! @brief Create dof table
    int dof_table_ij[4] = {1, 0, 0, 1};

    //! @brief Fill data for the measure (i,j1)
    Eigen::VectorXd mean_q_ij1(6);
    mean_q_ij1 << mean_q_i, mean_q_j1;
    Eigen::VectorXd stdv_q_ij1(2);
    stdv_q_ij1 << stdv_q_i, stdv_q_j1;
    Eigen::VectorXd xi_ij1(2);
    xi_ij1 << xi_i, xi_j1;
    Eigen::VectorXd sites_ij1(2);
    sites_ij1 << site_i, site_j1;
    AtomicSpecie spc_ij1[2] = {spc_i, spc_j1};

    //! Define function
    potential_function f_rho_ij = rho_ij_adp_MgHx_constructor();
    potential_function V_pair_ij = V_pair_ij_adp_MgHx_constructor();

    //! Create measure
    gaussian_measure_ctx measure_ij1 =
        fill_out_gaussian_measure(mean_q_ij1.data(), stdv_q_ij1.data(),
                                  xi_ij1.data(), spc_ij1, dof_table_ij, 2);

    //! @brief Compute gradient terms
    double mf_D_rho_ij1_Dxi_u;
    double D_V0_ij1_pair_Dxi_u;

    for (unsigned int direction = 0; direction < 2; direction++) {
      if (sites_ij1(direction) == site_u) {

        //! @brief Compute meanfield mean-q-grad energy density
        mf_D_rho_ij1_Dxi_u = 0.0;
        meanfield_integral_gh_3th_dxi(direction, &mf_D_rho_ij1_Dxi_u, f_rho_ij,
                                      &measure_ij1);

        //! @brief Compute meanfield mean-q-grad pairing term
        D_V0_ij1_pair_Dxi_u = 0.0;
        meanfield_integral_gh_3th_dxi(direction, &D_V0_ij1_pair_Dxi_u,
                                      V_pair_ij, &measure_ij1);

        //! @brief Add up partial contributions
        mf_D_rho_i_Dxi_u += mf_D_rho_ij1_Dxi_u;
        D_V0_i_pair_Dxi_u += D_V0_ij1_pair_Dxi_u;
      }
    }

    //! Remove integral measure (i,j1)
    destroy_gaussian_measure(&measure_ij1);

    //! @brief Loop in the neighborhood considering the simmetry of the
    //! opration a_ij1 * a_ij2 = a_ij2 * a_ij1
    for (unsigned int idx_j2 = idx_j1; idx_j2 < numneigh_site_i; idx_j2++) {

      //! @brief Get atomistic information of site j2
      unsigned int site_j2 = mech_neighs_i[idx_j2];
      AtomicSpecie spc_j2 = specie[site_j2];
      double xi_j2 = xi(site_j2);
      double stdv_q_j2 = stdv_q(site_j2);
      Eigen::Vector3d mean_q_j2 = mean_q.block<1, 3>(site_j2, 0);

      //! If the site is empty, skip from the evaluation
      if (xi_j2 < min_occupancy) {
        continue;
      }

      //! @brief Create dof table
      int dof_table_ij1j2[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

      //! Modify dof table for i, j1 and j2
      if (site_j1 == site_j2) {
        dof_table_ij1j2[5] = 1;
        dof_table_ij1j2[7] = 1;
      }

      //! @brief Factor to consider the simmetry of the opration
      //! a_ij1*a_ij2 = a_ij2*a_ij1
      double factor_j1j2 = idx_j1 == idx_j2 ? 1.0 : 2.0;

      //! @brief Fill data for the measure (i,j1,j2)
      Eigen::VectorXd mean_q_ij1j2(9);
      mean_q_ij1j2 << mean_q_i, mean_q_j1, mean_q_j2;
      Eigen::VectorXd stdv_q_ij1j2(3);
      stdv_q_ij1j2 << stdv_q_i, stdv_q_j1, stdv_q_j2;
      Eigen::VectorXd xi_ij1j2(3);
      xi_ij1j2 << xi_i, xi_j1, xi_j2;
      Eigen::VectorXd sites_ij1j2(3);
      sites_ij1j2 << site_i, site_j1, site_j2;
      AtomicSpecie spc_ij1j2[3] = {spc_i, spc_j1, spc_j2};

      //! Create measure/functions
      gaussian_measure_ctx measure_ij1j2 = fill_out_gaussian_measure(
          mean_q_ij1j2.data(), stdv_q_ij1j2.data(), xi_ij1j2.data(), spc_ij1j2,
          dof_table_ij1j2, 3);
      potential_function V_dipole_ij1j2 = V_dipole_ij1j2_adp_MgHx_constructor();
      potential_function V_quadrupole_ij1j2 =
          V_quadrupole_ij1j2_adp_MgHx_constructor();

      //! @brief Compute gradient terms
      double D_V0_ij1j2_dipo_Dxi_u;
      double D_V0_ij1j2_quad_Dxi_u;

      for (unsigned int direction = 0; direction < 3; direction++) {
        if (sites_ij1j2(direction) == site_u) {

          //! @brief Compute meanfield mean-q-grad dipole angular term
          D_V0_ij1j2_dipo_Dxi_u = 0.0;
          meanfield_integral_gh_3th_dxi(direction, &D_V0_ij1j2_dipo_Dxi_u,
                                        V_dipole_ij1j2, &measure_ij1j2);

          //! @brief Compute meanfield mean-q-grad quadrupole angular term
          D_V0_ij1j2_quad_Dxi_u = 0.0;
          meanfield_integral_gh_3th_dxi(direction, &D_V0_ij1j2_quad_Dxi_u,
                                        V_quadrupole_ij1j2, &measure_ij1j2);

          //! @brief Add up partial contributions
          D_V0_i_dipol_Dxi_u += factor_j1j2 * D_V0_ij1j2_dipo_Dxi_u;
          D_V0_i_quad_Dxi_u += factor_j1j2 * D_V0_ij1j2_quad_Dxi_u;
        }
      }

      //! Remove integral measure (i,j1,j2)
      destroy_gaussian_measure(&measure_ij1j2);
    }
  }

  //! @brief Compute embedding forces
  if ((spc_i == H) && (site_i == site_u)) {
    CubicSpline embed_ii = adp_HH.embed;
    D_V0_i_embed_Dxi_u = cubic_spline(&embed_ii, mf_rho_i);
  }

  if ((spc_i == H) && (site_i != site_u)) {
    CubicSpline embed_ii = adp_HH.embed;
    D_V0_i_embed_Dxi_u =
        xi_i * d_cubic_spline(&embed_ii, mf_rho_i) * mf_D_rho_i_Dxi_u;
  }

  if (spc_i == Mg) {
    CubicSpline embed_ii = adp_MgMg.embed;
    D_V0_i_embed_Dxi_u = d_cubic_spline(&embed_ii, mf_rho_i) * mf_D_rho_i_Dxi_u;
  }

  //! @brief Assembly the gradient of the potential V0 at the site i with
  //! respect the occupancy vector
  double dV0_i_dxi_u = D_V0_i_embed_Dxi_u +  //! Embedded
                       D_V0_i_pair_Dxi_u +   //! Pairing
                       D_V0_i_dipol_Dxi_u +  //! Dipole
                       D_V0_i_quad_Dxi_u;    //! Quadrupole

  return dV0_i_dxi_u;
}

/********************************************************************************/

Eigen::Matrix3d evaluate_DV0_i_dF_bulk_MgHx(
    unsigned int site_i,                 //!
    const Eigen::MatrixXd& mean_q,       //!
    const Eigen::MatrixXd& mean_q0,      //!
    const Eigen::VectorXd& stdv_q,       //!
    const Eigen::VectorXd& xi,           //!
    const Eigen::VectorXd& mf_rho,       //!
    const AtomicSpecie* specie,          //!
    const AtomTopology atom_topology_i)  //!
{

  unsigned int dim = NumberDimensions;

  //! @brief Define Integration rule
  void (*meanfield_integral)(double* integral_f, potential_function function,
                             void* ctx_measure);
  void (*meanfield_integral_dmq)(int direction, double* integral_grad_f,
                                 potential_function function,
                                 void* ctx_measure);
#if defined(MULTIPOLE_INTEGRAL)
  meanfield_integral = meanfield_integral_mp;
  meanfield_integral_dmq = meanfield_integral_mp_dmq;
#elif defined(GH3TH_INTEGRAL)
  meanfield_integral = meanfield_integral_gh3th;
  meanfield_integral_dmq = meanfield_integral_gh_3th_dmq;
#else
#error "Define MULTIPOLE_INTEGRAL or GH3TH_INTEGRAL"
#endif

  //! @brief Get atomistic information of site i
  AtomicSpecie spc_i = specie[site_i];
  double xi_i = xi(site_i);
  double stdv_q_i = stdv_q(site_i);
  Eigen::Vector3d mean_q_i = mean_q.block<1, 3>(site_i, 0);
  Eigen::Vector3d mean_q0_i = mean_q0.block<1, 3>(site_i, 0);

  //! If the site is empty, skip from the evaluation
  if (xi_i < min_occupancy) {
    return Eigen::Matrix3d::Zero();
  }

  //! @brief Get topologic information of site i
  unsigned int numneigh_site_i = atom_topology_i.numneigh;
  const PetscInt* mech_neighs_i = atom_topology_i.mech_neighs_ptr;

  //! @brief Initialize auxiliar variables
  double mf_rho_i = mf_rho(site_i);
  Eigen::Matrix3d D_V0_i_embed_DF = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d D_rho0_i_DF = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d D_V0_i_pair_DF = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d D_V0_i_dipol_DF = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d D_V0_i_quad_DF = Eigen::Matrix3d::Zero();

  //! @brief Compute the gradient of the meanfield potential with respect the
  //! mean value of the position at site i
  for (unsigned int idx_j1 = 0; idx_j1 < numneigh_site_i; idx_j1++) {

    //! @brief Get atomistic information of site j
    unsigned int site_j1 = mech_neighs_i[idx_j1];
    AtomicSpecie spc_j1 = specie[site_j1];
    double xi_j1 = xi(site_j1);
    double stdv_q_j1 = stdv_q(site_j1);
    Eigen::Vector3d mean_q_j1 = mean_q.block<1, 3>(site_j1, 0);
    Eigen::Vector3d mean_q0_j1 = mean_q0.block<1, 3>(site_j1, 0);

    //! If the site is empty, skip from the evaluation
    if ((spc_j1 == H) && (xi_j1 < min_occupancy)) {
      continue;
    }

    //! @brief Fill data for the measure (i,j1)
    Eigen::VectorXd mean_q_ij1(6);
    mean_q_ij1 << mean_q_i, mean_q_j1;
    Eigen::VectorXd mean_q0_ij1(6);
    mean_q0_ij1 << mean_q0_i, mean_q0_j1;
    Eigen::VectorXd stdv_q_ij1(2);
    stdv_q_ij1 << stdv_q_i, stdv_q_j1;
    Eigen::VectorXd xi_ij1(2);
    xi_ij1 << xi_i, xi_j1;
    Eigen::VectorXd sites_ij1(2);
    sites_ij1 << site_i, site_j1;
    AtomicSpecie spc_ij1[2] = {spc_i, spc_j1};

    //! @brief Create dof table
    int dof_table_ij[4] = {1, 0, 0, 1};

    //! Create integral measure (i,j1)
    gaussian_measure_ctx measure_ij1 =
        fill_out_gaussian_measure(mean_q_ij1.data(), stdv_q_ij1.data(),
                                  xi_ij1.data(), spc_ij1, dof_table_ij, 2);

    potential_function rho_ij = rho_ij_adp_MgHx_constructor();
    potential_function V_pair_ij = V_pair_ij_adp_MgHx_constructor();

    //! @brief Compute gradient terms
    Eigen::Vector3d mf_d_rho_ij1_dq;
    Eigen::Vector3d d_V0_pair_ij1_dq;
    Eigen::Vector3d d_mean_q_ij1_dF;

    for (unsigned int direction = 0; direction < 2; direction++) {
      //! @brief Get direction
      d_mean_q_ij1_dF << mean_q0_ij1.segment<3>(direction * 3);

      //! @brief Compute meanfield mean-q-grad energy density
      mf_d_rho_ij1_dq.setZero();
      meanfield_integral_dmq(direction, mf_d_rho_ij1_dq.data(), rho_ij,
                             &measure_ij1);

      //! @brief Compute meanfield mean-q-grad pairing term
      d_V0_pair_ij1_dq.setZero();
      meanfield_integral_dmq(direction, d_V0_pair_ij1_dq.data(), V_pair_ij,
                             &measure_ij1);

      //! @brief Add up partial contributions
      D_rho0_i_DF += mf_d_rho_ij1_dq * d_mean_q_ij1_dF.transpose();
      D_V0_i_pair_DF += d_V0_pair_ij1_dq * d_mean_q_ij1_dF.transpose();
    }

    //! Remove integral measure (i,j1)
    destroy_gaussian_measure(&measure_ij1);

    //! @brief Loop in the neighborhood considering the simmetry of the
    //! opration a_ij1 * a_ij2 = a_ij2 * a_ij1
    for (unsigned int idx_j2 = idx_j1; idx_j2 < numneigh_site_i; idx_j2++) {

      //! @brief Compute atomistic information of site j2
      unsigned int site_j2 = mech_neighs_i[idx_j2];
      AtomicSpecie spc_j2 = specie[site_j2];
      double xi_j2 = xi(site_j2);
      double stdv_q_j2 = stdv_q(site_j2);
      Eigen::Vector3d mean_q_j2 = mean_q.block<1, 3>(site_j2, 0);
      Eigen::Vector3d mean_q0_j2 = mean_q0.block<1, 3>(site_j2, 0);

      //! If the site is empty, skip from the evaluation
      if ((spc_j2 == H) && (xi_j2 < min_occupancy)) {
        continue;
      }

      //! @brief Create dof table
      int dof_table_ij1j2[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

      //! Modify dof table for i, j1 and j2
      if (site_j1 == site_j2) {
        dof_table_ij1j2[5] = 1;
        dof_table_ij1j2[7] = 1;
      }

      //! @brief Factor to consider the simmetry of the opration
      //! a_ij1*a_ij2 = a_ij2*a_ij1
      double factor_j1j2 = idx_j1 == idx_j2 ? 1.0 : 2.0;

      //! @brief Fill data for the measure (i,j1,j2)
      Eigen::VectorXd mean_q_ij1j2(9);
      mean_q_ij1j2 << mean_q_i, mean_q_j1, mean_q_j2;
      Eigen::VectorXd mean_q0_ij1j2(9);
      mean_q0_ij1j2 << mean_q0_i, mean_q0_j1, mean_q0_j2;
      Eigen::VectorXd stdv_q_ij1j2(3);
      stdv_q_ij1j2 << stdv_q_i, stdv_q_j1, stdv_q_j2;
      Eigen::VectorXd xi_ij1j2(3);
      xi_ij1j2 << xi_i, xi_j1, xi_j2;
      Eigen::VectorXd sites_ij1j2(3);
      sites_ij1j2 << site_i, site_j1, site_j2;
      AtomicSpecie spc_ij1j2[3] = {spc_i, spc_j1, spc_j2};

      //! @brief Create measure/functions
      gaussian_measure_ctx measure_ij1j2 = fill_out_gaussian_measure(
          mean_q_ij1j2.data(), stdv_q_ij1j2.data(), xi_ij1j2.data(), spc_ij1j2,
          dof_table_ij1j2, 3);

      potential_function V_dipole_ij1j2 = V_dipole_ij1j2_adp_MgHx_constructor();
      potential_function V_quadrupole_ij1j2 =
          V_quadrupole_ij1j2_adp_MgHx_constructor();

      //! @brief Compute gradient terms
      Eigen::Vector3d dV0_dipole_ij1j2_dq;
      Eigen::Vector3d dV0_quadrupole_ij1j2_dq;
      Eigen::Vector3d d_mean_q_ij1j2_dF;

      for (unsigned int direction = 0; direction < 3; direction++) {

        //! @brief Get direction
        d_mean_q_ij1j2_dF << mean_q0_ij1j2.segment<3>(direction * 3);

        //! @brief Compute meanfield mean-q-grad dipole angular term
        dV0_dipole_ij1j2_dq.setZero();
        meanfield_integral_dmq(direction, dV0_dipole_ij1j2_dq.data(),
                               V_dipole_ij1j2, &measure_ij1j2);

        //! @brief Compute meanfield mean-q-grad quadrupole angular term
        dV0_quadrupole_ij1j2_dq.setZero();
        meanfield_integral_dmq(direction, dV0_quadrupole_ij1j2_dq.data(),
                               V_quadrupole_ij1j2, &measure_ij1j2);

        //! @brief Add up partial contributions
        D_V0_i_dipol_DF +=
            factor_j1j2 * dV0_dipole_ij1j2_dq * d_mean_q_ij1j2_dF.transpose();
        D_V0_i_quad_DF += factor_j1j2 * dV0_quadrupole_ij1j2_dq *
                          d_mean_q_ij1j2_dF.transpose();
      }

      //! Remove integral measure (i,j1,j2)
      destroy_gaussian_measure(&measure_ij1j2);
    }
  }

  //! @brief Compute embedding forces
  CubicSpline embed_ii;
  if (spc_i == Mg) {
    embed_ii = adp_MgMg.embed;
  } else if (spc_i == H) {
    embed_ii = adp_HH.embed;
  }
  double d_F_embed_i = d_cubic_spline(&embed_ii, mf_rho_i);
  D_V0_i_embed_DF = xi_i * d_F_embed_i * D_rho0_i_DF;

  //! @brief Assembly the gradient of the potential V0 at the site i with
  //! respect the stretch tensor U
  Eigen::Matrix3d D_V0_i_DF =
      D_V0_i_embed_DF + D_V0_i_pair_DF + D_V0_i_dipol_DF + D_V0_i_quad_DF;

  return D_V0_i_DF;
}

/********************************************************************************/
