/**
 * @file AlCu-mf-V-bulk.hpp
 * @author Miguel Molinos (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2023-05-19
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef AlCu_mf_V_bulk_HPP
#define AlCu_mf_V_bulk_HPP

#include "ADP/AlCu-ADP.hpp"
#include "Macros.hpp"
#include <Eigen/Dense>

/**
 * @brief Function devoted to create the DMD function context of a Mg-Hx system
 *
 * @return dmd_equations
 */
dmd_equations DMD_AlCu_constructor();

/**
 * @brief
 *
 * @param site_i
 * @param mean_q:Mean value of q
 * @param xi Molar fraction
 * @param specie Atomic specie
 * @param atom_topology_i List of neighs
 * @return double
 */
double evaluate_rho_i_adp_AlCu(unsigned int site_i,           //!
                               const Eigen::MatrixXd &mean_q, //!
                               const Eigen::VectorXd &xi,     //!
                               const AtomicSpecie *specie,    //!
                               const AtomTopology atom_topology_i);

/**
 * @brief
 *
 * @param site_i_star
 * @param site_i
 * @param mean_q
 * @param xi
 * @param specie
 * @param atom_topology_i
 * @return Eigen::Vector3d
 */
Eigen::Vector3d evaluate_DV_i_Dq_u_adp_AlCu(unsigned int site_i_star,      //!
                                            unsigned int site_i,           //!
                                            const Eigen::MatrixXd &mean_q, //!
                                            const Eigen::VectorXd &xi,     //!
                                            const Eigen::VectorXd &rho,    //!
                                            const AtomicSpecie *specie,    //!
                                            const AtomTopology atom_topology_i);

/**
 * @brief
 *
 * @param site_i_star
 * @param site_i
 * @param mean_q
 * @param xi
 * @param specie
 * @param atom_topology_i
 * @return Eigen::Matrix3d
 */
Eigen::Matrix3d
evaluate_D2V_i_Dq2_u_AlCu(unsigned int site_i_star,            //!
                          unsigned int site_i,                 //!
                          const Eigen::MatrixXd &mean_q,       //!
                          const Eigen::VectorXd &xi,           //!
                          const Eigen::VectorXd &rho,          //!
                          const AtomicSpecie *specie,          //!
                          const AtomTopology atom_topology_i); //!

/**
 * @brief Compute the gradient of the potential with respect the stretch tensor
 * applied over the periodic box
 *
 * @param site_i Site to evalute the potential
 * @param mean_q Mean value of q
 * @param mean_q0 Reference value of the mean position
 * @param xi Molar fraction
 * @param specie
 * @param atom_topology_i
 * @return Eigen::Matrix3d
 */
Eigen::Matrix3d evaluate_DV_i_dF_AlCu(unsigned int site_i,                 //!
                                      const Eigen::MatrixXd &mean_q,       //!
                                      const Eigen::MatrixXd &mean_q0,      //!
                                      const Eigen::VectorXd &xi,           //!
                                      const Eigen::VectorXd &rho,          //!
                                      const AtomicSpecie *specie,          //!
                                      const AtomTopology atom_topology_i); //!

/**
 * @brief
 *
 * @param site_i
 * @param mean_q
 * @param stdv_q
 * @param xi
 * @param specie
 * @param atom_topology_i
 * @return double
 */
double evaluate_mf_rho_i_adp_AlCu(unsigned int site_i,           //!
                                  const Eigen::MatrixXd &mean_q, //!
                                  const Eigen::VectorXd &stdv_q, //!
                                  const Eigen::VectorXd &xi,     //!
                                  const AtomicSpecie *specie,    //!
                                  const AtomTopology atom_topology_i);

/**
 * @brief  Compute the mean-field potential at site i
 *
 * @param site_i Site to evalute the potential
 * @param mean_q Mean value of q
 * @param stdv_q Standard desviation of q
 * @param xi Molar fraction
 * @param mf_rho Meanfield energy density
 * @param specie
 * @param atom_topology_i
 * @return double
 */
double evaluate_V0_i_adp_AlCu(unsigned int site_i,                 //!
                              const Eigen::MatrixXd &mean_q,       //!
                              const Eigen::VectorXd &stdv_q,       //!
                              const Eigen::VectorXd &xi,           //!
                              const Eigen::VectorXd &mf_rho,       //!
                              const AtomicSpecie *specie,          //!
                              const AtomTopology atom_topology_i); //!

/**
 * @brief Compute meanfield entropy at site i
 *
 * @param site_i: Site to evaluate the potential
 * @param mean_q: Mean value of q
 * @param stdv_q: Standard desviation of q
 * @param xi: Molar fraction
 * @param mf_rho: Meanfield energy density
 * @param beta: Lagrange multiplier (thermal)
 * @param gamma: Lagrange multiplier (chemical)
 * @param specie: Atomic specie
 * @param atom_topology_i: Atom topology
 * @return double
 */
double evaluate_S0_i_adp_AlCu(unsigned int site_i,                 //!
                              const Eigen::MatrixXd &mean_q,       //!
                              const Eigen::VectorXd &stdv_q,       //!
                              const Eigen::VectorXd &xi,           //!
                              const Eigen::VectorXd &mf_rho,       //!
                              const Eigen::VectorXd &beta,         //!
                              const Eigen::VectorXd &gamma,        //!
                              const AtomicSpecie *specie,          //!
                              const AtomTopology atom_topology_i); //!

/**
 * @brief Evaluate the derivative of the meanfield potential with respect the
 * mean position of a site i
 *
 * @param site_i_star
 * @param site_i
 * @param mean_q Mean value of q
 * @param stdv_q Standard desviation of q
 * @param xi Molar fraction
 * @param mf_rho Meanfield energy density
 * @param specie
 * @param atom_topology_i
 * @return Eigen::Vector3d
 */
Eigen::Vector3d
evaluate_DV0_i_Dmeanq_u_AlCu(unsigned int site_i_star,            //!
                             unsigned int site_i,                 //!
                             const Eigen::MatrixXd &mean_q,       //!
                             const Eigen::VectorXd &stdv_q,       //!
                             const Eigen::VectorXd &xi,           //!
                             const Eigen::VectorXd &mf_rho,       //!
                             const AtomicSpecie *specie,          //!
                             const AtomTopology atom_topology_i); //!

/**
 * @brief  Evaluate the hessian of the meanfield potential with respect the
 * mean position of a site i
 *
 * @param site_i_star Site to evaluate the derivative of the potential
 * @param site_i Site to evaluate the potential
 * @param mean_q Mean position of the atomic positions in the updated
 * configuration
 * @param stdv_q Standard desviation of the position
 * @param xi
 * @param specie
 * @param atom_topology_i
 * @return Eigen::Matrix3d
 */
Eigen::Matrix3d
evaluate_D2V0_i_Dmeanq2_u_AlCu(unsigned int site_i_star,            //!
                               unsigned int site_i,                 //!
                               const Eigen::MatrixXd &mean_q,       //!
                               const Eigen::VectorXd &stdv_q,       //!
                               const Eigen::VectorXd &xi,           //!
                               const Eigen::VectorXd &mf_rho,       //!
                               const AtomicSpecie *specie,          //!
                               const AtomTopology atom_topology_i); //!

/**
 * @brief Evaluate the derivative of the meanfield potential with respect the
 * standard desviation of the position of a site i
 *
 * @param site_i_star
 * @param site_i
 * @param mean_q Mean position of the atomic positions in the updated
 * configuration
 * @param stdv_q Standard desviation of the position
 * @param xi Molar fraction
 * @param specie
 * @param atom_topology_i
 * @return double
 */
double evaluate_DV0_i_Dstdvq_u_AlCu(unsigned int site_i_star,            //!
                                    unsigned int site_i,                 //!
                                    const Eigen::MatrixXd &mean_q,       //!
                                    const Eigen::VectorXd &stdv_q,       //!
                                    const Eigen::VectorXd &xi,           //!
                                    const Eigen::VectorXd &mf_rho,       //!
                                    const AtomicSpecie *specie,          //!
                                    const AtomTopology atom_topology_i); //!

/**
 * @brief Evaluate the derivative of the meanfield potential with respect the
 * mean and standard desviation of the position at site i
 *
 * @param site_i_star Atomic site to derive with
 * @param site_i Atomic site where the potential is evaluated
 * @param mean_q Mean position of the atomic positions in the updated
 * configuration
 * @param stdv_q Standard desviation of the position
 * @param xi Molar fraction
 * @param mf_rho
 * @param specie
 * @param atom_topology_i
 * @return Eigen::Vector4d
 */
Eigen::Vector4d
evaluate_DV0_i_Dq_u_AlCu(unsigned int site_i_star,            //!
                         unsigned int site_i,                 //!
                         const Eigen::MatrixXd &mean_q,       //!
                         const Eigen::VectorXd &stdv_q,       //!
                         const Eigen::VectorXd &xi,           //!
                         const Eigen::VectorXd &mf_rho,       //!
                         const AtomicSpecie *specie,          //!
                         const AtomTopology atom_topology_i); //!

/**
 * @brief Evaluate the derivative of the meanfield potential with respect the
 * molar fraction of a site i
 *
 * @param site_i_star Site to evaluate the derivative of the potential
 * @param site_i
 * @param mean_q Mean position of the atomic positions in the updated
 * configuration
 * @param stdv_q Standard desviation of the position
 * @param xi Molar fraction
 * @param mf_rho: Mean field energy density
 * @param specie
 * @param atom_topology_i
 * @return double
 */
double evaluate_DV0_i_Dxi_u_adp_AlCu(unsigned int site_i_star,            //!
                                     unsigned int site_i,                 //!
                                     const Eigen::MatrixXd &mean_q,       //!
                                     const Eigen::VectorXd &stdv_q,       //!
                                     const Eigen::VectorXd &xi,           //!
                                     const Eigen::VectorXd &mf_rho,       //!
                                     const AtomicSpecie *specie,          //!
                                     const AtomTopology atom_topology_i); //!

/**
 * @brief Compute the gradient of the potential with respect the deformation
 * gradient tensor applied over the periodic box
 *
 * @param site_i
 * @param mean_q Mean position of the atomic sites in the updated configuration
 * @param mean_q0 Mean position of the atomic sites in the reference
 * configuration
 * @param stdv_q Standard desviation of the position
 * @param xi Molar fraction
 * @param specie
 * @param atom_topology_i
 * @return Eigen::Matrix3d
 */
Eigen::Matrix3d
evaluate_DV0_i_dF_bulk_AlCu(unsigned int site_i,                 //!
                            const Eigen::MatrixXd &mean_q,       //!
                            const Eigen::MatrixXd &mean_q0,      //!
                            const Eigen::VectorXd &stdv_q,       //!
                            const Eigen::VectorXd &xi,           //!
                            const Eigen::VectorXd &mf_rho,       //!
                            const AtomicSpecie *specie,          //!
                            const AtomTopology atom_topology_i); //!

#endif /* AlCu_mf_V_bulk_HPP */