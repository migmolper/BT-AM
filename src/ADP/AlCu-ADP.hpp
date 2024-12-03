/**
 * @file AlCu-ADP.hpp
 * @author J. M. Recio-Lopez ([jrecio1](https://github.com/jrecio1)), Miguel
 Molinos ([migmolper](https://github.com/migmolper)), Pilar Ariza
 ([mpariza](https://github.com/mpariza))
 * @brief Implementation of the Angular Dependant Potential for Al-Cu system
 * @version 0.1
 * @date 2024-08-31
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef ADP_AlCu_HPP
#define ADP_AlCu_HPP

#include "Atoms/Atom.hpp"
#include "Macros.hpp"
#include "Numerical/Quadrature-Measure.hpp"
#include "Numerical/cubic-spline.hpp"

enum species_comb_AlCu { AlAl, CuCu, AlCu };

/**
 * @brief Initialize the ADP potential for the Mg-Hx interaction.
 *
 * @param adp Structure containing the information requiered by the ADP
 * @param adp_material Pair selector (Al-Al, Al-Cu or Cu-Cu)
 * @param PotentialsFolder Relative adress to the folder containing the
 * potential
 * @return Error
 */
int init_adp_AlCu(adpPotential *adp, species_comb_AlCu adp_material,
                  const char *PotentialsFolder);

/**
 * @brief Destroy the ADP potential for the Al-Cu interaction.
 *
 * @param adp Structure containing the information requiered by the ADP
 */
void destroy_adp_AlCu(adpPotential *adp);

/**
 * @brief Energy density constructor. Where, \f$\bar{\rho}_i\f$ is
 \f[
 \bar{\rho}_i = \sum_{j, j \neq i} n_j \rho_j
 \f]
 * \f$n_i\f$ is the local occupancy of the site i. Taking derivatives of
 * \f$\bar{\rho}_{ij}\f$ with respect to \f$q_i\f$ and \f$q_j\f$
 \f[
 \frac{\partial \bar{\rho}_i}{\partial q_i} = \sum_{j, j \neq i} n_j\,
 d\rho_j\, \frac{r_{i,j}}{|r_{ij}|}
 \f]
 \f[
 \frac{\partial \bar{\rho}_i}{\partial q_j} = - \sum_{j, j \neq i} n_j\,
 d\rho_j\, \frac{r_{i,j}}{|r_{ij}|}
 \f]
 Second order derivatives of \f$\bar{\rho}_{ij}\f$
 \f[
 \frac{\partial^2 \bar{\rho}_i}{\partial q_i^2} = \sum_{j, j \neq i} n_j\,
 d^2\rho_j\, \frac{r_{i,j} \otimes r_{i,j}}{|r_{ij}|^2}\, +\, n_j\, d\rho_j\,
 \frac{\partial^2|r_{i,j}|}{\partial q_i^2}
 \f]
 \f[
 \frac{\partial^2 \bar{\rho}_i}{\partial q_i \partial q_j} = - \sum_{j, j \neq
 i} n_j\, d^2\rho_j\, \frac{r_{i,j} \otimes r_{i,j}}{|r_{ij}|^2}\, +\, n_j\,
 d\rho_j\, \frac{\partial^2|r_{i,j}|}{\partial q_i^2}
 \f]
 \f[
 \frac{\partial^2 \bar{\rho}_i}{\partial q_j \partial q_i} = - \sum_{j, j \neq
 i} n_j\, d^2\rho_j\, \frac{r_{i,j} \otimes r_{i,j}}{|r_{ij}|^2}\, +\, n_j\,
 d\rho_j\, \frac{\partial^2|r_{i,j}|}{\partial q_i^2}
 \f]
 \f[
 \frac{\partial^2 \bar{\rho}_i}{\partial q_j^2} = \sum_{j, j \neq i} n_j\,
 d^2\rho_j\, \frac{r_{i,j} \otimes r_{i,j}}{|r_{ij}|^2}\, +\, n_j\, d\rho_j\,
 \frac{\partial^2|r_{i,j}|}{\partial q_i^2}
 \f]
 * @return potential_function
 */
potential_function rho_ij_adp_AlCu_constructor();

/**
 * @brief Pair term contribution constructor
 \f[
 V^{\phi} = \frac{1}{2} \sum_i n_i n_j \phi(|r_{i,j}|) = \sum_i V^{\phi}_{ij}
 \f]
 Taking derivatives of \f$V^{\phi}_{ij}\f$ with respect to \f$q_i\f$ and
 \f$q_j\f$
 \f[
 \begin{split}
 \frac{\partial V^{\phi}_{ij}}{\partial q_i} = \frac{1}{2}\ (n_i n_j
 d\phi_{ij})\ \frac{r_{i,j}}{|r_{ij}|} \end{split}
 \f]
 \f[
 \begin{split}
 \frac{\partial V^{\phi}_{ij}}{\partial q_{j}} = - \frac{1}{2}\ (n_i n_j
 d\phi_{ij})\ \frac{r_{i,j}}{|r_{ij}|} \end{split}
 \f]
 Higher order derivatives of \f$V^{\phi}_{i,j}\f$
 \f[
 \frac{\partial^2 V^{\phi}_{i,j}}{\partial q_i^2} = \frac{1}{2}  (n_i n_j
 d^2\phi_{i,j}) \frac{r_{i,j} \otimes r_{i,j}}{|r_{i,j}|^2} + \frac{1}{2} (n_i
 n_j d\phi_{ij}) \frac{\partial^2|r_{i,j}|}{\partial q_i^2}
 \f]
 \f[
 \frac{\partial^2 V^{\phi}_{i,j}}{\partial q_i \partial q_j} = - \frac{1}{2}
 (n_i n_j d^2\phi_{i,j}) \frac{r_{i,j} \otimes r_{i,j}}{|r_{i,j}|^2} -
 \frac{1}{2} (n_i n_j d\phi_{ij}) \frac{\partial^2|r_{i,j}|}{\partial q_i^2}
 \f]
 \f[
 \frac{\partial^2 V^{\phi}_{i,j}}{\partial q_j \partial q_i} = - \frac{1}{2}
 (n_i n_j d^2\phi_{i,j}) \frac{r_{i,j} \otimes r_{i,j}}{|r_{i,j}|^2} -
 \frac{1}{2} (n_i n_j d\phi_{ij}) \frac{\partial^2|r_{i,j}|}{\partial q_i^2}
 \f]
 \f[
 \frac{\partial^2 V^{\phi}_{i,j}}{\partial q_j^2} = \frac{1}{2}  (n_i n_j
 d^2\phi_{i,j}) \frac{r_{i,j} \otimes r_{i,j}}{|r_{i,j}|^2} + \frac{1}{2} (n_i
 n_j d\phi_{ij}) \frac{\partial^2|r_{i,j}|}{\partial q_i^2}
 \f]
 * @return potential_function
 */
potential_function V_pair_ij_adp_AlCu_constructor();

/**
 * @brief Dipole term contribution constructor
 ### The Dipole contribution to the potential
 \f[
 V^{\mu} = \frac{1}{2} \sum_i \mu_i \cdot \mu_i,
 \f]
 where \f$\mu_i\f$ is
\f[
 \mu_i = \sum_{j, j \neq i} n_i n_j\ u(|r_{i,j}|)\ r_{i,j}
\f]
 substituting the expression of \f$\mu_i\f$ in \f$V^{\mu}\f$ results
 \f[
 V^{\mu} = \frac{1}{2} \sum_i \bigg(\sum_{j_1, j_1 \neq i} n_i
 n_{j_1}\ u(|r_{i,j_1}|)\ r_{i,j_1} \bigg) \cdot \bigg(\sum_{j_2, j_2 \neq i}
 n_i n_{j_2}\ u(|r_{i,j_2}|)\ r_{i,j_2} \bigg)
 \f]
 \f[
 V^{\mu} = \frac{1}{2} \sum_i \sum_{j_1\ ,\ j_2} V^{\mu}_{i,j_1,j_2} =
 \frac{1}{2} \sum_i \sum_{j_1\ ,\ j_2} \left(n_i
 n_{j_1}\ u_{i,j_1}\right) \left(n_i n_{j_2}\ u_{i,j_2}\right)\ \left(r_{i,j_1}
 \cdot r_{i,j_2} \right)
 \f]
 Taking derivatives of \f$V^{\mu}_{i,j_1,j_2}\f$ with respect to \f$q_i\f$,
 \f$q_{j1}\f$ and \f$q_{j2}\f$
 \f[
 \begin{split}
 \frac{\partial V^{\mu}_{i,j_1,j_2}}{\partial q_i}\ =\ &\left(n_i
 n_{j_1}\ du_{i,j_1}\right) \left(n_i n_{j_2}\ u_{i,j_2}\right)\ \left(r_{i,j_1}
 \cdot r_{i,j_2} \right) \frac{r_{i,j_i}}{|r_{i,j_1}|}\ +\ \\
 &\left(n_i n_{j_1}\ u_{i,j_1}\right) \left(n_i
 n_{j_2}\ du_{i,j_2}\right)\ \left(r_{i,j_1} \cdot r_{i,j_2} \right)
 \frac{r_{i,j_2}}{|r_{i,j_2}|}\ +\ \\
 &\left(n_i n_{j_1}\ u_{i,j_1}\right) \left(n_i
 n_{j_2}\ u_{i,j_2}\right)\ \left(r_{i,j_2} + r_{i,j_1} \right) \end{split}
 \f]
 \f[
 \begin{split}
 \frac{\partial V^{\mu}_{i,j_1,j_2}}{\partial q_{j_1}}\ =\ - &\left(n_i
 n_{j_1}\ du_{i,j_1}\right) \left(n_i n_{j_2}\ u_{i,j_2}\right)\ \left(r_{i,j_1}
 \cdot r_{i,j_2} \right) \frac{r_{i,j_i}}{|r_{i,j_1}|}\ -\ \\
 &\left(n_i n_{j_1}\ u_{i,j_1}\right) \left(n_i
 n_{j_2}\ u_{i,j_2}\right)\ r_{i,j_2} \end{split}
 \f]
 \f[
 \begin{split}
 \frac{\partial V^{\mu}_{i,j_1,j_2}}{\partial q_{j_2}}\ =\ - &\left(n_i
 n_{j_1}\ u_{i,j_1}\right) \left(n_i n_{j_2}\ du_{i,j_2}\right)\ \left(r_{i,j_1}
 \cdot r_{i,j_2} \right) \frac{r_{i,j_2}}{|r_{i,j_2}|}\ - \\
 &\left(n_i n_{j_1}\ u_{i,j_1}\right) \left(n_i
 n_{j_2}\ u_{i,j_2}\right)\ r_{i,j_1} \end{split}
 \f]

Higher order derivatives of \f$V^{\mu}_{i,j_1,j_2}\f$
\f[
\begin{split}
\frac{\partial^2 V^{\mu}_{i,j_1,j_2}}{\partial q_i^2}\ =\ &\left(n_i
n_{j_1}\ d^2u_{i,j_1}\right)\ \left(n_i n_{j_1}\ u_{i,j_2}\right)\ (r_{i,j_1}
\cdot r_{i,j_2}) \frac{(r_{i,j_1} \otimes r_{i,j_1})}{|r_{i,j_1}|^2}\ + \\
        &\left(n_i n_{j_1}\ du_{i,j_1}\right)\ \left(n_i
n_{j_2}\ du_{i,j_2}\right)\ (r_{i,j_1} \cdot r_{i,j_2}) \frac{(r_{i,j_1} \otimes
r_{i,j_2})}{|r_{i,j_1}| |r_{i,j_2}|}\ +\\
        &\left(n_i n_{j_1}\ du_{i,j_1}\right)\ \left(n_i
n_{j_2}\ u_{i,j_2}\right)\ \frac{(r_{i,j_1} \otimes
r_{i,j_2})}{|r_{i,j_1}|}\ +\\
        &\left(n_i n_{j_1}\ du_{i,j_1}\right)\ \left(n_i
n_{j_2}\ u_{i,j_2}\right)\ \frac{(r_{i,j_1} \otimes
r_{i,j_1})}{|r_{i,j_1}|}\ +\\
        &\left(n_i n_{j_1}\ du_{i,j_1}\right)\ \left(n_i
n_{j_2}\ u_{i,j_2}\right)\ (r_{i,j_1} \cdot
r_{i,j_2})\ \frac{\partial^2|r_{i,j_1}|}{\partial q_i^2}\ +\\
        &\left(n_i n_{j_1}\ du_{i,j_1}\right)\ \left(n_i
n_{j_2}\ du_{i,j_2}\right)\ (r_{i,j_1} \cdot r_{i,j_2}) \frac{(r_{i,j_2} \otimes
r_{i,j_1})}{|r_{i,j_1}| |r_{i,j_2}|} +\\
        &\left(n_i n_{j_1}\ u_{i,j_1}\right)\ \left(n_i
n_{j_2}\ d^2u_{i,j_2}\right)\ (r_{i,j_1} \cdot r_{i,j_2}) \frac{(r_{i,j_2}
\otimes r_{i,j_2})}{|r_{i,j_2}|^2} +\\
        &\left(n_i n_{j_1}\ u_{i,j_1}\right)\ \left(n_i
n_{j_2}\ du_{i,j_2}\right)\ \frac{(r_{i,j_2} \otimes r_{i,j_1})}{|r_{i,j_2}|}
+\\
        &\left(n_i n_{j_1}\ u_{i,j_1}\right)\ \left(n_i
n_{j_2}\ du_{i,j_2}\right)\  \frac{(r_{i,j_2} \otimes r_{i,j_2})}{|r_{i,j_2}|}
+\\
        &\left(n_i n_{j_1}\ u_{i,j_1}\right)\ \left(n_i
n_{j_2}\ du_{i,j_2}\right)\ (r_{i,j_1} \cdot
r_{i,j_2})\ \frac{\partial^2|r_{i,j_2}|}{\partial q_i^2}\ +\\
        &\left(n_i n_{j_1}\ du_{i,j_1}\right)\ \left(n_i
n_{j_2}\ u_{i,j_2}\right)\ \frac{1}{|r_{i,j_1}|} ((r_{i,j_1} \otimes r_{i,j_1})
+ (r_{i,j_2} \otimes r_{i,j_1})) +\\
        &\left(n_i n_{j_1}\ u_{i,j_1}\right)\ \left(n_i
n_{j_2}\ du_{i,j_2}\right)\ \frac{1}{|r_{i,j_2}|} ((r_{i,j_1} \otimes r_{i,j_2})
+ (r_{i,j_2} \otimes r_{i,j_2})) +\\ &2 \left(n_i
n_{j_1}\ u_{i,j_1}\right)\ \left(n_i n_{j_2}\ u_{i,j_2}\right)\ I \end{split}
\f]

\f[
\begin{split}
\frac{\partial^2 V^{\mu}_{i,j_1,j_2}}{\partial q_{j_1}^2}\ =\ &\left(n_i
n_{j_1}\ d^2u_{i,j_1}\right)\ \left(n_i n_{j_1}\ u_{i,j_2}\right)\ (r_{i,j_1}
\cdot r_{i,j_2}) \frac{(r_{i,j_1} \otimes r_{i,j_1})}{|r_{i,j_1}|^2}\ + \\
        &\left(n_i n_{j_1}\ du_{i,j_1}\right)\ \left(n_i
n_{j_2}\ u_{i,j_2}\right)\ (r_{i,j_1} \cdot
r_{i,j_2})\ \frac{\partial^2|r_{i,j_1}|}{\partial q_i^2}\ +\\
        &\left(n_i n_{j_1}\ du_{i,j_1}\right)\ \left(n_i
n_{j_2}\ u_{i,j_2}\right)\ \frac{(r_{i,j_1} \otimes
r_{i,j_2})}{|r_{i,j_1}|}\ +\\
        &\left(n_i n_{j_1}\ du_{i,j_1}\right)\ \left(n_i
n_{j_2}\ u_{i,j_2}\right)\ \frac{(r_{i,j_2} \otimes r_{i,j_1})}{|r_{i,j_1}|}
\end{split}
\f]

\f[
\begin{split}
\frac{\partial^2 V^{\mu}_{i,j_1,j_2}}{\partial q_{j_1} \partial
q_{j_2}}\ =\ &\left(n_i n_{j_1}\ du_{i,j_1}\right)\ \left(n_i
n_{j_2}\ du_{i,j_2}\right)\ (r_{i,j_1} \cdot r_{i,j_2}) \frac{(r_{i,j_1} \otimes
r_{i,j_2})}{|r_{i,j_1}| |r_{i,j_2}|}\ +\\
        &\left(n_i n_{j_1}\ du_{i,j_1}\right)\ \left(n_i
n_{j_2}\ u_{i,j_2}\right)\ \frac{(r_{i,j_1} \otimes
r_{i,j_1})}{|r_{i,j_1}|}\ +\\
        &\left(n_i n_{j_1}\ u_{i,j_1}\right)\ \left(n_i
n_{j_2}\ du_{i,j_2}\right)\  \frac{(r_{i,j_2} \otimes r_{i,j_2})}{|r_{i,j_2}|}
+\\
        &\left(n_i n_{j_1}\ u_{i,j_1}\right)\ \left(n_i
n_{j_2}\ u_{i,j_2}\right)\ I \end{split} \f]

\f[
\begin{split}
\frac{\partial^2 V^{\mu}_{i,j_1,j_2}}{\partial q_{j_2} \partial
q_{j_1}}\ =\ &\left(n_i n_{j_1}\ du_{i,j_1}\right)\ \left(n_i
n_{j_2}\ du_{i,j_2}\right)\ (r_{i,j_1} \cdot r_{i,j_2}) \frac{(r_{i,j_2} \otimes
r_{i,j_1})}{|r_{i,j_1}| |r_{i,j_2}|} +\\
        &\left(n_i n_{j_1}\ u_{i,j_1}\right)\ \left(n_i
n_{j_2}\ du_{i,j_2}\right)\  \frac{(r_{i,j_2} \otimes r_{i,j_2})}{|r_{i,j_2}|}
+\\
        &\left(n_i n_{j_1}\ du_{i,j_1}\right)\ \left(n_i
n_{j_2}\ u_{i,j_2}\right)\ \frac{(r_{i,j_1} \otimes
r_{i,j_1})}{|r_{i,j_1}|}\ +\\
        &\left(n_i n_{j_1}\ u_{i,j_1}\right)\ \left(n_i
n_{j_2}\ u_{i,j_2}\right)\ I \end{split} \f]

\f[
\begin{split}
\frac{\partial^2 V^{\mu}_{i,j_1,j_2}}{\partial q_{j_2}^2}\ =\ &\left(n_i
n_{j_1}\ u_{i,j_1}\right)\ \left(n_i n_{j_2}\ d^2u_{i,j_2}\right)\ (r_{i,j_1}
\cdot r_{i,j_2}) \frac{(r_{i,j_2} \otimes r_{i,j_2})}{|r_{i,j_2}|^2} +\\
        &\left(n_i n_{j_1}\ u_{i,j_1}\right)\ \left(n_i
n_{j_2}\ du_{i,j_2}\right)\ (r_{i,j_1} \cdot
r_{i,j_2})\ \frac{\partial^2|r_{i,j_2}|}{\partial q_i^2}\ +\\
        &\left(n_i n_{j_1}\ u_{i,j_1}\right)\ \left(n_i
n_{j_2}\ du_{i,j_2}\right)\ \frac{(r_{i,j_2} \otimes r_{i,j_1})}{|r_{i,j_2}|}
+\\
        &\left(n_i n_{j_1}\ u_{i,j_1}\right)\ \left(n_i
n_{j_2}\ du_{i,j_2}\right)\ \frac{(r_{i,j_1} \otimes r_{i,j_2})}{|r_{i,j_2}|}
\end{split}
\f]
 *
 * @return potential_function
 */
potential_function V_dipole_ij1j2_adp_AlCu_constructor();

/**
 * @brief Quadrupole term contribution constructor
 ### The Quadrupole contribution to the potential
 \f[
 V^{\lambda} = \frac{1}{2} \bigg(\sum_i \lambda_i : \lambda_i - \frac{1}{3}
 (\text{tr}\lambda_i)^2 \bigg),
 \f]
 where \f$\lambda_i\f$ is
 \f[
 \lambda_i = \sum_{j, j \neq i} n_i n_j\ w(|r_{i,j}|)\ (r_{i,j} \otimes r_{i,j})
 \f]
 substituting the expression of \f$\lambda_i\f$ in \f$V^{\lambda}\f$ results:
 \f[
 \begin{split}
 V^{\lambda} = \frac{1}{2} \sum_i &\left(\sum_{j_1, j_1 \neq i} n_i
 n_{j_1}\ w(|r_{i,j_1}|)\ (r_{i,j_1} \otimes r_{i,j_1}) \right) :
 \left(\sum_{j_2, j_2 \neq i} n_i n_{j_2}\ w(|r_{i,j_2}|)\ (r_{i,j_2} \otimes
 r_{i,j_2}) \right)\ -\\ \frac{1}{3} &\left(\sum_{j_1, j_1 \neq i} n_i
 n_{j_1}\ w(|r_{i,j_1}|)\ (r_{i,j_1} \cdot r_{i,j_1}) \right) \left(\sum_{j_2,
 j_2 \neq i} n_i n_{j_2}\ w(|r_{i,j_2}|)\ (r_{i,j_2} \cdot r_{i,j_2}) \right)
 \end{split}
 \f]
 \f[
 \begin{split}
 V^{\lambda} = \frac{1}{2} \sum_i \sum_{j_1\ ,\ j_2} V^{\lambda}_{i,j_1,j_2} =
 \frac{1}{2} \sum_i \sum_{j_1\ ,\ j_2} &\left(n_i
 n_{j_1}\ w_{i,j_1}\right) \left(n_i n_{j_2}\ w_{i,j_2}\right)\ \left(r_{i,j_1}
 \cdot r_{i,j_2} \right)^2 - \\ \frac{1}{3} &\left(n_i n_{j_1}\ w_{i,j_1}\right)
 \left(n_i n_{j_2}\ w_{i,j_2}\right) |r_{i,j_1}|^2 |r_{i,j_2}|^2 \end{split}
 \f]

Taking derivatives of \f$V^{\lambda}_{i,j_1,j_2}\f$ with respect to \f$q_i\f$,
\f$q_{j1}\f$ and \f$q_{j2}\f$ \f[ \begin{split}
\frac{\partial V^{\lambda}_{i,j_1,j_2}}{\partial q_i}\ =\ \bigg(&\left(n_i
n_{j_1}\ dw_{i,j_1}\right) \left(n_i n_{j_2}\ w_{i,j_2}\right)\ \left(r_{i,j_1}
\cdot r_{i,j_2} \right)^2 \frac{r_{i,j_i}}{|r_{i,j_1}|} +\ \\
&\left(n_i n_{j_1}\ w_{i,j_1}\right) \left(n_i
n_{j_2}\ dw_{i,j_2}\right)\ \left(r_{i,j_1} \cdot r_{i,j_2} \right)^2
\frac{r_{i,j_2}}{|r_{i,j_2}|} +\ \\
&\left(n_i n_{j_1}\ w_{i,j_1}\right) \left(n_i n_{j_2}\ w_{i,j_2}\right)\ 2
\left(r_{i,j_1} \cdot r_{i,j_2} \right) \left(r_{i,j_2} + r_{i,j_1}
\right)\bigg) \\
- \frac{1}{3}\bigg(&\left(n_i n_{j_1}\ dw_{i,j_1}\right)\ \left(n_i
n_{j_2}\ w_{i,j_2}\right)\ |r_{i,j_1}|^2 |r_{i,j_2}|^2
\frac{r_{i,j_1}}{|r_{i,j_1}|} +\ \\
&\left(n_i n_{j_1}\ w_{i,j_1}\right) \left(n_i
n_{j_2}\ dw_{i,j_2}\right)\ |r_{i,j_1}|^2 |r_{i,j_2}|^2
\frac{r_{i,j_2}}{|r_{i,j_2}|} +\ \\
&\left(n_i n_{j_1}\ w_{i,j_1}\right)\ \left(n_i n_{j_2}\ w_{i,j_2}\right)\ 2
\left(|r_{i,j_2}|^2 r_{i,j_1} + |r_{i,j_1}|^2 r_{i,j_2}\right)\bigg) \end{split}
\f]

\f[
\begin{split}
\frac{\partial V^{\lambda}_{i,j_1,j_2}}{\partial q_{j_1}}\ =\ - \bigg(&\left(n_i
n_{j_1}\ dw_{i,j_1}\right) \left(n_i n_{j_2}\ w_{i,j_2}\right)\ \left(r_{i,j_1}
\cdot r_{i,j_2} \right)^2 \frac{r_{i,j_i}}{|r_{i,j_1}|} +\ \\
&\left(n_i n_{j_1}\ w_{i,j_1}\right) \left(n_i n_{j_2}\ w_{i,j_2}\right)\ 2
\left(r_{i,j_1} \cdot r_{i,j_2} \right) r_{i,j_2}\bigg) \\
+ \frac{1}{3}\bigg(&\left(n_i n_{j_1}\ dw_{i,j_1}\right)\ \left(n_i
n_{j_2}\ w_{i,j_2}\right)\ |r_{i,j_1}|^2
|r_{i,j_2}|^2\ \frac{r_{i,j_1}}{|r_{i,j_1}|} +\ \\
&\left(n_i n_{j_1}\ w_{i,j_1}\right)\ \left(n_i n_{j_2}\ w_{i,j_2}\right)\ 2
|r_{i,j_2}|^2 r_{i,j_1} \bigg) \end{split} \f]

\f[
\begin{split}
\frac{\partial V^{\lambda}_{i,j_1,j_2}}{\partial q_{j_2}}\ =\ - \bigg(&\left(n_i
n_{j_1}\ w_{i,j_1}\right) \left(n_i n_{j_2}\ dw_{i,j_2}\right)\ \left(r_{i,j_1}
\cdot r_{i,j_2} \right)^2 \frac{r_{i,j_2}}{|r_{i,j_2}|} +\ \\
&\left(n_i n_{j_1}\ w_{i,j_1}\right) \left(n_i n_{j_2}\ w_{i,j_2}\right)\ 2
\left(r_{i,j_1} \cdot r_{i,j_2} \right) r_{i,j_1} \bigg) \\
+ \frac{1}{3}\bigg(&\left(n_i n_{j_1}\ w_{i,j_1}\right) \left(n_i
n_{j_2}\ dw_{i,j_2}\right)\ |r_{i,j_1}|^2 |r_{i,j_2}|^2
\frac{r_{i,j_2}}{|r_{i,j_2}|} +\ \\
&\left(n_i n_{j_1}\ w_{i,j_1}\right)\ \left(n_i n_{j_2}\ w_{i,j_2}\right)\ 2
|r_{i,j_1}|^2 r_{i,j_2}\bigg) \end{split} \f]

Higher order derivatives of \f$V^{\lambda}_{i,j_1,j_2}\f$
\f[
\begin{split}
\frac{\partial^2 V^{\lambda}_{i,j_1,j_2}}{\partial q_i^2}\ =\ \bigg(&\left(n_i
n_{j_1}\ d^2w_{i,j_1}\right)\ \left(n_i n_{j_2}\ w_{i,j_2}\right)\ (r_{i,j_1}
\cdot r_{i,j_2})^2\ \frac{(r_{i,j_1} \otimes r_{i,j_1})}{|r_{i,j_1}|^2} + \\
        &\left(n_i n_{j_1}\ dw_{i,j_1}\right)\ \left(n_i
n_{j_2}\ dw_{i,j_2}\right)\ (r_{i,j_1} \cdot r_{i,j_2})^2\ \frac{(r_{i,j_1}
\otimes r_{i,j_2})}{|r_{i,j_1}||r_{i,j_2}|} + \\
        &\left(n_i n_{j_1}\ dw_{i,j_1}\right)\ \left(n_i
n_{j_2}\ w_{i,j_2}\right)\ 2 (r_{i,j_1} \cdot r_{i,j_2})\ \frac{(r_{i,j_1}
\otimes r_{i,j_2})}{|r_{i,j_1}|} + \\
        &\left(n_i n_{j_1}\ dw_{i,j_1}\right)\ \left(n_i
n_{j_2}\ w_{i,j_2}\right)\ 2 (r_{i,j_1} \cdot r_{i,j_2})\ \frac{(r_{i,j_1}
\otimes r_{i,j_1})}{|r_{i,j_1}|} + \\
        &\left(n_i n_{j_1}\ dw_{i,j_1}\right)\ \left(n_i
n_{j_2}\ w_{i,j_2}\right)\ (r_{i,j_1} \cdot
r_{i,j_2})^2\  \frac{\partial^2|r_{i,j_1}|}{\partial q_i^2}\ + \\
        &\left(n_i n_{j_1}\ dw_{i,j_1}\right)\ \left(n_i
n_{j_2}\ dw_{i,j_2}\right)\ (r_{i,j_1} \cdot r_{i,j_2})^2\  \frac{(r_{i,j_2}
\otimes r_{i,j_1})}{ |r_{i,j_1}| |r_{i,j_2}|} + \\
        &\left(n_i n_{j_1}\ w_{i,j_1}\right)\ \left(n_i
n_{j_2}\ d^2w_{i,j_2}\right)\ (r_{i,j_1} \cdot r_{i,j_2})^2\ \frac{(r_{i,j_2}
\otimes r_{i,j_2})}{|r_{i,j_2}|^2} + \\
        &\left(n_i n_{j_1}\ w_{i,j_1}\right)\ \left(n_i
n_{j_2}\ dw_{i,j_2}\right)\ 2 (r_{i,j_1} \cdot r_{i,j_2})\ \frac{(r_{i,j_2}
\otimes r_{i,j_1})}{|r_{i,j_2}|} + \\
        &\left(n_i n_{j_1}\ w_{i,j_1}\right)\ \left(n_i
n_{j_2}\ dw_{i,j_2}\right)\ 2 (r_{i,j_1} \cdot r_{i,j_2})\ \frac{(r_{i,j_2}
\otimes r_{i,j_2})}{|r_{i,j_2}|} + \\
        &\left(n_i n_{j_1}\ w_{i,j_1}\right)\ \left(n_i
n_{j_2}\ dw_{i,j_2}\right)\ (r_{i,j_1} \cdot
r_{i,j_2})^2\  \frac{\partial^2|r_{i,j_2}|}{\partial q_i^2}\ + \\
        &\left(n_i n_{j_1}\ dw_{i,j_1}\right)\ \left(n_i
n_{j_2}\ w_{i,j_2}\right)\  2 \left(r_{i,j_1} \cdot r_{i,j_2}
\right)\ \frac{(r_{i,j_1}  \otimes r_{i,j_1})}{|r_{i,j_1}|}\  + \\
        &\left(n_i n_{j_1}\ dw_{i,j_1}\right)\ \left(n_i
n_{j_2}\ w_{i,j_2}\right)\ 2 \left(r_{i,j_1} \cdot r_{i,j_2}
\right)\ \frac{(r_{i,j_2} \otimes r_{i,j_1})}{|r_{i,j_1}|}\ + \\
        &\left(n_i n_{j_1}\ w_{i,j_1}\right)\ \left(n_i
n_{j_2}\ dw_{i,j_2}\right)\ 2 \left(r_{i,j_1} \cdot r_{i,j_2}
\right)\ \frac{(r_{i,j_1} \otimes r_{i,j_2})}{|r_{i,j_2}|}\ + \\
        &\left(n_i n_{j_1}\ w_{i,j_1}\right)\ \left(n_i
n_{j_2}\ dw_{i,j_2}\right)\ 2 \left(r_{i,j_1} \cdot r_{i,j_2}
\right)\ \frac{(r_{i,j_2} \otimes r_{i,j_2})}{|r_{i,j_2}|}\  + \\
        &\left(n_i n_{j_2}\ w_{i,j_2}\right)\ \left(n_i
n_{j_2}\ w_{i,j_2}\right)\ 2 (r_{i,j_1} \otimes r_{i,j_1})\ + \\
        &\left(n_i n_{j_2}\ w_{i,j_2}\right)\ \left(n_i
n_{j_2}\ w_{i,j_2}\right)\ 2 (r_{i,j_1} \otimes r_{i,j_2})\ + \\
        &\left(n_i n_{j_2}\ w_{i,j_2}\right)\ \left(n_i
n_{j_2}\ w_{i,j_2}\right)\ 2 (r_{i,j_2} \otimes r_{i,j_1})\ + \\
        &\left(n_i n_{j_2}\ w_{i,j_2}\right)\ \left(n_i
n_{j_2}\ w_{i,j_2}\right)\ 2 (r_{i,j_2} \otimes r_{i,j_2})\ + \\
        &\left(n_i n_{j_2}\ w_{i,j_2}\right)\ \left(n_i
n_{j_2}\ w_{i,j_2}\right)\ 4 (r_{i,j_1} \cdot r_{i,j_2}) \cdot I \bigg) \\
        - \frac{1}{3}\bigg(&\left(n_i n_{j_1}\ d^2w_{i,j_1}\right)\ \left(n_i
n_{j_2}\ w_{i,j_2}\right) |r_{i,j_2}|^2 (r_{i,j_1} \otimes r_{i,j_1})\ + \\
        &\left(n_i n_{j_1}\ dw_{i,j_1}\right)\ \left(n_i
n_{j_2}\ dw_{i,j_2}\right) |r_{i,j_1}| |r_{i,j_2}| (r_{i,j_1} \otimes
r_{i,j_2})\ +\\
        &\left(n_i n_{j_1}\ dw_{i,j_1}\right)\ \left(n_i
n_{j_2}\ w_{i,j_2}\right) 2 |r_{i,j_2}|^2 \frac{(r_{i,j_1} \otimes
r_{i,j_1})}{|r_{i,j_1}|} + \\
        &\left(n_i n_{j_1}\ dw_{i,j_1}\right)\ \left(n_i
n_{j_2}\ w_{i,j_2}\right) 2 |r_{i,j_1}|^2 \frac{(r_{i,j_1} \otimes
r_{i,j_2})}{|r_{i,j_1}|} + \\
        &\left(n_i n_{j_1}\ dw_{i,j_1}\right)\ \left(n_i
n_{j_2}\ w_{i,j_2}\right) |r_{i,j_1}|^2 |r_{i,j_2}|^2
\frac{\partial^2|r_{i,j_1}|}{\partial q_i^2} + \\
        &\left(n_i n_{j_1}\ dw_{i,j_1}\right)\ \left(n_i
n_{j_2}\ dw_{i,j_2}\right) |r_{i,j_1}| |r_{i,j_2}| (r_{i,j_2} \otimes r_{i,j_1})
+ \\
        &\left(n_i n_{j_1}\ w_{i,j_1}\right)\ \left(n_i
n_{j_2}\ d^2w_{i,j_2}\right) |r_{i,j_1}|^2 (r_{i,j_2} \otimes r_{i,j_2})\ + \\
        &\left(n_i n_{j_1}\ w_{i,j_1}\right)\ \left(n_i
n_{j_2}\ dw_{i,j_2}\right) 2 |r_{i,j_2}|^2 \frac{(r_{i,j_2} \otimes
r_{i,j_1})}{|r_{i,j_2}|} + \\
        &\left(n_i n_{j_1}\ w_{i,j_1}\right)\ \left(n_i
n_{j_2}\ dw_{i,j_2}\right) 2 |r_{i,j_1}|^2 \frac{(r_{i,j_2} \otimes
r_{i,j_2})}{|r_{i,j_2}|} + \\
        &\left(n_i n_{j_1}\ w_{i,j_1}\right)\ \left(n_i
n_{j_2}\ dw_{i,j_2}\right) |r_{i,j_1}|^2 |r_{i,j_2}|^2
\frac{\partial^2|r_{i,j_2}|}{\partial q_i^2} +\\
        &\left(n_i n_{j_1}\ dw_{i,j_1}\right)\ \left(n_i
n_{j_2}\ w_{i,j_2}\right)\ 2 |r_{i,j_2}|^2 \frac{\left(r_{i,j_1} \otimes
r_{i,j_1}\right)}{|r_{i,j_1}|} + \\
        &\left(n_i n_{j_1}\ dw_{i,j_1}\right)\ \left(n_i
n_{j_2}\ w_{i,j_2}\right)\ 2 |r_{i,j_1}| \left(r_{i,j_2} \otimes
r_{i,j_1}\right) +\\ 
        &\left(n_i n_{j_1}\ w_{i,j_1}\right)\ \left(n_i
n_{j_2}\ dw_{i,j_2}\right)\ 2 |r_{i,j_2}| \left(r_{i,j_1} \otimes
r_{i,j_2}\right)\ +\\
        &\left(n_i n_{j_1}\ w_{i,j_1}\right)\ \left(n_i
n_{j_2}\ dw_{i,j_2}\right)\ 2 |r_{i,j_1}|^2 \frac{\left(r_{i,j_2} \otimes
r_{i,j_2}\right)}{|r_{i,j_2}|} + \\
        &\left(n_i n_{j_1}\ w_{i,j_1}\right)\ \left(n_i
n_{j_2}\ w_{i,j_2}\right)\ 4 \left(r_{i,j_1} \otimes r_{i,j_2}\right) + \\ 
        &\left(n_i n_{j_1}\ w_{i,j_1}\right)\ \left(n_i
n_{j_2}\ w_{i,j_2}\right)\ 2 |r_{i,j_2}|^2\ \text{I} + \\
        &\left(n_i n_{j_1}\ w_{i,j_1}\right)\ \left(n_i
n_{j_2}\ w_{i,j_2}\right)\ 4 \left(r_{i,j_2} \otimes r_{i,j_1}\right) + \\
        &\left(n_i n_{j_1}\ w_{i,j_1}\right)\ \left(n_i
n_{j_2}\ w_{i,j_2}\right)\ 2 |r_{i,j_1}|^2\ \text{I}\bigg) \end{split} \f]

\f[
\begin{split}
\frac{\partial^2 V^{\lambda}_{i,j_1,j_2}}{\partial
q_{j_1}^2}\ =\ \bigg(&\left(n_i n_{j_1}\ d^2w_{i,j_1}\right)\ \left(n_i
n_{j_2}\ w_{i,j_2}\right)\ (r_{i,j_1} \cdot r_{i,j_2})^2\ \frac{(r_{i,j_1}
\otimes r_{i,j_1})}{|r_{i,j_1}|^2} + \\
        &\left(n_i n_{j_1}\ dw_{i,j_1}\right)\ \left(n_i
n_{j_2}\ w_{i,j_2}\right)\ 2 (r_{i,j_1} \cdot r_{i,j_2})\ \frac{(r_{i,j_1}
\otimes r_{i,j_2})}{|r_{i,j_1}|} + \\
        &\left(n_i n_{j_1}\ dw_{i,j_1}\right)\ \left(n_i
n_{j_2}\ w_{i,j_2}\right)\ (r_{i,j_1} \cdot
r_{i,j_2})^2\  \frac{\partial^2|r_{i,j_1}|}{\partial q_i^2}\ + \\
        &\left(n_i n_{j_1}\ dw_{i,j_1}\right)\ \left(n_i
n_{j_2}\ w_{i,j_2}\right)\ 2 \left(r_{i,j_1} \cdot r_{i,j_2}
\right)\ \frac{(r_{i,j_2} \otimes r_{i,j_1})}{|r_{i,j_1}|}\ + \\
        &\left(n_i n_{j_2}\ w_{i,j_2}\right)\ \left(n_i
n_{j_2}\ w_{i,j_2}\right)\ 2 (r_{i,j_2} \otimes r_{i,j_2}) \bigg) \\
        - \frac{1}{3}\bigg(&\left(n_i n_{j_1}\ d^2w_{i,j_1}\right)\ \left(n_i
n_{j_2}\ w_{i,j_2}\right) |r_{i,j_2}|^2 (r_{i,j_1} \otimes r_{i,j_1})\ + \\
        &\left(n_i n_{j_1}\ dw_{i,j_1}\right)\ \left(n_i
n_{j_2}\ w_{i,j_2}\right) 2 |r_{i,j_2}|^2 \frac{(r_{i,j_1} \otimes
r_{i,j_1})}{|r_{i,j_1}|} + \\
        &\left(n_i n_{j_1}\ dw_{i,j_1}\right)\ \left(n_i
n_{j_2}\ w_{i,j_2}\right) |r_{i,j_1}|^2 |r_{i,j_2}|^2
\frac{\partial^2|r_{i,j_1}|}{\partial q_i^2} + \\
        &\left(n_i n_{j_1}\ dw_{i,j_1}\right)\ \left(n_i
n_{j_2}\ w_{i,j_2}\right)\ 2 |r_{i,j_2}|^2 \frac{\left(r_{i,j_1} \otimes
r_{i,j_1}\right)}{|r_{i,j_1}|} + \\
        &\left(n_i n_{j_1}\ w_{i,j_1}\right)\ \left(n_i
n_{j_2}\ w_{i,j_2}\right)\ 2 |r_{i,j_2}|^2\ \text{I}\bigg) \end{split} \f]

\f[
\begin{split}
\frac{\partial^2 V^{\lambda}_{i,j_1,j_2}}{\partial q_{j_1} \partial
q_{j_2}}\ =\ \bigg( &\left(n_i n_{j_1}\ dw_{i,j_1}\right)\ \left(n_i
n_{j_2}\ dw_{i,j_2}\right)\ (r_{i,j_1} \cdot r_{i,j_2})^2\ \frac{(r_{i,j_1}
\otimes r_{i,j_2})}{|r_{i,j_1}||r_{i,j_2}|} + \\
        &\left(n_i n_{j_1}\ dw_{i,j_1}\right)\ \left(n_i
n_{j_2}\ w_{i,j_2}\right)\ 2 (r_{i,j_1} \cdot r_{i,j_2})\ \frac{(r_{i,j_1}
\otimes r_{i,j_1})}{|r_{i,j_1}|} + \\
        &\left(n_i n_{j_1}\ w_{i,j_1}\right)\ \left(n_i
n_{j_2}\ dw_{i,j_2}\right)\ 2 (r_{i,j_1} \cdot r_{i,j_2})\ \frac{(r_{i,j_2}
\otimes r_{i,j_2})}{|r_{i,j_2}|} + \\
        &\left(n_i n_{j_2}\ w_{i,j_2}\right)\ \left(n_i
n_{j_2}\ w_{i,j_2}\right)\ 2 (r_{i,j_2} \otimes r_{i,j_1})\ + \\
        &\left(n_i n_{j_2}\ w_{i,j_2}\right)\ \left(n_i
n_{j_2}\ w_{i,j_2}\right)\ 2 (r_{i,j_1} \cdot r_{i,j_2}) \cdot I \bigg) \\
        - \frac{1}{3}\bigg(&\left(n_i n_{j_1}\ dw_{i,j_1}\right)\ \left(n_i
n_{j_2}\ dw_{i,j_2}\right) |r_{i,j_1}| |r_{i,j_2}| (r_{i,j_1} \otimes
r_{i,j_2})\ +\\
        &\left(n_i n_{j_1}\ dw_{i,j_1}\right)\ \left(n_i
n_{j_2}\ w_{i,j_2}\right) 2 |r_{i,j_1}| (r_{i,j_1} \otimes r_{i,j_2}) + \\
        &\left(n_i n_{j_1}\ w_{i,j_1}\right)\ \left(n_i
n_{j_2}\ dw_{i,j_2}\right)\ 2 |r_{i,j_2}| \left(r_{i,j_1} \otimes
r_{i,j_2}\right)\ +\\
        &\left(n_i n_{j_1}\ w_{i,j_1}\right)\ \left(n_i
n_{j_2}\ w_{i,j_2}\right)\ 4 \left(r_{i,j_1} \otimes r_{i,j_2}\right) \bigg)
\end{split}
\f]

\f[
\begin{split}
\frac{\partial^2 V^{\lambda}_{i,j_1,j_2}}{\partial q_{j_2} \partial
q_{j_1}}\ =\ \bigg(&\left(n_i n_{j_1}\ dw_{i,j_1}\right)\ \left(n_i
n_{j_2}\ dw_{i,j_2}\right)\ (r_{i,j_1} \cdot r_{i,j_2})^2\  \frac{(r_{i,j_2}
\otimes r_{i,j_1})}{ |r_{i,j_1}| |r_{i,j_2}|} + \\
        &\left(n_i n_{j_1}\ w_{i,j_1}\right)\ \left(n_i
n_{j_2}\ dw_{i,j_2}\right)\ 2 (r_{i,j_1} \cdot r_{i,j_2})\ \frac{(r_{i,j_2}
\otimes r_{i,j_2})}{|r_{i,j_2}|} + \\
        &\left(n_i n_{j_1}\ dw_{i,j_1}\right)\ \left(n_i
n_{j_2}\ w_{i,j_2}\right)\ 2 (r_{i,j_1} \cdot r_{i,j_2})\ \frac{(r_{i,j_1}
\otimes r_{i,j_1})}{|r_{i,j_1}|} + \\
        &\left(n_i n_{j_2}\ w_{i,j_2}\right)\ \left(n_i
n_{j_2}\ w_{i,j_2}\right)\ 2 (r_{i,j_1} \otimes r_{i,j_2})\ + \\
        &\left(n_i n_{j_2}\ w_{i,j_2}\right)\ \left(n_i
n_{j_2}\ w_{i,j_2}\right)\ 2 (r_{i,j_1} \cdot r_{i,j_2}) \cdot I \bigg) \\
        - \frac{1}{3}\bigg(&\left(n_i n_{j_1}\ dw_{i,j_1}\right)\ \left(n_i
n_{j_2}\ dw_{i,j_2}\right) |r_{i,j_1}| |r_{i,j_2}| (r_{i,j_2} \otimes r_{i,j_1})
+ \\
        &\left(n_i n_{j_1}\ w_{i,j_1}\right)\ \left(n_i
n_{j_2}\ dw_{i,j_2}\right) 2 |r_{i,j_2}| (r_{i,j_2} \otimes r_{i,j_1}) + \\
        &\left(n_i n_{j_1}\ dw_{i,j_1}\right)\ \left(n_i
n_{j_2}\ w_{i,j_2}\right)\ 2 |r_{i,j_1}| \left(r_{i,j_2} \otimes
r_{i,j_1}\right) +\\ 
        &\left(n_i n_{j_1}\ w_{i,j_1}\right)\ \left(n_i
n_{j_2}\ w_{i,j_2}\right)\ 4 \left(r_{i,j_2} \otimes r_{i,j_1}\right) \bigg)
\end{split}
\f]

\f[
\begin{split}
\frac{\partial^2 V^{\lambda}_{i,j_1,j_2}}{\partial
q_{j_2}^2}\ =\ \bigg(&\left(n_i n_{j_1}\ w_{i,j_1}\right)\ \left(n_i
n_{j_2}\ d^2w_{i,j_2}\right)\ (r_{i,j_1} \cdot r_{i,j_2})^2\ \frac{(r_{i,j_2}
\otimes r_{i,j_2})}{|r_{i,j_2}|^2} + \\
        &\left(n_i n_{j_1}\ w_{i,j_1}\right)\ \left(n_i
n_{j_2}\ dw_{i,j_2}\right)\ 2 (r_{i,j_1} \cdot r_{i,j_2})\ \frac{(r_{i,j_2}
\otimes r_{i,j_1})}{|r_{i,j_2}|} + \\
        &\left(n_i n_{j_1}\ w_{i,j_1}\right)\ \left(n_i
n_{j_2}\ dw_{i,j_2}\right)\ (r_{i,j_1} \cdot
r_{i,j_2})^2\  \frac{\partial^2|r_{i,j_2}|}{\partial q_i^2}\ + \\
        &\left(n_i n_{j_1}\ w_{i,j_1}\right)\ \left(n_i
n_{j_2}\ dw_{i,j_2}\right)\ 2 \left(r_{i,j_1} \cdot r_{i,j_2}
\right)\ \frac{(r_{i,j_1} \otimes r_{i,j_2})}{|r_{i,j_2}|}\ + \\
        &\left(n_i n_{j_2}\ w_{i,j_2}\right)\ \left(n_i
n_{j_2}\ w_{i,j_2}\right)\ 2 (r_{i,j_1} \otimes r_{i,j_1}) \bigg) \\
        - \frac{1}{3}\bigg(&\left(n_i n_{j_1}\ w_{i,j_1}\right)\ \left(n_i
n_{j_2}\ d^2w_{i,j_2}\right) |r_{i,j_1}|^2 (r_{i,j_2} \otimes r_{i,j_2})\ + \\
        &\left(n_i n_{j_1}\ w_{i,j_1}\right)\ \left(n_i
n_{j_2}\ dw_{i,j_2}\right) 2 |r_{i,j_1}|^2 \frac{(r_{i,j_2} \otimes
r_{i,j_2})}{|r_{i,j_2}|} + \\
        &\left(n_i n_{j_1}\ w_{i,j_1}\right)\ \left(n_i
n_{j_2}\ dw_{i,j_2}\right) |r_{i,j_1}|^2 |r_{i,j_2}|^2
\frac{\partial^2|r_{i,j_2}|}{\partial q_i^2} +\\
        &\left(n_i n_{j_1}\ w_{i,j_1}\right)\ \left(n_i
n_{j_2}\ dw_{i,j_2}\right)\ 2 |r_{i,j_1}|^2 \frac{\left(r_{i,j_2} \otimes
r_{i,j_2}\right)}{|r_{i,j_2}|} + \\
        &\left(n_i n_{j_1}\ w_{i,j_1}\right)\ \left(n_i
n_{j_2}\ w_{i,j_2}\right)\ 2 |r_{i,j_1}|^2\ \text{I}\bigg) \end{split} \f]
 * @return potential_function
 */
potential_function V_quadrupole_ij1j2_adp_AlCu_constructor();

#endif // ADP_AlCu_HPP