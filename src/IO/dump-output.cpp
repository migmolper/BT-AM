/**
 * @file ovito-output.cpp
 * @author J.P. Mendez
 * @brief
 * @version 0.1
 * @date 2022-07-23
 *
 * @copyright Copyright (c) 2022
 *
 */

#if __APPLE__
#include <malloc/_malloc.h>
#endif
#ifdef USE_MPI
#include <mpi.h>
#endif
#include "Atoms/Atom.hpp"
#include "Macros.hpp"
#include <fstream>   // Para ofstream
#include <iostream>  // Para cout
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using namespace std;
extern int diffusion;

extern PetscMPIInt size_MPI;
extern PetscMPIInt rank_MPI;

extern Eigen::Vector3d lattice_x_B0;
extern Eigen::Vector3d lattice_y_B0;
extern Eigen::Vector3d lattice_z_B0;

extern Eigen::Vector3d lattice_x_Bn;
extern Eigen::Vector3d lattice_y_Bn;
extern Eigen::Vector3d lattice_z_Bn;

extern double box_x_min;
extern double box_x_max;
extern double box_y_min;
extern double box_y_max;
extern double box_z_min;
extern double box_z_max;

extern Eigen::Vector3d box_origin_0;

/********************************************************/

PetscErrorCode write_dump_information(int step, const char* Name_file_t,
                                      DMD* Simulation) {

  PetscFunctionBegin;

  unsigned int dim = NumberDimensions;

/*   MPI_Barrier(MPI_COMM_WORLD);

  MPI_File_open(MPI_COMM_WORLD, Name_file_t, MPI_MODE_WRONLY, MPI_INFO_NULL,
                &file); */

  //! Get mesh information from the DMD simulation
  PetscInt n_sites_global;
  PetscCall(DMSwarmGetSize(Simulation->atomistic_data, &n_sites_global));

  PetscInt n_sites_local;
  PetscCall(DMSwarmGetLocalSize(Simulation->atomistic_data, &n_sites_local));

  //!
  AtomicSpecie* specie_ptr;
  PetscCall(DMSwarmGetField(Simulation->atomistic_data, "specie", NULL, NULL,
                            (void**)&specie_ptr));

  double* mean_q_ptr;
  PetscCall(DMSwarmGetField(Simulation->atomistic_data, DMSwarmPICField_coor,
                            NULL, NULL, (void**)&mean_q_ptr));

  double* stdv_q_ptr;
  PetscCall(DMSwarmGetField(Simulation->atomistic_data, "stdv-q", NULL, NULL,
                            (void**)&stdv_q_ptr));

  double* xi_ptr;
  PetscCall(DMSwarmGetField(Simulation->atomistic_data, "molar-fraction", NULL,
                            NULL, (void**)&xi_ptr));

  double* beta_ptr;
  PetscCall(DMSwarmGetField(Simulation->atomistic_data, "beta", NULL, NULL,
                            (void**)&beta_ptr));

  double* gamma_ptr;
  PetscCall(DMSwarmGetField(Simulation->atomistic_data, "gamma", NULL, NULL,
                            (void**)&gamma_ptr));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Open, write and close simulation file
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ofstream fichero;
  fichero.open(Name_file_t, ios::trunc);

  fichero.precision(17);
  fichero.setf(ios::fixed);
  fichero.setf(ios::showpoint);

  Eigen::Vector3d box_dir_n = lattice_x_Bn + lattice_y_Bn + lattice_z_Bn;

  if (rank_MPI == 0) {
    fichero << "ITEM: TIMESTEP" << endl             //!
            << step << endl                         //!
            << "ITEM: NUMBER OF ATOMS " << endl     //!
            << n_sites_global << endl               //!
            << "ITEM: BOX BOUNDS pp pp pp" << endl  //!
            << box_origin_0(0) << " " << box_origin_0(0) + box_dir_n(0)
            << endl  //!
            << box_origin_0(1) << " " << box_origin_0(1) + box_dir_n(1)
            << endl  //!
            << box_origin_0(2) << " " << box_origin_0(2) + box_dir_n(2)
            << endl  //!
            << "ITEM: ATOMS type x y z Stdvq Molarfraction chemicalmultp "
               "thermalmultp"
            << endl;  //!
  }

  for (unsigned int site_i = 0; site_i < n_sites_local; site_i++) {

    fichero << (int)specie_ptr[site_i] + 1 << " "   //!
            << mean_q_ptr[site_i * dim + 0] << " "  //!
            << mean_q_ptr[site_i * dim + 1] << " "  //!
            << mean_q_ptr[site_i * dim + 2] << " "  //!
            << stdv_q_ptr[site_i] << " "            //!
            << xi_ptr[site_i] << " "                //!
            << gamma_ptr[site_i] << " "             //!
            << beta_ptr[site_i] << endl;            //!
  }

  fichero.close();

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Restore fields
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMSwarmRestoreField(Simulation->atomistic_data, "specie", NULL,
                                NULL, (void**)&specie_ptr));
  PetscCall(DMSwarmRestoreField(Simulation->atomistic_data,
                                DMSwarmPICField_coor, NULL, NULL,
                                (void**)&mean_q_ptr));
  PetscCall(DMSwarmRestoreField(Simulation->atomistic_data, "stdv-q", NULL,
                                NULL, (void**)&stdv_q_ptr));
  PetscCall(DMSwarmRestoreField(Simulation->atomistic_data, "molar-fraction",
                                NULL, NULL, (void**)&xi_ptr));
  PetscCall(DMSwarmRestoreField(Simulation->atomistic_data, "gamma", NULL, NULL,
                                (void**)&gamma_ptr));
  PetscCall(DMSwarmRestoreField(Simulation->atomistic_data, "beta", NULL, NULL,
                                (void**)&beta_ptr));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/********************************************************/

PetscErrorCode Write_restart_mpi(int step, const char* Name_file_t,
                                 DMD* Simulation) {

  PetscFunctionBegin;

  // definition of variable
  int n_types = 1;
  MPI_File file;
  MPI_Offset ofst = (int)0;
  MPI_Datatype datatype, filetype;
  int heading1[4];
  double heading2[9];
  int num_fixed_q = 0;  // (atoms->fixed_q).size();
  int num_fixed_x = 0;  // (atoms->fixed_x).size();
  MPI_Aint data_disps[3],
      file_disps[3];  // q_i[3](double), x_ik[n_types](double), omega_i(double)
  int array_blocklens[3];
  MPI_Datatype array_types[3] = {MPI_BYTE, MPI_BYTE, MPI_BYTE};

  Eigen::Vector3d box_dir_n = lattice_x_Bn + lattice_y_Bn + lattice_z_Bn;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Read atomistic information
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscInt n_sites_global;
  PetscCall(DMSwarmGetSize(Simulation->atomistic_data, &n_sites_global));

  PetscInt n_sites_local;
  PetscCall(DMSwarmGetLocalSize(Simulation->atomistic_data, &n_sites_local));

  //!
  AtomicSpecie* specie_ptr;
  PetscCall(DMSwarmGetField(Simulation->atomistic_data, "specie", NULL, NULL,
                            (void**)&specie_ptr));

  double* mean_q_ptr;
  PetscCall(DMSwarmGetField(Simulation->atomistic_data, DMSwarmPICField_coor,
                            NULL, NULL, (void**)&mean_q_ptr));

  double* stdv_q_ptr;
  PetscCall(DMSwarmGetField(Simulation->atomistic_data, "stdv-q", NULL, NULL,
                            (void**)&stdv_q_ptr));

  double* xi_ptr;
  PetscCall(DMSwarmGetField(Simulation->atomistic_data, "molar-fraction", NULL,
                            NULL, (void**)&xi_ptr));

  double* beta_ptr;
  PetscCall(DMSwarmGetField(Simulation->atomistic_data, "beta", NULL, NULL,
                            (void**)&beta_ptr));

  double* gamma_ptr;
  PetscCall(DMSwarmGetField(Simulation->atomistic_data, "gamma", NULL, NULL,
                            (void**)&gamma_ptr));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Open dump file
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  MPI_Barrier(MPI_COMM_WORLD);

  MPI_File_open(MPI_COMM_WORLD, Name_file_t, MPI_MODE_WRONLY, MPI_INFO_NULL,
                &file);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Write heading
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  if (rank_MPI == 0) {

    heading1[0] = n_sites_global;
    heading1[1] = n_types;
    heading1[2] = 0;  // atoms->diffusion;
    heading1[3] = 0;  // atoms->maxneigh;

    heading2[0] = 0.0;  // atoms->r2_cutoff;
    heading2[2] = 0.0;  // atoms->beta[0];
    heading2[3] = box_origin_0(0);
    heading2[4] = box_origin_0(0) + box_dir_n(0);
    heading2[5] = box_origin_0(1);
    heading2[6] = box_origin_0(1) + box_dir_n(1);
    heading2[7] = box_origin_0(2);
    heading2[8] = box_origin_0(2) + box_dir_n(2);

    MPI_File_write_at(
        file, (int)0, heading1, sizeof(int) * 4, MPI_BYTE,
        MPI_STATUS_IGNORE);  // n_atoms, n_types, diffusion, maxneigh
    MPI_File_write_at(file, sizeof(int) * 4, heading2, sizeof(double) * 9,
                      MPI_BYTE, MPI_STATUS_IGNORE);
  }

  ofst = sizeof(double) * 9 + sizeof(int) * (6 + num_fixed_q + num_fixed_x);

  // for data
  MPI_Get_address(mean_q_ptr, &data_disps[0]);
  MPI_Get_address(xi_ptr, &data_disps[1]);
  MPI_Get_address(stdv_q_ptr, &data_disps[2]);

  // for file
  file_disps[0] = ofst + rank_MPI * n_sites_local * sizeof(double) * 3;  // q[3]
  file_disps[1] =
      ofst + (n_sites_global * 3 + rank_MPI * n_sites_local * n_types) *
                 sizeof(double);  // x_i[n_types]
  file_disps[2] =
      ofst + (n_sites_global * (3 + n_types) + rank_MPI * n_sites_local) *
                 sizeof(double);  // w

  // define MPI_type for the data
  if (rank_MPI < size_MPI - 1 || size_MPI == 1) {
    if (n_sites_local > 0) {
      // q_i
      array_blocklens[0] = sizeof(double) * 3 * n_sites_local;
      // x_i[n_types]
      array_blocklens[1] = sizeof(double) * n_types * n_sites_local;
      // omega_i
      array_blocklens[2] = sizeof(double) * n_sites_local;
    } else {
      for (int i = 0; i < 3; i++) array_blocklens[i] = 1;
    }
  } else {  // for the rank_MPI==size-1 when size>1, write the rest
    if (n_sites_local > 0) {
      array_blocklens[0] = sizeof(double) * 3 *
                           (n_sites_global - n_sites_local * rank_MPI);  // q_i
      array_blocklens[1] =
          sizeof(double) * n_types *
          (n_sites_global - n_sites_local * rank_MPI);  // x_i[n_types]
      array_blocklens[2] =
          sizeof(double) *
          (n_sites_global - n_sites_local * rank_MPI);  // omega_i
    } else {
      for (int i = 0; i < 3; i++) array_blocklens[i] = 1;
    }
  }

  // define MPI_type for the data anf the file
  MPI_Type_create_struct(3, array_blocklens, data_disps, array_types,
                         &datatype);
  MPI_Type_create_struct(3, array_blocklens, file_disps, array_types,
                         &filetype);
  // commit MPI_type
  MPI_Type_commit(&datatype);
  MPI_Type_commit(&filetype);
  // write
  MPI_File_set_view(file, 0, MPI_BYTE, filetype, "native", MPI_INFO_NULL);
  MPI_File_write_all(file, 0, 1, datatype, MPI_STATUS_IGNORE);
  // free MPI_type
  MPI_Type_free(&datatype);
  MPI_Type_free(&filetype);

  // close file
  MPI_File_close(&file);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Restore fields
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMSwarmRestoreField(Simulation->atomistic_data, "specie", NULL,
                                NULL, (void**)&specie_ptr));
  PetscCall(DMSwarmRestoreField(Simulation->atomistic_data,
                                DMSwarmPICField_coor, NULL, NULL,
                                (void**)&mean_q_ptr));
  PetscCall(DMSwarmRestoreField(Simulation->atomistic_data, "stdv-q", NULL,
                                NULL, (void**)&stdv_q_ptr));
  PetscCall(DMSwarmRestoreField(Simulation->atomistic_data, "molar-fraction",
                                NULL, NULL, (void**)&xi_ptr));
  PetscCall(DMSwarmRestoreField(Simulation->atomistic_data, "gamma", NULL, NULL,
                                (void**)&gamma_ptr));
  PetscCall(DMSwarmRestoreField(Simulation->atomistic_data, "beta", NULL, NULL,
                                (void**)&beta_ptr));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/********************************************************/

#ifdef USE_MPI
void ConvertBinaryFiletoTextFile() {

  // definition of variable
  MPI_Status status;
  int errcode;
  MPI_File file;
  char filename[] = "results/q_w_x_binary";
  MPI_Offset ofst = (int)0, maxofst;
  int n_types = 1;
  int Nat, it;
  double q[3], type[n_types], omega;
  ofstream fichero;

  errcode = MPI_File_open(MPI_COMM_SELF, filename, MPI_MODE_RDONLY,
                          MPI_INFO_NULL, &file);
  if (errcode != MPI_SUCCESS) {
    cout << endl << "We could not open the binary file: " << filename << endl;
    return;
  }

  fichero.open("results/q_w_x_txt", ios::app);

  MPI_File_get_size(file, &maxofst);

  while (ofst < maxofst) {

    MPI_File_read_at(file, ofst, &Nat, sizeof(int), MPI_BYTE,
                     &status);  // Number of atoms
    ofst += sizeof(int);
    fichero << Nat << endl;

    MPI_File_read_at(file, ofst, &it, sizeof(int), MPI_BYTE,
                     &status);  // time step or iteration number
    ofst += sizeof(int);
    fichero << it << endl;

    for (int i = 0; i < Nat; i++) {

      MPI_File_read_at(file, ofst + i * 3 * sizeof(double), q,
                       sizeof(double) * 3, MPI_BYTE, &status);  // q_i[3]
      MPI_File_read_at(file, ofst + (3 * Nat + i * n_types) * sizeof(double),
                       &type, sizeof(double) * n_types, MPI_BYTE,
                       &status);  // x_ik[n_types]
      MPI_File_read_at(file, ofst + ((3 + n_types) * Nat + i) * sizeof(double),
                       &omega, sizeof(double), MPI_BYTE, &status);  // omega_i

      // id << qx << qy << qz << xi[n_types] << frequency
      fichero << i + 1 << "  " << q[0] << "  " << q[1] << "  " << q[2];
      fichero << "  " << type[0];
      fichero << "  " << omega << endl;
    }

    ofst += Nat * sizeof(double) * (4 + 1);
  }

  cout << endl << "Done converting binary file to text file.";

  fichero.close();
  MPI_File_close(&file);

  return;
}
#endif

/********************************************************/

#ifdef USE_MPI
void ConvertBinaryFiletoTextFile_Petsc() {

  // definition of variable
  int n_types = 1;
  MPI_Status status;
  int errcode;
  MPI_File file;
  char filename[] = "results/q_w_x_Petsc_binary";
  MPI_Offset ofst = (int)0, maxofst;
  int Nat, it;
  double q[3], type[n_types], omega, fq[3], fw;
  ofstream fichero;
  double time = 0;
  errcode = MPI_File_open(MPI_COMM_SELF, filename, MPI_MODE_RDONLY,
                          MPI_INFO_NULL, &file);
  if (errcode != MPI_SUCCESS) {
    cout << endl << "We could not open the binary file: " << filename << endl;
    return;
  }

  fichero.open("results/q_w_x_Petsc_txt", ios::app);

  MPI_File_get_size(file, &maxofst);

  while (ofst < maxofst) {

    MPI_File_read_at(file, ofst, &Nat, sizeof(int), MPI_BYTE,
                     &status);  // Number of atoms
    ofst += sizeof(int);
    //      fichero << Nat << endl;
    fichero << "ITEM: TIMESTEP" << endl
            << time << endl
            << "ITEM: BOX BOUNDS ss ss ss" << endl
            << 0 << " "
            << "3.8268000000000001e+01" << endl
            << 0 << " " << 4.4188000000000002e+01 << endl
            << 0 << " " << 0 << " " << 4.1660800000000002e+01 << endl
            << "ITEM: ATOMS id type x y z" << endl;
    MPI_File_read_at(file, ofst, &it, sizeof(int), MPI_BYTE,
                     &status);  // time step or iteration number
    ofst += sizeof(int);
    //        fichero << it << endl;

    for (int i = 0; i < Nat; i++) {

      MPI_File_read_at(file, ofst + i * 3 * sizeof(double), q,
                       sizeof(double) * 3, MPI_BYTE, &status);  // q_i[3]
      MPI_File_read_at(file, ofst + (3 * Nat + i * n_types) * sizeof(double),
                       &type, sizeof(double) * n_types, MPI_BYTE,
                       &status);  // x_ik[n_types]
      MPI_File_read_at(file, ofst + ((3 + n_types) * Nat + i) * sizeof(double),
                       &omega, sizeof(double), MPI_BYTE, &status);  // omega_i
      MPI_File_read_at(
          file,
          ofst + Nat * sizeof(double) * (4 + n_types) + 3 * sizeof(double) * i,
          fq, sizeof(double) * 3, MPI_BYTE, &status);  // atomic forces
      MPI_File_read_at(
          file,
          ofst + Nat * sizeof(double) * (7 + n_types) + sizeof(double) * i, &fw,
          sizeof(double), MPI_BYTE, &status);  // frequency force

      // id << qx << qy << qz << xi[n_types] << frequency << atomic force <<
      // frequency force
      fichero << i + 1 << "  " << q[0] << "  " << q[1] << "  " << q[2];
      fichero << "  " << type[0];
      fichero << "  " << omega << "  " << fq[0] << "  " << fq[1] << "  "
              << fq[2] << "  " << fw << endl;
    }

    ofst += Nat * sizeof(double) * (8 + n_types);
  }

  cout << endl << "Done converting binary file to text file.";

  fichero.close();
  MPI_File_close(&file);

  return;
}
#endif

/********************************************************/
