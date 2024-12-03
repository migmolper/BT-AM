#!/bin/bash

## Configure environments
MPI_RUN=~/petsc/arch-darwin-c-release/bin/mpirun
#MPI_RUN=~/petsc/arch-darwin-c-debug/bin/mpirun

MPI_P=8

${MPI_RUN} -np ${MPI_P} ./exe-BT-AM