#$ -S /bin/bash
#$ -N run-BT-AM
#$ -wd /home/mmolinos/BT-AM
#$ -m a
#$ -o run-BT-AM.salida
#$ -e run-BT-AM.err
#$ -q all.q@istorage-04
#$ -pe mpi 8

#PETSC_VERSION="Debug-3.22.1"
PETSC_VERSION="Release-3.21"


if [[ "$PETSC_VERSION" == "Release-3.21" ]]; then
    echo Release version
    module load gcc-10.2.0
    module load cmake-3.24.0
    module load petsc-3.21.0-openmpi-slepc-nodebug
    MPI_RUN=/home/software/petsc-3.21.0/installation/bin/mpiexec

elif [[ "$PETSC_VERSION" == "Debug-3.21" ]]; then
    echo Debug version
    module load gcc-10.2.0
    module load cmake-3.24.0
    module load petsc-3.21.0-openmpi-slepc-debug
    MPI_RUN=/home/software/petsc-3.21.0/installation-debug/bin/mpiexec
else
    echo "Unrecognised option" $PETSC_VERSION
    exit
fi

MPI_P=8
export OMP_NUM_THREADS=40

${MPI_RUN} -np ${MPI_P} ./exe-BT-AM