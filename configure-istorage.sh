#!/bin/bash

clear
module purge

#PETSC_VERSION="Release-3.20"
#PETSC_VERSION="Debug-3.20"

PETSC_VERSION="Release-3.21"
#PETSC_VERSION="Debug-3.21"

# Uncomment your preferred IDE
PLATFORM="Unix Makefiles" # Generates standard UNIX makefiles.

# Load modules release 3.20
if [[ "$PETSC_VERSION" == "Release-3.20" ]]
 then
 module load gcc-10.2.0
 module load cmake-3.24.0
 module load petsc-3.20.0-openmpi-nodebug
 export SOLERA_DIR=$HOME/DMD
 export PETSC_DIR=/home/software/petsc-3.20.0-openmpi-nodebug
 export PKG_CONFIG_PATH=$PETSC_DIR/lib/pkgconfig
 C_COMPILER=$PETSC_DIR/bin/mpicc
 CXX_COMPILER=$PETSC_DIR/bin/mpicxx
 MAKE=/home/software/petsc-3.20.0/src-git/arch-linux-c-opt/bin/make

# Load modules release 3.21
elif [[ "$PETSC_VERSION" == "Release-3.21" ]]
 then
 module load gcc-10.2.0
 module load cmake-3.24.0
 module load petsc-3.21.0-openmpi-slepc-nodebug
 export SOLERA_DIR=$HOME/DMD
 export PETSC_DIR=/home/software/petsc-3.21.0
 export PETSC_ARCH=arch-linux-c-opt
 export PKG_CONFIG_PATH=$PETSC_DIR/$PETSC_ARCH/lib/pkgconfig:$PETSC_DIR/installation/lib/pkgconfig
 C_COMPILER=$PETSC_DIR/installation/bin/mpicc
 CXX_COMPILER=$PETSC_DIR/installation/bin/mpicxx
 MAKE=make

# Load modules debug 3.20
elif [[ "$PETSC_VERSION" == "Debug-3.20" ]]
 then
 module load gcc-10.2.0
 module load cmake-3.24.0
 module load petsc-3.20.0-openmpi
 export SOLERA_DIR=$HOME/DMD
 export PETSC_DIR=/home/software/petsc-3.20.0-openmpi
 export PKG_CONFIG_PATH=$PETSC_DIR/lib/pkgconfig
 C_COMPILER=$PETSC_DIR/bin/mpicc
 CXX_COMPILER=$PETSC_DIR/bin/mpicxx
 MAKE=/home/software/petsc-3.20.0/src-git/arch-linux-c-debug/bin/make

# Load modules debug 3.21
elif [[ "$PETSC_VERSION" == "Debug-3.21" ]]
 then
 module load gcc-10.2.0
 module load cmake-3.24.0
 module load petsc-3.21.0-openmpi-slepc-debug
 export SOLERA_DIR=$HOME/DMD
 export PETSC_DIR=/home/software/petsc-3.21.0
 export PETSC_ARCH=arch-linux-c-opt
 export PKG_CONFIG_PATH=$PETSC_DIR/$PETSC_ARCH/lib/pkgconfig:$PETSC_DIR/installation-debug/lib/pkgconfig
 C_COMPILER=$PETSC_DIR/installation/bin/mpicc
 CXX_COMPILER=$PETSC_DIR/installation/bin/mpicxx
 MAKE=make

else
    echo "Unrecognised option" $PETSC_VERSION
    exit
fi

## Check mpi compiers
if ! command -v ${C_COMPILER} &> /dev/null
then
    echo -e ""${C_COMPILER}": "${RED}" False "${RESET}""
    echo "please, ask your administrator to install "${C_COMPILER}""
    exit   
else
    echo -e ""${C_COMPILER}": "${GREEN}" True "${RESET}""
fi

if ! command -v ${CXX_COMPILER} &> /dev/null
then
    echo -e ""${CXX_COMPILER}": "${RED}" False "${RESET}""
    echo "please, ask your administrator to install "${CXX_COMPILER}""
    exit   
else
    echo -e ""${CXX_COMPILER}": "${GREEN}" True "${RESET}""
fi

## Check if cmake is installed
if ! command -v cmake &> /dev/null
then
    echo "cmake could not be found"
    echo "please, ask your administrator to run sudo apt-get install make"
    exit   
fi

## If build does not exists, create it
BUILD_DIR=${SOLERA_DIR}/"build"
if [ ! -d "$BUILD_DIR" ]; then
  mkdir ${BUILD_DIR}
fi

## Navigate inside of build
cd ${BUILD_DIR}

if [ -f "$FILE" ]; then
    ${MAKE} -k
else 
    cmake .. \
    -DCMAKE_BUILD_TYPE="Release" \
    -DCMAKE_CXX_COMPILER=${CXX_COMPILER} \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
    -G "${PLATFORM}"    
fi

if [[ "$PLATFORM" == "Unix Makefiles" ]]
then
${MAKE} -j8
elif [[ "$PLATFORM" == "Ninja" ]]
then
ninja
fi

cd ..

 