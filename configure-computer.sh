
#!/bin/bash

clear

SOLERA_VERSION="Release"
#SOLERA_VERSION="Debug"

#PETSC_VERSION="Release-3.20"
#PETSC_VERSION="Debug-3.20"

PETSC_VERSION="Release-3.22.1"
#PETSC_VERSION="Debug-3.22.1"

# Generates standard UNIX makefiles.
PLATFORM="Unix Makefiles" 

export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"
export OpenMP_ROOT=$(brew --prefix)/opt/libomp
export C_INCLUDE_PATH=/usr/local/include
export CPLUS_INCLUDE_PATH=/usr/local/include
export SOLERA_DIR=$HOME/DMD
export PETSC_DIR=$HOME/petsc 
export SLEPC_DIR=$HOME/slepc 
if [[ "$PETSC_VERSION" == "Release-3.22.1" ]]
then
export PETSC_ARCH=arch-darwin-c-release
else
export PETSC_ARCH=arch-darwin-c-debug
fi
export PKG_CONFIG_PATH=$PETSC_DIR/$PETSC_ARCH/lib/pkgconfig:$SLEPC_DIR/$PETSC_ARCH/lib/pkgconfig
export VTK_DIR=/usr/local/include/vtk-9.2
C_COMPILER=$PETSC_DIR/$PETSC_ARCH/bin/mpicc
CXX_COMPILER=$PETSC_DIR/$PETSC_ARCH/bin/mpicxx
MAKE=make


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
BUILD_DIR="build"
if [ ! -d "$BUILD_DIR" ]; then
  mkdir ${BUILD_DIR}
fi

## Navigate inside of build
cd ${BUILD_DIR}

if [ -f "$FILE" ]; then
    ${MAKE} -k
else 
    cmake .. \
    -DCMAKE_BUILD_TYPE=${SOLERA_VERSION} \
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

 
