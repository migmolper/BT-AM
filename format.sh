#!/bin/bash

clear

## Check if clang-format is installed
if ! command -v clang-format &> /dev/null
then
    echo -e "clang-format: "${RED}" False "${RESET}""
    echo "please, ask your administrator to run ''sudo apt-get install clang-tools''"
    exit   
else 
    echo -e "clang-format: "${GREEN}" True "${RESET}" "
fi

## Format the project to make it prettier :-)
find ./src -iname *.hpp | xargs clang-format -i --verbose
find ./src -iname *.cpp | xargs clang-format -i --verbose

## Update submodules
git submodule update --init

## Check if doxygen is installed
if ! command -v doxygen &> /dev/null
then
    echo -e "doxygen: "${RED}" False "${RESET}""
    echo "please, ask your administrator to run ''sudo apt-get install doxygen''"
    exit   
else 
    echo -e "doxygen: "${GREEN}" True "${RESET}" "
fi

## Update documentation
doxygen
