#!/bin/bash -e
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

SRC=$PWD
COVERAGE=0
# by default output is the current directory
OUT=$(readlink -f $SRC)

# colors
RED="\033[0;31m"
GREEN="\033[0;32m"
NC="\033[0m" # No Color

# print a failure message
function failMsg
{
    echo -e "$RED$1$NC$2"
}

while (($#)); do
case $1 in
    -o|--output)
    shift
    if [ -z $1 ]; then
        failMsg "Missing argument"
        exit 1
    fi
    OUT=$(readlink -f $1)
    shift
    ;;

    -h|--help)
    echo "Usage: $0 [-o|--output]"
    echo ""
    echo " -o, --output"
    echo "  output directory, default is current directory"
    echo " -h, --help"
    echo "  display this help message"
    exit 1
    ;;

    *)
    failMsg "Unknown option '"$1"'"
    exit 1
    ;;
esac
done

mkdir -p $OUT
cd $OUT

# initial call to cmake to generate the build files
if [ ! -f CMakeCache.txt ]; then
    cmake $SRC
fi

# build
make -j$(nproc)
# install to package
cmake --install .
