#!/bin/bash
set -e

export PATH=$HOME/software/Firefly-RK3308_Linux_SDK_git_20181116/buildroot/output/firefly_rk3308_release/host/bin:$PATH

mkdir -p build-aarch64-linux-gnu
rm -rf build-aarch64-linux-gnu/*
cd build-aarch64-linux-gnu

cmake -DCMAKE_TOOLCHAIN_FILE=../cmake/aarch64-linux-gnu.toolchain.cmake \
-DNCNN_VULKAN=OFF \
-DCMAKE_BUILD_TYPE=Release \
-DBUILD_SYSTEM=aarch64-linux-gnu \
..

make -j8
