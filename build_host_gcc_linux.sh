#!/bin/bash
set -e

mkdir -p build-host-gcc-linux
rm -rf build-host-gcc-linux/*
cd build-host-gcc-linux
cmake -DNCNN_VULKAN=OFF -DCMAKE_BUILD_TYPE=Release -DBUILD_SYSTEM=linux ..
make -j8

cp ssdlite ..
cp ssdlite_192x192 ..

