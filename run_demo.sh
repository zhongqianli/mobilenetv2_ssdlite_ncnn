#!/bin/bash

export LD_LIBRARY_PATH=/media/home/zql/3rdparty/opencv/build-host-gcc-linux/install/lib:$LD_LIBRARY_PATH

./ssdlite models/mbv2_ssdlite images/bicycle.jpg
