#
# Author: zhongqianli <qianli_zh@qq.com>
# Date  : 2020-11-3
#

cmake_minimum_required(VERSION 2.8)

project(ssdlite)

# linux or aarch64-linux-gnu
option(BUILD_SYSTEM "build system")


SET(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

option(NCNN_OPENMP "openmp support" ON)
option(NCNN_STDIO "load model from external file" ON)
option(NCNN_STRING "plain and verbose string" ON)
option(NCNN_INSTALL_SDK "Install ncnn library and headers" ON)
option(NCNN_OPENCV "minimal opencv structure emulation" OFF)
option(NCNN_BENCHMARK "print benchmark information for every layer" OFF)
option(NCNN_PIXEL "convert and resize from/to image pixel" ON)
option(NCNN_PIXEL_ROTATE "rotate image pixel orientation" OFF)
option(NCNN_CMAKE_VERBOSE "print verbose cmake messages" OFF)
option(NCNN_VULKAN "vulkan compute support" OFF)
option(NCNN_REQUANT "auto merge int8 quant and dequant" OFF)
option(NCNN_AVX2 "optimize x86 platform with avx2" OFF)

if(ANDROID OR IOS)
    option(NCNN_DISABLE_RTTI "disable rtti" ON)
    option(NCNN_BUILD_TOOLS "build tools" OFF)
else()
    option(NCNN_DISABLE_RTTI "disable rtti" OFF)
    option(NCNN_BUILD_TOOLS "build tools" ON)
endif()

if(ANDROID OR IOS OR LINUX)
    option(NCNN_DISABLE_EXCEPTION "disable exception" ON)
else()
    option(NCNN_DISABLE_EXCEPTION "disable exception" OFF)
endif()


if(${BUILD_SYSTEM} STREQUAL "aarch64-linux-gnu")
    include_directories(
        /media/home/zql/3rdparty/ncnn/build-aarch64-linux-gnu/install/include/ncnn
        /media/home/zql/3rdparty/ncnn/build-aarch64-linux-gnu/install/include
        /media/home/zql/3rdparty/ncnn/src
        /media/home/zql/3rdparty/opencv/build-aarch64-linux-gnu/install/include/opencv4
        /media/home/zql/3rdparty/opencv/build-aarch64-linux-gnu/install/include/opencv4/opencv2
        /media/home/zql/3rdparty/opencv/build-aarch64-linux-gnu/install/include
    )
else()
    include_directories(
        /media/home/zql/3rdparty/ncnn/build-host-gcc-linux/install/include/ncnn
        /media/home/zql/3rdparty/ncnn/build-host-gcc-linux/install/include
        /media/home/zql/3rdparty/ncnn/src
        /media/home/zql/3rdparty/opencv/build-host-gcc-linux/install/include/opencv4
        /media/home/zql/3rdparty/opencv/build-host-gcc-linux/install/include/opencv4/opencv2
        /media/home/zql/3rdparty/opencv/build-host-gcc-linux/install/include
    )
endif()


if(${BUILD_SYSTEM} STREQUAL "aarch64-linux-gnu")
    link_directories(
        /media/home/zql/3rdparty/ncnn/build-aarch64-linux-gnu/install/lib
        /media/home/zql/3rdparty/opencv/build-aarch64-linux-gnu/install/lib
    )
else()
    link_directories(
        /media/home/zql/3rdparty/ncnn/build-host-gcc-linux/install/lib
        /media/home/zql/3rdparty/opencv/build-host-gcc-linux/install/lib
    )
endif()

link_libraries(
    ncnn
    opencv_core
    opencv_imgproc
    opencv_highgui
    opencv_imgcodecs
    opencv_videoio
)


add_executable(ssdlite mobilenetv2ssdlite.cpp)
add_executable(ssdlite_192x192 ssdlite_192x192.cpp)
