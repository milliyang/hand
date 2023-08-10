# CMake 最低版本号要求
cmake_minimum_required (VERSION 3.8.0)

# 项目信息
project (yolo)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11 -fPIE")

set(ROOT_DIR "${CMAKE_SOURCE_DIR}")
#set(CAFFE_DIR "${ROOT_DIR}/../caffe_hisi")
set(CAFFE_DIR "/home/leo/github/caffe")

#/home/leo/github/caffe/build/src/caffe/proto/caffe.pb.h

include_directories("${ROOT_DIR}/.")
include_directories("${CAFFE_DIR}/include")
include_directories("${CAFFE_DIR}/build/src")	#"caffe/proto/caffe.pb.h"
include_directories("${CAFFE_DIR}/build/install/include")	#"caffe/proto/caffe.pb.h"

include_directories("${CAFFE_DIR}/.build_release/include")	#"caffe/proto/caffe.pb.h"
include_directories("${CAFFE_DIR}/build/include")	#"caffe/proto/caffe.pb.h"

#include_directories("/usr/local/include")	#opencv

include_directories("/usr/local/cuda-12.2/targets/x86_64-linux/include/")

# include(cmake/FindOpenBLAS.cmake)
# include(cmake/FindAtlas.cmake)
# include(cmake/ProtoBuf.cmake)

# OPENCV
if(WIN32)
    if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
        set(OpenCV_DIR "F:\\finetune\\buildcv\\install")                        #MinGW build opencv
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
        set(CMAKE_PREFIX_PATH "C:\\Qt\\Qt5.13.0\\5.13.0\\msvc2015_64")          #just for MinGW
        # set(OpenCV_DIR "C:\\opencv\\build\\x64\\vc15\\lib")                     #msvc build opencv
        set(OpenCV_DIR "H:\\zone\\3rd\\opencv.install")                     #msvc build opencv
    endif()
    # when run, add windows system path:
    #  C:\opencv\build\x64\vc15\bin                 opencv_*.dll
    #  C:\Qt\Qt5.13.0\5.13.0\msvc2015_64\bin        Qt5*.dll   //MSCV
    #  F:\finetune\buildcv\install\x64\mingw\bin    Qt5*.dll   //GNU
    #  seting two version Qt5*.dll in PATH will lead to error !!!
else()
    #set binary output dir
    #set(EXECUTABLE_OUTPUT_PATH "bin")

    #set(OpenCV_DIR "${ROOT_DIR}/3rd/opencv.install/share/OpenCV/")
	set(OpenCV_DIR "/usr/local/share/OpenCV/")

    set(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/cmake/opencv4")   #u20.04
endif()

find_package( OpenCV REQUIRED )
#find_package( OpenCV 3.4.17 REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})
message(STATUS "OpenCV_LIBS = ${OpenCV_LIBS}")


# If you only built the library
list (APPEND CMAKE_PREFIX_PATH "/home/leo/hand/ai/caffe_hisi/.build_release")
list (APPEND CMAKE_PREFIX_PATH "/home/leo/github/caffe/build/install")
#list (APPEND CMAKE_PREFIX_PATH "/home/leo/hand/ai/caffe_hisi/.build_release")

# If you installed the library there
#list (APPEND CMAKE_PREFIX_PATH "/home/leo/hand/ai/caffe_hisi")
find_package(Caffe REQUIRED)

# u22.04:
#list (APPEND CMAKE_PREFIX_PATH "/usr/lib/x86_64-linux-gnu/cmake/Boost-1.74.0")
#find_package(boost_ALL REQUIRED COMPONENTS ALL) 

# u20.04:
list (APPEND CMAKE_PREFIX_PATH "/usr/lib/x86_64-linux-gnu/cmake/Boost-1.71.0")
find_package(Boost REQUIRED system thread filesystem python)

#find_package(Boost REQUIRED)


add_definitions(-DCONFIG_SPDLOG)

add_definitions(-DCPU_ONLY)


file(GLOB SRC "${ROOT_DIR}/*.cpp"
			  "${ROOT_DIR}/base/*.cpp"
)

if(LINUX)
	set(XLIB pthread)
else()
	set(XLIB "")
endif()


set(CAFFELIB caffe caffeproto)


# message(${SRC})
add_executable(yolo ${SRC})
#target_link_libraries(yolo ${XLIB} ${CMAKE_DL_LIBS}  ${OpenCV_LIBS} ${CAFFELIB} ${OpenBLAS_LIB} ${Atlas_LIBRARIES} ${PROTOBUF_LIBRARIES})
target_link_libraries(yolo ${XLIB} ${CMAKE_DL_LIBS}  ${OpenCV_LIBS} ${CAFFELIB} ${OpenBLAS_LIB} ${Atlas_LIBRARIES} ${PROTOBUF_LIBRARIES} ${Boost_LIBRARIES})

