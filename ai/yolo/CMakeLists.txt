# CMake 最低版本号要求
cmake_minimum_required (VERSION 3.8.0)

# 项目信息
project (yolo)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11 -fPIE")

set(ROOT_DIR "${CMAKE_SOURCE_DIR}")
set(CAFFE_DIR     "/home/leo/imvt/imvt.caffe")
set(CAFFE_INSTALL "/home/leo/imvt/imvt.caffe/build4090/install")

include_directories("${ROOT_DIR}/.")
include_directories("${CAFFE_DIR}/include")
include_directories("${CAFFE_INSTALL}/include") 	#"caffe/proto/caffe.pb.h"

include_directories("/usr/local/cuda-12.2/targets/x86_64-linux/include/")

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
list (APPEND CMAKE_PREFIX_PATH "${CAFFE_INSTALL}")
list (APPEND CMAKE_PREFIX_PATH "${CAFFE_INSTALL}/lib")
set(Caffe_DIR "${CAFFE_INSTALL}/share/Caffe")
find_package(Caffe REQUIRED)

# u22.04:
#list (APPEND CMAKE_PREFIX_PATH "/usr/lib/x86_64-linux-gnu/cmake/Boost-1.74.0")
#find_package(boost_ALL REQUIRED COMPONENTS ALL) 

# u20.04:
list (APPEND CMAKE_PREFIX_PATH "/usr/lib/x86_64-linux-gnu/cmake/Boost-1.71.0")
find_package(Boost REQUIRED system thread filesystem python)
#find_package(Boost REQUIRED)

add_definitions(-DCONFIG_SPDLOG)

#cmake ../yolo
#cmake ../yolo -DUSE_CUDA=yes
#cmake ../yolo -DUSE_CUDA=no
if (USE_CUDA)
    message(STATUS "USE_CUDA")
else()
    message(STATUS "USE_CPU")
    add_definitions(-DCPU_ONLY)
endif()

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

