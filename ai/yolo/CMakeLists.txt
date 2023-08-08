# CMake 最低版本号要求
cmake_minimum_required (VERSION 3.8.0)

# 项目信息
project (yolo)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11 -fPIE")

set(ROOT_DIR "${CMAKE_SOURCE_DIR}")
set(CAFFE_DIR "${ROOT_DIR}/../caffe_hisi")

include_directories("${ROOT_DIR}/.")
include_directories("${CAFFE_DIR}/include")
include_directories("${CAFFE_DIR}/build/src")	#"caffe/proto/caffe.pb.h"


include_directories("/usr/local/cuda-12.2/targets/x86_64-linux/include/")



add_definitions(-DCONFIG_SPDLOG)

file(GLOB SRC "${ROOT_DIR}/*.cpp"
			  "${ROOT_DIR}/base/*.cpp"
)

if(LINUX)
	set(XLIB pthread)
else()
	set(XLIB "")
endif()

# message(${SRC})
add_executable(yolo ${SRC})
target_link_libraries(yolo ${XLIB} ${CMAKE_DL_LIBS})

