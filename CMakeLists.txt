cmake_minimum_required (VERSION 2.6)
project (deepflow)

set (DEEPFLOW_VERSION_MAJOR 0)
set (DEEPFLOW_VERSION_MINOR 1)

find_package(CUDA QUIET REQUIRED)

set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_HOST_COMPILATION_CPP ON)

set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -gencode arch=compute_35,code=sm_35
    )

add_definitions(-DDEEPFLOW_DLL_EXPORT -D_CONSOLE -DGFLAGS_DLL_DECLARE_FLAG= -DGFLAGS_DLL_DEFINE_FLAG= -DGLOG_NO_ABBREVIATED_SEVERITIES -DGOOGLE_GLOG_DLL_DECL= -DGFLAGS_IS_A_DLL=0
		)

include_directories(
	include
	include/proto
	third-party/protobuf/src
	third-party/opencv/build/include
	third-party/gflags/build/include
	third-party/glog/src/windows
	third-party/cudnn-8.0-x64-v6.0/cuda/include	
	)

link_directories(
	${CMAKE_SOURCE_DIR}/third-party/protobuf/src
	${CMAKE_SOURCE_DIR}/third-party/protobuf/cmake-build/$(ConfigurationName)
	${CMAKE_SOURCE_DIR}/third-party/opencv/build/lib/$(ConfigurationName)
	${CMAKE_SOURCE_DIR}/third-party/glog/cmake-build/$(ConfigurationName)
	${CMAKE_SOURCE_DIR}/third-party/gflags/cmake-build/lib/$(ConfigurationName)	
	${CMAKE_SOURCE_DIR}/third-party/cudnn-8.0-x64-v6.0/cuda/lib/x64
	)

file(GLOB SOURCES_CORE
	"src/core/*.cpp"
	"src/core/*.cu"
	)

file(GLOB SOURCES_GENERATORS
	"src/generators/*.cu"
	"src/generators/*.cpp"
	)

file(GLOB SOURCES_INITIALIZERS
	"src/initializers/*.cu"
	"src/initializers/*.cpp"
	)

file(GLOB SOURCES_NODES
	"src/nodes/*.cu"
	"src/nodes/*.cpp"
	)

file(GLOB SOURCES_PROTO
	"src/proto/*.cc"
	)

file(GLOB SOURCES_SOLVERS
	"src/solvers/*.cu"
	"src/solvers/*.cpp"	
	)


file(GLOB INCLUDES_CORE
	"include/core/*.h"	
	)

file(GLOB INCLUDES_GENERATORS
	"include/generators/*.h"	
	)

file(GLOB INCLUDES_INITIALIZERS
	"include/initializers/*.h"	
	)

file(GLOB INCLUDES_NODES
	"include/nodes/*.h"	
	)

file(GLOB INCLUDES_PROTO
	"include/proto/*.h"
	)

file(GLOB INCLUDES_SOLVERS
	"include/solvers/*.h"	
	)



CUDA_ADD_LIBRARY(deepflow SHARED 
	${SOURCES_CORE}
	${SOURCES_GENERATORS}
	${SOURCES_INITIALIZERS}
	${SOURCES_NODES}
	${SOURCES_PROTO}
	${SOURCES_SOLVERS}
	)

source_group("Include Files\\core" FILES ${INCLUDES_CORE})
source_group("Include Files\\generators" FILES ${INCLUDES_GENERATORS})
source_group("Include Files\\initializers" FILES ${INCLUDES_INITIALIZERS})
source_group("Include Files\\nodes" FILES ${INCLUDES_NODES})
source_group("Include Files\\proto" FILES ${INCLUDES_PROTO})
source_group("Include Files\\solvers" FILES ${INCLUDES_SOLVERS})

source_group("Source Files\\core" FILES ${SOURCES_CORE})
source_group("Source Files\\generators" FILES ${SOURCES_GENERATORS})
source_group("Source Files\\initializers" FILES ${SOURCES_INITIALIZERS})
source_group("Source Files\\nodes" FILES ${SOURCES_NODES})
source_group("Source Files\\proto" FILES ${SOURCES_PROTO})
source_group("Source Files\\solvers" FILES ${SOURCES_SOLVERS})

CUDA_ADD_CUBLAS_TO_TARGET(deepflow)

target_link_libraries(deepflow 	shlwapi.lib	gflags_static.lib glog.lib cudnn.lib)
target_link_libraries(deepflow debug libprotobufd.lib optimized libprotobuf.lib)
target_link_libraries(deepflow debug opencv_imgcodecs320d.lib optimized opencv_imgcodecs320.lib)
target_link_libraries(deepflow debug opencv_imgproc320d.lib optimized opencv_imgproc320.lib)
target_link_libraries(deepflow debug opencv_core320d.lib optimized opencv_core320.lib)
target_link_libraries(deepflow debug opencv_highgui320d.lib optimized opencv_highgui320.lib)
