# CMake 工程，读者应该熟悉了，我就不一一注释了
cmake_minimum_required( VERSION 2.8 )
project( test_project )

set( CMAKE_BUILD_TYPE Release )
set( CMAKE_CXX_FLAGS "-std=c++11 -Wall -O2 -march=native" )

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

find_package( OpenCV REQUIRED )
find_package( Ceres REQUIRED )
find_package( Eigen3 REQUIRED )
find_package( Cholmod )
find_package( G2O REQUIRED )

include_directories( 
${EIGEN3_INCLUDE_DIR} 
${CHOLMOD_INCLUDE_DIR}
${CERES_INCLUDE_DIRS} 
)

add_executable( test test.cpp )
target_link_libraries( test
    ${OpenCV_LIBS}
    g2o_core g2o_types_slam3d g2o_solver_csparse g2o_stuff g2o_csparse_extension g2o_types_sba
    ${CHOLMOD_LIBRARIES}
    ${CERES_LIBRARIES}
    )
