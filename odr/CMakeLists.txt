cmake_minimum_required(VERSION 2.8.3)
project(odr)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(PCL 1.7 REQUIRED)
find_package(OpenCV REQUIRED)

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
  message("OPENMP FOUND")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# If you intend to use OpenCL, you need to find it
#find_package(OpenCL)
#SET(EXTRA_LIBS ${CMAKE_THREAD_LIBS_INIT} ${OpenCL_LIBRARIES})
#
## Or if you intend to use CUDA, you need it as well as NVVM:
#find_package(CUDA)
#find_package(NVVM) # this FIND script can be found in the ArrayFire CMake example repository
#SET(EXTRA_LIBS ${CMAKE_THREAD_LIBS_INIT} ${CUDA_LIBRARIES} ${NVVM_LIB})

find_package(catkin REQUIRED COMPONENTS
  image_transport
  pcl_conversions
  roscpp
  rospy
  std_msgs
  message_generation
  cv_bridge
)


## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)


## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend tag for "message_generation"
##   * add a build_depend and a run_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependency has been pulled in
##     but can be declared for certainty nonetheless:
##     * add a run_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

# Generate messages in the 'msg' folder
add_message_files(
  FILES
  Proposal.msg
  Feature.msg
  FrameID.msg
)

# Generate services in the 'srv' folder
add_service_files(
  FILES
  Detector.srv
  FeatureExtractor.srv
  Inference.srv
  NormalEstimator.srv
  Visualization.srv
)

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

# Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## To declare and build dynamic reconfigure parameters within this
## package, follow these steps:
## * In the file package.xml:
##   * add a build_depend and a run_depend tag for "dynamic_reconfigure"
## * In this file (CMakeLists.txt):
##   * add "dynamic_reconfigure" to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * uncomment the "generate_dynamic_reconfigure_options" section below
##     and list every .cfg file to be processed

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  LIBRARIES detector
  CATKIN_DEPENDS image_transport roscpp rospy std_msgs cv_bridge pcl_conversions
  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
#  ${ArrayFire_INCLUDE_DIRS}
)

## Includes
set(Sources
    src/utility.cpp
    src/detector.cpp
    src/cbf.cpp
)

#add_library(cbf src/cbf.cpp)
#target_link_libraries(cbf ${OpenCV_LIBS} ${PCL_LIBRARIES}) #${ArrayFire_LIBRARIES}

add_library(detector ${Sources})
target_link_libraries(detector ${OpenCV_LIBS} ${PCL_LIBRARIES}) #${ArrayFire_LIBRARIES}

#ADD_EXECUTABLE(filter src/filter.cpp)
#TARGET_LINK_LIBRARIES(filter ${ArrayFire_LIBRARIES})

add_executable(test_omp src/test_omp.cpp)

add_executable(detector_test src/test.cpp)
target_link_libraries(detector_test detector ${OpenCV_LIBS} ${PCL_LIBRARIES} )

add_executable(detection_server_kinect1 src/detection_server_kinect1.cpp)
target_link_libraries(detection_server_kinect1 detector ${catkin_LIBRARIES})
add_dependencies(detection_server_kinect1 detector_gencpp)

add_executable(detection_server_kinect2 src/detection_server_kinect2.cpp)
target_link_libraries(detection_server_kinect2 detector ${catkin_LIBRARIES})
add_dependencies(detection_server_kinect2 detector_gencpp)

add_executable(normal_estimation_server src/normal_estimation_server.cpp)
target_link_libraries(normal_estimation_server detector ${catkin_LIBRARIES})
add_dependencies(normal_estimation_server detector_gencpp)

add_executable(pointcloud_publisher src/pointcloud_publisher.cpp)
target_link_libraries(pointcloud_publisher detector ${catkin_LIBRARIES})
add_dependencies(pointcloud_publisher detector_gencpp)

add_executable(visualization_server src/visualization_server.cpp)
target_link_libraries(visualization_server detector ${catkin_LIBRARIES})
add_dependencies(visualization_server visualization_server)


## Declare a C++ library
# add_library(detector
#   src/${PROJECT_NAME}/detector.cpp
# )

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(detector ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
# add_executable(detector_node src/detector_node.cpp)

## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(detector_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
# target_link_libraries(detector_node
#   ${catkin_LIBRARIES}
# )

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
# install(TARGETS detector detector_node
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_detector.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
