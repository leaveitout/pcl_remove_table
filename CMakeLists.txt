cmake_minimum_required(VERSION 3.0)
project(pcl_remove_table)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/bin)

find_package(PCL 1.7 REQUIRED)
#find_package(OpenCV REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

include_directories(${CMAKE_HOME_DIRECTORY}/include)

set(SOURCE_FILES src/main.cpp src/Logger.cpp)

add_executable(pcl_remove_table ${SOURCE_FILES})

target_link_libraries(pcl_remove_table ${PCL_LIBRARIES})
#target_link_libraries(pcl_base_project ${OpenCV_LIBS})

