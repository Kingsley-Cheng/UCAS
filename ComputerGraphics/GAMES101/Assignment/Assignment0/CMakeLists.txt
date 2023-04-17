cmake_minimum_required(VERSION 3.0.0)
project(homework0 VERSION 0.1.0)

include(CTest)
enable_testing()

add_executable(homework0 main.cpp)
include_directories(/opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3/)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
