cmake_minimum_required(VERSION 3.0.0)
project(tinyRenderer VERSION 0.1.0)

include(CTest)
enable_testing()

add_executable(tinyRenderer main.cpp tgaimage.cpp tgaimage.h model.h model.cpp
        geometry.h geometry.cpp rasterizer.cpp rasterizer.h)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
