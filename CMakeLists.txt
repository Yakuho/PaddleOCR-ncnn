cmake_minimum_required(VERSION 3.14)
project(ppocr)

find_package(yaml-cpp REQUIRED)
find_package(OpenCV   REQUIRED)
find_package(gflags   REQUIRED)
find_package(ncnn     REQUIRED)
find_package(glog     REQUIRED)

include(FetchContent)
include(external-cmake/auto-log.cmake)
include_directories(${FETCHCONTENT_BASE_DIR}/extern_autolog-src)

file(GLOB_RECURSE PPOCR_SOURCES ${CMAKE_SOURCE_DIR}/src/*.cpp)

add_library(${PROJECT_NAME} SHARED ${PPOCR_SOURCES})

target_link_libraries(${PROJECT_NAME}
    PUBLIC
    yaml-cpp::yaml-cpp
    ${OpenCV_LIBS}
    glog::glog
    gflags
    ncnn
)

target_include_directories(${PROJECT_NAME}
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)
