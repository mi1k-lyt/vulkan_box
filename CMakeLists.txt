cmake_minimum_required(VERSION 3.22.1)
project(vulkan_box)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O2")

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)

set(VENDOR_PATH ${CMAKE_CURRENT_SOURCE_DIR}/vkCore/vendor)
set(SRC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(VKCORE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/vkCore)

add_subdirectory(${VKCORE_PATH})

link_directories(
    ${LIBRARY_OUTPUT_PATH}
    ${VKCORE_PATH}/bin
)

include_directories(
    ${VENDOR_PATH}/glfw/include
    ${VENDOR_PATH}/glm
    ${VKCORE_PATH}/src
)

file(GLOB_RECURSE FILES 
    ${SRC_PATH}/*.cpp
    ${SRC_PATH}/*.h
)

add_executable(vulkan_box
    ${FILES}
)

target_link_libraries(vulkan_box
    vkCore
)