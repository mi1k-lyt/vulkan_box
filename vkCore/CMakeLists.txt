cmake_minimum_required(VERSION 3.22.1)
project(vkCore)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O2")

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)

set(VENDOR_PATH ${CMAKE_CURRENT_SOURCE_DIR}/vendor)
set(SRC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_subdirectory(${VENDOR_PATH}/glfw)
add_subdirectory(${VENDOR_PATH}/spdlog)

if(WIN32)
    set(LIBS)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
elseif(UNIX AND NOT APPLE)
    set(LIBS vulkan dl pthread X11 Xxf86vm Xrandr Xi)
endif()

file(GLOB_RECURSE FILES 
    ${SRC_PATH}/*.cpp
    ${SRC_PATH}/*.h
)


include_directories(
    ${VENDOR_PATH}/glfw/include
    ${VENDOR_PATH}/spdlog/include
    ${VENDOR_PATH}/glm
)

add_library(vkCore 
    ${FILES}
)

link_directories(
    ${LIBRARY_OUTPUT_PATH}
)

add_dependencies(vkCore glfw)
add_dependencies(vkCore spdlog)

target_link_libraries(vkCore
    ${LIBRARY_OUTPUT_PATH}/libglfw3.a
    spdlog::spdlog $<$<BOOL:${MINGW}>:ws2_32>
    ${LIBS}
)