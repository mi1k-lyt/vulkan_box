#include <vulkan/vulkan.h>

#include <GLFW/glfw3.h>

#include <iostream>

#include "App.h"

int main() {
    AppInformation info = {};
    info.width = 640;
    info.height = 480;
    info.windowName = "Test";
    App::getInstance()->init(info);
    App::getInstance()->destory();
    return 0;
}