#include "App.h"

int main() {
    AppInformation info = {};
    info.width = 640;
    info.height = 480;
    info.windowName = "Vulkan_Box";
    App::getInstance()->init(info);
    App::getInstance()->loop();
    App::getInstance()->destory();
    return 0;
}