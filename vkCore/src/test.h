#include "App.h"
#include <memory>

int t() {
    spdlog::info("App test");
    AppInformation info = {};
    info.width = 640;
    info.height = 480;
    info.windowName = "Test";
    App::getInstance()->init(info);
    //window->destory();
    return 0;
}