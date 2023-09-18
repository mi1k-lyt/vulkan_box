#include "App.h"


App* App::appInstance = nullptr;

App::App() : appWindow(nullptr) {}
App::~App() {}

App* App::getInstance() {
    if(appInstance == nullptr) {
        appInstance = new App();
    }

    return appInstance;
}

bool App::init(AppInformation& info) {
    spdlog::info("Init App...");
    appInfo = info;
    if(initWindow()) {
        spdlog::info("Init App Successful!");
        return true;
    }
    spdlog::error("Init App Failed");
    return false;
}

void App::destory() {
    delete appInstance;
}

bool App::initWindow() {
    appWindow = std::make_shared<GLFWWindow>(appInfo.width, appInfo.height, appInfo.windowName);
    if(!appWindow->init()) {
        return false;
    }
    return true;
}
