#include "App.h"
#include "VkCore.h"
#include <memory>


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
    if(initWindow() && initVkCore()) {
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

bool App::initVkCore() {
    appCore = std::make_shared<VkCore>();
    if(appWindow) {
        appCore->setWindow(appWindow);
    }
    if(!appCore->init()) {
        return false;
    }

    return true;
}

void App::loop() {
    while (!glfwWindowShouldClose(appWindow->getWindow())) {
        glfwPollEvents();
        // glfwSwapBuffers(window_->getWindow());
        appCore->updateUniformBuffer();
        appCore->drawFrame();
    }
}


