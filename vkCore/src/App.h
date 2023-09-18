#pragma once
#include "GLFWWindow.h"
#include <memory>

struct AppInformation {
    int width;
    int height;
    std::string windowName;
};

class App {
private:
    App();
    ~App();
public:
    static App* appInstance;
    static App* getInstance();
    App(const App&) = delete;
    App& operator=(const App&) = delete;

    bool init(AppInformation& info);
    void loop();
    void destory();
private:
    bool initWindow();
    bool initVkCore();

private:
    AppInformation appInfo;
    std::shared_ptr<GLFWWindow> appWindow;
};
