#pragma once
#include <GLFW/glfw3.h>
#include "WindowUtils.h"


class GLFWWindow {
public:
    GLFWWindow(const int width, const int height, const std::string& windowName);
    ~GLFWWindow();

    bool init();
    void destory();

    GLFWwindow* getWindow() {
        return _window;
    }
    
    WindowUtil::Rect& getWindowRect() {
        return _windowRect;
    }


private:
    void initCallbacks();

private:
    float _width;
    float _height;
    float _xScale;
    std::string _windowName;
    WindowUtil::Rect _windowRect;
    GLFWwindow* _window;
};
