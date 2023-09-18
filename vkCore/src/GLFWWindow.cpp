#include "GLFWWindow.h"
#include "GLFW/glfw3.h"
#include "WindowUtils.h"
#include "spdlog/spdlog.h"

static void glfwErrowCallback(int error, const char* description) {
    spdlog::error("glfw error {0}: {1}", error, description);
}

GLFWWindow::GLFWWindow(const int width, const int height, const std::string& windowName)
    : _width(width)
    , _height(height)
    , _windowName(windowName)
    , _windowRect(WindowUtil::Rect(0, 0, width, height))
    , _window(nullptr) {}

GLFWWindow::~GLFWWindow() {
    destory();
}

bool GLFWWindow::init() {
    spdlog::info("Init GLFWWindow...");
    
    glfwSetErrorCallback(glfwErrowCallback);
    if(!glfwInit()) {
        spdlog::error("glfw init failed!");
        return false;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_SAMPLES, 4);

    _window = glfwCreateWindow(_width, _height, _windowName.c_str(), nullptr, nullptr);
    if(_window == nullptr) {
        spdlog::error("glfw create failed!");
    }

    initCallbacks();

    return true;
}

void GLFWWindow::destory() {
    spdlog::info("destory GLFWWindow...");
    glfwDestroyWindow(_window);
    glfwTerminate();
}

void GLFWWindow::initCallbacks() {
    glfwSetFramebufferSizeCallback(_window, [](GLFWwindow* window, int w, int h)->void{
        auto glfwWindow = static_cast<GLFWWindow*>(glfwGetWindowUserPointer(window));
        glfwWindow->getWindowRect() = WindowUtil::Rect(0, 0, w, h);
    });

    glfwSetMouseButtonCallback(_window, [](GLFWwindow* window, int button, int action, int modify)->void{

    });
}