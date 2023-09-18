#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include "GLFWWindow.h"
#include "vulkan/vulkan_core.h"
#include "glm/glm.hpp"



#ifdef DEBUG
const bool ENABLE_VALIDATIONLAYERS = true;
#else 
const bool ENABLE_VALIDATIONLAYERS = false;
#endif

struct UniformBufferObject {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
};

struct QueueFamilyIndices {
    int graphicsFamily = -1;
    int presentFamily = -1;
    bool isComplete() {
        return graphicsFamily >= 0 && presentFamily >= 0;
    }
};

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

struct Vertex {
    glm::vec3 pos;
    glm::vec3 color;
    glm::vec2 texCoord;

    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bindingDescription = {};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions = {};

        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Vertex, pos);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Vertex, color);

        attributeDescriptions[2].binding = 0;
        attributeDescriptions[2].location = 2;
        attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[2].offset = offsetof(Vertex, texCoord);

        return attributeDescriptions;
    }
};

class VkCore {
public:
    VkCore();
    ~VkCore();

    bool init();
    void updateUniformBuffer();
    void drawFrame();
    void destory();

    void setWindow(shared_ptr<GLFWWindow> window);
private:
    // init
    void _createInstance();
    void _setupDebugMessenger();
    void _createSurface();
    void _createSwapChain();
    void _createImageViews();
    void _pickPhysicalDevice();
    void _createLogicalDevice();

    void _createRenderPass();
    void _createDescriptorSetLayout();
    void _createGraphicsPipeline();
    void _createFramebuffers();
    void _createCommandPool();
    void _createDepthResources();
    void _createColorResources();
    void _createTextureImage();
    void _createTextureImageView();
    void _createTextureSampler();
    void _createVertexBuffer();
    void _createIndexBuffer();
    void _createUniformBuffer();
    void _createDescriptorPool();
    void _createDescriptorSet();
    void _createCommandBuffers();
    void _createSemaphores();

    // common
    QueueFamilyIndices _findQueueFamilies(VkPhysicalDevice device);
    std::vector<const char*> _getRequiredExtensions();
    void _populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo);
    void _cleanupSwapChain();

    // depth stencil
    VkFormat _findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling,
                                  VkFormatFeatureFlags features);
    VkFormat _findDepthFormat();
    bool _hasStencilComponent(VkFormat format);

    // check
    bool _isDeviceSuitable(VkPhysicalDevice device);
    bool _checkValidationLayerSupport();
    bool _checkDeviceExtensionSupport(VkPhysicalDevice device);
    VkSampleCountFlagBits _getMaxUsableSampleCount();

    // swapchain
    SwapChainSupportDetails _querySwapChainSupport(VkPhysicalDevice device);
    VkSurfaceFormatKHR _chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);
    VkPresentModeKHR _chooseSwapPresentMode(const std::vector<VkPresentModeKHR> availablePresentModes);
    VkExtent2D _chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);

    // GraphicsPipeline
    VkShaderModule _createShaderModule(const std::vector<char>& code);
    VkCommandBuffer _beginSingleTimeCommands();
    void _endSingleTimeCommands(VkCommandBuffer commandBuffer);
    void _transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout);
    void _copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);
    // vertex input
    void _createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer,
                       VkDeviceMemory& bufferMemory);
    void _copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
    uint32_t _findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);

    // update
    void _recreateSwapChain();

    void _createImage(uint32_t width, uint32_t height, VkSampleCountFlagBits numSamples, VkFormat format,
                      VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image,
                      VkDeviceMemory& imageMemory);
    VkImageView _createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags);

    // readpiexl
    void _checkBlitSupport();
    void _saveScreenshot(const char* filename, uint32_t imageIndex);

private:
    shared_ptr<GLFWWindow> window_;

    VkInstance vkInstance_;
    VkDevice device_;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkSurfaceKHR surface_;
    VkQueue graphicsQueue_;
    VkQueue presentQueue_;

    VkSwapchainKHR swapChain_;
    VkFormat swapChainImageFormat_;
    VkExtent2D swapChainExtent_;
    std::vector<VkImage> swapChainImages_;
    std::vector<VkImageView> swapChainImageViews_;
    std::vector<VkFramebuffer> swapChainFramebuffers_;

    VkImage offImage_;
    VkDeviceMemory offImageMemory_;
    VkImageView offImageView_;
    VkFramebuffer offFramebuffer_;

    VkRenderPass renderPass_;

    VkPipeline graphicsPipeline_;
    VkShaderModule vertShaderModule_;
    VkShaderModule fragShaderModule_;
    VkPipelineLayout pipelineLayout_;

    VkCommandPool commandPool_;
    std::vector<VkCommandBuffer> commandBuffers_;

    VkSemaphore imageAvailableSemaphore_;
    VkSemaphore renderFinishedSemaphore_;

    VkDescriptorSetLayout descriptorSetLayout_;
    VkDescriptorPool descriptorPool_;
    VkDescriptorSet descriptorSet_;

    VkBuffer vertexBuffer_;
    VkDeviceMemory vertexBufferMemory_;
    VkBuffer indexBuffer_;
    VkDeviceMemory indexBufferMemory_;
    VkBuffer uniformBuffer_;
    VkDeviceMemory uniformBufferMemory_;

    VkSampler textureSampler_;
    VkImage textureImage_;
    VkDeviceMemory textureImageMemory_;
    VkImageView textureImageView_;

    VkImage depthImage_;
    VkDeviceMemory depthImageMemory_;
    VkImageView depthImageView_;

    VkImage colorImage_;
    VkDeviceMemory colorImageMemory_;
    VkImageView colorImageView_;

    VkSampleCountFlagBits msaaSamples_ = VK_SAMPLE_COUNT_1_BIT;

    VkDebugUtilsMessengerEXT debugMessenger_;

    // readpiexl
    bool supportsBlit_ = false;
};