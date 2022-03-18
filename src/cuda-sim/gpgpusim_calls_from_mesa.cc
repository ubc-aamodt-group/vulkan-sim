#ifndef GPGPUSIM_CALLS_FROM_MESA_CC
#define GPGPUSIM_CALLS_FROM_MESA_CC

#include "vulkan_ray_tracing.h"
// #include "vulkan/anv_private.h"

extern "C" void gpgpusim_setPipelineInfo(VkRayTracingPipelineCreateInfoKHR* pCreateInfos)
{
    VulkanRayTracing::setPipelineInfo(pCreateInfos);
}

extern "C" void gpgpusim_setGeometries(VkAccelerationStructureGeometryKHR* pGeometries, uint32_t geometryCount)
{
    VulkanRayTracing::setGeometries(pGeometries, geometryCount);
}

extern "C" void gpgpusim_addTreelets(VkAccelerationStructureKHR accelerationStructure)
{
    VulkanRayTracing::setAccelerationStructure(accelerationStructure);
}

extern "C" void gpgpusim_testTraversal(struct anv_bvh_node* root)
{
    //VulkanRayTracing::
}

extern "C" uint32_t gpgpusim_registerShader(char * shaderPath, uint32_t shader_type)
{
    return VulkanRayTracing::registerShaders(shaderPath, gl_shader_stage(shader_type));
}

extern "C" void gpgpusim_vkCmdTraceRaysKHR(
                      void *raygen_sbt,
                      void *miss_sbt,
                      void *hit_sbt,
                      void *callable_sbt,
                      bool is_indirect,
                      uint32_t launch_width,
                      uint32_t launch_height,
                      uint32_t launch_depth,
                      uint64_t launch_size_addr)
{
    VulkanRayTracing::invoke_gpgpusim();
    VulkanRayTracing::vkCmdTraceRaysKHR(raygen_sbt, miss_sbt, hit_sbt, callable_sbt,
            is_indirect, launch_width, launch_height, launch_depth, launch_size_addr);
}

extern "C" void gpgpusim_setDescriptor(uint32_t setID, uint32_t descID, void *address, uint32_t size, VkDescriptorType type)
{
    VulkanRayTracing::setDescriptor(setID, descID, address, size, type);
}


// CPP externs
extern void gpgpusim_addTreelets_cpp(VkAccelerationStructureKHR accelerationStructure)
{
    VulkanRayTracing::setAccelerationStructure(accelerationStructure);
}

extern "C" void gpgpusim_setDescriptorSet(struct anv_descriptor_set *set)
{
    VulkanRayTracing::setDescriptorSet(set);
}



// CPP externs
extern uint32_t gpgpusim_registerShader_cpp(char * shaderPath, uint32_t shader_type)
{
    return VulkanRayTracing::registerShaders(shaderPath, gl_shader_stage(shader_type));
}

extern void gpgpusim_vkCmdTraceRaysKHR_cpp(
                      void *raygen_sbt,
                      void *miss_sbt,
                      void *hit_sbt,
                      void *callable_sbt,
                      bool is_indirect,
                      uint32_t launch_width,
                      uint32_t launch_height,
                      uint32_t launch_depth,
                      uint64_t launch_size_addr)
{
    VulkanRayTracing::invoke_gpgpusim();
    VulkanRayTracing::vkCmdTraceRaysKHR(raygen_sbt, miss_sbt, hit_sbt, callable_sbt,
            is_indirect, launch_width, launch_height, launch_depth, launch_size_addr);
}

extern void gpgpusim_setDescriptorSet_cpp(void *set)
{
    VulkanRayTracing::setDescriptorSet((struct anv_descriptor_set*) set);
}

extern void gpgpusim_setDescriptorSetFromLauncher_cpp(void *address, uint32_t setID, uint32_t descID)
{
    VulkanRayTracing::setDescriptorSetFromLauncher(address, setID, descID);
}

extern void gpgpusim_setStorageImageFromLauncher_cpp(void *address, 
                                                    uint32_t setID, 
                                                    uint32_t descID, 
                                                    uint32_t width,
                                                    uint32_t height,
                                                    VkFormat format,
                                                    uint32_t VkDescriptorTypeNum,
                                                    uint32_t n_planes,
                                                    uint32_t n_samples,
                                                    VkImageTiling tiling,
                                                    uint32_t isl_tiling_mode, 
                                                    uint32_t row_pitch_B)
{
    VulkanRayTracing::setStorageImageFromLauncher(address, setID, descID, width, height, format, VkDescriptorTypeNum, n_planes, n_samples, tiling, isl_tiling_mode, row_pitch_B);
}

extern void gpgpusim_setTextureFromLauncher_cpp(void *address, 
                                                uint32_t setID, 
                                                uint32_t descID, 
                                                uint64_t size,
                                                uint32_t width,
                                                uint32_t height,
                                                VkFormat format,
                                                uint32_t VkDescriptorTypeNum,
                                                uint32_t n_planes,
                                                uint32_t n_samples,
                                                VkImageTiling tiling,
                                                uint32_t isl_tiling_mode,
                                                uint32_t row_pitch_B)
{
    VulkanRayTracing::setTextureFromLauncher(address, setID, descID, size,width, height, format, VkDescriptorTypeNum, n_planes, n_samples, tiling, isl_tiling_mode, row_pitch_B);
}

extern "C" void gpgpusim_pass_child_addr(void *address)
{
    VulkanRayTracing::pass_child_addr(address);
}

#endif /* GPGPUSIM_CALLS_FROM_MESA_CC */
