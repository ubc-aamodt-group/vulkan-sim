#ifndef GPGPUSIM_CALLS_FROM_MESA_CC
#define GPGPUSIM_CALLS_FROM_MESA_CC

#include "vulkan_ray_tracing.h"

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

extern "C" void gpgpusim_vkCmdTraceRaysKHR(
                      const VkStridedDeviceAddressRegionKHR *raygen_sbt,
                      const VkStridedDeviceAddressRegionKHR *miss_sbt,
                      const VkStridedDeviceAddressRegionKHR *hit_sbt,
                      const VkStridedDeviceAddressRegionKHR *callable_sbt,
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

extern "C" void gpgpusim_setDescriptorSet(uint32_t setID, uint32_t descID, void *address, uint32_t size, VkDescriptorType type)
{
    VulkanRayTracing::setDescriptor(setID, descID, address, size, type);
}

#endif /* GPGPUSIM_CALLS_FROM_MESA_CC */
