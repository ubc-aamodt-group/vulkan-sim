#ifndef MESA_CALLS_H
#define MESA_CALLS_H

#include "vulkan_ray_tracing.h"

extern "C" void gpgpusim_setPipelineInfo(VkRayTracingPipelineCreateInfoKHR* pCreateInfos)
{
    VulkanRayTracing::setPipelineInfo(pCreateInfos);
}

extern "C" void gpgpusim_setGeometries(VkAccelerationStructureGeometryKHR* pGeometries, uint32_t geometryCount)
{
    VulkanRayTracing::setGeometries(pGeometries, geometryCount);
}

extern "C" void gpgpusim_setAccelerationStructure(VkAccelerationStructureKHR accelerationStructure)
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

#endif /* MESA_CALLS_H */
