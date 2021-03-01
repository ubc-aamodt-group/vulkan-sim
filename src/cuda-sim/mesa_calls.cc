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

extern "C" void gpgpusim_testTraversal(struct anv_bvh_node* root)
{
    //VulkanRayTracing::
}

#endif /* MESA_CALLS_H */
