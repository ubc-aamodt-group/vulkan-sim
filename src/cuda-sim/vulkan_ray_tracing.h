#ifndef VULKAN_RAY_TRACING_H
#define VULKAN_RAY_TRACING_H

#include "vulkan/vulkan.h"
#include "vulkan/vulkan_intel.h"

#include "vulkan/anv_acceleration_structure.h"
#include "vulkan/anv_public.h"

#include "ptx_ir.h"
//#include "vector-math.h"
#include "../../libcuda/gpgpu_context.h"

#define MAX(a,b) (((a)>(b))?(a):(b))
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MIN_MAX(a,b,c) MAX(MIN((a), (b)), (c))
#define MAX_MIN(a,b,c) MIN(MAX((a), (b)), (c))


class VulkanRayTracing
{
private:
    static VkRayTracingPipelineCreateInfoKHR* pCreateInfos;
    static VkAccelerationStructureGeometryKHR* pGeometries;
    static uint32_t geometryCount;
    static VkAccelerationStructureKHR topLevelAS;

private:
    static bool mt_ray_triangle_test(float3 p0, float3 p1, float3 p2, Ray ray_properties, float* thit);

public:
    static void traceRay( // called by raygen shader
                       VkAccelerationStructureKHR* _topLevelAS,
    				   uint rayFlags,
                       uint cullMask,
                       uint sbtRecordOffset,
                       uint sbtRecordStride,
                       uint missIndex,
                       float3 origin,
                       float Tmin,
                       float3 direction,
                       float Tmax,
                       int payload);

    static void setPipelineInfo(VkRayTracingPipelineCreateInfoKHR* pCreateInfos);
    static void setGeometries(VkAccelerationStructureGeometryKHR* pGeometries, uint32_t geometryCount);
    static void setAccelerationStructure(VkAccelerationStructureKHR accelerationStructure);
    static void invoke_gpgpusim();
    static void registerShaders();
    static void vkCmdTraceRaysKHR( // called by vulkan application
                      const VkStridedDeviceAddressRegionKHR *raygen_sbt,
                      const VkStridedDeviceAddressRegionKHR *miss_sbt,
                      const VkStridedDeviceAddressRegionKHR *hit_sbt,
                      const VkStridedDeviceAddressRegionKHR *callable_sbt,
                      bool is_indirect,
                      uint32_t launch_width,
                      uint32_t launch_height,
                      uint32_t launch_depth,
                      uint64_t launch_size_addr);
};

#endif /* VULKAN_RAY_TRACING_H */
