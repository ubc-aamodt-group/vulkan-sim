#ifndef VULKAN_RAY_TRACING_H
#define VULKAN_RAY_TRACING_H

#include "vulkan/vulkan.h"
#include "vulkan/vulkan_intel.h"

#include "vulkan/anv_acceleration_structure.h"
#include "vulkan/anv_public.h"

// #define HAVE_PTHREAD
// #define UTIL_ARCH_LITTLE_ENDIAN 1
// #define UTIL_ARCH_BIG_ENDIAN 0
// #include "util/u_endian.h"
// #include "vulkan/anv_private.h"
// #include "vk_object.h"

#include "ptx_ir.h"
//#include "vector-math.h"
#include "../../libcuda/gpgpu_context.h"
#include <fstream>

#define MAX(a,b) (((a)>(b))?(a):(b))
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MIN_MAX(a,b,c) MAX(MIN((a), (b)), (c))
#define MAX_MIN(a,b,c) MIN(MAX((a), (b)), (c))

enum class TransactionType {
    BVH_STRUCTURE,
    BVH_INTERNAL_NODE,
    BVH_INSTANCE_LEAF,
    BVH_PRIMITIVE_LEAF_DESCRIPTOR,
    BVH_QUAD_LEAF,
    BVH_PROCEDURAL_LEAF,
};

typedef struct MemoryTransactionRecord {
    MemoryTransactionRecord(void* address, uint32_t size, TransactionType type)
    : address(address), size(size), type(type) {}
    void* address;
    uint32_t size;
    TransactionType type;
} MemoryTransactionRecord;

typedef struct Descriptor
{
    uint32_t setID;
    uint32_t descID;
    void *address;
    uint32_t size;
    VkDescriptorType type;
} Descriptor;

typedef struct variable_decleration_entry{
  uint64_t type;
  std::string name;
  uint64_t address;
  uint32_t size;
} variable_decleration_entry;

typedef struct Hit_data{
    uint32_t geometry_id;
    float3 intersection_point;
    float3 barycentric_coordinates;
} Hit_data;


struct Vulkan_RT_thread_data{
    std::vector<variable_decleration_entry> variable_decleration_table;
    bool hit_geometry;
    Hit_data closest_hit;

    variable_decleration_entry* get_variable_decleration_entry(uint64_t type, std::string name, uint32_t size) {
        if(type == 8192)
            return get_hitAttribute();
        
        for (int i = 0; i < variable_decleration_table.size(); i++) {
            if (variable_decleration_table[i].name == name) {
                assert (variable_decleration_table[i].address != NULL);
                return &(variable_decleration_table[i]);
            }
        }
        return NULL;
    }

    uint64_t add_variable_decleration_entry(uint64_t type, std::string name, uint32_t size) {
        variable_decleration_entry entry;
        entry.type = type;
        entry.name = name;
        entry.address = (uint64_t) malloc(size);;
        entry.size = size;
        variable_decleration_table.push_back(entry);

        return entry.address;
    }

    variable_decleration_entry* get_hitAttribute() {
        variable_decleration_entry* hitAttribute = NULL;
        for (int i = 0; i < variable_decleration_table.size(); i++) {
            if (variable_decleration_table[i].type == 8192) {
                assert (variable_decleration_table[i].address != NULL);
                assert (hitAttribute == NULL); // There should be only 1 hitAttribute
                hitAttribute = &(variable_decleration_table[i]);
            }
        }
        return hitAttribute;
    }

    void set_hitAttribute(float3 barycentric) {
        variable_decleration_entry* hitAttribute = get_hitAttribute();
        float* address;
        if(hitAttribute == NULL) {
            address = (float*)add_variable_decleration_entry(8192, "attribs", 12);
        }
        else {
            assert (hitAttribute->type == 8192);
            assert (hitAttribute->address != NULL);
            // hitAttribute->name = name;
            address = (float*)(hitAttribute->address);
        }
        address[0] = barycentric.x;
        address[1] = barycentric.y;
        address[2] = barycentric.z;
    }
};

class VulkanRayTracing
{
private:
    static VkRayTracingPipelineCreateInfoKHR* pCreateInfos;
    static VkAccelerationStructureGeometryKHR* pGeometries;
    static uint32_t geometryCount;
    static VkAccelerationStructureKHR topLevelAS;
    static std::vector<std::vector<Descriptor> > descriptors;
    static std::ofstream imageFile;
    static bool firstTime;

private:
    static bool mt_ray_triangle_test(float3 p0, float3 p1, float3 p2, Ray ray_properties, float* thit);
    static float3 Barycentric(float3 p, float3 a, float3 b, float3 c);


public:
    static void traceRay( // called by raygen shader
                       VkAccelerationStructureKHR _topLevelAS,
    				   uint rayFlags,
                       uint cullMask,
                       uint sbtRecordOffset,
                       uint sbtRecordStride,
                       uint missIndex,
                       float3 origin,
                       float Tmin,
                       float3 direction,
                       float Tmax,
                       int payload,
                       uint32_t *hit_geometry,
                       const ptx_instruction *pI,
                       ptx_thread_info *thread);
    
    static void load_descriptor(const ptx_instruction *pI, ptx_thread_info *thread);

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
    static void callShader(const ptx_instruction *pI, ptx_thread_info *thread, function_info *target_func);
    static void callMissShader(const ptx_instruction *pI, ptx_thread_info *thread);
    static void callClosestHitShader(const ptx_instruction *pI, ptx_thread_info *thread);
    static void callIntersectionShader(const ptx_instruction *pI, ptx_thread_info *thread);
    static void callAnyHitShader(const ptx_instruction *pI, ptx_thread_info *thread);
    static void setDescriptor(uint32_t setID, uint32_t descID, void *address, uint32_t size, VkDescriptorType type);
    static void* getDescriptorAddress(uint32_t setID, uint32_t descID);

    static void image_store(void* image, uint32_t gl_LaunchIDEXT_X, uint32_t gl_LaunchIDEXT_Y, uint32_t gl_LaunchIDEXT_Z, uint32_t gl_LaunchIDEXT_W, 
              float hitValue_X, float hitValue_Y, float hitValue_Z, float hitValue_W, const ptx_instruction *pI, ptx_thread_info *thread);
};

#endif /* VULKAN_RAY_TRACING_H */
