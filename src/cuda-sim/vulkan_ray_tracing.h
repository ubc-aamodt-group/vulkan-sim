#ifndef VULKAN_RAY_TRACING_H
#define VULKAN_RAY_TRACING_H

#include "vulkan/vulkan.h"
#include "vulkan/vulkan_intel.h"

#include "vulkan/anv_acceleration_structure.h"
#include "intersection_table.h"
#include "compiler/spirv/spirv.h"

// #include "ptx_ir.h"
#include "ptx_ir.h"
#include "../../libcuda/gpgpu_context.h"
#include "compiler/shader_enums.h"
#include <fstream>
#include <cmath>

#define MAX(a,b) (((a)>(b))?(a):(b))
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MIN_MAX(a,b,c) MAX(MIN((a), (b)), (c))
#define MAX_MIN(a,b,c) MIN(MAX((a), (b)), (c))

// typedef struct float4 {
//     float x, y, z, w;
// } float4;

typedef struct float4x4 {
  float m[4][4];

  float4 operator*(const float4& _vec) const
  {
    float vec[] = {_vec.x, _vec.y, _vec.z, _vec.w};
    float res[] = {0, 0, 0, 0};
    for(int i = 0; i < 4; i++)
        for(int j = 0; j < 4; j++)
            res[i] += this->m[j][i] * vec[j];
    return {res[0], res[1], res[2], res[3]};
  }
} float4x4;

typedef struct RayDebugGPUData
{
    bool valid;
    int launchIDx;
    int launchIDy;
    int instanceCustomIndex;
    int primitiveID;
    float3 v0pos;
    float3 v1pos;
    float3 v2pos;
    float3 attribs;
    float3 P_object;
    float3 P; //world intersection point
    float3 N_object;
    float3 N;
    float NdotL;
    float3 hitValue;
} RayDebugGPUData;

// float4 operator*(const float4& _vec, const float4x4& matrix)
// {
//     float vec[] = {_vec.x, _vec.y, _vec.z, _vec.w};
//     float res[] = {0, 0, 0, 0};
//     for(int i = 0; i < 4; i++)
//         for(int j = 0; j < 4; j++)
//             res[i] += matrix.m[j][i] * vec[j];
//     return {res[0], res[1], res[2], res[3]};
// }

enum class TransactionType {
    BVH_STRUCTURE,
    BVH_INTERNAL_NODE,
    BVH_INSTANCE_LEAF,
    BVH_PRIMITIVE_LEAF_DESCRIPTOR,
    BVH_QUAD_LEAF,
    BVH_PROCEDURAL_LEAF,
    Intersection_Table_Write,
    Intersection_Table_Load,
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
    VkGeometryTypeKHR geometryType;
    float world_min_thit;
    uint32_t geometry_index;
    uint32_t primitive_index;
    float3 intersection_point;
    float3 barycentric_coordinates;
    int32_t hitGroupIndex; // Shader ID of the closest hit for procedural geometries

    uint32_t instance_index;
    float4x4 worldToObjectMatrix;
    float4x4 objectToWorldMatrix;
} Hit_data;

typedef struct shader_stage_info {
    uint32_t ID;
    gl_shader_stage type;
    char* function_name;
} shader_stage_info;

typedef struct Traversal_data {
    bool hit_geometry;
    Hit_data closest_hit;
    float3 ray_world_direction;
    float3 ray_world_origin;
    float Tmin;
    float Tmax;
    int32_t current_shader_counter; // set to shader_counter in call_intersection and -1 in call_miss and call_closest_hit

    uint32_t rayFlags;
    uint32_t cullMask;
    uint32_t sbtRecordOffset;
    uint32_t sbtRecordStride;
    uint32_t missIndex;
} Traversal_data;


typedef struct Vulkan_RT_thread_data {
    std::vector<variable_decleration_entry> variable_decleration_table;

    std::vector<Traversal_data> traversal_data;


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
        entry.address = (uint64_t) malloc(size);
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
} Vulkan_RT_thread_data;

struct anv_descriptor_set;
struct anv_descriptor;

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
    static struct anv_descriptor_set *descriptorSet;
public:
    // static RayDebugGPUData rayDebugGPUData[2000][2000];
    static warp_intersection_table intersection_table[120][2160];

private:
    static bool mt_ray_triangle_test(float3 p0, float3 p1, float3 p2, Ray ray_properties, float* thit);
    static float3 Barycentric(float3 p, float3 a, float3 b, float3 c);
    static std::vector<shader_stage_info> shaders;


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
                       const ptx_instruction *pI,
                       ptx_thread_info *thread);
    static void endTraceRay(const ptx_instruction *pI, ptx_thread_info *thread);
    
    static void load_descriptor(const ptx_instruction *pI, ptx_thread_info *thread);

    static void setPipelineInfo(VkRayTracingPipelineCreateInfoKHR* pCreateInfos);
    static void setGeometries(VkAccelerationStructureGeometryKHR* pGeometries, uint32_t geometryCount);
    static void setAccelerationStructure(VkAccelerationStructureKHR accelerationStructure);
    static void setDescriptorSet(struct anv_descriptor_set *set);
    static void invoke_gpgpusim();
    static uint32_t registerShaders(char * shaderPath, gl_shader_stage shaderType);
    static void vkCmdTraceRaysKHR( // called by vulkan application
                      void *raygen_sbt,
                      void *miss_sbt,
                      void *hit_sbt,
                      void *callable_sbt,
                      bool is_indirect,
                      uint32_t launch_width,
                      uint32_t launch_height,
                      uint32_t launch_depth,
                      uint64_t launch_size_addr);
    static void callShader(const ptx_instruction *pI, ptx_thread_info *thread, function_info *target_func);
    static void callMissShader(const ptx_instruction *pI, ptx_thread_info *thread);
    static void callClosestHitShader(const ptx_instruction *pI, ptx_thread_info *thread);
    static void callIntersectionShader(const ptx_instruction *pI, ptx_thread_info *thread, uint32_t shader_counter);
    static void callAnyHitShader(const ptx_instruction *pI, ptx_thread_info *thread);
    static void setDescriptor(uint32_t setID, uint32_t descID, void *address, uint32_t size, VkDescriptorType type);
    static void* getDescriptorAddress(uint32_t setID, uint32_t binding);

    static void image_store(struct anv_descriptor* desc, uint32_t gl_LaunchIDEXT_X, uint32_t gl_LaunchIDEXT_Y, uint32_t gl_LaunchIDEXT_Z, uint32_t gl_LaunchIDEXT_W, 
              float hitValue_X, float hitValue_Y, float hitValue_Z, float hitValue_W, const ptx_instruction *pI, ptx_thread_info *thread);
    static void getTexture(struct anv_descriptor *desc, float x, float y, float lod, float &c0, float &c1, float &c2, float &c3);
    static void image_load(struct anv_descriptor *desc, uint32_t x, uint32_t y, float &c0, float &c1, float &c2, float &c3);
};

#endif /* VULKAN_RAY_TRACING_H */
