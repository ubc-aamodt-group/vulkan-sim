#include "vulkan_ray_tracing.h"

#include "vector-math.h"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#define BOOST_FILESYSTEM_VERSION 3
#define BOOST_FILESYSTEM_NO_DEPRECATED 
#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

#define __CUDA_RUNTIME_API_H__
// clang-format off
#include "host_defines.h"
#include "builtin_types.h"
#include "driver_types.h"
#include "../../libcuda/cuda_api.h"
#include "cudaProfiler.h"
// clang-format on
#if (CUDART_VERSION < 8000)
#include "__cudaFatFormat.h"
#endif

#include "../../libcuda/gpgpu_context.h"
#include "../../libcuda/cuda_api_object.h"
#include "../gpgpu-sim/gpu-sim.h"
#include "../cuda-sim/ptx_loader.h"
#include "../cuda-sim/cuda-sim.h"
#include "../cuda-sim/ptx_ir.h"
#include "../cuda-sim/ptx_parser.h"
#include "../gpgpusim_entrypoint.h"
#include "../stream_manager.h"
#include "../abstract_hardware_model.h"
#include "vulkan_acceleration_structure_util.h"

VkRayTracingPipelineCreateInfoKHR* VulkanRayTracing::pCreateInfos = NULL;
VkAccelerationStructureGeometryKHR* VulkanRayTracing::pGeometries = NULL;
uint32_t VulkanRayTracing::geometryCount = 0;
VkAccelerationStructureKHR VulkanRayTracing::topLevelAS = NULL;

float magic_max7(float a0, float a1, float b0, float b1, float c0, float c1, float d)
{
	float t1 = MIN_MAX(a0, a1, d);
	float t2 = MIN_MAX(b0, b1, t1);
	float t3 = MIN_MAX(c0, c1, t2);
	return t3;
}

float magic_min7(float a0, float a1, float b0, float b1, float c0, float c1, float d)
{
	float t1 = MAX_MIN(a0, a1, d);
	float t2 = MAX_MIN(b0, b1, t1);
	float t3 = MAX_MIN(c0, c1, t2);
	return t3;
}

float3 get_t_bound(float3 box, float3 origin, float3 idirection)
{
    // // Avoid div by zero, returns 1/2^80, an extremely small number
    // const float ooeps = exp2f(-80.0f);

    // // Calculate inverse direction
    // float3 idir;
    // idir.x = 1.0f / (fabsf(direction.x) > ooeps ? direction.x : copysignf(ooeps, direction.x));
    // idir.y = 1.0f / (fabsf(direction.y) > ooeps ? direction.y : copysignf(ooeps, direction.y));
    // idir.z = 1.0f / (fabsf(direction.z) > ooeps ? direction.z : copysignf(ooeps, direction.z));

    // Calculate bounds
    float3 result;
    result.x = (box.x - origin.x) * idirection.x;
    result.y = (box.y - origin.y) * idirection.y;
    result.z = (box.z - origin.z) * idirection.z;

    // Return
    return result;
}

float3 calculate_idir(float3 direction) {
    // Avoid div by zero, returns 1/2^80, an extremely small number
    const float ooeps = exp2f(-80.0f);

    // Calculate inverse direction
    float3 idir;
    idir.x = 1.0f / (fabsf(direction.x) > ooeps ? direction.x : copysignf(ooeps, direction.x));
    idir.y = 1.0f / (fabsf(direction.y) > ooeps ? direction.y : copysignf(ooeps, direction.y));
    idir.z = 1.0f / (fabsf(direction.z) > ooeps ? direction.z : copysignf(ooeps, direction.z));

    return idir;
}

bool ray_box_test(float3 low, float3 high, float3 idirection, float3 origin, float tmin, float tmax, float& thit)
{
	// const float3 lo = Low * InvDir - Ood;
	// const float3 hi = High * InvDir - Ood;
    float3 lo = get_t_bound(low, origin, idirection);
    float3 hi = get_t_bound(high, origin, idirection);

    // QUESTION: max value does not match rtao benchmark, rtao benchmark converts float to int with __float_as_int
    // i.e. __float_as_int: -110.704826 => -1025677090, -24.690834 => -1044019502

	// const float slabMin = tMinFermi(lo.x, hi.x, lo.y, hi.y, lo.z, hi.z, TMin);
	// const float slabMax = tMaxFermi(lo.x, hi.x, lo.y, hi.y, lo.z, hi.z, TMax);
    float min = magic_max7(lo.x, hi.x, lo.y, hi.y, lo.z, hi.z, tmin);
    float max = magic_min7(lo.x, hi.x, lo.y, hi.y, lo.z, hi.z, tmax);

	// OutIntersectionDist = slabMin;
    thit = min;

	// return slabMin <= slabMax;
    return (min <= max);
}

void VulkanRayTracing::traceRay(VkAccelerationStructureKHR* _topLevelAS,
				   uint rayFlags,
                   uint cullMask,
                   uint sbtRecordOffset,
                   uint sbtRecordStride,
                   uint missIndex,
                   float3 origin,
                   float Tmin,
                   float3 direction,
                   float Tmax,
                   int payload)
{
	assert(cullMask == 0xff);
	assert(payload == 0);

	// Global memory
    // memory_space *mem=NULL;
    // mem = thread->get_global_memory();

	// Create ray
	Ray ray;
	ray.make_ray(origin, direction, Tmin, Tmax);

	// TODO: Get geometry data
	VkAccelerationStructureBuildGeometryInfoKHR* pInfos;

	// Get bottom-level AS
    uint8_t* topLevelASAddr = (uint8_t *)get_anv_accel_address(topLevelAS);
    GEN_RT_BVH topBVH;
    GEN_RT_BVH_unpack(&topBVH, topLevelASAddr);
    
    uint8_t* topRootAddr = topLevelASAddr + topBVH.RootNodeOffset;

    std::list<uint8_t *> traversal_stack;
	std::list<uint8_t *> leaf_stack;

    // start traversing top level BVH
    {
        uint8_t *node_addr = NULL;
        uint8_t *next_node_addr = topRootAddr;
        

        while (next_node_addr > 0)
        {
            node_addr = next_node_addr;
            next_node_addr = NULL;
            struct GEN_RT_BVH_INTERNAL_NODE node;
            GEN_RT_BVH_INTERNAL_NODE_unpack(&node, node_addr);

            bool child_hit[6];
            float thit[6];
            for(int i = 0; i < 6; i++)
            {
                float3 idir = calculate_idir(ray.get_direction());
                float3 lo, hi;
                lo = {node.ChildLowerXBound[i], node.ChildLowerYBound[i], node.ChildLowerYBound[i]}; //TODO: change this
                hi = {node.ChildUpperXBound[i], node.ChildUpperYBound[i], node.ChildUpperZBound[i]};

                child_hit[i] = ray_box_test(lo, hi, idir, ray.get_origin(), ray.get_tmin(), ray.get_tmax(), thit[i]);
            }

            uint8_t *child_addr = node_addr + (node.ChildOffset * 64);
            for(int i = 0; i < 6; i++)
            {
                if(child_hit[i])
                {
                    uint8_t *child_addr = node_addr + (node.ChildOffset * 64) * (i + 1);
                    if(node.ChildType[i] != NODE_TYPE_INTERNAL)
                        leaf_stack.push_back(child_addr);
                    else
                    {
                        if(next_node_addr == 0)
                            next_node_addr = child_addr; // TODO: sort by thit
                        else
                            traversal_stack.push_back(child_addr);
                    }
                }
            }

            // Miss
            if (next_node_addr == NULL) {
                if (traversal_stack.empty()) {
                    break;
                }

                // Pop next node from stack
                next_node_addr = traversal_stack.back();
                traversal_stack.pop_back();
            }
        }
    }

    { // leaf nodes
        // Set thit to max
        float min_thit = ray.dir_tmax.w;
        uint32_t min_geometry_id;

        for (auto const& leaf_addr : leaf_stack)
        {
            GEN_RT_BVH_INSTANCE_LEAF leaf;
            GEN_RT_BVH_INSTANCE_LEAF_unpack(&leaf, leaf_addr);

            //TODO: apply transformation matrix
            traversal_stack.push_back((uint8_t *)anv_address_map(*(struct anv_address *)(leaf.StartNodeAddress)));
        }
    }



    //traverse bottom AS
    if(!traversal_stack.empty())
    {
        uint8_t* node_addr = NULL;
        uint8_t* next_node_addr = traversal_stack.back();
        traversal_stack.pop_back();
        

        while (next_node_addr > 0)
        {
            node_addr = next_node_addr;
            next_node_addr = NULL;
            struct GEN_RT_BVH_INTERNAL_NODE node;
            GEN_RT_BVH_INTERNAL_NODE_unpack(&node, node_addr);

            bool child_hit[6];
            float thit[6];
            for(int i = 0; i < 6; i++)
            {
                float3 idir = calculate_idir(ray.get_direction());
                float3 lo, hi;
                lo = {node.ChildLowerXBound[i], node.ChildLowerYBound[i], node.ChildLowerZBound[i]}; //TODO: change this
                hi = {node.ChildUpperXBound[i], node.ChildUpperYBound[i], node.ChildUpperZBound[i]};

                child_hit[i] = ray_box_test(lo, hi, idir, ray.get_origin(), ray.get_tmin(), ray.get_tmax(), thit[i]);
            }

            for(int i = 0; i < 6; i++)
            {
                if(child_hit[i])
                {
                    uint8_t *child_addr = node_addr + (node.ChildOffset * 64) * (i + 1);
                    if(node.ChildType[i] != NODE_TYPE_INTERNAL)
                        leaf_stack.push_back(child_addr);
                    else
                    {
                        if(next_node_addr == 0)
                            next_node_addr = child_addr; // TODO: sort by thit
                        else
                            traversal_stack.push_back(child_addr);
                    }
                }
            }

            // Miss
            if (next_node_addr == NULL) {
                if (traversal_stack.empty()) {
                    break;
                }

                // Pop next node from stack
                next_node_addr = traversal_stack.back();
                traversal_stack.pop_back();
            }
        }
    }

    // Set thit to max
    float min_thit = ray.dir_tmax.w;
    uint32_t min_geometry_id;
    {
        for (auto const& leaf_addr : leaf_stack)
        {
            struct GEN_RT_BVH_QUAD_LEAF leaf;
            GEN_RT_BVH_QUAD_LEAF_unpack(&leaf, leaf_addr);

            float3 p[3];
            for(int i = 0; i < 3; i++)
            {
                p[i].x = leaf.QuadVertex[i].X;
                p[i].y = leaf.QuadVertex[i].Y;
                p[i].z = leaf.QuadVertex[i].Z;
            }

            // Triangle intersection algorithm
            float thit;
            bool hit = VulkanRayTracing::mt_ray_triangle_test(p[0], p[1], p[2], ray, &thit);

            if(thit < min_thit)
            {
                min_thit = thit;
                min_geometry_id = leaf.LeafDescriptor.GeometryIndex;
            }
        }
    }


	if (min_thit != ray.get_tmax())
	{
		// Run closest hit shader
	}
	else
	{
		// Run miss shader
	}
}

bool VulkanRayTracing::mt_ray_triangle_test(float3 p0, float3 p1, float3 p2, Ray ray_properties, float* thit)
{
    // Moller Trumbore algorithm (from scratchapixel.com)
    float3 v0v1 = p1 - p0;
    float3 v0v2 = p2 - p0;
    float3 pvec = cross(ray_properties.get_direction(), v0v2);
    float det = dot(v0v1, pvec);

    float idet = 1 / det;

    float3 tvec = ray_properties.get_origin() - p0;
    float u = dot(tvec, pvec) * idet;

    if (u < 0 || u > 1) return false;

    float3 qvec = cross(tvec, v0v1);
    float v = dot(ray_properties.get_direction(), qvec) * idet;

    if (v < 0 || (u + v) > 1) return false;

    *thit = dot(v0v2, qvec) * idet;
    return true;
}


void VulkanRayTracing::setPipelineInfo(VkRayTracingPipelineCreateInfoKHR* pCreateInfos)
{
    VulkanRayTracing::pCreateInfos = pCreateInfos;
	std::cout << "gpgpusim: set pipeline" << std::endl;
}


void VulkanRayTracing::setGeometries(VkAccelerationStructureGeometryKHR* pGeometries, uint32_t geometryCount)
{
    VulkanRayTracing::pGeometries = pGeometries;
    VulkanRayTracing::geometryCount = geometryCount;
	std::cout << "gpgpusim: set geometry" << std::endl;
}

void VulkanRayTracing::setAccelerationStructure(VkAccelerationStructureKHR accelerationStructure)
{
    std::cout << "gpgpusim: set AS" << std::endl;
    VulkanRayTracing::topLevelAS = accelerationStructure;
}

static bool invoked = false;

void VulkanRayTracing::registerShaders()
{
    // std::ifstream  src("/home/mrs/emerald-ray-tracing/MESA_SHADER_RAYGEN_0.ptx", std::ios::binary);
    // std::ofstream  dst("/home/mrs/emerald-ray-tracing/mesagpgpusimShaders/MESA_SHADER_RAYGEN_0.ptx",   std::ios::binary);
    // dst << src.rdbuf();



    gpgpu_context *ctx;
    ctx = GPGPU_Context();
    CUctx_st *context = GPGPUSim_Context(ctx);

    // Register all the ptx files in $MESA_ROOT/gpgpusimShaders by looping through them
    std::vector <std::string> ptx_list;

    // Add ptx file names in gpgpusimShaders folder to ptx_list
    char *mesa_root = getenv("MESA_ROOT");
    char *gpgpusim_root = getenv("GPGPUSIM_ROOT");
    char *filePath = "gpgpusimShaders/";
    char fullPath[200];
    snprintf(fullPath, sizeof(fullPath), "%s%s", mesa_root, filePath);
    std::string fullPathString(fullPath);

    for (auto &p : fs::recursive_directory_iterator(fullPathString))
    {
        if (p.path().extension() == ".ptx")
        {
            //std::cout << p.path().string() << '\n';
            ptx_list.push_back(p.path().string());
        }
    }
    
    // Register each ptx file in ptx_list
    symbol_table *symtab;
    unsigned num_ptx_versions = 0;
    unsigned max_capability = 20;
    unsigned selected_capability = 20;
    bool found = false;
    
    unsigned source_num = 1;
    unsigned long long fat_cubin_handle = 1;

    for(auto itr : ptx_list)
    {
        printf("############### adding %s\n", itr.c_str());
        // PTX File
        //std::cout << itr << std::endl;
        symtab = ctx->gpgpu_ptx_sim_load_ptx_from_filename(itr.c_str());
        context->add_binary(symtab, fat_cubin_handle);
        // need to add all the magic registers to ptx.l to special_register, reference ayub ptx.l:225

        // PTX info
        // Run the python script and get ptxinfo
        std::cout << "GPGPUSIM: Generating PTXINFO for" << itr.c_str() << "info" << std::endl;
        char command[400];
        snprintf(command, sizeof(command), "python3 %s/scripts/generate_rt_ptxinfo.py %s", gpgpusim_root, itr.c_str());
        int result = system(command);
        if (result != 0) {
            printf("GPGPU-Sim PTX: ERROR ** while loading PTX (b) %d\n", result);
            printf("               Ensure ptxas is in your path.\n");
            exit(1);
        }
        
        char ptxinfo_filename[400];
        snprintf(ptxinfo_filename, sizeof(ptxinfo_filename), "%sinfo", itr.c_str());
        ctx->gpgpu_ptx_info_load_from_external_file(ptxinfo_filename); // TODO: make a version where it just loads my ptxinfo instead of generating a new one

        if (itr.find("RAYGEN") != std::string::npos)
        {
            printf("############### registering %s\n", itr.c_str());
            context->register_function(fat_cubin_handle, "raygen_shader", "MESA_SHADER_RAYGEN_main");
        }

        source_num++;
        fat_cubin_handle++;
    }

}


void VulkanRayTracing::invoke_gpgpusim()
{
    gpgpu_context *ctx;
    ctx = GPGPU_Context();
    CUctx_st *context = GPGPUSim_Context(ctx);

    if(!invoked)
    {
        registerShaders();
        invoked = true;
    }
}

void VulkanRayTracing::vkCmdTraceRaysKHR(
                      const VkStridedDeviceAddressRegionKHR *raygen_sbt,
                      const VkStridedDeviceAddressRegionKHR *miss_sbt,
                      const VkStridedDeviceAddressRegionKHR *hit_sbt,
                      const VkStridedDeviceAddressRegionKHR *callable_sbt,
                      bool is_indirect,
                      uint32_t launch_width,
                      uint32_t launch_height,
                      uint32_t launch_depth,
                      uint64_t launch_size_addr) {
    gpgpu_context *ctx;
    ctx = GPGPU_Context();
    CUctx_st *context = GPGPUSim_Context(ctx);

    printf("vkCmdTraceRaysKHR\n");
    function_info *entry = context->get_kernel("raygen_shader");
    printf("################ number of args = %d\n", entry->num_args());

    gpgpu_ptx_sim_arg_list_t args;
    kernel_info_t *grid = ctx->api->gpgpu_cuda_ptx_sim_init_grid(
      "raygen_shader", args, dim3(1, 1, 1), dim3(16, 1, 1),
      context);
    
    struct CUstream_st *stream = 0;
    stream_operation op(grid, ctx->func_sim->g_ptx_sim_mode, stream);
    ctx->the_gpgpusim->g_stream_manager->push(op);

    // for (unsigned i = 0; i < entry->num_args(); i++) {
    //     std::pair<size_t, unsigned> p = entry->get_param_config(i);
    //     cudaSetupArgumentInternal(args[i], p.first, p.second);
    // }
}