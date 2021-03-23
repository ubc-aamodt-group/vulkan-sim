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

VkRayTracingPipelineCreateInfoKHR* VulkanRayTracing::pCreateInfos = NULL;
VkAccelerationStructureGeometryKHR* VulkanRayTracing::pGeometries = NULL;
uint32_t VulkanRayTracing::geometryCount = 0;

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
	assert(cullMask = 0xff);
	assert(payload = 0);

	// Global memory
    //memory_space *mem=NULL;
    //mem = thread->get_global_memory();

	// Create ray
	Ray ray;
	ray.make_ray(origin, direction, Tmin, Tmax);

	// TODO: Get geometry data
	VkAccelerationStructureBuildGeometryInfoKHR* pInfos;

	// Get bottom-level AS
	struct anv_bvh_node* topLevelASRoot =  (struct anv_bvh_node*)(_topLevelAS);

	assert(topLevelASRoot->children[0]->is_leaf);
	uint32_t bottomLevelGeometryID = ((struct anv_bvh_leaf*)(topLevelASRoot->children[0]))->geometry_id;

	assert(pInfos->pGeometries[bottomLevelGeometryID].geometryType == VK_GEOMETRY_TYPE_INSTANCES_KHR);
	VkAccelerationStructureGeometryInstancesDataKHR ASInstance = pInfos->pGeometries[bottomLevelGeometryID].geometry.instances;
	struct anv_bvh_node* root = (struct anv_bvh_node*)(ASInstance.data.hostAddress);



	std::list<struct anv_bvh_node*> traversal_stack;
	std::list<struct anv_bvh_leaf*> leaf_stack;

	// Initialize
    struct anv_bvh_node* next_node = 0;

    while (next_node > 0)
    {
		bool child_hit[6];
		struct anv_bvh_node** child_addr = next_node->children;
		float thit[6];
		for(int i = 0; i < 6; i++)
		{
            float3 idir = calculate_idir(ray.get_direction());
			float3 lo, hi;
			lo = {next_node->lower_x[i], next_node->lower_y[i], next_node->lower_z[i]};
			hi = {next_node->upper_x[i], next_node->upper_y[i], next_node->upper_z[i]};

			child_hit[i] = ray_box_test(lo, hi, idir, ray.get_origin(), ray.get_tmin(), ray.get_tmax(), thit[i]);
		}

		next_node = 0;
		for(int i = 0; i < 6; i++)
		{
			if(child_hit[i])
			{
				if(child_addr[i]->is_leaf == 1)
					leaf_stack.push_back((struct anv_bvh_leaf*)(child_addr[i]));
				else
				{
					if(next_node == 0)
						next_node = child_addr[i]; // sort by thit
					else
						traversal_stack.push_back(child_addr[i]);
				}
			}
		}

        // Miss
        if (next_node == 0) {
            if (traversal_stack.empty()) {
                break;
            }

            // Pop next node from stack
            next_node = traversal_stack.back();
            traversal_stack.pop_back();
        }
    }

	#ifdef DEBUG_PRINT
	printf("Transition to leaf nodes.\n");
	#endif

	// Initialize
    struct anv_bvh_leaf* next_leaf = 0;

	// Set thit to max
    float min_thit = ray.dir_tmax.w;
	uint32_t min_geometry_id;


	for (auto const& leaf_addr : leaf_stack)
	{
	    // Load vertices
		uint32_t geometry_id = leaf_addr->geometry_id;

		assert(pInfos->pGeometries[geometry_id].geometryType == VK_GEOMETRY_TYPE_TRIANGLES_KHR);
		const VkAccelerationStructureGeometryTrianglesDataKHR triangleData = pInfos->pGeometries[bottomLevelGeometryID].geometry.triangles;

		assert(triangleData.maxVertex == 3);
		assert(triangleData.vertexStride == 12);
		float* vertexData = (float*)(triangleData.vertexData.hostAddress);

		float3 p[3];
		for(int i = 0; i < 3; i++)
		{
			p[i].x = vertexData[i];
			p[i].y = vertexData[i + 1];
			p[i].z = vertexData[i + 2];
			vertexData += 3; //TODO: +3 or +12
		}


	    // Triangle intersection algorithm
		float thit;
	    bool hit = VulkanRayTracing::mt_ray_triangle_test(p[0], p[1], p[2], ray, &thit);

		if(thit < min_thit)
		{
			min_thit = thit;
			min_geometry_id = geometry_id;
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

static bool invoked = false;

void VulkanRayTracing::registerShaders()
{
    gpgpu_context *ctx;
    ctx = GPGPU_Context();
    CUctx_st *context = GPGPUSim_Context(ctx);

    // Register all the ptx files in $MESA_ROOT/gpgpusimShaders by looping through them
    std::vector <std::string> ptx_list;

    // Add ptx file names in gpgpusimShaders folder to ptx_list
    char *mesa_root = getenv("MESA_ROOT");
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
        // PTX File
        //std::cout << itr << std::endl;
        symtab = ctx->gpgpu_ptx_sim_load_ptx_from_filename(itr.c_str());
        context->add_binary(symtab, fat_cubin_handle);
        // need to add all the magic registers to ptx.l to special_register, reference ayub ptx.l:225

        // PTX info
        // run the python script and get it in
        //ctx->gpgpu_ptx_info_load_from_filename(itr.c_str(), source_num, max_capability, ptx_list.size());

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