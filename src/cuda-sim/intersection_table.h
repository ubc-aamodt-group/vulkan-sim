#ifndef INTERSECTION_TABLE_H
#define INTERSECTION_TABLE_H

#include <stdint.h>
#include <assert.h>
#include <vector>

typedef struct warp_intersection_entry {
    warp_intersection_entry() {
        for(int i = 0; i < 32; i++) {
            thread_mask[i] = false;
        }
    }

    uint32_t GeometryIndex;
    bool thread_mask[32];

    struct {
    uint32_t primitiveID;
    uint32_t instanceID;
    } shader_data[32];
} warp_intersection_entry;

class warp_intersection_table {
    std::vector<warp_intersection_entry> table;

public:
    void add_to_baseline_table(uint32_t geometry_id, uint32_t tid, uint32_t primitiveID, uint32_t instanceID) {
        assert(tid < 32);
        if (table.size() > 0 && table.back().GeometryIndex == geometry_id)
            if (!table.back().thread_mask[tid])
            {
                table.back().thread_mask[tid] = true;
                table.back().shader_data[tid].primitiveID = primitiveID;
                table.back().shader_data[tid].instanceID = instanceID;
                return;
            }

        warp_intersection_entry entry;
        entry.GeometryIndex = geometry_id;
        entry.thread_mask[tid] = true;
        entry.shader_data[tid].primitiveID = primitiveID;
        entry.shader_data[tid].instanceID = instanceID;
        table.push_back(entry);
    }

    void add_to_coalescing_table(uint32_t geometry_id, uint32_t tid, uint32_t primitiveID, uint32_t instanceID) {
        assert(tid < 32);
        for (int i = 0; i < table.size(); i++) {
            if (table[i].GeometryIndex == geometry_id)
                if (!table[i].thread_mask[tid])
                {
                    table[i].thread_mask[tid] = true;
                    table[i].shader_data[tid].primitiveID = primitiveID;
                    table[i].shader_data[tid].instanceID = instanceID;
                    return;
                }
        }

        warp_intersection_entry entry;
        entry.GeometryIndex = geometry_id;
        entry.thread_mask[tid] = true;
        entry.shader_data[tid].primitiveID = primitiveID;
        entry.shader_data[tid].instanceID = instanceID;
        table.push_back(entry);
    }

    bool shader_exists(uint32_t tid, uint32_t shader_counter) {
        return shader_counter < table.size() && table[shader_counter].thread_mask[tid];
    }

    bool exit_shaders(uint32_t shader_counter) {
        return shader_counter >= table.size();
    }

    uint32_t get_primitiveID(uint32_t shader_counter, uint32_t tid) {
        return table[shader_counter].shader_data[tid].primitiveID;
    }

    uint32_t get_instanceID(uint32_t shader_counter, uint32_t tid) {
        return table[shader_counter].shader_data[tid].instanceID;
    }

    void clear() {
        table.clear();
    }
};

#endif /* INTERSECTION_TABLE_H */
