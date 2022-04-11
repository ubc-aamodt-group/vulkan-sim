#ifndef INTERSECTION_TABLE_H
#define INTERSECTION_TABLE_H

#include <stdint.h>
#include <assert.h>
#include <vector>

enum class IntersectionTableType {
    Baseline,
    Function_Call_Coalescing,
};

typedef struct warp_intersection_entry {
    warp_intersection_entry() {
        for(int i = 0; i < 32; i++) {
            thread_mask[i] = false;
        }
    }

    uint32_t hitGroupIndex;
    bool thread_mask[32];

    struct {
    uint32_t primitiveID;
    uint32_t instanceID;
    } shader_data[32];
} warp_intersection_entry;

class warp_intersection_table {
    std::vector<warp_intersection_entry> table;

public:
    void add_to_baseline_table(uint32_t index, uint32_t hit_group_index, uint32_t tid, uint32_t primitiveID, uint32_t instanceID) {
        assert(tid < 32);
        if (index < table.size()) {
            assert(hit_group_index == table[index].hitGroupIndex);
            assert(!table[index].thread_mask[tid]);

            table[index].thread_mask[tid] = true;
            table[index].shader_data[tid].primitiveID = primitiveID;
            table[index].shader_data[tid].instanceID = instanceID;
        }
        else {
            assert(index == table.size());
            warp_intersection_entry entry;
            entry.hitGroupIndex = hit_group_index;
            entry.thread_mask[tid] = true;
            entry.shader_data[tid].primitiveID = primitiveID;
            entry.shader_data[tid].instanceID = instanceID;
            table.push_back(entry);
        }
    }

    void add_to_coalescing_table(uint32_t hit_group_index, uint32_t tid, uint32_t primitiveID, uint32_t instanceID) {
        assert(tid < 32);
        for (int i = 0; i < table.size(); i++) {
            if (table[i].hitGroupIndex == hit_group_index)
                if (!table[i].thread_mask[tid])
                {
                    table[i].thread_mask[tid] = true;
                    table[i].shader_data[tid].primitiveID = primitiveID;
                    table[i].shader_data[tid].instanceID = instanceID;
                    return;
                }
        }

        warp_intersection_entry entry;
        entry.hitGroupIndex = hit_group_index;
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

    uint32_t get_hitGroupIndex(uint32_t shader_counter) {
        return table[shader_counter].hitGroupIndex;
    }

    uint32_t size() {
        return table.size();
    }

    void clear() {
        table.clear();
    }
};

#endif /* INTERSECTION_TABLE_H */
