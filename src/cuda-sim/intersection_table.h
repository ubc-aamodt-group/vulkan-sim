#ifndef INTERSECTION_TABLE_H
#define INTERSECTION_TABLE_H

#include <stdint.h>
#include <assert.h>
#include <vector>

#define INTERSECTION_TABLE_MAX_LENGTH 100

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

struct MemoryTransactionRecord;

class warp_intersection_table {
    warp_intersection_entry* table;
    uint32_t index[32];
    uint32_t tableSize;

    static const IntersectionTableType intersectionTableType = IntersectionTableType::Baseline;

    std::vector<MemoryTransactionRecord> add_to_baseline_table(uint32_t hit_group_index, uint32_t tid, uint32_t primitiveID, uint32_t instanceID);
    std::vector<MemoryTransactionRecord> add_to_coalescing_table(uint32_t hit_group_index, uint32_t tid, uint32_t primitiveID, uint32_t instanceID);

public:
    warp_intersection_table();
    ~warp_intersection_table();

    std::vector<MemoryTransactionRecord> add_intersection(uint32_t hit_group_index, uint32_t tid, uint32_t primitiveID, uint32_t instanceID);

    bool shader_exists(uint32_t tid, uint32_t shader_counter) {
        return shader_counter < tableSize && table[shader_counter].thread_mask[tid];
    }

    bool exit_shaders(uint32_t shader_counter) {
        return shader_counter >= tableSize;
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
        return tableSize;
    }

    void clear() {
        for(int i = 0; i < 32; i++)
            index[i] = 0;
        tableSize = 0;

        delete table;
        table = new warp_intersection_entry[INTERSECTION_TABLE_MAX_LENGTH];
    }
};

#endif /* INTERSECTION_TABLE_H */
