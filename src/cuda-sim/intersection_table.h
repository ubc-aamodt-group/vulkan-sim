#ifndef INTERSECTION_TABLE_H
#define INTERSECTION_TABLE_H

#include <stdint.h>
#include <assert.h>
#include <vector>
#include <utility>
#include <limits.h>

#define INTERSECTION_TABLE_MAX_LENGTH 100

enum class IntersectionTableType {
    Baseline,
    Function_Call_Coalescing,
};

struct MemoryTransactionRecord;
struct MemoryStoreTransactionRecord;

class warp_intersection_table {
public:
    // virtual warp_intersection_table() {}
    // virtual ~warp_intersection_table() {}
    virtual std::pair<std::vector<MemoryTransactionRecord>, std::vector<MemoryStoreTransactionRecord> >
            add_intersection(uint32_t hit_group_index, uint32_t tid, uint32_t primitiveID, uint32_t instanceID) = 0;
    
    virtual void clear() = 0;
    virtual bool shader_exists(uint32_t tid, uint32_t shader_counter) = 0;
    virtual bool exit_shaders(uint32_t shader_counter, uint32_t tid) = 0;
    virtual uint32_t get_primitiveID(uint32_t shader_counter, uint32_t tid) = 0;
    virtual uint32_t get_instanceID(uint32_t shader_counter, uint32_t tid) = 0;
    virtual uint32_t get_hitGroupIndex(uint32_t shader_counter, uint32_t tid) = 0;
    virtual void* get_shader_data_address(uint32_t shader_counter, uint32_t tid) = 0;
};


typedef struct Coalescing_Entry {
    Coalescing_Entry() {
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
} Coalescing_Entry;

class Coalescing_warp_intersection_table : public warp_intersection_table {
    Coalescing_Entry* table;
    uint32_t tableSize;

public:
    Coalescing_warp_intersection_table();
    ~Coalescing_warp_intersection_table() {
        delete table;
    }

    std::pair<std::vector<MemoryTransactionRecord>, std::vector<MemoryStoreTransactionRecord> >
            add_intersection(uint32_t hit_group_index, uint32_t tid, uint32_t primitiveID, uint32_t instanceID);
    

    bool shader_exists(uint32_t tid, uint32_t shader_counter) {
        return shader_counter < tableSize && table[shader_counter].thread_mask[tid];
    }

    bool exit_shaders(uint32_t shader_counter, uint32_t tid) {
        return shader_counter >= tableSize;
    }

    uint32_t get_primitiveID(uint32_t shader_counter, uint32_t tid) {
        return table[shader_counter].shader_data[tid].primitiveID;
    }

    uint32_t get_instanceID(uint32_t shader_counter, uint32_t tid) {
        return table[shader_counter].shader_data[tid].instanceID;
    }

    uint32_t get_hitGroupIndex(uint32_t shader_counter, uint32_t tid) {
        return table[shader_counter].hitGroupIndex;
    }

    void* get_shader_data_address(uint32_t shader_counter, uint32_t tid) {
        return (void*)&table[shader_counter].shader_data[tid];
    }

    void clear() {
        tableSize = 0;
        delete table;
        table = new Coalescing_Entry[INTERSECTION_TABLE_MAX_LENGTH];
    }

};


typedef struct Baseline_Entry {
    uint32_t hitGroupIndex[32];

    struct {
    uint32_t primitiveID;
    uint32_t instanceID;
    } shader_data[32];
} Baseline_Entry;

class Baseline_warp_intersection_table : public warp_intersection_table {
    Baseline_Entry* table;
    uint32_t index[32];

public:
    Baseline_warp_intersection_table();
    ~Baseline_warp_intersection_table() {
        delete table;
    }

    std::pair<std::vector<MemoryTransactionRecord>, std::vector<MemoryStoreTransactionRecord> >
            add_intersection(uint32_t hit_group_index, uint32_t tid, uint32_t primitiveID, uint32_t instanceID);

    bool shader_exists(uint32_t tid, uint32_t shader_counter) {
        return shader_counter < index[tid];
    }

    bool exit_shaders(uint32_t shader_counter, uint32_t tid) {
        return shader_counter >= index[tid];
    }

    uint32_t get_primitiveID(uint32_t shader_counter, uint32_t tid) {
        return table[shader_counter].shader_data[tid].primitiveID;
    }

    uint32_t get_instanceID(uint32_t shader_counter, uint32_t tid) {
        return table[shader_counter].shader_data[tid].instanceID;
    }

    uint32_t get_hitGroupIndex(uint32_t shader_counter, uint32_t tid) {
        return table[shader_counter].hitGroupIndex[tid];
    }

    void* get_shader_data_address(uint32_t shader_counter, uint32_t tid) {
        return (void*)&table[shader_counter].shader_data[tid];
    }

    void clear() {
        for(int i = 0; i < 32; i++)
            index[i] = 0;

        delete table;
        table = new Baseline_Entry[INTERSECTION_TABLE_MAX_LENGTH];
    }
};


#endif /* INTERSECTION_TABLE_H */
