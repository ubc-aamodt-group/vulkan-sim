// Copyright (c) 2022, Mohammadreza Saed, Yuan Hsi Chou, Lufei Liu, Tor M. Aamodt,
// The University of British Columbia
// All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:

// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The University of British Columbia nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifndef INTERSECTION_TABLE_H
#define INTERSECTION_TABLE_H

#include "ptx_ir.h"
#include "../../libcuda/gpgpu_context.h"
#include "../abstract_hardware_model.h"

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
            add_intersection(uint32_t hit_group_index, uint32_t tid, uint32_t primitiveID, uint32_t instanceID,
                            const ptx_instruction *pI, ptx_thread_info *thread) = 0;
    
    virtual void clear() = 0;
    virtual bool shader_exists(uint32_t tid, uint32_t shader_counter, const ptx_instruction *pI, ptx_thread_info *thread) = 0;
    virtual bool exit_shaders(uint32_t shader_counter, uint32_t tid) = 0;
    virtual uint32_t get_primitiveID(uint32_t shader_counter, uint32_t tid, const ptx_instruction *pI, ptx_thread_info *thread) = 0;
    virtual uint32_t get_instanceID(uint32_t shader_counter, uint32_t tid, const ptx_instruction *pI, ptx_thread_info *thread) = 0;
    virtual uint32_t get_hitGroupIndex(uint32_t shader_counter, uint32_t tid, const ptx_instruction *pI, ptx_thread_info *thread) = 0;
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
            add_intersection(uint32_t hit_group_index, uint32_t tid, uint32_t primitiveID, uint32_t instanceID,
                            const ptx_instruction *pI, ptx_thread_info *thread);
    void clear();
    bool shader_exists(uint32_t tid, uint32_t shader_counter, const ptx_instruction *pI, ptx_thread_info *thread);
    bool exit_shaders(uint32_t shader_counter, uint32_t tid);
    uint32_t get_primitiveID(uint32_t shader_counter, uint32_t tid, const ptx_instruction *pI, ptx_thread_info *thread);
    uint32_t get_instanceID(uint32_t shader_counter, uint32_t tid, const ptx_instruction *pI, ptx_thread_info *thread);
    uint32_t get_hitGroupIndex(uint32_t shader_counter, uint32_t tid, const ptx_instruction *pI, ptx_thread_info *thread);
    void* get_shader_data_address(uint32_t shader_counter, uint32_t tid);
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

    void clear();

    std::pair<std::vector<MemoryTransactionRecord>, std::vector<MemoryStoreTransactionRecord> >
            add_intersection(uint32_t hit_group_index, uint32_t tid, uint32_t primitiveID, uint32_t instanceID,
                            const ptx_instruction *pI, ptx_thread_info *thread);
    bool shader_exists(uint32_t tid, uint32_t shader_counter, const ptx_instruction *pI, ptx_thread_info *thread);
    bool exit_shaders(uint32_t shader_counter, uint32_t tid);
    uint32_t get_primitiveID(uint32_t shader_counter, uint32_t tid, const ptx_instruction *pI, ptx_thread_info *thread);
    uint32_t get_instanceID(uint32_t shader_counter, uint32_t tid, const ptx_instruction *pI, ptx_thread_info *thread);
    uint32_t get_hitGroupIndex(uint32_t shader_counter, uint32_t tid, const ptx_instruction *pI, ptx_thread_info *thread);
    void* get_shader_data_address(uint32_t shader_counter, uint32_t tid);
};


#endif /* INTERSECTION_TABLE_H */
