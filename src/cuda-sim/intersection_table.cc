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

#include <intersection_table.h>
#include <stdio.h>
#include "vulkan_ray_tracing.h"
#include "../abstract_hardware_model.h"



Coalescing_warp_intersection_table::Coalescing_warp_intersection_table()
{
    // table = new Coalescing_Entry[INTERSECTION_TABLE_MAX_LENGTH];
    table = (Coalescing_Entry*) VulkanRayTracing::gpgpusim_alloc(sizeof(Coalescing_Entry) * INTERSECTION_TABLE_MAX_LENGTH);
    tableSize = 0;
}

std::pair<std::vector<MemoryTransactionRecord>, std::vector<MemoryStoreTransactionRecord> >
Coalescing_warp_intersection_table::add_intersection(uint32_t hit_group_index, uint32_t tid, uint32_t primitiveID, uint32_t instanceID,
                                                    const ptx_instruction *pI, ptx_thread_info *thread)
{
    memory_space *mem = thread->get_global_memory();

    assert(tid < 32);
    std::vector<MemoryTransactionRecord> loads;
    std::vector<MemoryStoreTransactionRecord> stores;

    for (int i = 0; i < tableSize; i++) {
        loads.push_back(MemoryTransactionRecord(&table[i].hitGroupIndex, 4, TransactionType::Intersection_Table_Load));
        
        uint32_t hitGroupIndex;
        mem->read(&hitGroupIndex, sizeof(uint32_t), &(table[i].hitGroupIndex));
        // mem->read(&table[i], sizeof(Coalescing_Entry), &entry);
        if (hitGroupIndex == hit_group_index)
        {
            bool thread_mask_tid;
            mem->read(&thread_mask_tid, sizeof(bool), &(table[i].thread_mask[tid]));
            // loads.push_back(MemoryTransactionRecord(&table[i].thread_mask[tid], 1, TransactionType::Intersection_Table_Load));
            if (!table[i].thread_mask[tid])
            {
                thread_mask_tid = true;
                mem->write(&(table[i].thread_mask[tid]), sizeof(bool), &thread_mask_tid, thread, pI);
                mem->write(&(table[i].shader_data[tid].primitiveID), sizeof(uint32_t), &primitiveID, thread, pI);
                mem->write(&(table[i].shader_data[tid].instanceID), sizeof(uint32_t), &instanceID, thread, pI);

                stores.push_back(MemoryStoreTransactionRecord(&table[i].thread_mask[tid], 1, StoreTransactionType::Intersection_Table_Store));
                stores.push_back(MemoryStoreTransactionRecord(&table[i].shader_data[tid], 8, StoreTransactionType::Intersection_Table_Store));
                return std::make_pair(loads, stores);
            }
        }
    }

    bool thread_mask_tid = true;
    mem->write(&(table[tableSize].hitGroupIndex), sizeof(uint32_t), &hit_group_index, thread, pI);
    mem->write(&(table[tableSize].thread_mask[tid]), sizeof(bool), &thread_mask_tid, thread, pI);
    mem->write(&(table[tableSize].shader_data[tid].primitiveID), sizeof(uint32_t), &primitiveID, thread, pI);
    mem->write(&(table[tableSize].shader_data[tid].instanceID), sizeof(uint32_t), &instanceID, thread, pI);

    stores.push_back(MemoryStoreTransactionRecord(&table[tableSize].hitGroupIndex, 4, StoreTransactionType::Intersection_Table_Store));
    stores.push_back(MemoryStoreTransactionRecord(&table[tableSize].thread_mask[tid], 1, StoreTransactionType::Intersection_Table_Store));
    stores.push_back(MemoryStoreTransactionRecord(&table[tableSize].shader_data[tid], 8, StoreTransactionType::Intersection_Table_Store));


    tableSize++;

    // if(tableSize > maxTableSize)
    // {
    //     maxTableSize = tableSize;
    //     printf("max table size = %d\n", maxTableSize);
    // }

    return std::make_pair(loads, stores);
}


void Coalescing_warp_intersection_table::clear() {
    for (int i = 0; i < tableSize; i++) 
        for(int i = 0; i < 32; i++) {
            table->thread_mask[i] = false;
        }
    tableSize = 0;
}

bool Coalescing_warp_intersection_table::shader_exists(uint32_t tid, uint32_t shader_counter, const ptx_instruction *pI, ptx_thread_info *thread) {
    memory_space *mem = thread->get_global_memory();
    bool thread_mask_tid;
    mem->read(&(table[shader_counter].thread_mask[tid]), sizeof(bool), &thread_mask_tid);
    return shader_counter < tableSize && thread_mask_tid;
}

bool Coalescing_warp_intersection_table::exit_shaders(uint32_t shader_counter, uint32_t tid) {
    return shader_counter >= tableSize;
}

uint32_t Coalescing_warp_intersection_table::get_primitiveID(uint32_t shader_counter, uint32_t tid, const ptx_instruction *pI, ptx_thread_info *thread) {
    memory_space *mem = thread->get_global_memory();
    uint32_t primitiveID;
    mem->read(&(table[shader_counter].shader_data[tid].primitiveID), sizeof(uint32_t), &primitiveID);
    return primitiveID;
}

uint32_t Coalescing_warp_intersection_table::get_instanceID(uint32_t shader_counter, uint32_t tid, const ptx_instruction *pI, ptx_thread_info *thread) {
    memory_space *mem = thread->get_global_memory();
    uint32_t instanceID;
    mem->read(&(table[shader_counter].shader_data[tid].instanceID), sizeof(uint32_t), &instanceID);
    return instanceID;
}

uint32_t Coalescing_warp_intersection_table::get_hitGroupIndex(uint32_t shader_counter, uint32_t tid, const ptx_instruction *pI, ptx_thread_info *thread) {
    memory_space *mem = thread->get_global_memory();
    uint32_t hitGroupIndex;
    mem->read(&(table[shader_counter].hitGroupIndex), sizeof(uint32_t), &hitGroupIndex);
    return hitGroupIndex;
}

void* Coalescing_warp_intersection_table::get_shader_data_address(uint32_t shader_counter, uint32_t tid) {
    return (void*)&table[shader_counter].shader_data[tid];
}







Baseline_warp_intersection_table::Baseline_warp_intersection_table()
{
    // table = new Baseline_Entry[INTERSECTION_TABLE_MAX_LENGTH];
    table = (Baseline_Entry*) VulkanRayTracing::gpgpusim_alloc(sizeof(Baseline_Entry) * INTERSECTION_TABLE_MAX_LENGTH);
    for(int i = 0; i < 32; i++)
        index[i] = 0;
}

// static int maxTableSize = 0;

std::pair<std::vector<MemoryTransactionRecord>, std::vector<MemoryStoreTransactionRecord> > 
Baseline_warp_intersection_table::add_intersection(uint32_t hit_group_index, uint32_t tid, uint32_t primitiveID, uint32_t instanceID,
                                                    const ptx_instruction *pI, ptx_thread_info *thread)
{
    assert(tid < 32);
    assert(index[tid] < INTERSECTION_TABLE_MAX_LENGTH);

    memory_space *mem = thread->get_global_memory();

    std::vector<MemoryTransactionRecord> loads;
    std::vector<MemoryStoreTransactionRecord> stores;

    mem->write(&(table[index[tid]].hitGroupIndex[tid]), sizeof(uint32_t), &hit_group_index, thread, pI);
    mem->write(&(table[index[tid]].shader_data[tid].primitiveID), sizeof(uint32_t), &primitiveID, thread, pI);
    mem->write(&(table[index[tid]].shader_data[tid].instanceID), sizeof(uint32_t), &instanceID, thread, pI);

    stores.push_back(MemoryStoreTransactionRecord(&table[index[tid]].hitGroupIndex[tid], 4, StoreTransactionType::Intersection_Table_Store));
    stores.push_back(MemoryStoreTransactionRecord(&table[index[tid]].shader_data[tid], 8, StoreTransactionType::Intersection_Table_Store));

    index[tid]++;

    return std::make_pair(loads, stores);
}

void Baseline_warp_intersection_table::clear() {
    for(int i = 0; i < 32; i++)
        index[i] = 0;
}

std::pair<std::vector<MemoryTransactionRecord>, std::vector<MemoryStoreTransactionRecord> >
Baseline_warp_intersection_table::add_intersection(uint32_t hit_group_index, uint32_t tid, uint32_t primitiveID, uint32_t instanceID,
                                                    const ptx_instruction *pI, ptx_thread_info *thread);

bool Baseline_warp_intersection_table::shader_exists(uint32_t tid, uint32_t shader_counter, const ptx_instruction *pI, ptx_thread_info *thread) {
    return shader_counter < index[tid];
}

bool Baseline_warp_intersection_table::exit_shaders(uint32_t shader_counter, uint32_t tid) {
    return shader_counter >= index[tid];
}

uint32_t Baseline_warp_intersection_table::get_primitiveID(uint32_t shader_counter, uint32_t tid, const ptx_instruction *pI, ptx_thread_info *thread) {
    memory_space *mem = thread->get_global_memory();
    uint32_t primitiveID;
    mem->read(&(table[shader_counter].shader_data[tid].primitiveID), sizeof(uint32_t), &primitiveID);
    return primitiveID;
}

uint32_t Baseline_warp_intersection_table::get_instanceID(uint32_t shader_counter, uint32_t tid, const ptx_instruction *pI, ptx_thread_info *thread) {
    memory_space *mem = thread->get_global_memory();
    uint32_t instanceID;
    mem->read(&(table[shader_counter].shader_data[tid].instanceID), sizeof(uint32_t), &instanceID);
    return instanceID;
}

uint32_t Baseline_warp_intersection_table::get_hitGroupIndex(uint32_t shader_counter, uint32_t tid, const ptx_instruction *pI, ptx_thread_info *thread) {
    memory_space *mem = thread->get_global_memory();
    uint32_t hitGroupIndex;
    mem->read(&(table[shader_counter].hitGroupIndex[tid]), sizeof(uint32_t), &hitGroupIndex);
    return hitGroupIndex;
}

void* Baseline_warp_intersection_table::get_shader_data_address(uint32_t shader_counter, uint32_t tid) {
    return (void*)&table[shader_counter].shader_data[tid];
}