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
    table = new Coalescing_Entry[INTERSECTION_TABLE_MAX_LENGTH];
    tableSize = 0;
}

std::pair<std::vector<MemoryTransactionRecord>, std::vector<MemoryStoreTransactionRecord> >
Coalescing_warp_intersection_table::add_intersection(uint32_t hit_group_index, uint32_t tid, uint32_t primitiveID, uint32_t instanceID)
{
    assert(tid < 32);
    std::vector<MemoryTransactionRecord> loads;
    std::vector<MemoryStoreTransactionRecord> stores;

    for (int i = 0; i < tableSize; i++) {
        loads.push_back(MemoryTransactionRecord(&table[i].hitGroupIndex, 4, TransactionType::Intersection_Table_Load));
        if (table[i].hitGroupIndex == hit_group_index)
        {
            // loads.push_back(MemoryTransactionRecord(&table[i].thread_mask[tid], 1, TransactionType::Intersection_Table_Load));
            if (!table[i].thread_mask[tid])
            {
                table[i].thread_mask[tid] = true;
                table[i].shader_data[tid].primitiveID = primitiveID;
                table[i].shader_data[tid].instanceID = instanceID;

                stores.push_back(MemoryStoreTransactionRecord(&table[i].thread_mask[tid], 1, StoreTransactionType::Intersection_Table_Store));
                stores.push_back(MemoryStoreTransactionRecord(&table[i].shader_data[tid], 8, StoreTransactionType::Intersection_Table_Store));
                return std::make_pair(loads, stores);
            }
        }
    }

    table[tableSize].hitGroupIndex = hit_group_index;
    table[tableSize].thread_mask[tid] = true;
    table[tableSize].shader_data[tid].primitiveID = primitiveID;
    table[tableSize].shader_data[tid].instanceID = instanceID;

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




Baseline_warp_intersection_table::Baseline_warp_intersection_table()
{
    table = new Baseline_Entry[INTERSECTION_TABLE_MAX_LENGTH];
    for(int i = 0; i < 32; i++)
        index[i] = 0;
}

// static int maxTableSize = 0;

std::pair<std::vector<MemoryTransactionRecord>, std::vector<MemoryStoreTransactionRecord> > 
Baseline_warp_intersection_table::add_intersection(uint32_t hit_group_index, uint32_t tid, uint32_t primitiveID, uint32_t instanceID)
{
    assert(tid < 32);
    assert(index[tid] < INTERSECTION_TABLE_MAX_LENGTH);
    std::vector<MemoryTransactionRecord> loads;
    std::vector<MemoryStoreTransactionRecord> stores;

    table[index[tid]].hitGroupIndex[tid] = hit_group_index;
    table[index[tid]].shader_data[tid].primitiveID = primitiveID;
    table[index[tid]].shader_data[tid].instanceID = instanceID;

    stores.push_back(MemoryStoreTransactionRecord(&table[index[tid]].hitGroupIndex[tid], 4, StoreTransactionType::Intersection_Table_Store));
    stores.push_back(MemoryStoreTransactionRecord(&table[index[tid]].shader_data[tid], 8, StoreTransactionType::Intersection_Table_Store));

    index[tid]++;

    return std::make_pair(loads, stores);
}