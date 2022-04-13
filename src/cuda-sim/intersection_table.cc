#include <intersection_table.h>
#include <stdio.h>
#include "vulkan_ray_tracing.h"

warp_intersection_table::warp_intersection_table()
{
    table = new warp_intersection_entry[INTERSECTION_TABLE_MAX_LENGTH];
    for(int i = 0; i < 32; i++)
        index[i] = 0;
    tableSize = 0;
}

warp_intersection_table::~warp_intersection_table()
{
    delete table;
}

// static int maxTableSize = 0;

std::vector<MemoryTransactionRecord> warp_intersection_table::add_to_baseline_table(uint32_t hit_group_index, uint32_t tid, uint32_t primitiveID, uint32_t instanceID)
{
    assert(tid < 32);
    std::vector<MemoryTransactionRecord> transactions;

    uint32_t &index = this->index[tid];
    bool new_entry_needed = true;
    while(index < tableSize)
    {
        transactions.push_back(MemoryTransactionRecord(&table[index].hitGroupIndex, 4, TransactionType::Intersection_Table_Load));
        if(table[index].hitGroupIndex == hit_group_index)
        {
            assert(!table[index].thread_mask[tid]);
            table[index].thread_mask[tid] = true;
            table[index].shader_data[tid].primitiveID = primitiveID;
            table[index].shader_data[tid].instanceID = instanceID;

            transactions.push_back(MemoryTransactionRecord(&table[index].thread_mask[tid], 1, TransactionType::Intersection_Table_Write));
            transactions.push_back(MemoryTransactionRecord(&table[index].shader_data[tid], 8, TransactionType::Intersection_Table_Write));

            new_entry_needed = false;
            index++;
            break;
        }
        index++;
    }
    if(new_entry_needed)
    {
        if(index != tableSize)
            printf("this is where things go wrong\n");
        assert(index == tableSize);
        table[tableSize].hitGroupIndex = hit_group_index;
        table[tableSize].thread_mask[tid] = true;
        table[tableSize].shader_data[tid].primitiveID = primitiveID;
        table[tableSize].shader_data[tid].instanceID = instanceID;

        transactions.push_back(MemoryTransactionRecord(&table[tableSize].hitGroupIndex, 4, TransactionType::Intersection_Table_Write));
        transactions.push_back(MemoryTransactionRecord(&table[tableSize].thread_mask[tid], 1, TransactionType::Intersection_Table_Write));
        transactions.push_back(MemoryTransactionRecord(&table[tableSize].shader_data[tid], 8, TransactionType::Intersection_Table_Write));
        
        index++;
        tableSize++;

        // if(tableSize > maxTableSize)
        // {
        //     maxTableSize = tableSize;
        //     printf("max table size = %d\n", maxTableSize);
        // }
    }

    return transactions;
}


std::vector<MemoryTransactionRecord> warp_intersection_table::add_to_coalescing_table(uint32_t hit_group_index, uint32_t tid, uint32_t primitiveID, uint32_t instanceID)
{
    assert(tid < 32);
    std::vector<MemoryTransactionRecord> transactions;

    for (int i = 0; i < tableSize; i++) {
        transactions.push_back(MemoryTransactionRecord(&table[i].hitGroupIndex, 4, TransactionType::Intersection_Table_Load));
        if (table[i].hitGroupIndex == hit_group_index)
        {
            transactions.push_back(MemoryTransactionRecord(&table[i].thread_mask[tid], 1, TransactionType::Intersection_Table_Load));
            if (!table[i].thread_mask[tid])
            {
                table[i].thread_mask[tid] = true;
                table[i].shader_data[tid].primitiveID = primitiveID;
                table[i].shader_data[tid].instanceID = instanceID;

                transactions.push_back(MemoryTransactionRecord(&table[i].thread_mask[tid], 1, TransactionType::Intersection_Table_Write));
                transactions.push_back(MemoryTransactionRecord(&table[i].shader_data[tid], 8, TransactionType::Intersection_Table_Write));
                return transactions;
            }
        }
    }

    table[tableSize].hitGroupIndex = hit_group_index;
    table[tableSize].thread_mask[tid] = true;
    table[tableSize].shader_data[tid].primitiveID = primitiveID;
    table[tableSize].shader_data[tid].instanceID = instanceID;

    transactions.push_back(MemoryTransactionRecord(&table[tableSize].hitGroupIndex, 4, TransactionType::Intersection_Table_Write));
    transactions.push_back(MemoryTransactionRecord(&table[tableSize].thread_mask[tid], 1, TransactionType::Intersection_Table_Write));
    transactions.push_back(MemoryTransactionRecord(&table[tableSize].shader_data[tid], 8, TransactionType::Intersection_Table_Write));


    tableSize++;

    // if(tableSize > maxTableSize)
    // {
    //     maxTableSize = tableSize;
    //     printf("max table size = %d\n", maxTableSize);
    // }

    return transactions;
}

std::vector<MemoryTransactionRecord> warp_intersection_table::add_intersection(uint32_t hit_group_index, uint32_t tid, uint32_t primitiveID, uint32_t instanceID)
{
    if(warp_intersection_table::intersectionTableType == IntersectionTableType::Baseline)
        return add_to_baseline_table(hit_group_index, tid, primitiveID, instanceID);
    else
        return add_to_coalescing_table(hit_group_index, tid, primitiveID, instanceID);
}