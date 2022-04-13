#include <intersection_table.h>
#include <stdio.h>

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

static int maxTableSize = 0;

void warp_intersection_table::add_to_baseline_table(uint32_t hit_group_index, uint32_t tid, uint32_t primitiveID, uint32_t instanceID)
{
    assert(tid < 32);
    uint32_t &index = this->index[tid];
    bool new_entry_needed = true;
    while(index < tableSize)
    {
        if(table[index].hitGroupIndex == hit_group_index)
        {
            assert(!table[index].thread_mask[tid]);
            table[index].thread_mask[tid] = true;
            table[index].shader_data[tid].primitiveID = primitiveID;
            table[index].shader_data[tid].instanceID = instanceID;
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
        warp_intersection_entry entry;
        entry.hitGroupIndex = hit_group_index;
        entry.thread_mask[tid] = true;
        entry.shader_data[tid].primitiveID = primitiveID;
        entry.shader_data[tid].instanceID = instanceID;
        table[tableSize++] = entry;
        index++;

        if(tableSize > maxTableSize)
        {
            maxTableSize = tableSize;
            printf("max table size = %d\n", maxTableSize);
        }
    }
}


void warp_intersection_table::add_to_coalescing_table(uint32_t hit_group_index, uint32_t tid, uint32_t primitiveID, uint32_t instanceID)
{
    assert(tid < 32);
    for (int i = 0; i < tableSize; i++) {
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
    table[tableSize++] = entry;

    if(tableSize > maxTableSize)
    {
        maxTableSize = tableSize;
        printf("max table size = %d\n", maxTableSize);
    }
}

void warp_intersection_table::add_intersection(uint32_t hit_group_index, uint32_t tid, uint32_t primitiveID, uint32_t instanceID)
{
    if(warp_intersection_table::intersectionTableType == IntersectionTableType::Baseline)
        add_to_baseline_table(hit_group_index, tid, primitiveID, instanceID);
    else
        add_to_coalescing_table(hit_group_index, tid, primitiveID, instanceID);
}