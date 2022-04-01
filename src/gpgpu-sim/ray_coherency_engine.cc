#include "ray_coherency_engine.h"
#include "../../libcuda/gpgpu_context.h"


ray_coherence_engine::ray_coherence_engine(unsigned sid, struct ray_coherence_config config, coherence_stats *stats, shader_core_ctx *core) {
  m_core = core;
  m_sid = sid;
  m_config = config;
  m_initialized = false;
  m_active = false;
  m_stats = stats;
  m_schedule_packet_id = 0;

  m_scheduled_packets.resize(m_config.max_packets);
}

void ray_coherence_engine::set_world(float3 min, float3 max) {
  COHERENCE_DPRINTF("Shader %d: Set world coordinates\n", m_sid);
  world_min = min;
  world_max = max;
}

void ray_coherence_engine::insert(warp_inst_t inst) {
  assert(!inst.empty());

  m_last_insertion_cycle = GPGPU_Context()->the_gpgpusim->g_the_gpu->gpu_tot_sim_cycle + GPGPU_Context()->the_gpgpusim->g_the_gpu->gpu_sim_cycle;
  COHERENCE_DPRINTF("Shader %d: New warp inserted (%ld)\n", m_sid, m_last_insertion_cycle);
  
  unsigned num_rays = 0;

  // Copy rays into engine
  for (unsigned i=0; i<m_config.warp_size; i++) {
    if (inst.rt_mem_accesses_empty(i)) continue;

    // Create ray
    coherence_ray ray;
    ray.origin_thread_id = i;
    ray.origin_warp_uid = inst.get_uid();
    ray.ray_properties = inst.get_thread_info(i).ray_properties;
    ray.RT_mem_accesses = inst.get_thread_info(i).RT_mem_accesses;
    ray.latency_delay = inst.get_thread_latency(i);

    // Get ray hash
    ray_hash hash = get_ray_hash(ray.ray_properties);

    // Add ray to pool
    if (m_ray_pool.find(hash) == m_ray_pool.end()) {
      COHERENCE_DPRINTF("Shader %d: New coherence packet created for hash 0x%x\n", m_sid, hash);
      coherence_packet packet;
      m_ray_pool[hash] = packet;
      m_stats->total_packets++;
    }

    m_ray_pool[hash].push_back(ray);
    m_total_rays++;
    m_stats->total_rays++;
    num_rays++;
    m_num_ray_pool_rays++;
  }
  
  COHERENCE_DPRINTF("Shader %d: %d rays added (%d rays total)\n", m_sid, num_rays, m_total_rays);
  if (m_total_rays > m_stats->max_rays) {
    m_stats->max_rays = m_total_rays;
  }
}

bool ray_coherence_engine::is_empty(coherence_packet packet) {
  for (auto it=packet.cbegin(); it!=packet.cend(); it++) {
    coherence_ray ray = *it;
    if (!ray.empty()) {
      return false;
    }
  }
  return true;
}

bool ray_coherence_engine::is_stalled() {
  // Check all the coherence packets
  for (auto i=m_scheduled_packets.cbegin(); i!=m_scheduled_packets.cend(); i++) {
    coherence_packet packet = *i;
    if (!is_stalled(packet)) return false;
  }
  // Stalled
  return true;
}

bool ray_coherence_engine::is_stalled(const coherence_packet packet) {
  // Check all the rays
  for (auto it=packet.cbegin(); it!=packet.cend(); it++) {
    coherence_ray ray = *it;
    if (!ray.empty()) {
      if (ray.next_status() == RT_MEM_UNMARKED && ray.latency_delay == 0) return false;
    }
  }
  // Stalled
  return true;
}

bool ray_coherence_engine::check_scheduled() {
  for (unsigned i=0; i<m_config.max_packets; i++) {
    if (!m_scheduled_packets[i].empty()) return true;
  }
  // No packets currently scheduled
  return false;
}

bool ray_coherence_engine::scheduled_full() {
  for (unsigned i=0; i<m_config.max_packets; i++) {
    if (m_scheduled_packets[i].empty()) return false;
  }
  // All packets full
  return true;
}

void ray_coherence_engine::cycle() {
  unsigned long long current_cycle = GPGPU_Context()->the_gpgpusim->g_the_gpu->gpu_tot_sim_cycle + GPGPU_Context()->the_gpgpusim->g_the_gpu->gpu_sim_cycle;
  
  // Turn engine off if there are no more rays
  if (m_total_rays == 0) m_active = false;
  // Turn engine off if stalled
  else if (scheduled_full() && is_stalled()) {
    m_active = false;
    m_stats->stalled_cycles++;
  }
  // If not stalled and there are scheduled packets, turn on.
  else if (check_scheduled()) {
    m_active = true;
  }
  // Turn engine on if there are enough rays
  else if (m_total_rays > m_config.min_rays) {
    if (!m_active) m_stats->activate_by_rays++;
    m_active = true;
  }
  // Turn engine on if timer expires
  else if (current_cycle - m_last_insertion_cycle > m_config.max_cycles) {
    if (!m_active) m_stats->activate_by_timer++;
    m_active = true;
  }

  if (m_total_rays != 0) m_stats->total_cycles++;
  if (m_active) {
    m_stats->active_cycles++;

    // Check all the coherence packets
    unsigned active_packets = 0;
    for (auto i=m_ray_pool.cbegin(); i!=m_ray_pool.cend(); i++) {
      ray_hash hash = i->first;
      coherence_packet packet = i->second;
      if (!is_stalled(packet) && !is_empty(packet)) {
        active_packets++;
      }
    }
    m_stats->average_stat(coherence_stats_type::ACTIVE_PACKETS, active_packets);

    // Schedule packets
    if (m_num_ray_pool_rays > 0) {
      for (unsigned i=0; i<m_config.max_packets; i++) {
        // If there is an empty packet, fill it
        if (m_scheduled_packets[i].empty()) {
          // Find largest packet
          ray_hash hash;
          coherence_packet *selected_packet = get_largest_packet(hash);
          COHERENCE_DPRINTF("Shader %d: Scheduling new packet [%d] with 0x%x\n", m_sid, i, hash);

          // Move rays (schedule)
          for (unsigned r=0; r<m_config.warp_size; r++) {
            if (selected_packet->empty()) break;
            m_scheduled_packets[i].push_back(selected_packet->front());
            selected_packet->pop_front();
            m_num_scheduled_rays++;
            m_num_ray_pool_rays--;
          }
        }
      }
    }

    if (is_stalled()) m_active = false;
  }
  assert(m_num_ray_pool_rays + m_num_scheduled_rays == m_total_rays);
}

coherence_packet * ray_coherence_engine::get_largest_packet(ray_hash &hash) {
  // Find the largest coherence packet
  unsigned largest_packet = 0;
  for (auto it=m_ray_pool.cbegin(); it!=m_ray_pool.cend(); it++) {
    if (it->second.size() > largest_packet) {
      hash = it->first;
      largest_packet = it->second.size();
    }
  }

  return &m_ray_pool[hash];
}

unsigned ray_coherence_engine::schedule_next_warp() {
  assert(m_active);

  // Iterate through packet to find non-stalled packet
  while (is_stalled(m_scheduled_packets[m_schedule_packet_id])) {
    m_schedule_packet_id = (m_schedule_packet_id + 1) % m_config.max_packets;
  }
  COHERENCE_DPRINTF("Shader %d: Scheduling next access (packet %d ", m_sid, m_schedule_packet_id);
  coherence_packet &selected_packet = m_scheduled_packets[m_schedule_packet_id];

  // Choose the most common request
  // map (addr, size)->occurrences
  std::map<addr_size_pair, unsigned> requests;
  // Gather all the addresses
  for (auto it=selected_packet.cbegin(); it!=selected_packet.cend(); it++) {
    coherence_ray ray = *it;

    if (!ray.empty()) {
      // Check if address is already in progress or ray is not ready yet
      if (ray.next_status() != RT_MEM_AWAITING && ray.latency_delay == 0) {
        addr_size_pair request = addr_size_pair(ray.next_access().address, ray.next_access().size);
        if (requests.find(request) == requests.end()) {
          requests[request] = 0;
        }
        requests[request]++;
      }
    }
  }

  assert(!requests.empty());

  // Find the most common
  unsigned occurrences = 0;
  addr_size_pair next_request;
  for (auto it=requests.cbegin(); it!=requests.cend(); it++) {
    if (it->second > occurrences) {
      occurrences = it->second;
      next_request = it->first;
    }
  }

  m_stats->average_stat(coherence_stats_type::COALESCED_REQUESTS, occurrences);

  COHERENCE_DPRINTF("addr 0x%x size %d ", next_request.first, next_request.second);


  // Find thread
  for (auto it=selected_packet.cbegin(); it!=selected_packet.cend(); it++) {
    coherence_ray ray = *it;
    if (!ray.empty() && ray.latency_delay == 0) {
      if (ray.next_access().address == next_request.first && ray.next_access().size == next_request.second) {
        m_active_thread = ray.origin_thread_id;
        m_active_warp = ray.origin_warp_uid;
        m_active_record = ray.next_access();

        COHERENCE_DPRINTF("warp %d thread %d)\n", m_active_warp, m_active_thread);
        return m_active_warp;
      }
    }
  }
  assert(0);
}

RTMemoryTransactionRecord ray_coherence_engine::get_next_access() {
  coherence_packet &selected_packet = m_scheduled_packets[m_schedule_packet_id];
  // Mark memory record status
  for (coherence_ray &ray : selected_packet) {
    if (!ray.empty()) {
      if (ray.next_addr() == m_active_record.address &&
          ray.next_access().size == m_active_record.size &&
          ray.latency_delay == 0) {
        ray.RT_mem_accesses.front().status = RT_MEM_AWAITING;
        COHERENCE_DPRINTF("Shader %d: Mark mem awaiting for warp %d thread %d\n", m_sid, ray.origin_warp_uid, ray.origin_thread_id);
      }
    }
  }

  // Mark request as sent
  if (m_request_mshr.find(m_active_record.address) == m_request_mshr.end()) {
    std::set<unsigned> outstanding_packets;
    m_request_mshr[m_active_record.address] = outstanding_packets;
  }
  COHERENCE_DPRINTF("Shader %d: Inserting MSHR entry for packet %d at addr 0x%x\n", m_sid, m_schedule_packet_id, m_active_record.address);
  m_request_mshr[m_active_record.address].insert(m_schedule_packet_id);

  // Create MSHR for chunks
  if (m_active_record.size > 32) {
    COHERENCE_DPRINTF("Shader %d: Memory request > 32B. Inserting MSHR entries\n", m_sid);
    // Create the memory chunks and push to mem_access_q
    for (unsigned i=1; i<((m_active_record.size+31)/32); i++) {
      if (m_request_mshr.find(m_active_record.address + (i * 32)) == m_request_mshr.end()) {
        std::set<unsigned> outstanding_packets;
        m_request_mshr[m_active_record.address + (i * 32)] = outstanding_packets;
      }
      COHERENCE_DPRINTF("Shader %d: Inserting MSHR entry for packet %d at addr 0x%x\n", m_sid, m_schedule_packet_id, m_active_record.address + (i * 32));
      m_request_mshr[m_active_record.address + (i * 32)].insert(m_schedule_packet_id);
    }
  }

  return m_active_record;
}

void ray_coherence_engine::undo_access(new_addr_type addr) {
  coherence_packet &selected_packet = m_scheduled_packets[m_schedule_packet_id];
  // Assume that this was the most recent request
  assert(m_active_record.address == addr);
  
  for (coherence_ray &ray : selected_packet) {
    if (!ray.empty()) {
      if (ray.next_addr() == m_active_record.address &&
          ray.next_access().size == m_active_record.size &&
          ray.RT_mem_accesses.front().status == RT_MEM_AWAITING) {
        ray.RT_mem_accesses.front().status = RT_MEM_UNMARKED;
        COHERENCE_DPRINTF("Shader %d: Undo mem awaiting for warp &d thread %d\n", m_sid, ray.origin_warp_uid, ray.origin_thread_id);
      }
    }
  }

  // Remove the most recent hash from the MSHR
  assert(m_request_mshr.find(addr) != m_request_mshr.end());
  assert(m_request_mshr[addr].erase(m_schedule_packet_id) > 0);
  COHERENCE_DPRINTF("Shader %d: Undoing MSHR entry for packet %d at addr 0x%x\n", m_sid, m_schedule_packet_id, addr);
}

void ray_coherence_engine::process_response(mem_fetch *mf, std::map<unsigned, warp_inst_t *> &m_current_warps, warp_inst_t *pipe_reg) {
  new_addr_type uncoalesced_addr = mf->get_uncoalesced_addr();
  new_addr_type uncoalesced_base_addr = mf->get_uncoalesced_base_addr();
  COHERENCE_DPRINTF("Shader %d: Processing memory response for addr 0x%x\n", m_sid, uncoalesced_addr);

  unsigned found = 0;
  
  if (m_request_mshr.find(uncoalesced_addr) != m_request_mshr.end()) {
    std::set<unsigned> packets = m_request_mshr[uncoalesced_addr];
    COHERENCE_DPRINTF("Shader %d: Found %d MSHR ray coherency packets for addr 0x%x\n", m_sid, packets.size(), uncoalesced_addr);

    // Mark memory response for all hashes
    for (unsigned p : packets) {
      // Find the appropriate threads
      coherence_packet &packet = m_scheduled_packets[p];
      
      // Go through each ray in the packet
      for (coherence_ray &ray : packet) {
        if (!ray.empty() && ray.latency_delay == 0) {
          unsigned thread_id = ray.origin_thread_id;
          unsigned warp_uid = ray.origin_warp_uid;

          assert(pipe_reg->get_uid() == warp_uid || m_current_warps.find(warp_uid) != m_current_warps.end());
          if (uncoalesced_base_addr == ray.next_addr()) {
            COHERENCE_DPRINTF("Shader %d: Ray coherency packet includes warp %d thread %d\n", m_sid, warp_uid, thread_id);
            bool mem_record_done = (!pipe_reg->empty() && pipe_reg->get_uid() == warp_uid) ?
               pipe_reg->process_returned_mem_access(mf, thread_id) :
               m_current_warps[warp_uid]->process_returned_mem_access(mf, thread_id);

            if (mem_record_done) {
              assert(ray.next_addr() == mf->get_uncoalesced_base_addr());
              ray.RT_mem_accesses.pop_front();
              ray.latency_delay = (!pipe_reg->empty() && pipe_reg->get_uid() == warp_uid) ?
                 pipe_reg->get_thread_latency(thread_id):
                 m_current_warps[warp_uid]->get_thread_latency(thread_id);
            }
          }
        }
      }
    }

    // Remove address from MSHR 
    m_request_mshr.erase(uncoalesced_addr);
  }
  if (is_stalled()) m_active = false;
}

void ray_coherence_engine::dec_thread_latency() {
  for (unsigned i=0; i<m_config.max_packets; i++) {
    unsigned index = 0;
    std::deque<unsigned> index_list;
    coherence_packet &packet = m_scheduled_packets[i];
    for (coherence_ray &ray : packet) {
      if (ray.latency_delay > 0) ray.latency_delay--;
      else if (ray.empty()) {
        COHERENCE_DPRINTF("Shader %d: Ray (w%d:t%d) complete!\n", m_sid, ray.origin_warp_uid, ray.origin_thread_id);
        index_list.push_back(index);
      }
      index++;
    }

    // Delete completed rays
    index = 0;
    for (auto iter=index_list.begin(); iter!=index_list.end(); iter++) {
      unsigned delete_index = *iter;
      unsigned adjusted_index = delete_index - index;
      packet.erase(packet.begin() + adjusted_index);
      index++;
      m_num_scheduled_rays--;
      m_total_rays--;
    }
  }
}

uint64_t ray_coherence_engine::hash_comp(float x, uint32_t num_bits) {
  uint32_t mask = UINT32_MAX >> (32 - num_bits);

  uint32_t o_x = *((uint32_t*) &x);

  uint64_t sign_bit_x = o_x >> 31;
  uint64_t exp_x = (o_x >> (31 - num_bits)) & mask;
  uint64_t mant_x = (o_x >> (23 - num_bits)) & mask;

  return (sign_bit_x << (2 * num_bits)) | (exp_x << num_bits) | mant_x;
}

// Quantize direction to a sphere - xyz to theta and phi
// `theta_bits` is used for theta, `theta_bits` + 1 is used for phi, for a total of
// 2 * `theta_bits` + 1 bits
uint64_t ray_coherence_engine::hash_direction_spherical(const float3 &d) {
  uint32_t num_sphere_bits = m_config.hash_sphere_bits;
  uint32_t theta_bits = num_sphere_bits;
  uint32_t phi_bits = theta_bits + 1;

  uint64_t theta = std::acos(clamp(d.z, -1.f, 1.f)) / PI * 180;
  uint64_t phi = (std::atan2(d.y, d.x) + PI) / PI * 180;
  uint64_t q_theta = theta >> (8 - theta_bits);
  uint64_t q_phi = phi >> (9 - phi_bits);

  return (q_phi << theta_bits) | q_theta;
}

// Quantize origin to a grid
// Each component uses `num_bits`, for a total of 3 * `num_bits` bits
uint64_t ray_coherence_engine::hash_origin_grid(const float3& o, uint32_t num_bits) {
  uint32_t grid_size = 1 << num_bits;

  uint64_t hash_o_x = clamp((o.x - world_min.x) / (world_max.x - world_min.x) * grid_size, 0.f, (float)grid_size - 1);
  uint64_t hash_o_y = clamp((o.y - world_min.y) / (world_max.y - world_min.y) * grid_size, 0.f, (float)grid_size - 1);
  uint64_t hash_o_z = clamp((o.z - world_min.z) / (world_max.z - world_min.z) * grid_size, 0.f, (float)grid_size - 1);
  return (hash_o_x << (2 * num_bits)) | (hash_o_y << num_bits) | hash_o_z;
}

ray_hash ray_coherence_engine::hash_francois(const Ray &ray) {
  uint32_t num_bits = m_config.hash_francois_bits;
  // Each component has 1 bit sign, `num_bits` mantissa, `num_bits` exponent
  uint32_t num_comp_bits = 2 * num_bits + 1;
  uint64_t hash_d =
    (hash_comp(ray.get_direction().z, num_bits) << (2 * num_comp_bits)) |
    (hash_comp(ray.get_direction().y, num_bits) << num_comp_bits) |
     hash_comp(ray.get_direction().x, num_bits);
  uint64_t hash_o =
    (hash_comp(ray.get_origin().x, num_bits) << (2 * num_comp_bits)) |
    (hash_comp(ray.get_origin().y, num_bits) << num_comp_bits) |
     hash_comp(ray.get_origin().z, num_bits);
  return (ray_hash)hash_o ^ hash_d;
}

ray_hash ray_coherence_engine::hash_grid_spherical(const Ray &ray)
{
  uint32_t num_sphere_bits = m_config.hash_sphere_bits;
  uint32_t num_grid_bits = m_config.hash_grid_bits;
  uint64_t hash_d = hash_direction_spherical(ray.get_direction());
  uint64_t hash_o = hash_origin_grid(ray.get_origin(), num_grid_bits);
  uint64_t hash = hash_o ^ hash_d;

  return (ray_hash)hash;
}

ray_hash ray_coherence_engine::hash_francois_grid_spherical(const Ray &ray) {
  return (ray_hash)hash_grid_spherical(ray) ^ hash_francois(ray);
}

ray_hash ray_coherence_engine::hash_two_point(const Ray &ray) {
  uint64_t hash_1 = hash_origin_grid(ray.get_origin(), m_config.hash_grid_bits);
  float3 d = world_max - world_min;
  float max_extent_length = std::max(std::max(d.x, d.y), d.z);
  float3 est_target = ray.get_origin() + m_config.hash_two_point_est_length_ratio * max_extent_length * ray.get_direction();
  uint64_t hash_2 = hash_origin_grid(est_target, m_config.hash_grid_bits);
  return (ray_hash)hash_1 ^ hash_2;
}

ray_hash ray_coherence_engine::hash_direction_only(const Ray &ray) {
  uint64_t hash_d = hash_direction_spherical(ray.get_direction());
  return (ray_hash)hash_d;
}


unsigned long long ray_coherence_engine::compute_index(ray_hash hash, unsigned num_bits) const {
  uint64_t mask = UINT64_MAX >> (64 - num_bits);

  uint64_t index = 0;
  while (hash > 0) {
    index ^= (hash & mask);
    hash >>= num_bits;
  }
  return index;
}


ray_hash ray_coherence_engine::get_ray_hash(const Ray &ray) {

  switch (m_config.hash) {
    // Francois's hash
    case 'f':
        return hash_francois(ray);

    // Grid-Spherical
    case 'g':
      return hash_grid_spherical(ray);

    // Two-Point
    case 't':
      return hash_two_point(ray);

    // Direction-Only
    case 'd':
      return hash_direction_only(ray);
    
    default:
      assert(0);
  }
}

void ray_coherence_engine::print(FILE *fout) {
  fprintf(fout, "\nRAY_COHERENCE_ENGINE: (%sactive)\n", m_active ? "" : "in");

  fprintf(fout, "Rays (%d/%d):\n", m_total_rays, m_num_ray_pool_rays);
  for (auto it=m_ray_pool.begin(); it!=m_ray_pool.end(); it++) {
    ray_hash hash = it->first;
    fprintf(fout, "[0x%x] (%d)\t", hash, is_stalled(it->second));
    for (coherence_ray ray : it->second) {
      if (!ray.empty())
        fprintf(fout, "w%d:t%d\t", ray.origin_warp_uid, ray.origin_thread_id);
    }
    fprintf(fout, "\n");
  }

  fprintf(fout, "Scheduled Packets (%d):\n", m_num_scheduled_rays);
  for (unsigned i=0; i<m_config.max_packets; i++) {
    if (i == m_schedule_packet_id) fprintf(fout, "*");
    fprintf(fout, "[%d] (%d)\t", i, is_stalled(m_scheduled_packets[i]));
    for (coherence_ray ray : m_scheduled_packets[i]) {
      if (!ray.empty())
        fprintf(fout, "w%d:t%d\t", ray.origin_warp_uid, ray.origin_thread_id);
    }
    fprintf(fout, "\n");
  }

  fprintf(fout, "Outstanding requests:\n");
  for (auto it=m_request_mshr.begin(); it!=m_request_mshr.end(); it++) {
    fprintf(fout, "[0x%x]\t", it->first);
    for (unsigned i : it->second) {
      fprintf(fout, "%d\t", i);
    }
    fprintf(fout, "\n");
  }
}

void ray_coherence_engine::print(ray_hash &hash, FILE *fout) {
  if (m_ray_pool.find(hash)!=m_ray_pool.end()) {
    print(m_ray_pool[hash], fout);
  }
  else {
    fprintf(fout, "0x%x not found!\n", hash);
  }
}

void ray_coherence_engine::print(coherence_packet &packet, FILE *fout) const {
  for (coherence_ray ray : packet) {
    ray.print(fout);
  }
}

void ray_coherence_engine::print_full(FILE *fout) {
  fprintf(fout, "\nRAY_COHERENCE_ENGINE: (%sactive)\n", m_active ? "" : "in");

  fprintf(fout, "Rays (%d):\n", m_total_rays);
  for (auto it=m_ray_pool.begin(); it!=m_ray_pool.end(); it++) {
    ray_hash hash = it->first;
    fprintf(fout, "Hash [0x%x] (%s)\n", hash, is_stalled(it->second) ? "s" : " ");
    print(it->second, fout);
  }

  fprintf(fout, "Scheduled Packets:\n");
  for (unsigned i=0; i<m_config.max_packets; i++) {
    if (i == m_schedule_packet_id) fprintf(fout, "*");
    fprintf(fout, "[%d] (%s)\n", i, is_stalled(m_scheduled_packets[i]) ? "s" : " ");
    print(m_scheduled_packets[i], fout);
  }

  fprintf(fout, "Outstanding requests:\n");
  for (auto it=m_request_mshr.begin(); it!=m_request_mshr.end(); it++) {
    fprintf(fout, "[0x%x]\t", it->first);
    for (unsigned i : it->second) {
      fprintf(fout, "%d\t", i);
    }
    fprintf(fout, "\n");
  }
}

void ray_coherence_engine::print_stats(FILE *fout) {
  m_stats->print(fout);
}


void coherence_stats::print(FILE *fout) {
  fprintf(fout, "coherence_packets = %d\n", total_packets);
  fprintf(fout, "total_rays = %d\n", total_rays);
  fprintf(fout, "max_coherence_rays = %d\n", max_rays);
  fprintf(fout, "active_cycles = %d\n", active_cycles);
  fprintf(fout, "stalled_cycles = %d\n", stalled_cycles);
  fprintf(fout, "total_cycles = %d\n", total_cycles);
  fprintf(fout, "activate_by_rays = %d\n", activate_by_rays);
  fprintf(fout, "activate_by_timer = %d\n", activate_by_timer);

  // Number of rays added to an already scheduled packet (currently stalled)
  fprintf(fout, "stalled_addition = %d\n", stalled_addition);

  // Average stats
  fprintf(fout, "Average Stats:\n");
  for (unsigned i=0; i<(int)coherence_stats_type::TOTAL_TYPES; i++) {
    fprintf(fout, "%.3f\t", avg_stats[i]);
  }
  fprintf(fout, "\n");
}

void coherence_stats::average_stat(coherence_stats_type type, unsigned value) {
  int stat = (int)type;
  float update = avg_stats[stat] * avg_counter[stat];
  update += value;
  avg_counter[stat]++;
  avg_stats[stat] = update / avg_counter[stat];
}