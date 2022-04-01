#ifndef RAY_COHERENCE_INCLUDED
#define RAY_COHERENCE_INCLUDED

#include "../abstract_hardware_model.h"
#include <cmath>
#include "vector-math.h"

typedef uint64_t(*HashFunc)(const Ray&, const float3&, const float3&);


struct {
  Ray ray_properties;
  std::deque<RTMemoryTransactionRecord> RT_mem_accesses;
  unsigned origin_warp_uid;
  unsigned origin_thread_id;
  unsigned latency_delay;

  bool empty() {
    return RT_mem_accesses.empty();
  }
  void print(FILE* fout) {
    fprintf(fout, "\t[%d:%d] [%d]- ", origin_warp_uid, origin_thread_id, latency_delay);
    for (RTMemoryTransactionRecord record : RT_mem_accesses) {
      fprintf(fout, "0x%x (%d-%s-<%s>)\t", record.address, record.size, record.status == RT_MEM_AWAITING ? "A" : "U", record.mem_chunks.to_string().c_str()); 
    }
    fprintf(fout, "\n");
  }
  RTMemoryTransactionRecord next_access() {
    assert(!RT_mem_accesses.empty());
    return RT_mem_accesses.front();
  }
  new_addr_type next_addr() {
    assert(!RT_mem_accesses.empty());
    return RT_mem_accesses.front().address;
  }
  RTMemStatus next_status() {
    assert(!RT_mem_accesses.empty());
    return RT_mem_accesses.front().status;
  }
} typedef coherence_ray;

typedef std::pair<new_addr_type, unsigned> addr_size_pair;
typedef std::deque<coherence_ray> coherence_packet;
typedef unsigned long long ray_hash;

enum class coherence_stats_type {
  COALESCED_REQUESTS = 0,
  ACTIVE_PACKETS,
  TOTAL_TYPES
};

class coherence_stats {
  public:
    coherence_stats() {
      total_packets = 0;
      total_rays = 0;
      max_rays = 0;
      total_cycles = 0;
      stalled_cycles = 0;
      active_cycles = 0;
      activate_by_rays = 0;
      activate_by_timer = 0;
      stalled_addition = 0;
    }
    ~coherence_stats();

    void print(FILE *fout);
    void average_stat(coherence_stats_type type, unsigned value);

    unsigned total_packets;
    unsigned total_rays;
    unsigned max_rays;
    unsigned long long total_cycles;
    unsigned long long stalled_cycles;
    unsigned long long active_cycles;
    unsigned activate_by_rays;
    unsigned activate_by_timer;
    unsigned stalled_addition;
  
  private:
    unsigned avg_counter[(int)coherence_stats_type::TOTAL_TYPES] = {0};
    float avg_stats[(int)coherence_stats_type::TOTAL_TYPES] = {0};
};

class ray_coherence_engine {
  public:
    ray_coherence_engine(unsigned sid, struct ray_coherence_config config, coherence_stats *stats, shader_core_ctx *core);
    ~ray_coherence_engine();
    
    void cycle();
    void insert(warp_inst_t new_warp);
    unsigned schedule_next_warp();
    RTMemoryTransactionRecord get_next_access();
    void undo_access(new_addr_type addr);
    void process_response(mem_fetch *mf, std::map<unsigned, warp_inst_t *> &m_current_warps, warp_inst_t *pipe_reg);
    void dec_thread_latency();

    // Backwards pointer
    shader_core_ctx *m_core;
    unsigned m_sid;
    ray_coherence_config m_config;

    bool m_initialized;

    void set_world(float3 min, float3 max);
    bool active() const { return m_active; }
    void print_full(FILE *fout);
    void print(FILE *fout);
    void print(ray_hash &hash, FILE *fout);
    void print(coherence_packet &packet, FILE *fout) const;
    void print_stats(FILE *fout);

  private:
    unsigned m_total_rays;
    unsigned m_num_ray_pool_rays;
    unsigned m_num_scheduled_rays;
    unsigned long long m_last_insertion_cycle;

    // map [hash]->[group of rays]
    std::map<ray_hash, coherence_packet> m_ray_pool;

    std::vector<coherence_packet> m_scheduled_packets;

    // map [addr]->[set of hashes (coherence_packets)]
    std::map<new_addr_type, std::set<unsigned> > m_request_mshr;

    float3 world_min;
    float3 world_max;

    coherence_stats* m_stats;

    bool m_active;
    unsigned m_schedule_packet_id;

    ray_hash m_active_hash;
    unsigned m_active_warp;
    unsigned m_active_thread;
    RTMemoryTransactionRecord m_active_record;

    bool is_empty(coherence_packet packet);
    bool is_stalled();
    bool is_stalled(coherence_packet packet);
    bool check_scheduled();
    bool scheduled_full();

    coherence_packet * get_largest_packet(ray_hash &hash);
    unsigned long long compute_index(ray_hash hash, unsigned num_bits) const;
    ray_hash get_ray_hash(const Ray &ray);

    // Hash functions
    uint64_t hash_comp(float x, uint32_t num_bits);
    uint64_t hash_direction_spherical(const float3 &d);
    uint64_t hash_origin_grid(const float3& o, uint32_t num_bits);

    ray_hash hash_francois(const Ray &ray);
    ray_hash hash_grid_spherical(const Ray &ray);
    ray_hash hash_francois_grid_spherical(const Ray &ray);
    ray_hash hash_two_point(const Ray &ray);
    ray_hash hash_direction_only(const Ray &ray);

};

#endif
