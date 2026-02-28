#pragma once

#include "llama-kv-cache.h"

#include <vector>

//
// llama_kv_cache_paged
//
// Wraps a standard llama_kv_cache and adds block-based management on top.
// The underlying KV cache is allocated as a flat pool, partitioned into
// fixed-size blocks. Sequences are mapped via block tables (logical block
// index -> physical block index), enabling future dynamic allocation and
// Copy-on-Write optimizations.
//
// In this initial implementation the full pool is pre-allocated (pool mode),
// so total memory usage is the same as a standard cache. The key additions
// are block tracking infrastructure and memory utilization reporting.
//
// Context objects are returned directly from the inner llama_kv_cache,
// because the graph builder uses static_cast<llama_kv_cache_context *>
// and requires that exact type.
//

class llama_kv_cache_paged : public llama_memory_i {
  public:
    llama_kv_cache_paged(const llama_model &     model,
                         ggml_type               type_k,
                         ggml_type               type_v,
                         bool                    v_trans,
                         bool                    offload,
                         bool                    unified,
                         uint32_t                kv_size,
                         uint32_t                n_seq_max,
                         uint32_t                n_pad,
                         uint32_t                n_swa,
                         llama_swa_type          swa_type,
                         const layer_filter_cb & filter,
                         const layer_reuse_cb &  reuse,
                         uint32_t                block_size = 256);

    ~llama_kv_cache_paged() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override;

    void clear(bool data) override;

    bool seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) override;
    void seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id) override;
    void seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load
    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read(llama_io_read_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    //
    // llama_kv_cache_paged specific API
    //

    // get the inner KV cache (for graph building via context)
    llama_kv_cache * get_kv() const;

    // block management queries
    uint32_t get_block_size() const;
    uint32_t get_n_blocks_total() const;
    uint32_t get_n_blocks_used() const;
    uint32_t get_n_blocks_free() const;

    // memory utilization ratio: used_blocks / total_blocks
    float get_utilization() const;

  private:
    // the inner KV cache that holds all the actual tensor data
    std::unique_ptr<llama_kv_cache> kv;

    // block management
    uint32_t block_size;  // tokens per block (default: 256)
    uint32_t n_blocks;    // total number of blocks

    // track which blocks contain used cells
    void update_block_usage();

    std::vector<bool> block_used;  // block_used[i] = true if block i has any used cells
};
