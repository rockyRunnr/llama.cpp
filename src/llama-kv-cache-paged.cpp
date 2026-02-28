#include "llama-kv-cache-paged.h"

#include "llama-batch.h"
#include "llama-impl.h"
#include "llama-model.h"

#include <algorithm>
#include <cassert>

//
// llama_kv_cache_paged
//

llama_kv_cache_paged::llama_kv_cache_paged(const llama_model &     model,
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
                                           uint32_t                block_size) :
    block_size(block_size) {
    // align kv_size to block_size boundary
    const uint32_t kv_size_aligned = ((kv_size + block_size - 1) / block_size) * block_size;

    n_blocks = kv_size_aligned / block_size;

    LLAMA_LOG_INFO("%s: creating paged KV cache, size = %u cells, block_size = %u, n_blocks = %u\n", __func__,
                   kv_size_aligned, block_size, n_blocks);

    // create the inner KV cache with aligned size
    // the inner cache is a flat pool — we manage block-level tracking on top
    kv = std::make_unique<llama_kv_cache>(model, type_k, type_v, v_trans, offload, unified, kv_size_aligned, n_seq_max,
                                          n_pad, n_swa, swa_type, filter, reuse);

    // initialize block usage tracking
    block_used.resize(n_blocks, false);

    LLAMA_LOG_INFO("%s: paged KV cache ready, %u blocks x %u tokens = %u total capacity\n", __func__, n_blocks,
                   block_size, kv_size_aligned);
}

void llama_kv_cache_paged::clear(bool data) {
    kv->clear(data);

    std::fill(block_used.begin(), block_used.end(), false);
}

bool llama_kv_cache_paged::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    bool res = kv->seq_rm(seq_id, p0, p1);

    update_block_usage();

    return res;
}

void llama_kv_cache_paged::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    kv->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_kv_cache_paged::seq_keep(llama_seq_id seq_id) {
    kv->seq_keep(seq_id);

    update_block_usage();
}

void llama_kv_cache_paged::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    kv->seq_add(seq_id, p0, p1, shift);
}

void llama_kv_cache_paged::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    kv->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_kv_cache_paged::seq_pos_min(llama_seq_id seq_id) const {
    return kv->seq_pos_min(seq_id);
}

llama_pos llama_kv_cache_paged::seq_pos_max(llama_seq_id seq_id) const {
    return kv->seq_pos_max(seq_id);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache_paged::memory_breakdown() const {
    return kv->memory_breakdown();
}

llama_memory_context_ptr llama_kv_cache_paged::init_batch(llama_batch_allocr & balloc,
                                                          uint32_t             n_ubatch,
                                                          bool                 embd_all) {
    // delegate directly to the inner cache
    // the inner cache returns a llama_kv_cache_context which is what
    // the graph builder expects via static_cast
    return kv->init_batch(balloc, n_ubatch, embd_all);
}

llama_memory_context_ptr llama_kv_cache_paged::init_full() {
    return kv->init_full();
}

llama_memory_context_ptr llama_kv_cache_paged::init_update(llama_context * lctx, bool optimize) {
    return kv->init_update(lctx, optimize);
}

bool llama_kv_cache_paged::get_can_shift() const {
    return kv->get_can_shift();
}

void llama_kv_cache_paged::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    kv->state_write(io, seq_id, flags);
}

void llama_kv_cache_paged::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    kv->state_read(io, seq_id, flags);

    update_block_usage();
}

llama_kv_cache * llama_kv_cache_paged::get_kv() const {
    return kv.get();
}

uint32_t llama_kv_cache_paged::get_block_size() const {
    return block_size;
}

uint32_t llama_kv_cache_paged::get_n_blocks_total() const {
    return n_blocks;
}

uint32_t llama_kv_cache_paged::get_n_blocks_used() const {
    uint32_t count = 0;
    for (bool used : block_used) {
        if (used) {
            count++;
        }
    }
    return count;
}

uint32_t llama_kv_cache_paged::get_n_blocks_free() const {
    return n_blocks - get_n_blocks_used();
}

float llama_kv_cache_paged::get_utilization() const {
    if (n_blocks == 0) {
        return 0.0f;
    }
    return (float) get_n_blocks_used() / (float) n_blocks;
}

void llama_kv_cache_paged::update_block_usage() {
    // scan the inner cache's cells to determine which blocks are in use
    // a block is "used" if it contains any non-empty cells
    //
    // NOTE: this is O(kv_size) but only called on seq_rm/seq_keep/state_read
    //       operations, not on the hot path

    std::fill(block_used.begin(), block_used.end(), false);

    // conservative: mark all blocks as used if cache has any capacity
    // TODO: implement proper per-block usage tracking by integrating with
    //       the inner cache's cell state
    const uint32_t total = kv->get_size();
    if (total > 0) {
        for (uint32_t b = 0; b < n_blocks; ++b) {
            block_used[b] = true;
        }
    }
}
