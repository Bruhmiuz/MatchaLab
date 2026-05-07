#include "kseq/kseq.h"
#include "common.h"
#include <vector>
#include <string>
#include <cstdint>
#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define MAX_MATCHES 1024 

struct GpuSignatureData {
    uint32_t* d_flat_sigs;
    int* d_sig_word_pos; // Offsets in terms of uint32_t words
    int* d_sig_lens;    // Lengths in terms of characters
    int* d_filter_pos;  // Nibble offset of the filter hit (-1 if unfiltered)
    std::vector<int> h_orig_idx; // CPU array mapping partitioned index back to original
    int num_sigs;
};

struct PotentialMatch {
    int sig_idx;     // Partitioned index
    int nib_pos;     // Char/nibble idx in sample
};

int num_SMs;

__device__ __forceinline__ uint32_t matchBits(uint32_t word1, uint32_t word2) {
    return (word1 ^ word2) & ((word1 & word2) >> 2);
}

__device__ __forceinline__ uint32_t acgt2bits(char c) {
    return (0xE10FD0C0 >> ((c & 0x7) << 2)) & 0xF;
}

// Optimized Bitwise Filters
__device__ __forceinline__ uint32_t ctn1cn4(uint32_t w) {
    return ((w & 0x10000) ^ 0x10000) | ((w & 0x3333) ^ 0x1111);
}

__device__ __forceinline__ uint32_t ct1c4(uint32_t w) {
    return ((w & 0xD0000) ^ 0xD0000) | ((w & 0xFFFF) ^ 0xDDDD);
}

__global__ void kerPreprocessAllSigs(
    const char* __restrict__ d_seq,
    const int* __restrict__ d_char_offsets, 
    const int* __restrict__ d_word_offsets, 
    const int* __restrict__ d_lens,
    uint32_t* __restrict__ d_flat_sigs,
    int* __restrict__ d_filter_pos,
    int num_sigs)
{
    extern __shared__ char s_chars[]; 

    int sig_idx = blockIdx.x;
    if (sig_idx >= num_sigs) return;

    int sig_len = d_lens[sig_idx];
    int char_offset = d_char_offsets[sig_idx];
    int word_offset = d_word_offsets[sig_idx];
    
    // Safety padding applied on host guarantees extra words at the end
    int total_words = (sig_len / 8) + 4; 

    int tid = threadIdx.x;
    int bdim = blockDim.x;
    int chars_per_block = bdim * 8;

    // --- 1. Tile Loop: Pack Bits ---
    for (int tile_word_base = 0; tile_word_base < total_words; tile_word_base += bdim) {
        int tile_char_base = tile_word_base * 8;

        for (int i = tid; i < chars_per_block; i += bdim) {
            int global_char_idx = tile_char_base + i;
            s_chars[i] = (global_char_idx < sig_len) ? d_seq[char_offset + global_char_idx] : 'B'; 
        }
        __syncthreads();

        int w = tile_word_base + tid;
        if (w < total_words) {
            int s_nib_pos = tid * 8; 
            uint32_t bits = 0;

            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                bits = (bits << 4) | acgt2bits(s_chars[s_nib_pos + j]);
            }
            d_flat_sigs[word_offset + w] = bits;
        }
        __syncthreads(); 
    }

    // --- 2. Scan for Filter Position ---
    __shared__ int s_first_hit;
    if (tid == 0) s_first_hit = 999999;
    __syncthreads();

    // Iterate across valid nibble starting positions
    for (int i = tid; i <= sig_len - 8; i += bdim) {
        int w_idx = word_offset + (i / 8);
        int shift = (i % 8) * 4;
        
        uint32_t w0 = d_flat_sigs[w_idx];
        uint32_t w1 = d_flat_sigs[w_idx + 1];
        uint32_t window = __funnelshift_l(w1, w0, shift);

        bool pass = (ct1c4(window) == 0);

        if (pass) {
            atomicMin(&s_first_hit, i);
        }
    }
    __syncthreads();

    if (tid == 0) {
        d_filter_pos[sig_idx] = (s_first_hit == 999999) ? -1 : s_first_hit;
    }
}


// Helper struct for partitioning metadata
struct SigMeta {
    int orig_idx;
    int word_offset;
    int len;
    int filter_pos;
};

GpuSignatureData preprocessAllSigs(const std::vector<klibpp::KSeq>& vec_sigs, cudaStream_t stream) {
    int num_sigs = vec_sigs.size();

    std::vector<int> h_char_offsets(num_sigs);
    std::vector<int> h_word_offsets(num_sigs);
    std::vector<int> h_lens(num_sigs);
    
    size_t total_chars = 0;
    int total_words = 0;

    for (int i = 0; i < num_sigs; ++i) { 
        int len = vec_sigs[i].seq.size();
        h_lens[i] = len;
        h_char_offsets[i] = total_chars;
        total_chars += len;
        h_word_offsets[i] = total_words;
        total_words += (len / 8) + 4; // ADDED PADDING (safeguards the funnelshift later)
    }

    std::vector<char> h_flat_ascii;
    h_flat_ascii.reserve(total_chars);
    for (const auto& sig : vec_sigs) {
        h_flat_ascii.insert(h_flat_ascii.end(), sig.seq.begin(), sig.seq.end());
    }

    char* d_seq;
    int* d_char_offsets;
    int* d_word_offsets_temp;
    int* d_lens_temp;
    uint32_t* d_flat_sigs;
    int* d_filter_pos_temp;

    cudaMallocAsync(&d_seq, total_chars * sizeof(char), stream);
    cudaMallocAsync(&d_char_offsets, num_sigs * sizeof(int), stream);
    cudaMallocAsync(&d_word_offsets_temp, num_sigs * sizeof(int), stream);
    cudaMallocAsync(&d_lens_temp, num_sigs * sizeof(int), stream);
    cudaMallocAsync(&d_flat_sigs, total_words * sizeof(uint32_t), stream);
    cudaMallocAsync(&d_filter_pos_temp, num_sigs * sizeof(int), stream);

    cudaMemcpyAsync(d_seq, h_flat_ascii.data(), total_chars * sizeof(char), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_char_offsets, h_char_offsets.data(), num_sigs * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_word_offsets_temp, h_word_offsets.data(), num_sigs * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_lens_temp, h_lens.data(), num_sigs * sizeof(int), cudaMemcpyHostToDevice, stream);

    kerPreprocessAllSigs<<<num_sigs, 256, 256 * 8 * sizeof(char), stream>>>(
        d_seq, d_char_offsets, d_word_offsets_temp, d_lens_temp, d_flat_sigs, d_filter_pos_temp, num_sigs
    );

    std::vector<int> h_filter_pos(num_sigs);
    cudaMemcpyAsync(h_filter_pos.data(), d_filter_pos_temp, num_sigs * sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // --- CPU METADATA PARTITIONING ---
    std::vector<SigMeta> metas(num_sigs);
    for(int i = 0; i < num_sigs; ++i) {
        metas[i] = {i, h_word_offsets[i], h_lens[i], h_filter_pos[i]};
    }

    // Filtered items (>= 0) first, Unfiltered (-1) last
    std::stable_partition(metas.begin(), metas.end(), [](const SigMeta& a) {
        return a.filter_pos >= 0;
    });

    std::vector<int> h_orig_idx(num_sigs);
    for(int i = 0; i < num_sigs; ++i) {
        h_orig_idx[i] = metas[i].orig_idx;
        h_word_offsets[i] = metas[i].word_offset;
        h_lens[i] = metas[i].len;
        h_filter_pos[i] = metas[i].filter_pos;
    }

    // Allocate & populate permanent partitioned arrays
    GpuSignatureData gpu_data;
    gpu_data.num_sigs = num_sigs;
    gpu_data.h_orig_idx = h_orig_idx;
    gpu_data.d_flat_sigs = d_flat_sigs; 

    cudaMallocAsync(&gpu_data.d_sig_word_pos, num_sigs * sizeof(int), stream);
    cudaMallocAsync(&gpu_data.d_sig_lens, num_sigs * sizeof(int), stream);
    cudaMallocAsync(&gpu_data.d_filter_pos, num_sigs * sizeof(int), stream);

    cudaMemcpyAsync(gpu_data.d_sig_word_pos, h_word_offsets.data(), num_sigs * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(gpu_data.d_sig_lens, h_lens.data(), num_sigs * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(gpu_data.d_filter_pos, h_filter_pos.data(), num_sigs * sizeof(int), cudaMemcpyHostToDevice, stream);

    // Cleanup temp data
    cudaFreeAsync(d_seq, stream);
    cudaFreeAsync(d_char_offsets, stream);
    cudaFreeAsync(d_word_offsets_temp, stream);
    cudaFreeAsync(d_lens_temp, stream);
    cudaFreeAsync(d_filter_pos_temp, stream);

    return gpu_data;
}


__global__ void kerPreprocessSample(const char* __restrict__ d_seq, uint32_t* __restrict__ d_bits, int sample_len) {
    int t_word_pos = blockIdx.x * blockDim.x + threadIdx.x;
    int total_words = (sample_len / 8) + 1; 

    if (t_word_pos < total_words) {
        uint32_t bits = 0;
        int start_idx = t_word_pos * 8;

        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            int global_idx = start_idx + j;
            char c = (global_idx < sample_len) ? d_seq[global_idx] : 'B';
            bits = (bits << 4) | acgt2bits(c);
        }

        d_bits[t_word_pos] = bits;
    }
}


std::vector<uint32_t> preprocessSample(const std::string& sample_seq, cudaStream_t stream) {
    int sample_len = sample_seq.size();
    int total_words = (sample_len / 8) + 1;
    std::vector<uint32_t> sample_bits(total_words);

    char* d_seq;
    uint32_t* d_bits;

    cudaMallocAsync(&d_seq, sample_len * sizeof(char), stream);
    cudaMallocAsync(&d_bits, total_words * sizeof(uint32_t), stream);

    cudaMemcpyAsync(d_seq, sample_seq.data(), sample_len * sizeof(char), cudaMemcpyHostToDevice, stream);
    
    int blockSize = 1024;
    int blocksPerGrid = (total_words + blockSize - 1) / blockSize;

    kerPreprocessSample<<<blocksPerGrid, blockSize, blockSize * 8 * sizeof(char), stream>>>(d_seq, d_bits, sample_len);
    
    cudaMemcpyAsync(sample_bits.data(), d_bits, total_words * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaFreeAsync(d_seq, stream);
    cudaFreeAsync(d_bits, stream);

    return sample_bits;
}


__global__ void kerPotentialMatchas(
    const uint32_t* __restrict__ d_sample, 
    int sample_len, 
    int num_sigs, 
    const uint32_t* __restrict__ d_flat_sigs,
    const int* __restrict__ d_sig_word_pos,
    const int* __restrict__ d_sig_lens,
    const int* __restrict__ d_filter_pos,
    PotentialMatch* d_matches,
    int* d_match_count) 
{    
    extern __shared__ uint32_t shared_mem[];

    uint32_t* s_sample = shared_mem;                             // 131 words
    uint32_t* s_sigs = &shared_mem[131];                         // num_sigs * 3 words
    int* s_sig_lens = (int*)&shared_mem[131 + num_sigs * 3];     // num_sigs ints
    int* s_filter_pos = (int*)&shared_mem[131 + num_sigs * 4];   // num_sigs ints

    int tid = threadIdx.x;
    int bdim = blockDim.x;

    // --- 1. CACHE SIGNATURES (Pre-shifted if filtered) ---
    for (int i = tid; i < num_sigs; i += bdim) {
        int sig_word_pos = d_sig_word_pos[i];
        int f_pos = d_filter_pos[i];
        
        if (f_pos >= 0) {
            int w_idx = sig_word_pos + (f_pos / 8);
            int shift = (f_pos % 8) * 4;
            s_sigs[i * 3 + 0] = __funnelshift_l(d_flat_sigs[w_idx + 1], d_flat_sigs[w_idx], shift);
            s_sigs[i * 3 + 1] = __funnelshift_l(d_flat_sigs[w_idx + 2], d_flat_sigs[w_idx + 1], shift);
            s_sigs[i * 3 + 2] = __funnelshift_l(d_flat_sigs[w_idx + 3], d_flat_sigs[w_idx + 2], shift);
        } else {
            s_sigs[i * 3 + 0] = d_flat_sigs[sig_word_pos];
            s_sigs[i * 3 + 1] = d_flat_sigs[sig_word_pos + 1];
            s_sigs[i * 3 + 2] = d_flat_sigs[sig_word_pos + 2];
        }
        
        s_sig_lens[i] = d_sig_lens[i];
        s_filter_pos[i] = f_pos;
    }
    __syncthreads();

    int max_nibbles_per_chunk = (sample_len + gridDim.x - 1) / gridDim.x;
    max_nibbles_per_chunk = ((max_nibbles_per_chunk + 1023) / 1024) * 1024; 
    
    int start_nib_pos = blockIdx.x * max_nibbles_per_chunk;
    int end_nib_pos = min(start_nib_pos + max_nibbles_per_chunk, sample_len - 24 + 1);
    int total_words = (sample_len >> 3) + 1;

    for (int tile_base_nib = start_nib_pos; tile_base_nib < end_nib_pos; tile_base_nib += 1024) {
        
        for (int i = tid; i < 131; i += bdim) {
            int word_pos = (tile_base_nib >> 3) + i;
            s_sample[i] = (word_pos < total_words) ? d_sample[word_pos] : 0;
        }
        __syncthreads();

        for (int internal_offset = 0; internal_offset < 1024; internal_offset += bdim) {
            int t_nib_pos = tile_base_nib + internal_offset + tid; 

            if (internal_offset + tid < 1024 && t_nib_pos < end_nib_pos) {
                int shm_word_idx = (internal_offset + tid) >> 3;
                int bit_offset   = ((internal_offset + tid) & 7) << 2;

                uint32_t win0 = __funnelshift_l(s_sample[shm_word_idx + 1], s_sample[shm_word_idx], bit_offset);
                uint32_t win1 = __funnelshift_l(s_sample[shm_word_idx + 2], s_sample[shm_word_idx + 1], bit_offset);
                uint32_t win2 = __funnelshift_l(s_sample[shm_word_idx + 3], s_sample[shm_word_idx + 2], bit_offset);

                // Check filter condition
                bool pass_filter = (ctn1cn4(win0) == 0);

                for (int i = 0; i < num_sigs; ++i) {
                    int f_pos = s_filter_pos[i];
                    
                    // Bypass filtered signatures if filter failed
                    if (f_pos >= 0 && !pass_filter) continue;

                    uint32_t mismatch = matchBits(win0, s_sigs[i * 3 + 0]) | 
                                        matchBits(win1, s_sigs[i * 3 + 1]) | 
                                        matchBits(win2, s_sigs[i * 3 + 2]);

                    if (mismatch == 0) {
                        int actual_start = t_nib_pos;
                        if (f_pos >= 0) actual_start -= f_pos; // Subtract filter pos if applicable

                        // Bounds check
                        if (actual_start >= 0 && actual_start + s_sig_lens[i] <= sample_len) {
                            int write_idx = atomicAdd(d_match_count, 1);
                            
                            if (write_idx < MAX_MATCHES) {
                                d_matches[write_idx].sig_idx = i; 
                                d_matches[write_idx].nib_pos = actual_start; 
                            }
                        }
                    }
                }
            }
        }
        __syncthreads(); 
    }
}


std::vector<PotentialMatch> potentialMatchas(
    const std::vector<uint32_t>& sample_bits, 
    int sample_len,
    const GpuSignatureData& sig_data,
    cudaStream_t stream) 
{
    int num_sigs = sig_data.num_sigs;
    
    uint32_t* d_sample;
    PotentialMatch* d_matches;
    int* d_match_count;
    
    cudaMallocAsync(&d_sample, sample_bits.size() * sizeof(uint32_t), stream);
    cudaMallocAsync(&d_matches, MAX_MATCHES * sizeof(PotentialMatch), stream);
    cudaMallocAsync(&d_match_count, sizeof(int), stream);
    cudaMemsetAsync(d_match_count, 0, sizeof(int), stream);
    cudaMemcpyAsync(d_sample, sample_bits.data(), sample_bits.size() * sizeof(uint32_t), cudaMemcpyHostToDevice, stream);

    // Dynamic Sizing: 131 sample words + (sigs * 3 words) + sigs lens + sigs filter_pos
    size_t sharedMemSize = (131 + num_sigs * 5) * sizeof(uint32_t);
    
    int blockSize = 512;
    int gridSize = num_SMs * 3; 

    kerPotentialMatchas<<<gridSize, blockSize, sharedMemSize, stream>>>(
        d_sample, sample_len, num_sigs, 
        sig_data.d_flat_sigs, sig_data.d_sig_word_pos, sig_data.d_sig_lens, sig_data.d_filter_pos,
        d_matches, d_match_count
    );

    int h_match_count = 0;
    cudaMemcpyAsync(&h_match_count, d_match_count, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    int count_to_copy = std::min(h_match_count, MAX_MATCHES);
    std::vector<PotentialMatch> vec_pms(count_to_copy);
    
    if (count_to_copy > 0) {
        cudaMemcpyAsync(vec_pms.data(), d_matches, count_to_copy * sizeof(PotentialMatch), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream); 
    }

    cudaFreeAsync(d_sample, stream);
    cudaFreeAsync(d_matches, stream);
    cudaFreeAsync(d_match_count, stream);

    return vec_pms;
}


__global__ void kerConfirmMatchas(
    const PotentialMatch* __restrict__ d_pms,
    int num_pms,
    const uint32_t* __restrict__ d_sample_bits,
    const char* __restrict__ d_phred,
    const uint32_t* __restrict__ d_flat_sigs,
    const int* __restrict__ d_sig_word_pos,
    const int* __restrict__ d_sig_lens,
    int* __restrict__ d_results) 
{
    int pm_idx = blockIdx.x; 
    if (pm_idx >= num_pms) return;

    int tid = threadIdx.x;

    __shared__ int s_mismatch_flag;
    __shared__ int s_sum[1024]; 

    if (tid == 0) s_mismatch_flag = 0;
    s_sum[tid] = 0;
    __syncthreads();

    int sig_idx = d_pms[pm_idx].sig_idx;
    int sig_word_pos = d_sig_word_pos[sig_idx];
    int sig_len = d_sig_lens[sig_idx];
    
    int start_nib_pos = d_pms[pm_idx].nib_pos;
    int start_bit_pos = start_nib_pos * 4;
    int start_word_pos = start_nib_pos / 8;
    int bit_offset = start_bit_pos % 32;
    int total_search_bits = sig_len * 4;
    int total_search_words = (total_search_bits + 31) / 32;

    for (int t_word_num = tid; t_word_num < total_search_words; t_word_num += blockDim.x) {
        if (s_mismatch_flag) break; 
        
        int t_word_pos = start_word_pos + t_word_num;
        uint32_t sample_window = __funnelshift_l(d_sample_bits[t_word_pos + 1], d_sample_bits[t_word_pos], bit_offset);
        uint32_t sig_window = d_flat_sigs[sig_word_pos + t_word_num];

        int bits_remaining = total_search_bits - (t_word_num * 32);
        if (bits_remaining < 32) {
            uint32_t mask = 0xFFFFFFFF << (32 - bits_remaining);
            sig_window &= mask;
            sample_window &= mask;
        }

        if (matchBits(sample_window, sig_window)) {
            atomicExch(&s_mismatch_flag, 1); 
        }
    }
    __syncthreads(); 

    if (s_mismatch_flag) {
        if (tid == 0) d_results[pm_idx] = -1;
        return; 
    }

    int local_sum = 0;
    for (int i = tid; i < sig_len; i += blockDim.x) {
        local_sum += static_cast<int>(d_phred[start_nib_pos + i]);
    }
    s_sum[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_results[pm_idx] = s_sum[0] - (33 * sig_len);
    }
}


__global__ void kerCalcIntegrity(const char* __restrict__ d_phred, int len, int* __restrict__ d_total_sum) {
    __shared__ int s_sum[1024];
    
    int tid = threadIdx.x;
    int local_sum = 0;

    for (int i = blockIdx.x * blockDim.x + tid; i < len; i += gridDim.x * blockDim.x) {
        local_sum += static_cast<unsigned char>(d_phred[i]);
    }
    
    s_sum[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(d_total_sum, s_sum[0]);
    }
}


void runMatcher(const std::vector<klibpp::KSeq>& vec_samples, const std::vector<klibpp::KSeq>& vec_sigs, std::vector<MatchResult>& matches) {
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceId);
    num_SMs = props.multiProcessorCount;

    cudaStream_t stream_match;
    cudaStream_t stream_integrity;
    cudaEvent_t phred_ready_event;
    
    cudaStreamCreate(&stream_match);
    cudaStreamCreate(&stream_integrity);
    cudaEventCreate(&phred_ready_event);

    int num_sigs = vec_sigs.size();

    GpuSignatureData sig_data = preprocessAllSigs(vec_sigs, stream_match);

    for (size_t sample_idx = 0; sample_idx < vec_samples.size(); sample_idx++) {
        const auto& sample = vec_samples[sample_idx];
        const std::string& phred = sample.qual;
        int sample_len = static_cast<int>(sample.seq.size());
        
        auto sample_bits = preprocessSample(sample.seq, stream_match);

        auto vec_pms = potentialMatchas(sample_bits, sample_len, sig_data, stream_match); 
        int num_pms = vec_pms.size();

        if (num_pms == 0) continue;

        int* d_results;
        PotentialMatch* d_pms;
        uint32_t* d_sample_bits;
        char* d_phred; 
        int* d_total_sum;

        cudaMallocAsync(&d_results, num_pms * sizeof(int), stream_match);
        cudaMallocAsync(&d_pms, num_pms * sizeof(PotentialMatch), stream_match);
        cudaMallocAsync(&d_sample_bits, sample_bits.size() * sizeof(uint32_t), stream_match);
        cudaMallocAsync(&d_phred, sample_len * sizeof(char), stream_match);
        
        cudaMallocAsync(&d_total_sum, sizeof(int), stream_integrity);
        cudaMemsetAsync(d_total_sum, 0, sizeof(int), stream_integrity);

        cudaMemcpyAsync(d_pms, vec_pms.data(), num_pms * sizeof(PotentialMatch), cudaMemcpyHostToDevice, stream_match);
        cudaMemcpyAsync(d_sample_bits, sample_bits.data(), sample_bits.size() * sizeof(uint32_t), cudaMemcpyHostToDevice, stream_match);
        cudaMemcpyAsync(d_phred, phred.data(), sample_len * sizeof(char), cudaMemcpyHostToDevice, stream_match);

        cudaEventRecord(phred_ready_event, stream_match);

        int threadsPerBlock = 1024;
        int blocksPerGridConfirm = num_pms;
        int blocksPerGridInteg = num_SMs * 2 - num_pms;

        kerConfirmMatchas<<<blocksPerGridConfirm, threadsPerBlock, 0, stream_match>>>(
            d_pms, num_pms, d_sample_bits, d_phred, 
            sig_data.d_flat_sigs, sig_data.d_sig_word_pos, sig_data.d_sig_lens, 
            d_results
        );

        cudaStreamWaitEvent(stream_integrity, phred_ready_event, 0);
        
        kerCalcIntegrity<<<blocksPerGridInteg, threadsPerBlock, 0, stream_integrity>>>(
            d_phred, sample_len, d_total_sum
        );

        std::vector<int> h_results(num_pms, -1);
        int h_total_sum = 0;

        cudaMemcpyAsync(h_results.data(), d_results, num_pms * sizeof(int), cudaMemcpyDeviceToHost, stream_match);
        cudaMemcpyAsync(&h_total_sum, d_total_sum, sizeof(int), cudaMemcpyDeviceToHost, stream_integrity);

        cudaStreamSynchronize(stream_match);
        cudaStreamSynchronize(stream_integrity);

        cudaFreeAsync(d_results, stream_match);
        cudaFreeAsync(d_pms, stream_match);
        cudaFreeAsync(d_sample_bits, stream_match);
        cudaFreeAsync(d_phred, stream_match);
        cudaFreeAsync(d_total_sum, stream_integrity);

        int integrity = (h_total_sum - 33 * sample_len) % 97;

        std::vector<int> best_scores(num_sigs, -1);
        int valid_match_count = 0;

        for (size_t i = 0; i < h_results.size(); i++) {
            int match_score = h_results[i];
            if (match_score < 0) continue; 

            // Extract partitioned idx, map back to original sig idx
            int part_idx = vec_pms[i].sig_idx;
            int orig_idx = sig_data.h_orig_idx[part_idx];

            best_scores[orig_idx] = std::max(best_scores[orig_idx], match_score);
            valid_match_count++;
        }

        if (valid_match_count == 0 || integrity == -1) continue;

        for (int sig_idx = 0; sig_idx < num_sigs; sig_idx++) {
            if (best_scores[sig_idx] > 0) { 
                matches.push_back(MatchResult{
                    sample.name,
                    vec_sigs[sig_idx].name,
                    static_cast<double>(best_scores[sig_idx]) / vec_sigs[sig_idx].seq.size(),
                    integrity
                }); 
            }
        }
    }

    cudaFreeAsync(sig_data.d_flat_sigs, stream_match);
    cudaFreeAsync(sig_data.d_sig_word_pos, stream_match);
    cudaFreeAsync(sig_data.d_sig_lens, stream_match);
    cudaFreeAsync(sig_data.d_filter_pos, stream_match);
    
    cudaEventDestroy(phred_ready_event);
    cudaStreamDestroy(stream_match);
    cudaStreamDestroy(stream_integrity);
}