/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 *
 * Fused DFlash context K RMSNorm + RoPE kernel.
 *
 * Replaces the DFlash precompute sequence:
 *
 *   for layer in layers:
 *       k_normed[layer] = rms_norm(k[layer], k_norm_weight[layer])
 *   rotary_embedding(context_positions.repeat(num_layers), k_normed.view(...))
 *
 * with one warp per (layer, token, kv_head). Cache insertion intentionally
 * stays in the backend-specific do_kv_cache_update path.
 */

#include <cmath>
#include <cuda_runtime.h>
#include <type_traits>

#include "torch_utils.h"

#include "../cuda_compat.h"
#include "../type_convert.cuh"
#include "dispatch_utils.h"

#define CHECK_TYPE(x, st)                                                  \
  STD_TORCH_CHECK(x.scalar_type() == st, #x " dtype is ", x.scalar_type(), \
                  ", while ", st, " is expected")
#define CHECK_TH_CUDA(x) \
  STD_TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  STD_TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_TH_CUDA(x);    \
  CHECK_CONTIGUOUS(x)

#ifdef USE_ROCM
  #define FINAL_MASK 0xffffffffffffffffULL
#else
  #define FINAL_MASK 0xffffffff
#endif

namespace vllm::dflash_fused_ops {

template <typename T, int num>
struct packed_as;

template <>
struct packed_as<uint, 1> {
  using type = uint;
};

template <>
struct packed_as<uint, 2> {
  using type = uint2;
};

template <>
struct packed_as<uint, 4> {
  using type = uint4;
};

template <typename T>
__inline__ __device__ T warpReduceSum(T val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  }
  return val;
}

template <typename scalar_t_in, typename scalar_t_cache, int head_dim,
          bool interleave>
__global__ void dflashKNormRopeKernel(
    void const* all_k_void, void* all_k_out_void,
    void const* k_norm_weights_void, void const* cos_sin_cache_void,
    int64_t const* positions, int const num_layers, int const num_ctx,
    int const num_kv_heads, int64_t const all_k_stride0,
    int64_t const all_k_stride1, int64_t const all_k_stride2,
    float const eps, int const rotary_dim) {
#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800) && !defined(USE_ROCM)
  if constexpr ((std::is_same_v<scalar_t_in, c10::BFloat16>) ||
                std::is_same_v<scalar_t_cache, c10::BFloat16>) {
    return;
  } else {
#endif
    using Converter = vllm::_typeConvert<scalar_t_in>;
    static_assert(Converter::exists,
                  "Input dtype is not supported for this CUDA architecture.");
    using T_in = typename Converter::hip_type;
    using T2_in = typename Converter::packed_hip_type;

    using CacheConverter = vllm::_typeConvert<scalar_t_cache>;
    static_assert(CacheConverter::exists,
                  "RoPE cache dtype is not supported for this CUDA architecture.");
    using T_cache = typename CacheConverter::hip_type;

    T_in const* all_k = reinterpret_cast<T_in const*>(all_k_void);
    T_in* all_k_out = reinterpret_cast<T_in*>(all_k_out_void);
    T_in const* k_norm_weights =
        reinterpret_cast<T_in const*>(k_norm_weights_void);
    T_cache const* cos_sin_cache =
        reinterpret_cast<T_cache const*>(cos_sin_cache_void);

    int const warps_per_block = blockDim.x / 32;
    int const warp_id = threadIdx.x / 32;
    int const lane_id = threadIdx.x % 32;
    int const global_warp_idx = blockIdx.x * warps_per_block + warp_id;

    int const heads_per_token = num_kv_heads;
    int const heads_per_layer = num_ctx * heads_per_token;
    int const total_heads = num_layers * heads_per_layer;
    if (global_warp_idx >= total_heads) return;

    int const layer_idx = global_warp_idx / heads_per_layer;
    int const rem = global_warp_idx % heads_per_layer;
    int const token_idx = rem / heads_per_token;
    int const head_idx = rem % heads_per_token;

    static_assert(head_dim % (32 * 2) == 0,
                  "head_dim must be divisible by 64.");
    constexpr int num_elems_per_thread = head_dim / 32;
    constexpr int elem_size_bytes = num_elems_per_thread * sizeof(__nv_bfloat16);
    static_assert(elem_size_bytes % 4 == 0,
                  "elem_size_bytes must be a multiple of 4.");
    constexpr int vec_size = elem_size_bytes / 4;
    using vec_T = typename packed_as<uint, vec_size>::type;

    int64_t const in_row_offset =
        static_cast<int64_t>(layer_idx) * all_k_stride0 +
        static_cast<int64_t>(token_idx) * all_k_stride1 +
        static_cast<int64_t>(head_idx) * all_k_stride2;
    int64_t const out_row_offset =
        (((static_cast<int64_t>(layer_idx) * num_ctx + token_idx) *
              num_kv_heads +
          head_idx) *
         head_dim);
    int64_t const in_thread_offset =
        in_row_offset + lane_id * num_elems_per_thread;
    int64_t const out_thread_offset =
        out_row_offset + lane_id * num_elems_per_thread;

    float elements[num_elems_per_thread];
    float sum_squares = 0.0f;

    {
      vec_T vec = *reinterpret_cast<vec_T const*>(&all_k[in_thread_offset]);
      constexpr int num_packed_elems = elem_size_bytes / sizeof(T2_in);
#pragma unroll
      for (int i = 0; i < num_packed_elems; i++) {
        T2_in packed_val = *(reinterpret_cast<T2_in*>(&vec) + i);
        float2 vals = Converter::convert(packed_val);
        sum_squares += vals.x * vals.x;
        sum_squares += vals.y * vals.y;
        elements[2 * i] = vals.x;
        elements[2 * i + 1] = vals.y;
      }
    }

    sum_squares = warpReduceSum(sum_squares);
    float const rms_rcp =
        rsqrtf(sum_squares / static_cast<float>(head_dim) + eps);

#pragma unroll
    for (int i = 0; i < num_elems_per_thread; i++) {
      int const dim = lane_id * num_elems_per_thread + i;
      float const weight =
          Converter::convert(k_norm_weights[layer_idx * head_dim + dim]);
      elements[i] *= rms_rcp * weight;
    }

    // Match the unfused path's materialization boundary:
    // ops.rms_norm writes K back to half/bfloat before rotary_embedding reads it.
#pragma unroll
    for (int i = 0; i < num_elems_per_thread; i += 2) {
      T2_in rounded =
          Converter::convert(make_float2(elements[i], elements[i + 1]));
      float2 vals = Converter::convert(rounded);
      elements[i] = vals.x;
      elements[i + 1] = vals.y;
    }

    int64_t const pos_id = positions[token_idx];
    T_cache const* cache_ptr = cos_sin_cache + pos_id * rotary_dim;
    int const embed_dim = rotary_dim / 2;
    T_cache const* cos_ptr = cache_ptr;
    T_cache const* sin_ptr = cache_ptr + embed_dim;
    int const rotary_lanes = rotary_dim / num_elems_per_thread;

    if (lane_id < rotary_lanes) {
      if constexpr (interleave) {
#pragma unroll
        for (int i = 0; i < num_elems_per_thread / 2; ++i) {
          int const idx0 = 2 * i;
          int const idx1 = 2 * i + 1;
          int const dim_idx = lane_id * num_elems_per_thread + idx0;
          float const val0 = elements[idx0];
          float const val1 = elements[idx1];
          int const half_dim = dim_idx / 2;
          float const cos_val =
              CacheConverter::convert(VLLM_LDG(cos_ptr + half_dim));
          float const sin_val =
              CacheConverter::convert(VLLM_LDG(sin_ptr + half_dim));
          elements[idx0] = val0 * cos_val - val1 * sin_val;
          elements[idx1] = val0 * sin_val + val1 * cos_val;
        }
      } else {
        __syncwarp();
        int const pair_offset = (rotary_dim / 2) / num_elems_per_thread;
        float elements2[num_elems_per_thread];
#pragma unroll
        for (int i = 0; i < num_elems_per_thread; i++) {
          elements2[i] =
              __shfl_xor_sync(FINAL_MASK, elements[i], pair_offset);
          if (lane_id < pair_offset) {
            elements2[i] = -elements2[i];
          }
          int dim_idx = lane_id * num_elems_per_thread + i;
          dim_idx = (dim_idx * 2) % rotary_dim;
          int const half_dim = dim_idx / 2;
          float const cos_val =
              CacheConverter::convert(VLLM_LDG(cos_ptr + half_dim));
          float const sin_val =
              CacheConverter::convert(VLLM_LDG(sin_ptr + half_dim));
          elements[i] = elements[i] * cos_val + elements2[i] * sin_val;
        }
        __syncwarp();
      }
    }

    {
      vec_T vec;
      constexpr int num_packed_elems = elem_size_bytes / sizeof(T2_in);
#pragma unroll
      for (int i = 0; i < num_packed_elems; i++) {
        T2_in packed_val = Converter::convert(
            make_float2(elements[2 * i], elements[2 * i + 1]));
        *(reinterpret_cast<T2_in*>(&vec) + i) = packed_val;
      }
      *reinterpret_cast<vec_T*>(&all_k_out[out_thread_offset]) = vec;
    }
#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800) && !defined(USE_ROCM)
  }
#endif
}

template <typename scalar_t_in, typename scalar_t_cache>
void launchDFlashKNormRope(
    void const* all_k, void* all_k_out, void const* k_norm_weights,
    void const* cos_sin_cache, int64_t const* positions, int num_layers,
    int num_ctx, int num_kv_heads, int head_dim, int64_t all_k_stride0,
    int64_t all_k_stride1, int64_t all_k_stride2, int rotary_dim, float eps,
    bool interleave, cudaStream_t stream) {
  constexpr int block_size = 256;
  constexpr int warps_per_block = block_size / 32;
  int const total_warps = num_layers * num_ctx * num_kv_heads;
  dim3 grid((total_warps + warps_per_block - 1) / warps_per_block);
  dim3 block(block_size);

#define LAUNCH_HEAD_DIM(HEAD_DIM)                                          \
  do {                                                                     \
    if (interleave) {                                                      \
      dflashKNormRopeKernel<scalar_t_in, scalar_t_cache, (HEAD_DIM), true> \
          <<<grid, block, 0, stream>>>(                                    \
              all_k, all_k_out, k_norm_weights, cos_sin_cache, positions,  \
              num_layers, num_ctx, num_kv_heads, all_k_stride0,             \
              all_k_stride1, all_k_stride2, eps, rotary_dim);              \
    } else {                                                               \
      dflashKNormRopeKernel<scalar_t_in, scalar_t_cache, (HEAD_DIM),       \
                            false><<<grid, block, 0, stream>>>(           \
          all_k, all_k_out, k_norm_weights, cos_sin_cache, positions,      \
          num_layers, num_ctx, num_kv_heads, all_k_stride0, all_k_stride1, \
          all_k_stride2, eps, rotary_dim);                                 \
    }                                                                      \
  } while (0)

  switch (head_dim) {
    case 64:
      LAUNCH_HEAD_DIM(64);
      break;
    case 128:
      LAUNCH_HEAD_DIM(128);
      break;
    case 256:
      LAUNCH_HEAD_DIM(256);
      break;
    default:
      STD_TORCH_CHECK(false, "Unsupported DFlash head dimension: ", head_dim);
  }
#undef LAUNCH_HEAD_DIM
}

template <typename scalar_t_in, typename scalar_t_cache, int head_dim,
          bool interleave>
__global__ void dflashKNormRopeCacheUpdateKernel(
    void const* all_k_void, void const* all_v_void, void* key_cache_void,
    void* value_cache_void, void const* k_norm_weight_void,
    void const* cos_sin_cache_void, int64_t const* positions,
    int64_t const* slot_mapping, int const num_ctx, int const num_kv_heads,
    int64_t const all_k_stride0, int64_t const all_k_stride1,
    int64_t const all_v_stride0, int64_t const all_v_stride1,
    int64_t const key_cache_stride0, int64_t const key_cache_stride1,
    int64_t const key_cache_stride2, int64_t const value_cache_stride0,
    int64_t const value_cache_stride1, int64_t const value_cache_stride2,
    int const block_size, float const eps, int const rotary_dim) {
#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800) && !defined(USE_ROCM)
  if constexpr ((std::is_same_v<scalar_t_in, c10::BFloat16>) ||
                std::is_same_v<scalar_t_cache, c10::BFloat16>) {
    return;
  } else {
#endif
    using Converter = vllm::_typeConvert<scalar_t_in>;
    static_assert(Converter::exists,
                  "Input dtype is not supported for this CUDA architecture.");
    using T_in = typename Converter::hip_type;
    using T2_in = typename Converter::packed_hip_type;

    using CacheConverter = vllm::_typeConvert<scalar_t_cache>;
    static_assert(CacheConverter::exists,
                  "RoPE cache dtype is not supported for this CUDA architecture.");
    using T_cache = typename CacheConverter::hip_type;

    T_in const* all_k = reinterpret_cast<T_in const*>(all_k_void);
    T_in const* all_v = reinterpret_cast<T_in const*>(all_v_void);
    T_in* key_cache = reinterpret_cast<T_in*>(key_cache_void);
    T_in* value_cache = reinterpret_cast<T_in*>(value_cache_void);
    T_in const* k_norm_weight =
        reinterpret_cast<T_in const*>(k_norm_weight_void);
    T_cache const* cos_sin_cache =
        reinterpret_cast<T_cache const*>(cos_sin_cache_void);

    int const warps_per_block = blockDim.x / 32;
    int const warp_id = threadIdx.x / 32;
    int const lane_id = threadIdx.x % 32;
    int const global_warp_idx = blockIdx.x * warps_per_block + warp_id;
    int const total_heads = num_ctx * num_kv_heads;
    if (global_warp_idx >= total_heads) return;

    int const token_idx = global_warp_idx / num_kv_heads;
    int const head_idx = global_warp_idx % num_kv_heads;
    int64_t const slot_idx = slot_mapping[token_idx];
    if (slot_idx < 0) return;

    int64_t const block_idx = slot_idx / block_size;
    int64_t const block_offset = slot_idx % block_size;

    static_assert(head_dim % (32 * 2) == 0,
                  "head_dim must be divisible by 64.");
    constexpr int num_elems_per_thread = head_dim / 32;
    constexpr int elem_size_bytes = num_elems_per_thread * sizeof(__nv_bfloat16);
    static_assert(elem_size_bytes % 4 == 0,
                  "elem_size_bytes must be a multiple of 4.");
    constexpr int vec_size = elem_size_bytes / 4;
    using vec_T = typename packed_as<uint, vec_size>::type;

    int64_t const k_row_offset =
        static_cast<int64_t>(token_idx) * all_k_stride0 +
        static_cast<int64_t>(head_idx) * all_k_stride1;
    int64_t const v_row_offset =
        static_cast<int64_t>(token_idx) * all_v_stride0 +
        static_cast<int64_t>(head_idx) * all_v_stride1;
    int64_t const cache_row_offset =
        block_idx * key_cache_stride0 + block_offset * key_cache_stride1 +
        static_cast<int64_t>(head_idx) * key_cache_stride2;
    int64_t const value_cache_row_offset =
        block_idx * value_cache_stride0 + block_offset * value_cache_stride1 +
        static_cast<int64_t>(head_idx) * value_cache_stride2;
    int64_t const dim_offset = lane_id * num_elems_per_thread;

    float elements[num_elems_per_thread];
    float sum_squares = 0.0f;

    {
      vec_T vec =
          *reinterpret_cast<vec_T const*>(&all_k[k_row_offset + dim_offset]);
      constexpr int num_packed_elems = elem_size_bytes / sizeof(T2_in);
#pragma unroll
      for (int i = 0; i < num_packed_elems; i++) {
        T2_in packed_val = *(reinterpret_cast<T2_in*>(&vec) + i);
        float2 vals = Converter::convert(packed_val);
        sum_squares += vals.x * vals.x;
        sum_squares += vals.y * vals.y;
        elements[2 * i] = vals.x;
        elements[2 * i + 1] = vals.y;
      }
    }

    sum_squares = warpReduceSum(sum_squares);
    float const rms_rcp =
        rsqrtf(sum_squares / static_cast<float>(head_dim) + eps);

#pragma unroll
    for (int i = 0; i < num_elems_per_thread; i++) {
      int const dim = lane_id * num_elems_per_thread + i;
      float const weight = Converter::convert(k_norm_weight[dim]);
      elements[i] *= rms_rcp * weight;
    }

#pragma unroll
    for (int i = 0; i < num_elems_per_thread; i += 2) {
      T2_in rounded =
          Converter::convert(make_float2(elements[i], elements[i + 1]));
      float2 vals = Converter::convert(rounded);
      elements[i] = vals.x;
      elements[i + 1] = vals.y;
    }

    int64_t const pos_id = positions[token_idx];
    T_cache const* cache_ptr = cos_sin_cache + pos_id * rotary_dim;
    int const embed_dim = rotary_dim / 2;
    T_cache const* cos_ptr = cache_ptr;
    T_cache const* sin_ptr = cache_ptr + embed_dim;
    int const rotary_lanes = rotary_dim / num_elems_per_thread;

    if (lane_id < rotary_lanes) {
      if constexpr (interleave) {
#pragma unroll
        for (int i = 0; i < num_elems_per_thread / 2; ++i) {
          int const idx0 = 2 * i;
          int const idx1 = 2 * i + 1;
          int const dim_idx = lane_id * num_elems_per_thread + idx0;
          float const val0 = elements[idx0];
          float const val1 = elements[idx1];
          int const half_dim = dim_idx / 2;
          float const cos_val =
              CacheConverter::convert(VLLM_LDG(cos_ptr + half_dim));
          float const sin_val =
              CacheConverter::convert(VLLM_LDG(sin_ptr + half_dim));
          elements[idx0] = val0 * cos_val - val1 * sin_val;
          elements[idx1] = val0 * sin_val + val1 * cos_val;
        }
      } else {
        __syncwarp();
        int const pair_offset = (rotary_dim / 2) / num_elems_per_thread;
        float elements2[num_elems_per_thread];
#pragma unroll
        for (int i = 0; i < num_elems_per_thread; i++) {
          elements2[i] =
              __shfl_xor_sync(FINAL_MASK, elements[i], pair_offset);
          if (lane_id < pair_offset) {
            elements2[i] = -elements2[i];
          }
          int dim_idx = lane_id * num_elems_per_thread + i;
          dim_idx = (dim_idx * 2) % rotary_dim;
          int const half_dim = dim_idx / 2;
          float const cos_val =
              CacheConverter::convert(VLLM_LDG(cos_ptr + half_dim));
          float const sin_val =
              CacheConverter::convert(VLLM_LDG(sin_ptr + half_dim));
          elements[i] = elements[i] * cos_val + elements2[i] * sin_val;
        }
        __syncwarp();
      }
    }

    {
      vec_T vec;
      constexpr int num_packed_elems = elem_size_bytes / sizeof(T2_in);
#pragma unroll
      for (int i = 0; i < num_packed_elems; i++) {
        T2_in packed_val = Converter::convert(
            make_float2(elements[2 * i], elements[2 * i + 1]));
        *(reinterpret_cast<T2_in*>(&vec) + i) = packed_val;
      }
      *reinterpret_cast<vec_T*>(
          &key_cache[cache_row_offset + dim_offset]) = vec;
    }

    {
      vec_T vec =
          *reinterpret_cast<vec_T const*>(&all_v[v_row_offset + dim_offset]);
      *reinterpret_cast<vec_T*>(
          &value_cache[value_cache_row_offset + dim_offset]) = vec;
    }
#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800) && !defined(USE_ROCM)
  }
#endif
}

template <typename scalar_t_in, typename scalar_t_cache>
void launchDFlashKNormRopeCacheUpdate(
    void const* all_k, void const* all_v, void* key_cache, void* value_cache,
    void const* k_norm_weight, void const* cos_sin_cache,
    int64_t const* positions, int64_t const* slot_mapping, int num_ctx,
    int num_kv_heads, int head_dim, int64_t all_k_stride0,
    int64_t all_k_stride1, int64_t all_v_stride0, int64_t all_v_stride1,
    int64_t key_cache_stride0, int64_t key_cache_stride1,
    int64_t key_cache_stride2, int64_t value_cache_stride0,
    int64_t value_cache_stride1, int64_t value_cache_stride2, int block_size,
    int rotary_dim, float eps, bool interleave, cudaStream_t stream) {
  constexpr int block_threads = 256;
  constexpr int warps_per_block = block_threads / 32;
  int const total_warps = num_ctx * num_kv_heads;
  dim3 grid((total_warps + warps_per_block - 1) / warps_per_block);
  dim3 block(block_threads);

#define LAUNCH_HEAD_DIM(HEAD_DIM)                                      \
  do {                                                                 \
    if (interleave) {                                                  \
      dflashKNormRopeCacheUpdateKernel<scalar_t_in, scalar_t_cache,    \
                                       (HEAD_DIM), true>               \
          <<<grid, block, 0, stream>>>(                                \
              all_k, all_v, key_cache, value_cache, k_norm_weight,     \
              cos_sin_cache, positions, slot_mapping, num_ctx,         \
              num_kv_heads, all_k_stride0, all_k_stride1,              \
              all_v_stride0, all_v_stride1, key_cache_stride0,         \
              key_cache_stride1, key_cache_stride2,                    \
              value_cache_stride0, value_cache_stride1,                \
              value_cache_stride2, block_size, eps, rotary_dim);       \
    } else {                                                           \
      dflashKNormRopeCacheUpdateKernel<scalar_t_in, scalar_t_cache,    \
                                       (HEAD_DIM), false>              \
          <<<grid, block, 0, stream>>>(                                \
              all_k, all_v, key_cache, value_cache, k_norm_weight,     \
              cos_sin_cache, positions, slot_mapping, num_ctx,         \
              num_kv_heads, all_k_stride0, all_k_stride1,              \
              all_v_stride0, all_v_stride1, key_cache_stride0,         \
              key_cache_stride1, key_cache_stride2,                    \
              value_cache_stride0, value_cache_stride1,                \
              value_cache_stride2, block_size, eps, rotary_dim);       \
    }                                                                  \
  } while (0)

  switch (head_dim) {
    case 64:
      LAUNCH_HEAD_DIM(64);
      break;
    case 128:
      LAUNCH_HEAD_DIM(128);
      break;
    case 256:
      LAUNCH_HEAD_DIM(256);
      break;
    default:
      STD_TORCH_CHECK(false, "Unsupported DFlash head dimension: ", head_dim);
  }
#undef LAUNCH_HEAD_DIM
}

}  // namespace vllm::dflash_fused_ops

void dflash_k_norm_rope(
    torch::stable::Tensor const& all_k,
    torch::stable::Tensor& all_k_out,
    torch::stable::Tensor const& k_norm_weights,
    torch::stable::Tensor const& positions,
    torch::stable::Tensor const& cos_sin_cache,
    int64_t rope_head_size,
    bool is_neox,
    double eps) {
  CHECK_TH_CUDA(all_k);
  CHECK_INPUT(all_k_out);
  CHECK_INPUT(k_norm_weights);
  CHECK_INPUT(positions);
  CHECK_INPUT(cos_sin_cache);
  CHECK_TYPE(positions, torch::headeronly::ScalarType::Long);

  STD_TORCH_CHECK(all_k.dim() == 4,
                  "all_k must be [num_layers, num_ctx, num_kv_heads, head_dim]");
  STD_TORCH_CHECK(all_k_out.dim() == 4 && all_k_out.size(0) == all_k.size(0) &&
                      all_k_out.size(1) == all_k.size(1) &&
                      all_k_out.size(2) == all_k.size(2) &&
                      all_k_out.size(3) == all_k.size(3),
                  "all_k_out must have the same shape as all_k");
  STD_TORCH_CHECK(k_norm_weights.dim() == 2,
                  "k_norm_weights must be [num_layers, head_dim]");
  STD_TORCH_CHECK(positions.dim() == 1, "positions must be [num_ctx]");
  STD_TORCH_CHECK(cos_sin_cache.dim() == 2,
                  "cos_sin_cache must be [max_position, rotary_dim]");
  STD_TORCH_CHECK(all_k.scalar_type() == all_k_out.scalar_type(),
                  "all_k and all_k_out must have the same dtype");
  STD_TORCH_CHECK(all_k.scalar_type() == k_norm_weights.scalar_type(),
                  "all_k and k_norm_weights must have the same dtype");

  int64_t const num_layers = all_k.size(0);
  int64_t const num_ctx = all_k.size(1);
  int64_t const num_kv_heads = all_k.size(2);
  int64_t const head_dim = all_k.size(3);
  int64_t const rotary_dim = cos_sin_cache.size(1);

  STD_TORCH_CHECK(k_norm_weights.size(0) == num_layers &&
                      k_norm_weights.size(1) == head_dim,
                  "k_norm_weights shape must match [num_layers, head_dim]");
  STD_TORCH_CHECK(all_k.stride(3) == 1 && all_k.stride(2) == head_dim,
                  "all_k must have contiguous [num_kv_heads, head_dim] "
                  "inner dimensions");
  STD_TORCH_CHECK(all_k_out.is_contiguous(),
                  "all_k_out must be contiguous");
  STD_TORCH_CHECK(positions.size(0) == num_ctx,
                  "positions length must match num_ctx");
  STD_TORCH_CHECK(rotary_dim % 2 == 0, "rotary_dim must be even");
  STD_TORCH_CHECK(rotary_dim <= rope_head_size,
                  "rotary_dim must be <= rope_head_size");
  STD_TORCH_CHECK(rope_head_size == head_dim,
                  "DFlash fused K norm + RoPE currently expects rope_head_size "
                  "to match head_dim");
  STD_TORCH_CHECK(head_dim % 64 == 0,
                  "DFlash fused K norm + RoPE requires head_dim divisible by 64");
  STD_TORCH_CHECK(rotary_dim % (head_dim / 32) == 0,
                  "DFlash fused K norm + RoPE requires rotary_dim to be "
                  "divisible by head_dim / 32");
  STD_TORCH_CHECK(all_k.get_device_index() == all_k_out.get_device_index() &&
                      all_k.get_device_index() ==
                          k_norm_weights.get_device_index() &&
                      all_k.get_device_index() == positions.get_device_index() &&
                      all_k.get_device_index() ==
                          cos_sin_cache.get_device_index(),
                  "all inputs must be on the same CUDA device");

  const torch::stable::accelerator::DeviceGuard device_guard(
      all_k.get_device_index());
  auto stream = get_current_cuda_stream(all_k.get_device_index());

  VLLM_STABLE_DISPATCH_HALF_TYPES(
      all_k.scalar_type(), "dflash_k_norm_rope_kernel", [&] {
        using k_scalar_t = scalar_t;
        VLLM_STABLE_DISPATCH_FLOATING_TYPES(
            cos_sin_cache.scalar_type(), "dflash_k_norm_rope_cache", [&] {
              using cache_scalar_t = scalar_t;
              vllm::dflash_fused_ops::launchDFlashKNormRope<
                  k_scalar_t, cache_scalar_t>(
                  all_k.const_data_ptr(), all_k_out.mutable_data_ptr(),
                  k_norm_weights.const_data_ptr(),
                  cos_sin_cache.const_data_ptr(),
                  positions.const_data_ptr<int64_t>(),
                  static_cast<int>(num_layers), static_cast<int>(num_ctx),
                  static_cast<int>(num_kv_heads), static_cast<int>(head_dim),
                  all_k.stride(0), all_k.stride(1), all_k.stride(2),
                  static_cast<int>(rotary_dim), static_cast<float>(eps),
                  !is_neox, stream);
            });
      });
}

void dflash_k_norm_rope_cache_update(
    torch::stable::Tensor const& all_k,
    torch::stable::Tensor const& all_v,
    torch::stable::Tensor& key_cache,
    torch::stable::Tensor& value_cache,
    torch::stable::Tensor const& k_norm_weight,
    torch::stable::Tensor const& positions,
    torch::stable::Tensor const& cos_sin_cache,
    torch::stable::Tensor const& slot_mapping,
    int64_t rope_head_size,
    bool is_neox,
    double eps) {
  CHECK_TH_CUDA(all_k);
  CHECK_TH_CUDA(all_v);
  CHECK_TH_CUDA(key_cache);
  CHECK_TH_CUDA(value_cache);
  CHECK_INPUT(k_norm_weight);
  CHECK_INPUT(positions);
  CHECK_INPUT(cos_sin_cache);
  CHECK_INPUT(slot_mapping);
  CHECK_TYPE(positions, torch::headeronly::ScalarType::Long);
  CHECK_TYPE(slot_mapping, torch::headeronly::ScalarType::Long);

  STD_TORCH_CHECK(all_k.dim() == 3,
                  "all_k must be [num_ctx, num_kv_heads, head_dim]");
  STD_TORCH_CHECK(all_v.dim() == 3 && all_v.size(0) == all_k.size(0) &&
                      all_v.size(1) == all_k.size(1) &&
                      all_v.size(2) == all_k.size(2),
                  "all_v must have the same shape as all_k");
  STD_TORCH_CHECK(key_cache.dim() == 4 && value_cache.dim() == 4,
                  "key_cache and value_cache must be 4D");
  STD_TORCH_CHECK(k_norm_weight.dim() == 1,
                  "k_norm_weight must be [head_dim]");
  STD_TORCH_CHECK(positions.dim() == 1, "positions must be [num_ctx]");
  STD_TORCH_CHECK(slot_mapping.dim() == 1,
                  "slot_mapping must be [num_ctx]");
  STD_TORCH_CHECK(cos_sin_cache.dim() == 2,
                  "cos_sin_cache must be [max_position, rotary_dim]");

  int64_t const num_ctx = all_k.size(0);
  int64_t const num_kv_heads = all_k.size(1);
  int64_t const head_dim = all_k.size(2);
  int64_t const rotary_dim = cos_sin_cache.size(1);

  STD_TORCH_CHECK(k_norm_weight.size(0) == head_dim,
                  "k_norm_weight length must match head_dim");
  STD_TORCH_CHECK(positions.size(0) == num_ctx,
                  "positions length must match num_ctx");
  STD_TORCH_CHECK(slot_mapping.size(0) == num_ctx,
                  "slot_mapping length must match num_ctx");
  STD_TORCH_CHECK(key_cache.size(2) == num_kv_heads &&
                      value_cache.size(2) == num_kv_heads &&
                      key_cache.size(3) == head_dim &&
                      value_cache.size(3) == head_dim,
                  "cache shape must be [num_blocks, block_size, "
                  "num_kv_heads, head_dim]");
  STD_TORCH_CHECK(key_cache.size(1) == value_cache.size(1),
                  "key_cache and value_cache block sizes must match");
  STD_TORCH_CHECK(all_k.scalar_type() == all_v.scalar_type() &&
                      all_k.scalar_type() == key_cache.scalar_type() &&
                      all_k.scalar_type() == value_cache.scalar_type() &&
                      all_k.scalar_type() == k_norm_weight.scalar_type(),
                  "all K/V/cache tensors and k_norm_weight must share dtype");
  STD_TORCH_CHECK(all_k.stride(2) == 1 && all_k.stride(1) == head_dim,
                  "all_k must have contiguous [num_kv_heads, head_dim] "
                  "inner dimensions");
  STD_TORCH_CHECK(all_v.stride(2) == 1 && all_v.stride(1) == head_dim,
                  "all_v must have contiguous [num_kv_heads, head_dim] "
                  "inner dimensions");
  STD_TORCH_CHECK(key_cache.stride(3) == 1 &&
                      value_cache.stride(3) == 1,
                  "cache head_dim must be contiguous");
  STD_TORCH_CHECK(rotary_dim % 2 == 0, "rotary_dim must be even");
  STD_TORCH_CHECK(rotary_dim <= rope_head_size,
                  "rotary_dim must be <= rope_head_size");
  STD_TORCH_CHECK(rope_head_size == head_dim,
                  "DFlash fused cache update expects rope_head_size to match "
                  "head_dim");
  STD_TORCH_CHECK(head_dim % 64 == 0,
                  "DFlash fused cache update requires head_dim divisible by 64");
  STD_TORCH_CHECK(rotary_dim % (head_dim / 32) == 0,
                  "DFlash fused cache update requires rotary_dim to be "
                  "divisible by head_dim / 32");
  STD_TORCH_CHECK(all_k.get_device_index() == all_v.get_device_index() &&
                      all_k.get_device_index() == key_cache.get_device_index() &&
                      all_k.get_device_index() ==
                          value_cache.get_device_index() &&
                      all_k.get_device_index() ==
                          k_norm_weight.get_device_index() &&
                      all_k.get_device_index() == positions.get_device_index() &&
                      all_k.get_device_index() ==
                          cos_sin_cache.get_device_index() &&
                      all_k.get_device_index() ==
                          slot_mapping.get_device_index(),
                  "all inputs must be on the same CUDA device");

  const torch::stable::accelerator::DeviceGuard device_guard(
      all_k.get_device_index());
  auto stream = get_current_cuda_stream(all_k.get_device_index());

  VLLM_STABLE_DISPATCH_HALF_TYPES(
      all_k.scalar_type(), "dflash_k_norm_rope_cache_update_kernel", [&] {
        using k_scalar_t = scalar_t;
        VLLM_STABLE_DISPATCH_FLOATING_TYPES(
            cos_sin_cache.scalar_type(), "dflash_k_norm_rope_cache", [&] {
              using cache_scalar_t = scalar_t;
              vllm::dflash_fused_ops::launchDFlashKNormRopeCacheUpdate<
                  k_scalar_t, cache_scalar_t>(
                  all_k.const_data_ptr(), all_v.const_data_ptr(),
                  key_cache.mutable_data_ptr(),
                  value_cache.mutable_data_ptr(),
                  k_norm_weight.const_data_ptr(),
                  cos_sin_cache.const_data_ptr(),
                  positions.const_data_ptr<int64_t>(),
                  slot_mapping.const_data_ptr<int64_t>(),
                  static_cast<int>(num_ctx), static_cast<int>(num_kv_heads),
                  static_cast<int>(head_dim), all_k.stride(0),
                  all_k.stride(1), all_v.stride(0), all_v.stride(1),
                  key_cache.stride(0), key_cache.stride(1),
                  key_cache.stride(2), value_cache.stride(0),
                  value_cache.stride(1), value_cache.stride(2),
                  static_cast<int>(key_cache.size(1)),
                  static_cast<int>(rotary_dim), static_cast<float>(eps),
                  !is_neox, stream);
            });
      });
}
