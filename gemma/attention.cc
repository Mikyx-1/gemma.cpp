// Copyright 2025 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stddef.h>
#include <stdint.h>

#include <vector>

#include "compression/types.h"  // GEMMA_DISABLED_TARGETS
#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS GEMMA_DISABLED_TARGETS
#endif  // HWY_DISABLED_TARGETS

#include "gemma/activations.h"
#include "gemma/configs.h"  // kMaxQKVDim
#include "gemma/kv_cache.h"
#include "gemma/query.h"
#include "gemma/weights.h"
#include "hwy/base.h"
#include "hwy/profiler.h"
#include "ops/matmul.h"
#include "util/threading.h"
#include "util/threading_context.h"
#include "util/zones.h"

// Compiles this file for multiple architectures via "foreach_target.h", to
// which we pass the filename via macro 'argument'.
// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "gemma/attention.cc"  // NOLINT
// clang-format on
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
// After highway.h
#include "compression/compress-inl.h"
#include "gemma/attention.h"  // includes highway.h
#include "gemma/flash_attention.h"
#include "gemma/gemma-inl.h"
#include "ops/ops-inl.h"

HWY_BEFORE_NAMESPACE();
namespace gcpp {
namespace HWY_NAMESPACE {

// Transposes a single row of the kv cache into the k-cache and v-cache.
void TransposeKVCacheRow(const KV_t* HWY_RESTRICT kv, KV_t* HWY_RESTRICT k,
                         KV_t* HWY_RESTRICT v, size_t qkv_dim) {
  // This is inefficient, as the writes are scattered over cache lines, but it
  // is a tiny fraction of the overall computation, and it is linear in the
  // token length.
  const size_t kFloatsPerTile = 2 * FloatsPerVector();
  const size_t kRoundedQkvDim = hwy::RoundUpTo(qkv_dim, kMaxBF16PerVector);
  for (size_t i = 0; i < qkv_dim; i += 2) {
    k[i * kFloatsPerTile] = kv[i];
    k[i * kFloatsPerTile + 1] = kv[i + 1];
  }
  for (size_t i = qkv_dim; i < kRoundedQkvDim; i += 2) {
    k[i * kFloatsPerTile] = hwy::ConvertScalarTo<KV_t>(0.0f);
    k[i * kFloatsPerTile + 1] = hwy::ConvertScalarTo<KV_t>(0.0f);
  }
  for (size_t i = 0; i < qkv_dim; i += kFloatsPerTile) {
    if (i + kFloatsPerTile <= qkv_dim) {
      for (size_t j = 0; j < kFloatsPerTile; j++) {
        v[i * kFloatsPerTile + j] = kv[i + j + qkv_dim];
      }
    } else {
      for (size_t j = 0; j < qkv_dim - i; j++) {
        v[i * kFloatsPerTile + j] = kv[i + j + qkv_dim];
      }
      for (size_t j = qkv_dim - i; j < kFloatsPerTile; j++) {
        v[i * kFloatsPerTile + j] = hwy::ConvertScalarTo<KV_t>(0.0f);
      }
    }
  }
  for (size_t i = hwy::RoundUpTo(qkv_dim, kFloatsPerTile); i < kRoundedQkvDim;
       i += kFloatsPerTile) {
    for (size_t j = 0; j < kFloatsPerTile; j++) {
      v[i * kFloatsPerTile + j] = hwy::ConvertScalarTo<KV_t>(0.0f);
    }
  }
}

void TransposeKVCacheRow_KEqV(const KV_t* HWY_RESTRICT kv, KV_t* HWY_RESTRICT k,
                              KV_t* HWY_RESTRICT v, size_t qkv_dim) {
  const size_t kFloatsPerTile = 2 * FloatsPerVector();
  const size_t kRoundedQkvDim = hwy::RoundUpTo(qkv_dim, kMaxBF16PerVector);
  for (size_t i = 0; i < qkv_dim; i += 2) {
    k[i * kFloatsPerTile] = kv[i];
    k[i * kFloatsPerTile + 1] = kv[i + 1];
  }
  for (size_t i = qkv_dim; i < kRoundedQkvDim; i += 2) {
    k[i * kFloatsPerTile] = hwy::ConvertScalarTo<KV_t>(0.0f);
    k[i * kFloatsPerTile + 1] = hwy::ConvertScalarTo<KV_t>(0.0f);
  }
  for (size_t i = 0; i < qkv_dim; i += kFloatsPerTile) {
    if (i + kFloatsPerTile <= qkv_dim) {
      for (size_t j = 0; j < kFloatsPerTile; j++) {
        v[i * kFloatsPerTile + j] = kv[i + j];
      }
    } else {
      for (size_t j = 0; j < qkv_dim - i; j++) {
        v[i * kFloatsPerTile + j] = kv[i + j];
      }
      for (size_t j = qkv_dim - i; j < kFloatsPerTile; j++) {
        v[i * kFloatsPerTile + j] = hwy::ConvertScalarTo<KV_t>(0.0f);
      }
    }
  }
  for (size_t i = hwy::RoundUpTo(qkv_dim, kFloatsPerTile); i < kRoundedQkvDim;
       i += kFloatsPerTile) {
    for (size_t j = 0; j < kFloatsPerTile; j++) {
      v[i * kFloatsPerTile + j] = hwy::ConvertScalarTo<KV_t>(0.0f);
    }
  }
}

// Zeros out a part of k and v that corresponds to out-of-bounds cache
// positions.
void TransposeOOBKVCacheRow(KV_t* HWY_RESTRICT k, KV_t* HWY_RESTRICT v,
                            size_t qkv_dim) {
  const size_t kFloatsPerTile = 2 * FloatsPerVector();
  const size_t kRoundedQkvDim = hwy::RoundUpTo(qkv_dim, kMaxBF16PerVector);
  for (size_t i = 0; i < kRoundedQkvDim; i += 2) {
    k[i * kFloatsPerTile] = hwy::ConvertScalarTo<KV_t>(0.0f);
    k[i * kFloatsPerTile + 1] = hwy::ConvertScalarTo<KV_t>(0.0f);
  }
  for (size_t i = 0; i < kRoundedQkvDim; i += kFloatsPerTile) {
    for (size_t j = 0; j < kFloatsPerTile; j++) {
      v[i * kFloatsPerTile + j] = hwy::ConvertScalarTo<KV_t>(0.0f);
    }
  }
}

void PositionalEncodingQK(float* qk, const size_t layer_idx,
                          const AttentionActivationsPtrs& activations,
                          ThreadingContext& ctx, const size_t worker,
                          const size_t pos, const float mul) {
  const LayerConfig& layer_config = activations.config.layer_configs[layer_idx];
  const size_t qkv_dim = layer_config.qkv_dim;
  const PostQKType& post_qk = layer_config.post_qk;
  // qk is either q or k, so qkv_dim is the length we operate on.
  const float* inv_timescale = activations.inv_timescale.PackedScale1();
  const bool is_global_layer = activations.config.IsGlobalLayer(layer_idx);
  if (is_global_layer && activations.config.use_global_timescale) {
    inv_timescale = activations.inv_timescale_global.PackedScale1();
  }
  // PostQKType::Rope
  if (post_qk == PostQKType::HalfRope) {
    Rope(qk, qkv_dim / 2, inv_timescale, pos, ctx, worker);
    if (mul != 1.0f) MulByConst(mul, qk, qkv_dim);
  } else {
    RopeAndMulBy(mul, qk, qkv_dim, inv_timescale, pos, ctx, worker);
  }
}

// Different functions use different naming conventions for the number of
// tokens. Functions that are query-independent, such as RMSNorm*, call the
// count `num_interleaved`. Functions that are query-dependent, such as
// `Attention`, use separate `num_tokens` and `num_queries`. `num_tokens` is the
// number of tokens from one query: 1 for decode, otherwise prefill_tbatch_size.

// Fills activations.q and writes to KV cache.
// Applies exactly the native cache processing after Q/KV projection. The
// projected BF16 KV rows are already in their final flat cache locations.
static HWY_INLINE void PrepareGemmaProjectedKVCache(
    size_t num_tokens, size_t layer_idx, const LayerWeightsPtrs& layer,
    AttentionActivationsPtrs& activations, const QBatch& qbatch,
    MatMulEnv& env) {
  const hwy::Divisor div_qbatch(qbatch.Size());
  const LayerConfig& layer_config = layer.layer_config;
  const size_t qkv_dim = layer_config.qkv_dim;
  const size_t kv_heads = layer_config.kv_heads;
  const size_t kv_layer_idx =
      layer_config.kv_share_layer_idx >= 0
          ? static_cast<size_t>(layer_config.kv_share_layer_idx)
          : layer_idx;
  const size_t cache_layer_size =
      activations.config.layer_configs[kv_layer_idx].CacheLayerSize();
  for (size_t qi = 0; qi < qbatch.Size(); ++qi) {
    MaybeReshapeCache(qbatch.KV(qi).cache->KOrVDefaultCols(),
                      qbatch.KV(qi).k_cache);
    MaybeReshapeCache(qbatch.KV(qi).cache->KOrVDefaultCols(),
                      qbatch.KV(qi).v_cache);
  }
  const size_t kFloatsPerVector = FloatsPerVector();
  const size_t kRoundedTokens =
      hwy::RoundUpTo(num_tokens, 2 * kFloatsPerVector);
  const size_t kRoundedNumInterleaved =
      kRoundedTokens * div_qbatch.GetDivisor();

  // Apply positional encodings for K.
  // Note that 2D parallelism is not worth the fork/join overhead because the
  // tasks are very lightweight.
  ParallelFor(
      Parallelism::kFlat, kv_heads * kRoundedNumInterleaved, env.ctx,
      /*cluster_idx=*/0, Callers::kAttComputeQKV,
      [&](size_t task, size_t worker) HWY_ATTR {
        const size_t head = task % kv_heads;
        const size_t interleaved_idx = task / kv_heads;
        const size_t qi = div_qbatch.Remainder(interleaved_idx);
        const size_t token_idx = div_qbatch.Divide(interleaved_idx);
        const size_t cache_pos = qbatch.Pos(qi) + token_idx;
        if (token_idx >= kRoundedTokens) {
          return;
        }
        // The innermost dimension of v is 2NF values from qkv_dim because they
        // will be loaded into a BF16 vector to be scaled and added to the
        // cached attention output in 2 NF-sized registers.
        auto& k_cache = qbatch.KV(qi).k_cache;
        KV_t* HWY_RESTRICT k =
            k_cache.Row(cache_pos / (2 * kFloatsPerVector)) +
            qbatch.KV(qi).cache->KOffset(kv_layer_idx, head, kFloatsPerVector,
                                         cache_pos);
        auto& v_cache = qbatch.KV(qi).v_cache;
        KV_t* HWY_RESTRICT v =
            v_cache.Row(cache_pos / (2 * kFloatsPerVector)) +
            qbatch.KV(qi).cache->VOffset(kv_layer_idx, head, kFloatsPerVector,
                                         cache_pos);
        if (token_idx >= num_tokens) {
          // Create a zero-filled K/V pair for padding for out-of-sequence
          // tokens.
          TransposeOOBKVCacheRow(k, v, qkv_dim);
          return;
        }
        // --seq_len must be large enough to avoid wraparound.
        HWY_DASSERT(cache_pos < activations.SeqLen());
        auto& kv_cache = qbatch.KV(qi).kv_cache;
        const size_t layer_offset =
            qbatch.KV(qi).cache->layer_flat_offsets.empty()
                ? kv_layer_idx * cache_layer_size
                : qbatch.KV(qi).cache->layer_flat_offsets[kv_layer_idx];
        KV_t* HWY_RESTRICT kv =
            kv_cache.Row(cache_pos) + layer_offset + head * qkv_dim * 2;
        // Note that k_cache and v_cache are different shapes.
        // The innermost dimension of k is 2 values from qkv_dim because they
        // are going to be used in a BF16 dot product involving pairs of
        // values over NF k positions.

        HWY_ALIGN float kv_f32[2 * kMaxQKVDim];
        const hn::ScalableTag<float> df;
        DecompressAndZeroPad(df, MakeSpan(kv, 2 * qkv_dim), 0, kv_f32,
                             2 * qkv_dim);

        // Apply further processing to K.
        if (layer.key_norm_scale.HasPtr()) {
          CallUpcasted(&layer.key_norm_scale, [&](const auto* weights_t) {
            RMSNormInplace(weights_t->PackedScale1(), /*w_ofs=*/0, kv_f32,
                           qkv_dim, env.ctx, worker);
          });
        } else if (layer_config.post_qk == PostQKType::NormLocalRope ||
                   layer_config.use_qk_norm) {
          RMSNormNoScaleInplace(kv_f32, qkv_dim, env.ctx, worker);
        }

        // Normalize V projections before caching.
        if (layer_config.norm_v) {
          RMSNormNoScaleInplace(kv_f32 + qkv_dim, qkv_dim, env.ctx, worker);
        }

        constexpr size_t offset = 0;  // placeholder, do not remove
        PositionalEncodingQK(kv_f32, layer_idx, activations, env.ctx, worker,
                             cache_pos + offset,
                             /*mul=*/1.0f);
        CompressPerThread tls;
        Compress(kv_f32, 2 * qkv_dim, tls, MakeSpan(kv, 2 * qkv_dim), 0);
        // This is inefficient, as multiple threads are writing the same K
        // cache line, but the input is generated by a matmul, so it is
        // difficult to change, and it probably isn't significant.
        TransposeKVCacheRow(kv, k, v, qkv_dim);
      });
}

static HWY_INLINE void ComputeQKV(size_t num_tokens, const size_t layer_idx,
                                  const LayerWeightsPtrs& layer,
                                  AttentionActivationsPtrs& activations,
                                  const QBatch& qbatch, const int flags,
                                  MatMulEnv& env) {
  GCPP_ZONE(env.ctx, hwy::Profiler::GlobalIdx(),
            Zones::kGenAttentionComputeQKV);

  const hwy::Divisor div_qbatch(qbatch.Size());
  const size_t num_interleaved = num_tokens * div_qbatch.GetDivisor();
  const LayerConfig& layer_config = layer.layer_config;
  const size_t active_qkv_dim = layer_config.heads * layer_config.qkv_dim;
  activations.q.OverrideCols(active_qkv_dim);
  activations.q_bf.OverrideCols(active_qkv_dim);
  activations.att_out.OverrideCols(active_qkv_dim);
  activations.att_out_reps.OverrideCols(active_qkv_dim);

  // Resolve KV cache layer index and skip flag
  const size_t kv_layer_idx =
      (layer_config.kv_share_layer_idx >= 0)
          ? static_cast<size_t>(layer_config.kv_share_layer_idx)
          : layer_idx;
  const bool skip_kv =
      (layer_config.kv_share_layer_idx >= 0) || (flags & kSkipKV);
  const size_t cache_layer_size =
      activations.config.layer_configs[kv_layer_idx].CacheLayerSize();

  // The original qkv_einsum_w has shape [(heads + kv_heads * 2), qkv_dim,
  // model_dim], which we reshaped to (heads + kv_heads * 2) * qkv_dim rows.
  CallMatMul(activations.pre_att_rms_out, layer.qkv_einsum_w1,
             /*add=*/nullptr, env, activations.q);

  if (skip_kv) return;
  // Set up MatMul row pointers for writing to KV, which consists of
  // `kv_heads` pairs of (k, v) vectors. This safely handles wraparound
  // because rows are computed modulo seq_len.
  MatPtrT<KV_t> kv_rows("kv", Extents2D(activations.pre_att_rms_out.Rows(),
                                        layer.qkv_einsum_w2.Rows()));
  for (size_t interleaved_idx = 0; interleaved_idx < num_interleaved;
       ++interleaved_idx) {
    // Index into qbatch, within [0, qbatch.Size()]
    const size_t qi = div_qbatch.Remainder(interleaved_idx);
    const size_t token_idx = div_qbatch.Divide(interleaved_idx);
    const size_t cache_pos = qbatch.Pos(qi) + token_idx;
    // --seq_len must be large enough to avoid wraparound.
    HWY_DASSERT(cache_pos < activations.SeqLen());

    const size_t layer_offset =
        qbatch.KV(qi).cache->layer_flat_offsets.empty()
            ? kv_layer_idx * cache_layer_size
            : qbatch.KV(qi).cache->layer_flat_offsets[kv_layer_idx];

    env.row_ptrs[0][interleaved_idx] = reinterpret_cast<uint8_t*>(
        qbatch.KV(qi).kv_cache.Row(cache_pos) + layer_offset);
  }
  kv_rows.AttachRowPtrs(env.row_ptrs[0].get());
  CallMatMul(activations.pre_att_rms_out, layer.qkv_einsum_w2,
             /*add=*/nullptr, env, kv_rows);

  PrepareGemmaProjectedKVCache(num_tokens, layer_idx, layer, activations,
                               qbatch, env);
}

// Internal token-prefill helper. The caller has already checked the model,
// final-layer position, flat cache, token count, no sharing, and fixed tuning.
// Existing ComputeQKV<false> call sites retain their original behavior.

// Copies only committed input rows. Flat KV already contains native key norm,
// RoPE and BF16 rounding; transposing it must not repeat those operations.

// Commits one raw projected row, then executes the same M1 cache normalization,
// RoPE, BF16 compression, transposition, and future padding as native decode.
// Returns false without writes when the caller needs the tiled/native path.
bool GemmaPrepareKVFromProjectedM1(size_t layer_idx,
                                   const LayerWeightsPtrs& layer,
                                   const MatPtrT<BF16>& projected_kv,
                                   AttentionActivationsPtrs& activations,
                                   QBatch& qbatch, MatMulEnv& env,
                                   AttentionImpl attention_impl) {
  if (attention_impl != AttentionImpl::kFlash || qbatch.Size() != 1 ||
      projected_kv.Rows() != 1)
    return false;
  const LayerConfig& layer_config = layer.layer_config;
  if (layer_config.kv_share_layer_idx >= 0) return true;
  const size_t cols = layer_config.kv_heads * 2 * layer_config.qkv_dim;
  HWY_ASSERT(projected_kv.Cols() == cols);
  HWY_ASSERT(qbatch.Pos(0) < activations.SeqLen());
  const size_t layer_offset =
      qbatch.KV(0).cache->layer_flat_offsets.empty()
          ? layer_idx * layer_config.CacheLayerSize()
          : qbatch.KV(0).cache->layer_flat_offsets[layer_idx];
  KV_t* const destination =
      qbatch.KV(0).kv_cache.Row(qbatch.Pos(0)) + layer_offset;
  hwy::CopyBytes(projected_kv.Row(0), destination, cols * sizeof(BF16));
  PrepareGemmaProjectedKVCache(1, layer_idx, layer, activations, qbatch, env);
  return true;
}

static HWY_INLINE void RunGemmaProjectedFlashAttention(
    size_t num_tokens, size_t layer_idx, const LayerWeightsPtrs& layer,
    AttentionActivationsPtrs& activations, QBatch& qbatch, MatMulEnv& env,
    AttentionImpl attention_impl) {
  FlashAttention(num_tokens,
                 /*target_parallelism=*/env.ctx.pools.MaxWorkers() *
                     AttentionActivations::kThreadReplicationFactor,
                 layer_idx, layer.query_norm_scale, activations, qbatch,
                 env.ctx, attention_impl);
}

// Q has the exact raw F32 projection; KV for this row has just been committed.
// This mutates Q through native query norm/RoPE and writes float att_out.
// The caller batches SumHeads only after every row completed M1 attention.
bool GemmaAttentionFromProjectedM1(size_t layer_idx,
                                   const LayerWeightsPtrs& layer,
                                   AttentionActivationsPtrs& activations,
                                   QBatch& qbatch, MatMulEnv& env,
                                   AttentionImpl attention_impl) {
  if (attention_impl != AttentionImpl::kFlash || qbatch.Size() != 1 ||
      activations.q.Rows() != 1)
    return false;
  const size_t active_qkv_dim =
      layer.layer_config.heads * layer.layer_config.qkv_dim;
  activations.q.OverrideCols(active_qkv_dim);
  activations.q_bf.OverrideCols(active_qkv_dim);
  activations.att_out.OverrideCols(active_qkv_dim);
  activations.att_out_reps.OverrideCols(active_qkv_dim);
  RunGemmaProjectedFlashAttention(1, layer_idx, layer, activations, qbatch, env,
                                  attention_impl);
  return true;
}


// Batches independent native-M1 attention rows without changing the Flash
// tile/reduction selected for any row. All declines precede view/data writes.
bool GemmaAttentionFromProjectedBatchM1(
    size_t num_tokens, size_t layer_idx, const LayerWeightsPtrs& layer,
    const MatPtrT<BF16>& projected_kv,
    AttentionActivationsPtrs& activations, QBatch& qbatch, MatMulEnv& env,
    AttentionImpl attention_impl) {
  const auto& config = activations.config;
  if (attention_impl != AttentionImpl::kFlash || qbatch.Size() != 1 ||
      num_tokens < 2 || num_tokens > 16 || FloatsPerVector() != 8 ||
      hn::Lanes(hn::ScalableTag<float>()) != 8 ||
      env.ctx.pools.MaxWorkers() != 6 || qbatch.PrefixEnd(0) != 0 ||
      (config.model != Model::GEMMA3_270M && config.model != Model::GEMMA3_1B) ||
      config.is_encoder_decoder || config.num_mtp_layers != 0 ||
      layer_idx >= config.layer_configs.size() ||
      config.attention_window_sizes.size() != config.layer_configs.size() ||
      config.attention_window_sizes[layer_idx] == 0 ||
      activations.div_heads.GetDivisor() != 4) return false;
  constexpr size_t kTile = 16;
  const size_t start = qbatch.Pos(0);
  // The bound both prevents overflow and proves the native M1 split test is
  // ceil(4 * max_tiles / 6) <= ceil(4 * 22 / 6) == 15 < 16.
  if (start > 352 || num_tokens > 352 - start) return false;
  const size_t end = start + num_tokens;
  const size_t padded_end = end + kTile - 1;
  const auto& lc = config.layer_configs[layer_idx];
  const auto& weight_lc = layer.layer_config;
  if (lc.type != LayerAttentionType::kGemma || lc.heads != 4 ||
      lc.kv_heads == 0 || lc.heads % lc.kv_heads != 0 || lc.IsMHA() ||
      lc.qkv_dim == 0 || lc.qkv_dim > kMaxQKVDim || lc.qkv_dim % kTile != 0 ||
      lc.kv_share_layer_idx >= 0 || weight_lc.type != lc.type ||
      weight_lc.heads != lc.heads || weight_lc.kv_heads != lc.kv_heads ||
      weight_lc.qkv_dim != lc.qkv_dim || weight_lc.kv_share_layer_idx >= 0 ||
      weight_lc.post_qk != lc.post_qk ||
      weight_lc.use_qk_norm != lc.use_qk_norm || weight_lc.norm_v != lc.norm_v)
    return false;
  const size_t q_cols = lc.heads * lc.qkv_dim;
  const size_t kv_cols = lc.kv_heads * 2 * lc.qkv_dim;
  const auto valid_rows = [](const auto& matrix, size_t rows, size_t cols) {
    if (!matrix.HasPtr() || matrix.Rows() != rows || matrix.Cols() < cols ||
        matrix.Stride() < cols || matrix.Scale() != 1.0f ||
        matrix.GetLayout() != MatPtr::Layout::kFlat) return false;
    // MatMul may write through attached pointers; Flash reads normal Row().
    // Standard Activations attach the same contiguous rows, which are valid.
    if (matrix.GetRowPtrs() != nullptr) {
      for (size_t r = 0; r < rows; ++r)
        if (matrix.GetRowPtrs()[r] != matrix.RowBytes(r)) return false;
    }
    return true;
  };
  if (projected_kv.Cols() != kv_cols ||
      !valid_rows(projected_kv, num_tokens, kv_cols) ||
      !valid_rows(activations.q, num_tokens, q_cols) ||
      !valid_rows(activations.q_bf, num_tokens, q_cols) ||
      !valid_rows(activations.att_out, num_tokens, q_cols) ||
      !valid_rows(activations.softmax_max, num_tokens, lc.heads) ||
      !valid_rows(activations.softmax_d, num_tokens, lc.heads) ||
      !activations.att_out_reps.HasPtr() ||
      activations.att_out_reps.Cols() < q_cols ||
      activations.att_out_reps.Stride() < q_cols) return false;
  const auto valid_norm = [&](const MatPtr& weights) {
    return !weights.HasPtr() || (weights.Rows() == 1 &&
        weights.Cols() >= lc.qkv_dim && weights.IsPacked() &&
        weights.Scale() == 1.0f);
  };
  if (!valid_norm(layer.key_norm_scale) || !valid_norm(layer.query_norm_scale))
    return false;
  const auto& timescale = config.IsGlobalLayer(layer_idx) &&
                                 config.use_global_timescale
                             ? activations.inv_timescale_global
                             : activations.inv_timescale;
  const size_t rope_cols = lc.post_qk == PostQKType::HalfRope
                              ? lc.qkv_dim / 4 : lc.qkv_dim / 2;
  if (!valid_rows(timescale, 1, rope_cols) || !timescale.IsPacked()) return false;
  auto& cache = qbatch.KV(0);
  // kFlash also allocates optional tiled buffers, so IsTiled() reflects
  // allocation metadata, not the selected runtime path. Validate the actual
  // flat buffers and their physical row count under the explicit kFlash guard.
  if (cache.cache == nullptr || !cache.kv_cache.HasPtr() ||
      activations.SeqLen() != cache.kv_cache.Rows() ||
      end > cache.kv_cache.Rows()) return false;
  const auto& owner = *cache.cache;
  const size_t layers = config.layer_configs.size();
  if (owner.num_layers != layers || owner.layer_flat_offsets.size() != layers ||
      owner.layer_k_v_offsets.size() != layers ||
      owner.rounded_qkv_dims.size() != layers ||
      !owner.kv_cache.HasPtr() || !owner.k_cache.HasPtr() ||
      !owner.v_cache.HasPtr()) return false;
  size_t flat_cols = 0, transposed_cols = 0;
  for (size_t i = 0; i < layers; ++i) {
    const auto& c = config.layer_configs[i];
    if (c.type != LayerAttentionType::kGemma || c.heads != 4 ||
        c.kv_share_layer_idx >= 0 || c.kv_heads == 0 || c.kv_heads > 4 ||
        c.heads % c.kv_heads != 0 || c.qkv_dim == 0 ||
        c.qkv_dim > kMaxQKVDim || c.qkv_dim % 2 != 0 ||
        owner.layer_flat_offsets[i] != flat_cols ||
        owner.layer_k_v_offsets[i] != transposed_cols) return false;
    const size_t rounded = hwy::RoundUpTo(c.qkv_dim, kMaxBF16PerVector);
    if (owner.rounded_qkv_dims[i] != rounded) return false;
    flat_cols += c.kv_heads * 2 * c.qkv_dim;
    transposed_cols += c.kv_heads * rounded;
  }
  if (owner.KOrVDefaultCols() != transposed_cols ||
      !valid_rows(cache.kv_cache, owner.kv_cache.Rows(), flat_cols) ||
      cache.kv_cache.Cols() != flat_cols ||
      !cache.kv_cache.SameShape(owner.kv_cache) ||
      cache.kv_cache.Stride() != owner.kv_cache.Stride() ||
      cache.kv_cache.Row(0) != owner.kv_cache.Row(0)) return false;
  const auto valid_transposed = [&](const MatPtrT<KV_t>& matrix,
                                    const MatPtrT<KV_t>& owned) {
    if (!matrix.HasPtr() || matrix.Rows() == 0 || !matrix.IsPacked() ||
        matrix.Scale() != 1.0f ||
        matrix.GetRowPtrs() != nullptr ||
        matrix.GetLayout() != MatPtr::Layout::kFlat ||
        matrix.Row(0) != owned.Row(0) ||
        matrix.Rows() * matrix.Cols() != owned.Rows() * owned.Cols()) return false;
    if (matrix.Cols() == transposed_cols)
      return matrix.Rows() % kTile == 0 && matrix.Rows() >= padded_end;
    return matrix.Cols() == transposed_cols * kTile &&
           matrix.Rows() >= hwy::DivCeil(padded_end, kTile);
  };
  if (!valid_transposed(cache.k_cache, owner.k_cache) ||
      !valid_transposed(cache.v_cache, owner.v_cache)) return false;

  // No rejection after this point. The original preparation applies identical
  // norm/RoPE/BF16 operations independently to every projected cache row.
  const size_t layer_offset = owner.layer_flat_offsets[layer_idx];
  for (size_t r = 0; r < num_tokens; ++r)
    hwy::CopyBytes(projected_kv.Row(r),
                   cache.kv_cache.Row(start + r) + layer_offset,
                   kv_cols * sizeof(BF16));
  PrepareGemmaProjectedKVCache(num_tokens, layer_idx, layer, activations,
                               qbatch, env);
  // Batched Prepare clears [end,start+16). Repeated M1 calls additionally
  // clear through end+15. Match the complete final transposed cache, using
  // at most15 tiny serial rows without adding another worker barrier.
  for (size_t pos = start + kTile; pos < padded_end; ++pos) {
    for (size_t head = 0; head < lc.kv_heads; ++head) {
      KV_t* const k = cache.k_cache.Row(pos / kTile) +
                     owner.KOffset(layer_idx, head, 8, pos);
      KV_t* const v = cache.v_cache.Row(pos / kTile) +
                     owner.VOffset(layer_idx, head, 8, pos);
      TransposeOOBKVCacheRow(k, v, lc.qkv_dim);
    }
  }
  activations.q.OverrideCols(q_cols);
  activations.q_bf.OverrideCols(q_cols);
  activations.att_out.OverrideCols(q_cols);
  activations.att_out_reps.OverrideCols(q_cols);
  // Scaling by rows preserves Vtile1. At least8 independent params fill the
  // six workers, so the batch never introduces a K split. Future K logits
  // are masked before max/softmax; Vtile1 limits V reads to each row's last.
  FlashAttention(num_tokens, env.ctx.pools.MaxWorkers() *
                     AttentionActivations::kThreadReplicationFactor * num_tokens,
                 layer_idx, layer.query_norm_scale, activations, qbatch,
                 env.ctx, attention_impl);
  return true;
}

void GemmaAttention(size_t num_tokens, const size_t layer_idx,
                    const LayerWeightsPtrs& layer,
                    AttentionActivationsPtrs& activations, QBatch& qbatch,
                    MatMulEnv& env, AttentionImpl attention_impl, int flags) {
  GCPP_ZONE(env.ctx, hwy::Profiler::GlobalIdx(), Zones::kGenAttention);

  const LayerConfig& layer_config = layer.layer_config;
  HWY_DASSERT(!layer_config.IsMHA());  // No longer supported.
  HWY_DASSERT_M((layer_config.heads % layer_config.kv_heads) == 0,
                "query heads must be a multiple of key-value heads");
  (void)layer_config;  // only used in HWY_DASSERT

  ComputeQKV(num_tokens, layer_idx, layer, activations, qbatch, flags, env);
  RunGemmaProjectedFlashAttention(num_tokens, layer_idx, layer, activations,
                                  qbatch, env, attention_impl);
  SumHeads(layer, activations, env);
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace gcpp
HWY_AFTER_NAMESPACE();
