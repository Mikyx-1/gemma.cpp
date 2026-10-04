// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0
// Include inside the per-target namespace before PrefillTBatch. This private
// opt-in helper is only called from token-batched prefill, never Transformer.
#if defined(THIRD_PARTY_GEMMA_CPP_PREFILL_LAST_KV_TOGGLE) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef THIRD_PARTY_GEMMA_CPP_PREFILL_LAST_KV_TOGGLE
#undef THIRD_PARTY_GEMMA_CPP_PREFILL_LAST_KV_TOGGLE
#else
#define THIRD_PARTY_GEMMA_CPP_PREFILL_LAST_KV_TOGGLE
#endif

// Read-only: do not add keys, initialize candidates or advance tuning trials.
// Every skipped operation, and the retained KV projection, must already use a
// settled I8 configuration for this exact prefill M bucket and num_B policy.
static bool PrefillLastKVSettled(const MatPtr& b, size_t m, size_t num_b,
                                 const MatMulEnv& env) {
  const auto& cache = MMI8WeightCache::Get();
  if (!cache.Eligible(b) || env.per_cluster.empty()) return false;
  const MMActivation activation =
      cache.QuantBlockSize(b) ? MMActivation::kI8Block : MMActivation::kI8;
  const auto wanted =
      MMKeys::KeyFromDims(m, b.Cols(), b.Rows(), num_b, activation);
  const auto& cluster = env.per_cluster[0];
  const auto keys = cluster.keys.Keys();
  for (size_t i = 0; i < keys.size(); ++i) {
    if (keys[i] != wanted) continue;
    return i < cluster.per_key.size() &&
           cluster.per_key[i].autotune.Best() != nullptr;
  }
  return false;
}

static bool TryPrefillLastLayerKV(size_t num_tokens, size_t layer_idx,
                                  const ModelConfig& config,
                                  const RuntimeConfig& runtime,
                                  const WeightsPtrs& weights,
                                  Activations& activations, QBatch& qbatch,
                                  MatMulEnv& env) {
  static const bool enabled = MMI8Flag("GEMMA_MM_I8_PREFILL_LAST_KV");
  if (!enabled) return false;
  if ((config.model != Model::GEMMA3_270M &&
       config.model != Model::GEMMA3_1B) ||
      config.is_encoder_decoder || config.ple_dim != 0 ||
      layer_idx + 1 != config.layer_configs.size() ||
      weights.c_layers.size() != config.layer_configs.size() ||
      num_tokens == 0 || num_tokens != activations.x.Rows() ||
      qbatch.Size() != 1 || qbatch.PrefixEnd(0) != 0 ||
      activations.attention_impl != AttentionImpl::kFlash ||
      runtime.attention_impl != AttentionImpl::kFlash ||
      runtime.image_tokens != nullptr || runtime.use_mtp ||
      runtime.layers_output || runtime.activations_observer ||
      GCPP_TENSOR_STATS || MMI8CalibrationCaptureEnabled() || env.autotune)
    return false;
  const auto& cache = MMI8WeightCache::Get();
  if (!cache.Enabled() || cache.ScalingEnabled()) return false;
  const auto& layer = *weights.GetLayer(layer_idx);
  const auto& lc = layer.layer_config;
  if (lc.type != LayerAttentionType::kGemma || lc.IsMoE() || lc.IsMHA() ||
      lc.ple_dim != 0 || lc.ff_biases || lc.kv_share_layer_idx >= 0 ||
      layer.skip_scale.HasPtr() || !layer.pre_attention_norm_scale.HasPtr() ||
      layer.qkv_einsum_w1.Cols() != config.model_dim ||
      layer.qkv_einsum_w1.Rows() != lc.heads * lc.qkv_dim ||
      layer.qkv_einsum_w2.Cols() != config.model_dim ||
      layer.qkv_einsum_w2.Rows() != 2 * lc.kv_heads * lc.qkv_dim ||
      qbatch.Pos(0) >= activations.attention.SeqLen() ||
      num_tokens > activations.attention.SeqLen() - qbatch.Pos(0))
    return false;
  constexpr size_t num_ffn_b = GEMMA_FUSED_FFN ? 2 : 1;
  if (!PrefillLastKVSettled(layer.qkv_einsum_w1, num_tokens, 1, env) ||
      !PrefillLastKVSettled(layer.qkv_einsum_w2, num_tokens, 1, env) ||
      !PrefillLastKVSettled(layer.att_weights, num_tokens, 1, env) ||
      !PrefillLastKVSettled(layer.gating_einsum_w1, num_tokens, num_ffn_b,
                            env) ||
      !PrefillLastKVSettled(layer.gating_einsum_w2, num_tokens, num_ffn_b,
                            env) ||
      !PrefillLastKVSettled(layer.linear_w, num_tokens, 1, env))
    return false;

  // Scaling is disabled, so this is exactly TransformerLayer's pre-attention
  // norm. The flat helper reuses native projection, BF16 stores, key norm,
  // RoPE, transposition and future padding at the original batch size.
  RMSNormBatched(activations.x, layer.pre_attention_norm_scale,
                 activations.attention.pre_att_rms_out, env.ctx);
  GemmaPrefillKVOnly(num_tokens, layer_idx, layer, activations.attention,
                     qbatch, env);
  return true;
}
#endif
