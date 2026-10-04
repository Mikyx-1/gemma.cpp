// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// CPU speculative decoding. Include in gemma.cc's target namespace after
// head_spec-inl.h, exact_i8_project-inl.h and exact_dense_body-inl.h. The
// ordinary body_spec_registry.h must be included outside the target namespace.
#if defined(THIRD_PARTY_GEMMA_CPP_GEMMA_BODY_SPEC_TOGGLE) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef THIRD_PARTY_GEMMA_CPP_GEMMA_BODY_SPEC_TOGGLE
#undef THIRD_PARTY_GEMMA_CPP_GEMMA_BODY_SPEC_TOGGLE
#else
#define THIRD_PARTY_GEMMA_CPP_GEMMA_BODY_SPEC_TOGGLE
#endif

// The kFlash constructor also allocates optional tiled buffers, so IsTiled()
// reports allocation metadata rather than the attention path selected here.
// Runtime/activation kFlash guards below select these flat KV/K/V buffers.
static bool BodySpecHasFlatFlashCache(const KVCachePtr& kv) {
  return kv.cache != nullptr && kv.kv_cache.HasPtr() &&
         kv.kv_cache.Rows() != 0 && kv.k_cache.HasPtr() && kv.v_cache.HasPtr();
}

// Native prompt-body operations for the restricted Gemma3 draft, with no
// sampling, stream callbacks, observers or response construction. The last
// prompt token remains the previous token for the first generated step.
static void BodySpecPrefillDraft(const ModelConfig& config,
                                 const WeightsPtrs& weights,
                                 size_t max_batch_size,
                                 Activations& activations, QBatch& query,
                                 MatMulEnv& env) {
  const PromptTokens& prompt = query.Prompt(0);
  HWY_ASSERT(query.Size() == 1 && query.InitialPos(0) == 0 &&
             query.PrefixEnd(0) == 0 && prompt.size() != 0);
  for (size_t start = 0; start + 1 < prompt.size();) {
    const size_t count = HWY_MIN(max_batch_size, prompt.size() - 1 - start);
    activations.SetBatchSize(count);
    activations.token_ids.resize(count);
    query.MutablePos(0) = start;
    for (size_t row = 0; row < count; ++row) {
      const int token = prompt[start + row];
      activations.token_ids[row] = token;
      EmbedMMToken(token, row, start + row, start + row, config, weights,
                   activations.x, env.ctx, nullptr, 0);
    }
    for (size_t layer = 0; layer < config.layer_configs.size(); ++layer) {
      TransformerLayer(count, layer, *weights.GetLayer(layer), activations,
                       query, env);
    }
    start += count;
  }
  activations.SetBatchSize(1);
  query.MutablePos(0) = prompt.size() - 1;
  query.PrevToken(0) = prompt[prompt.size() - 1];
}

#include "gemma/speculative/body_spec_ann-inl.h"

static bool TryGenerateBodySpec(const ModelConfig& config,
                                const RuntimeConfig& runtime,
                                const WeightsPtrs& weights,
                                Activations& activations, QBatch& qbatch,
                                MatMulEnv& env, TimingInfo& timing,
                                hwy::BitSet4096<>& non_eos, size_t max_steps,
                                const SampleFunc& sample) {
#if GEMMA_HEAD_SPEC_HNSW
  return TryGenerateBodySpecAnn(config, runtime, weights, activations, qbatch,
                                env, timing, non_eos, max_steps, sample);
#else
  return false;
#endif
}
#endif
