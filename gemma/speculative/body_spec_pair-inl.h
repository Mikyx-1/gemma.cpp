// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0
// Private270M catch-up and its already-known next input, using settled M1 math.
#if defined(THIRD_PARTY_GEMMA_CPP_BODY_SPEC_PAIR_TOGGLE) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef THIRD_PARTY_GEMMA_CPP_BODY_SPEC_PAIR_TOGGLE
#undef THIRD_PARTY_GEMMA_CPP_BODY_SPEC_PAIR_TOGGLE
#else
#define THIRD_PARTY_GEMMA_CPP_BODY_SPEC_PAIR_TOGGLE
#endif
#include "gemma/speculative/body_spec_pair_control-inl.h"

class BodySpecPairedDraft {
 public:
  bool Try(bool active, bool failed, size_t generated, size_t max_steps,
           size_t catchup_position, int catchup_input, size_t next_position,
           int next_input, const ModelConfig& config,
           const RuntimeConfig& runtime, const WeightsPtrs& weights,
           Activations& native, QBatch& query, MatMulEnv& env) {
    ++attempts;
    // Reject before modifying hidden/KV/query state. Missing plans retain the
    // original eager catch-up; no tuning key is created to force eligibility.
    if (state_.Pending() ||
        !BodySpecPairedRowState::Eligible(
            true, active, failed, generated, max_steps, catchup_position,
            catchup_input, next_position, next_input, config.vocab_size) ||
        config.model != Model::GEMMA3_270M || query.Size() != 1 ||
        query.PrefixEnd(0) != 0 || query.Pos(0) != catchup_position ||
        query.PrevToken(0) != catchup_input ||
        !BodySpecHasFlatFlashCache(query.KV(0)) ||
        next_position >= query.KV(0).kv_cache.Rows() ||
        runtime.attention_impl != AttentionImpl::kFlash ||
        native.attention_impl != AttentionImpl::kFlash ||
        native.x.Cols() != config.model_dim || !native.x.HasPtr() ||
        env.autotune || !HeadSpecNativeBodySettled(env) ||
        runtime.layers_output || runtime.activations_observer ||
        runtime.image_tokens != nullptr || runtime.use_mtp ||
        MMI8WeightCache::Get().ScalingEnabled() ||
        MMI8CalibrationCaptureEnabled() || GCPP_TENSOR_STATS || COMPRESS_STATS)
      return false;
    if (!prepared_) {
      const double begin = hwy::platform::Now();
      body_ = std::make_unique<ExactDenseBody>(
          config, runtime, 2, query.KV(0).kv_cache.Rows(), env);
      prepared_ = body_->Prepare(config, weights, native, env);
      preparation_seconds += hwy::platform::Now() - begin;
      if (!prepared_) {
        body_.reset();
        return false;
      }
      cache_ = query.KV(0).cache;
    }
    if (cache_ != query.KV(0).cache) return false;
    const std::vector<int> inputs{catchup_input, next_input};
    const double begin = hwy::platform::Now();
    body_->Run(inputs, 2, catchup_position, config, runtime, weights, native,
               query, env);
    batch_seconds += hwy::platform::Now() - begin;
    HWY_ASSERT(query.Pos(0) == catchup_position &&
               query.PrevToken(0) == catchup_input);
    if (!state_.Publish(next_position, next_input))
      HWY_ABORT("Paired draft row already pending");
    ++batches;
    return true;
  }

  bool Pending() const { return state_.Pending(); }
  bool Consume(size_t position, int input, Activations& native) {
    if (!state_.Pending()) return false;
    // No callbacks occur between publication and the next round. A mismatch
    // indicates a driver bug; never silently reuse a row under another ID.
    HWY_ASSERT(prepared_ && state_.Matches(position, input));
    const double begin = hwy::platform::Now();
    const auto& result = body_->Result();
    native.SetBatchSize(1);
    HWY_ASSERT(native.x.Cols() == result.x.Cols());
    hwy::CopyBytes(result.x.Row(1), native.x.Row(0),
                   result.x.Cols() * sizeof(float));
    native.x.SetScale(result.x.Scale());
    if (!state_.Consume(position, input))
      HWY_ABORT("Paired draft input identity changed");
    copy_seconds += hwy::platform::Now() - begin;
    ++consumed;
    return true;
  }

  size_t attempts = 0, batches = 0, consumed = 0;
  double preparation_seconds = 0, batch_seconds = 0, copy_seconds = 0;

 private:
  BodySpecPairedRowState state_;
  std::unique_ptr<ExactDenseBody> body_;
  const KVCache* cache_ = nullptr;
  bool prepared_ = false;
};
#endif
