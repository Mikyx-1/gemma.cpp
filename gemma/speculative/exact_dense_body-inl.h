// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Native verifier. Include in gemma.cc's target namespace after
// exact_i8_project-inl.h and the native Transformer helpers.
#if defined(THIRD_PARTY_GEMMA_CPP_EXACT_DENSE_BODY_TOGGLE) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef THIRD_PARTY_GEMMA_CPP_EXACT_DENSE_BODY_TOGGLE
#undef THIRD_PARTY_GEMMA_CPP_EXACT_DENSE_BODY_TOGGLE
#else
#define THIRD_PARTY_GEMMA_CPP_EXACT_DENSE_BODY_TOGGLE
#endif

#include "gemma/speculative/exact_stopped_leaf-inl.h"

struct ExactBodyProjection {
  const MMI8B* weights = nullptr;
  MMConfig schedule;
};

// Initialization may pack previously unseen weights but never changes the
// native tuning tables. All projections must be ready before any KV writes.
static bool ExactBodyPrepareProjection(const MatPtr& weights, size_t num_b,
                                       MatMulEnv& env,
                                       ExactBodyProjection& projection) {
  auto& cache = MMI8WeightCache::Get();
  if (!cache.Eligible(weights)) return false;
  projection.weights = CallUpcasted(
      &weights, [&](const auto* typed) { return cache.Lookup(*typed, env); });
  return projection.weights != nullptr &&
         ExactI8PackedProjectionSupported(*projection.weights) &&
         ExactI8CopyNativeConfig(*projection.weights, num_b, env,
                                 projection.schedule);
}

// The default policy retains the original exact helper calls and schedules.
struct ExactI8BodyProjectionPolicy {
  using Plan = ExactBodyProjection;
  static bool SupportsRows(size_t rows) { return rows > 0; }
  static bool Prepare(const MatPtr& weights, size_t num_b, MatMulEnv& env,
                      Plan& plan) {
    return ExactBodyPrepareProjection(weights, num_b, env, plan);
  }
  template <typename TA, typename TC>
  static bool Project(const MatPtrT<TA>& input, const Plan& plan,
                      MatPtrT<TC>& output, MMI8AStorage& storage,
                      MatMulEnv& env) {
    return ExactI8ProjectWithConfig<false>(
        input, *plan.weights, nullptr, plan.schedule,
        {IndexRange(0, plan.weights->Rows())}, output, storage, env, nullptr,
        MMOptions());
  }
  template <typename TA>
  static bool Pair(const MatPtrT<TA>& input, const Plan& gate, const Plan& up,
                   MatPtrT<BF16>& output, MMI8AStorage& storage, MatMulEnv& env,
                   MMOptions options) {
    return ExactI8ProjectWithConfig<true>(
        input, *gate.weights, up.weights, gate.schedule,
        {IndexRange(0, gate.weights->Rows())}, output, storage, env, nullptr,
        options);
  }
};

struct DenseBodyLayer {
  ExactBodyProjection q, kv, attention, gate, up, down;
};

static size_t ExactBodyMaxKV(const ModelConfig& config) {
  size_t cols = 0;
  for (const auto& layer : config.layer_configs)
    cols = HWY_MAX(cols, size_t{2} * layer.kv_heads * layer.qkv_dim);
  return cols;
}

static size_t ExactBodyMaxK(const ModelConfig& config) {
  size_t cols = config.model_dim;
  for (const auto& layer : config.layer_configs) {
    cols = HWY_MAX(cols, static_cast<size_t>(layer.ff_hidden_dim));
    cols = HWY_MAX(cols, static_cast<size_t>(layer.heads * layer.qkv_dim));
  }
  return cols;
}

class ExactDenseBody {
 public:
  ExactDenseBody(const ModelConfig& config, const RuntimeConfig& runtime,
                 size_t horizon, size_t seq_len, MatMulEnv& env)
      : horizon_(horizon),
        batch_(runtime, config, horizon, seq_len, env.ctx, row_ptrs_),
        raw_kv_("verify_raw_kv", Extents2D(horizon, ExactBodyMaxKV(config)),
                env.ctx.allocator, MatPadding::kOdd),
        quantized_(horizon, ExactBodyMaxK(config), env.ctx.allocator) {}

  bool Prepare(const ModelConfig& config, const WeightsPtrs& weights,
               const Activations& native, MatMulEnv& env) {
    if (!ExactI8BodyProjectionPolicy::SupportsRows(horizon_) ||
        (config.model != Model::GEMMA3_270M &&
         config.model != Model::GEMMA3_1B) ||
        native.attention_impl != AttentionImpl::kFlash ||
        MMI8WeightCache::Get().ScalingEnabled() || config.ple_dim != 0 ||
        GCPP_TENSOR_STATS || MMI8CalibrationCaptureEnabled())
      return false;
    layers_.resize(weights.c_layers.size());
    for (size_t i = 0; i < layers_.size(); ++i) {
      const auto& layer = *weights.GetLayer(i);
      const auto& lc = layer.layer_config;
      if (lc.type != LayerAttentionType::kGemma || lc.IsMoE() || lc.ff_biases ||
          lc.kv_share_layer_idx >= 0 || layer.skip_scale.HasPtr())
        return false;
      auto& plan = layers_[i];
      if (!ExactI8BodyProjectionPolicy::Prepare(layer.qkv_einsum_w1, 1, env,
                                                plan.q) ||
          !ExactI8BodyProjectionPolicy::Prepare(layer.qkv_einsum_w2, 1, env,
                                                plan.kv) ||
          !ExactI8BodyProjectionPolicy::Prepare(layer.att_weights, 1, env,
                                                plan.attention) ||
          !ExactI8BodyProjectionPolicy::Prepare(layer.linear_w, 1, env,
                                                plan.down))
        return false;
      constexpr size_t num_b = GEMMA_FUSED_FFN ? 2 : 1;
      if (!ExactI8BodyProjectionPolicy::Prepare(layer.gating_einsum_w1, num_b,
                                                env, plan.gate) ||
          !ExactI8BodyProjectionPolicy::Prepare(layer.gating_einsum_w2, num_b,
                                                env, plan.up))
        return false;
    }
    return true;
  }

  // Inputs are the previous committed token followed by private draft tokens.
  // Earlier KV is unchanged. Before wrap, rejected future rows can be replaced
  // by the next native/verified call. Logical query position is restored here.
  void Run(const std::vector<int>& inputs, size_t count, size_t start,
           const ModelConfig& config, const RuntimeConfig& runtime,
           const WeightsPtrs& weights, Activations& native, QBatch& qbatch,
           MatMulEnv& env) {
    HWY_ASSERT(qbatch.Size() == 1 && count > 0 && count <= inputs.size());
    HWY_ASSERT(count <= horizon_ &&
               ExactI8BodyProjectionPolicy::SupportsRows(count));
    HWY_ASSERT(start + count <= qbatch.KV(0).SeqLen());
    const size_t old_pos = qbatch.Pos(0);
    const int old_token = qbatch.PrevToken(0);
    batch_.SetBatchSize(count);
    raw_kv_.OverrideRows(count);
    native.SetBatchSize(1);
    // This changes only the exact verifier's attention scheduling. Private
    // Observers retain the serial route.
    static const bool batch_attention_enabled =
        MMI8Flag("GEMMA_MM_I8_EXACT_BATCH_ATTENTION", false);
    const bool try_batch_attention =
        batch_attention_enabled && !runtime.layers_output &&
        !runtime.activations_observer && runtime.image_tokens == nullptr &&
        !runtime.use_mtp && !kObserver && !GCPP_TENSOR_STATS &&
        !COMPRESS_STATS && !MMI8CalibrationCaptureEnabled();
    for (size_t r = 0; r < count; ++r)
      EmbedMMToken(inputs[r], r, start + r, 0, config, weights, batch_.x,
                   env.ctx, nullptr, 0);
    for (size_t i = 0; i < layers_.size(); ++i) {
      const auto& layer = *weights.GetLayer(i);
      const auto& lc = layer.layer_config;
      const auto& plan = layers_[i];
      const size_t q_cols = lc.heads * lc.qkv_dim;
      const size_t kv_cols = lc.kv_heads * 2 * lc.qkv_dim;
      batch_.attention.q.OverrideCols(q_cols);
      batch_.attention.att_out.OverrideCols(q_cols);
      raw_kv_.OverrideCols(kv_cols);
      RMSNormBatched(batch_.x, layer.pre_attention_norm_scale,
                     batch_.attention.pre_att_rms_out, env.ctx);
      Project(batch_.attention.pre_att_rms_out, plan.q, batch_.attention.q,
              env);
      Project(batch_.attention.pre_att_rms_out, plan.kv, raw_kv_, env);
      bool batched_attention = false;
      if (try_batch_attention) {
        // The previous layer's serial fallback can leave Pos at the last row.
        qbatch.MutablePos(0) = start;
        qbatch.PrevToken(0) = inputs[0];
        batched_attention = GemmaAttentionFromProjectedBatchM1(
            count, i, layer, raw_kv_, batch_.attention, qbatch, env,
            native.attention_impl);
        if (batched_attention) {
          static const bool reported = [=]() {
            fprintf(stderr,
                    "Exact batch attention: executed=1 rows=%zu "
                    "heads=4 workers=6 float_lanes=8 max_end=352\n",
                    count);
            return true;
          }();
          (void)reported;
        }
      }
      if (!batched_attention) {
        for (size_t r = 0; r < count; ++r) {
          qbatch.MutablePos(0) = start + r;
          qbatch.PrevToken(0) = inputs[r];
          native.attention.q.OverrideCols(q_cols);
          hwy::CopyBytes(batch_.attention.q.Row(r), native.attention.q.Row(0),
                         q_cols * sizeof(float));
          MatPtrT<BF16> kv_row("verify_kv_row", Extents2D(1, kv_cols));
          kv_row.SetPtr(raw_kv_.Row(r), raw_kv_.Stride());
          HWY_ASSERT(GemmaPrepareKVFromProjectedM1(i, layer, kv_row,
                                                   native.attention, qbatch,
                                                   env, native.attention_impl));
          HWY_ASSERT(GemmaAttentionFromProjectedM1(
              i, layer, native.attention, qbatch, env, native.attention_impl));
          hwy::CopyBytes(native.attention.att_out.Row(0),
                         batch_.attention.att_out.Row(r),
                         q_cols * sizeof(float));
        }
      }
      Project(batch_.attention.att_out, plan.attention,
              batch_.attention.att_sums, env);
      PostNorm(lc.post_norm, layer.post_attention_norm_scale,
               batch_.attention.att_sums, env.ctx);
      ResidualConnection(batch_.attention.att_sums, batch_.x, layer, true,
                         env.ctx);
      RMSNormBatched(batch_.x, layer.pre_ffw_norm_scale, batch_.pre_ffw_rms_out,
                     env.ctx);
      batch_.C1.OverrideCols(lc.ff_hidden_dim);
      batch_.C2.OverrideCols(lc.ff_hidden_dim);
#if GEMMA_FUSED_FFN
      const auto fused = [&](RowPtrsBF c1, IndexRange rows, IndexRange columns,
                             StridedViewBF c2, size_t worker) {
        Activation(lc.activation, c1, rows, columns, c2, env.ctx, worker);
      };
      MMOptions options;
      options.SetFunc(fused);
      HWY_ASSERT(ExactI8BodyProjectionPolicy::Pair(
          batch_.pre_ffw_rms_out, plan.gate, plan.up, batch_.C1, quantized_,
          env, options));
#else
      Project(batch_.pre_ffw_rms_out, plan.gate, batch_.C1, env);
      Project(batch_.pre_ffw_rms_out, plan.up, batch_.C2, env);
      ActivationBatched(lc.activation, batch_.C1, &batch_.C2, env.ctx);
#endif
      Project(batch_.C1, plan.down, batch_.ffw_out, env);
      PostNorm(lc.post_norm, layer.post_ffw_norm_scale, batch_.ffw_out,
               env.ctx);
      ResidualConnection(batch_.ffw_out, batch_.x, layer, false, env.ctx);
    }
    FinalNormBatched(config, weights, batch_, env);
    qbatch.MutablePos(0) = old_pos;
    qbatch.PrevToken(0) = old_token;
  }

  // Optional terminal sibling; original linear Run above is unchanged.
  bool RunStoppedSecondLeaf(const std::vector<int>& inputs, size_t count,
                            size_t start, int second_id,
                            const ModelConfig& config,
                            const RuntimeConfig& runtime,
                            const WeightsPtrs& weights, Activations& native,
                            QBatch& qbatch, MatMulEnv& env,
                            ExactStoppedLeafKV& snapshot) {
    // All decline checks precede target writes. This method branches only the
    // final headless sibling; its absolute position is start+count-1.
    if ((count != 2 && count != 3) || count > inputs.size() ||
        count + 1 > horizon_ ||
        !ExactI8BodyProjectionPolicy::SupportsRows(count + 1) ||
        qbatch.Size() != 1 || qbatch.PrefixEnd(0) != 0 ||
        qbatch.Pos(0) != start || qbatch.PrevToken(0) != inputs[0] ||
        config.model != Model::GEMMA3_1B || start > 352 ||
        count > 352 - start || second_id < 0 ||
        static_cast<size_t>(second_id) >= config.vocab_size ||
        second_id == inputs[count - 1] || layers_.empty() ||
        layers_.size() != config.layer_configs.size() ||
        layers_.size() != weights.c_layers.size() ||
        runtime.attention_impl != AttentionImpl::kFlash ||
        native.attention_impl != AttentionImpl::kFlash ||
        env.ctx.pools.MaxWorkers() != 6 || runtime.layers_output ||
        runtime.activations_observer || runtime.image_tokens != nullptr ||
        runtime.use_mtp || kObserver || GCPP_TENSOR_STATS || COMPRESS_STATS ||
        MMI8CalibrationCaptureEnabled() ||
        MMI8WeightCache::Get().ScalingEnabled() ||
        native.attention.SeqLen() != qbatch.KV(0).kv_cache.Rows())
      return false;
    for (size_t r = 0; r < count; ++r)
      if (inputs[r] < 0 || static_cast<size_t>(inputs[r]) >= config.vocab_size)
        return false;
    for (size_t i = 0; i < layers_.size(); ++i) {
      const auto& lc = config.layer_configs[i];
      const auto& wc = weights.GetLayer(i)->layer_config;
      const auto& plan = layers_[i];
      if (wc.type != lc.type || wc.heads != lc.heads ||
          wc.kv_heads != lc.kv_heads || wc.qkv_dim != lc.qkv_dim ||
          wc.kv_share_layer_idx >= 0 || wc.post_qk != lc.post_qk ||
          wc.use_qk_norm != lc.use_qk_norm || wc.norm_v != lc.norm_v ||
          plan.q.weights == nullptr || plan.kv.weights == nullptr ||
          plan.attention.weights == nullptr || plan.gate.weights == nullptr ||
          plan.up.weights == nullptr || plan.down.weights == nullptr)
        return false;
    }
    if (!snapshot.Prepare(config, qbatch.KV(0), start + count - 1))
      return false;
    const size_t old_pos = qbatch.Pos(0);
    const int old_token = qbatch.PrevToken(0);
    batch_.SetBatchSize(count + 1);
    raw_kv_.OverrideRows(count + 1);
    native.SetBatchSize(1);
    // This changes only the exact verifier's attention scheduling. Private
    // Observers retain the serial route.
    static const bool batch_attention_enabled =
        MMI8Flag("GEMMA_MM_I8_EXACT_BATCH_ATTENTION", false);
    const bool try_batch_attention =
        batch_attention_enabled && !runtime.layers_output &&
        !runtime.activations_observer && runtime.image_tokens == nullptr &&
        !runtime.use_mtp && !kObserver && !GCPP_TENSOR_STATS &&
        !COMPRESS_STATS && !MMI8CalibrationCaptureEnabled();
    for (size_t r = 0; r < count; ++r)
      EmbedMMToken(inputs[r], r, start + r, 0, config, weights, batch_.x,
                   env.ctx, nullptr, 0);
    EmbedMMToken(second_id, count, start + count - 1, 0, config, weights,
                 batch_.x, env.ctx, nullptr, 0);
    for (size_t i = 0; i < layers_.size(); ++i) {
      const auto& layer = *weights.GetLayer(i);
      const auto& lc = layer.layer_config;
      const auto& plan = layers_[i];
      const size_t q_cols = lc.heads * lc.qkv_dim;
      const size_t kv_cols = lc.kv_heads * 2 * lc.qkv_dim;
      batch_.attention.q.OverrideCols(q_cols);
      batch_.attention.att_out.OverrideCols(q_cols);
      raw_kv_.OverrideCols(kv_cols);
      RMSNormBatched(batch_.x, layer.pre_attention_norm_scale,
                     batch_.attention.pre_att_rms_out, env.ctx);
      Project(batch_.attention.pre_att_rms_out, plan.q, batch_.attention.q,
              env);
      Project(batch_.attention.pre_att_rms_out, plan.kv, raw_kv_, env);
      bool batched_attention = false;
      auto spine_attention = batch_.attention;
      spine_attention.SetBatchSize(count);
      MatPtrT<BF16> spine_kv("leaf_spine_kv", Extents2D(count, kv_cols));
      spine_kv.SetPtr(raw_kv_.Row(0), raw_kv_.Stride());
      if (try_batch_attention) {
        // The previous layer's serial fallback can leave Pos at the last row.
        qbatch.MutablePos(0) = start;
        qbatch.PrevToken(0) = inputs[0];
        batched_attention = GemmaAttentionFromProjectedBatchM1(
            count, i, layer, spine_kv, spine_attention, qbatch, env,
            native.attention_impl);
        if (batched_attention) snapshot.NoteBatchAttention();
      }
      if (!batched_attention) {
        for (size_t r = 0; r < count; ++r) {
          qbatch.MutablePos(0) = start + r;
          qbatch.PrevToken(0) = inputs[r];
          native.attention.q.OverrideCols(q_cols);
          hwy::CopyBytes(batch_.attention.q.Row(r), native.attention.q.Row(0),
                         q_cols * sizeof(float));
          MatPtrT<BF16> kv_row("verify_kv_row", Extents2D(1, kv_cols));
          kv_row.SetPtr(raw_kv_.Row(r), raw_kv_.Stride());
          HWY_ASSERT(GemmaPrepareKVFromProjectedM1(i, layer, kv_row,
                                                   native.attention, qbatch,
                                                   env, native.attention_impl));
          HWY_ASSERT(GemmaAttentionFromProjectedM1(
              i, layer, native.attention, qbatch, env, native.attention_impl));
          hwy::CopyBytes(native.attention.att_out.Row(0),
                         batch_.attention.att_out.Row(r),
                         q_cols * sizeof(float));
        }
      }
      // Both choices occupy the same final position. Preserve the processed
      // top1 sibling, then replace only that row with the raw second-leaf KV.
      HWY_ASSERT(snapshot.CaptureLayer(i, qbatch.KV(0)));
      qbatch.MutablePos(0) = start + count - 1;
      qbatch.PrevToken(0) = second_id;
      native.attention.q.OverrideCols(q_cols);
      hwy::CopyBytes(batch_.attention.q.Row(count), native.attention.q.Row(0),
                     q_cols * sizeof(float));
      MatPtrT<BF16> leaf_kv("leaf_kv", Extents2D(1, kv_cols));
      leaf_kv.SetPtr(raw_kv_.Row(count), raw_kv_.Stride());
      HWY_ASSERT(GemmaPrepareKVFromProjectedM1(i, layer, leaf_kv,
                                               native.attention, qbatch, env,
                                               native.attention_impl));
      HWY_ASSERT(GemmaAttentionFromProjectedM1(
          i, layer, native.attention, qbatch, env, native.attention_impl));
      hwy::CopyBytes(native.attention.att_out.Row(0),
                     batch_.attention.att_out.Row(count),
                     q_cols * sizeof(float));
      Project(batch_.attention.att_out, plan.attention,
              batch_.attention.att_sums, env);
      PostNorm(lc.post_norm, layer.post_attention_norm_scale,
               batch_.attention.att_sums, env.ctx);
      ResidualConnection(batch_.attention.att_sums, batch_.x, layer, true,
                         env.ctx);
      RMSNormBatched(batch_.x, layer.pre_ffw_norm_scale, batch_.pre_ffw_rms_out,
                     env.ctx);
      batch_.C1.OverrideCols(lc.ff_hidden_dim);
      batch_.C2.OverrideCols(lc.ff_hidden_dim);
#if GEMMA_FUSED_FFN
      const auto fused = [&](RowPtrsBF c1, IndexRange rows, IndexRange columns,
                             StridedViewBF c2, size_t worker) {
        Activation(lc.activation, c1, rows, columns, c2, env.ctx, worker);
      };
      MMOptions options;
      options.SetFunc(fused);
      HWY_ASSERT(ExactI8BodyProjectionPolicy::Pair(
          batch_.pre_ffw_rms_out, plan.gate, plan.up, batch_.C1, quantized_,
          env, options));
#else
      Project(batch_.pre_ffw_rms_out, plan.gate, batch_.C1, env);
      Project(batch_.pre_ffw_rms_out, plan.up, batch_.C2, env);
      ActivationBatched(lc.activation, batch_.C1, &batch_.C2, env.ctx);
#endif
      Project(batch_.C1, plan.down, batch_.ffw_out, env);
      PostNorm(lc.post_norm, layer.post_ffw_norm_scale, batch_.ffw_out,
               env.ctx);
      ResidualConnection(batch_.ffw_out, batch_.x, layer, false, env.ctx);
    }
    FinalNormBatched(config, weights, batch_, env);
    qbatch.MutablePos(0) = old_pos;
    qbatch.PrevToken(0) = old_token;
    return true;
  }

  Activations& Result() { return batch_; }

 private:
  template <typename TA, typename TC>
  void Project(const MatPtrT<TA>& input, const ExactBodyProjection& plan,
               MatPtrT<TC>& output, MatMulEnv& env) {
    HWY_ASSERT(ExactI8BodyProjectionPolicy::Project(input, plan, output,
                                                    quantized_, env));
  }
  const size_t horizon_;
  // The owning row pointers must outlive all views in batch_.
  std::vector<hwy::AlignedFreeUniquePtr<uint8_t*[]>> row_ptrs_;
  Activations batch_;
  MatStorageT<BF16> raw_kv_;
  MMI8AStorage quantized_;
  std::vector<DenseBodyLayer> layers_;
};

#endif
