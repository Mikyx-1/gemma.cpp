// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Include inside gemma.cc's target namespace after ChooseSampleFunc and before
// GenerateT. This path batches only the output head. Every body step is
// the original M1 Transformer; drafts never invoke user callbacks.
#if defined(THIRD_PARTY_GEMMA_CPP_GEMMA_HEAD_SPEC_TOGGLE) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef THIRD_PARTY_GEMMA_CPP_GEMMA_HEAD_SPEC_TOGGLE
#undef THIRD_PARTY_GEMMA_CPP_GEMMA_HEAD_SPEC_TOGGLE
#else
#define THIRD_PARTY_GEMMA_CPP_GEMMA_HEAD_SPEC_TOGGLE
#endif

// Speculative rejections must not consume tuning trials: that could change
// the configurations used by later committed tokens versus serial generation.
static bool HeadSpecNativeBodySettled(const MatMulEnv& env) {
  for (const auto& cluster : env.per_cluster) {
    const auto keys = cluster.keys.Keys();
    for (size_t i = 0; i < keys.size(); ++i) {
      if ((keys[i] & uint64_t{0xFFFF}) != MMKeys::BucketM(1)) continue;
      const auto& key = cluster.per_key[i];
      if (key.autotune.Best() == nullptr) return false;
      if (key.autotune_par_a.Best() == nullptr &&
          key.autotune_par_a.HasCandidates())
        return false;
    }
  }
  return true;
}

// Copy a settled native M1 head configuration without creating a tuning key.
static bool HeadSpecNativeConfig(const MatPtr& head, MatMulEnv& env,
                                 MMConfig& config) {
  const auto key = MMKeys::KeyFromDims(1, head.Cols(), head.Rows(), 1,
                                       MMActivation::kI8Block);
  const auto keys = env.per_cluster[0].keys.Keys();
  for (size_t i = 0; i < keys.size(); ++i) {
    if (keys[i] != key) continue;
    const auto* best = env.per_cluster[0].per_key[i].autotune.Best();
    if (best == nullptr) return false;
    config = *best;
    return true;
  }
  return false;
}

#if GEMMA_HEAD_SPEC_HNSW
#include "gemma/speculative/head_spec_ann-inl.h"
#endif

// Only read the original B representation. Group scales still use full B.Rows
// even when the draft evaluates a prefix and a few disconnected tail ranges.
// Explicit ranges also prevent M1..M3 tuning buckets from forcing MC=1.
static void HeadSpecProject(const MatPtrT<BF16>& input, const MMI8B& B,
                            const MMConfig& native,
                            const std::vector<IndexRange>& columns,
                            MatStorageT<float>& output, MMI8AStorage& storage,
                            MatMulEnv& env) {
  const size_t m = input.Rows(), k = input.Cols(), n = B.Rows();
  MMI8AView residual;
  const auto av =
      QuantizeA(input, storage, env.ctx, 0, B.a_pre_scale, B.block_size,
                MMI8UseDualA(B, 1) ? &residual : nullptr);
  MMAutoTune<MMConfig> tuner;
  tuner.SetCandidates({native}, false);
  const MMArgs args(env, m, k, n, 1.0f, nullptr, MMOptions(), tuner, native);
  std::vector<IndexRange> tasks;
  size_t total = 0;
  for (const auto& range : columns) total += range.Num();
  const size_t workers = HWY_MAX(size_t{1}, env.ctx.pools.MaxWorkers());
  const size_t grain =
      HWY_MAX(size_t{8}, hwy::RoundUpTo(hwy::DivCeil(total, workers * 4), 8));
  for (const auto& range : columns) {
    for (size_t begin = range.begin(); begin < range.end(); begin += grain)
      tasks.emplace_back(
          begin, HWY_MIN(begin + grain, static_cast<size_t>(range.end())));
  }
  const auto ranges_kc = native.RangesOfKC(k);
  ParallelFor(
      Parallelism::kFlat, tasks.size(), env.ctx, 0, Callers::kMMClusterForN,
      [&](size_t task, size_t /*worker*/) HWY_ATTR {
        const auto rn = tasks[task];
        const StridedView<float> out(output.Row(0) + rn.begin(), rn.Num(),
                                     output.Stride());
        for (size_t ki = 0; ki < ranges_kc.NumTasks(); ++ki) {
          if (ki == 0)
            MMI8Kernel::B3A2C0(av, B, IndexRange(0, m), ranges_kc.Range(ki), rn,
                               args, MMSetC(), out);
          else
            MMI8Kernel::B3A2C0(av, B, IndexRange(0, m), ranges_kc.Range(ki), rn,
                               args, MMAddC(), out);
        }
      });
}

static int HeadSpecDraftArgmax(const MatStorageT<float>& logits,
                               const std::vector<IndexRange>& ranges) {
  size_t best = ranges.front().begin();
  float value = logits.Row(0)[best];
  for (const auto& range : ranges) {
    for (size_t n : range) {
      if (logits.Row(0)[n] > value) {
        best = n;
        value = logits.Row(0)[n];
      }
    }
  }
  return static_cast<int>(best);
}

// Before the first physical cache wrap, discarded future rows cannot destroy
// any committed KV entries. Past that point, use native generation.
static size_t HeadSpecFirstWrap(const KVCachePtr& kv) {
  size_t end = kv.SeqLen();
  if (kv.IsTiled()) {
    for (const auto& ptr : kv.cache->kv_head_ptrs)
      end = HWY_MIN(end, ptr.Rows() * KVCache::kTileSize);
  }
  return end;
}

// Returns false before doing any work if this request needs native fallback.
// Called after prefill, after streaming the last prompt token, and after the
// caller starts generation timing. It does not start/stop the timing itself.
static bool TryGenerateHeadSpec(const ModelConfig& config,
                                const RuntimeConfig& runtime,
                                const WeightsPtrs& weights,
                                Activations& activations, QBatch& qbatch,
                                MatMulEnv& env, TimingInfo& timing,
                                hwy::BitSet4096<>& non_eos, size_t max_steps,
                                const SampleFunc& sample) {
#if !GEMMA_HEAD_SPEC_HNSW
  return false;
#else
  if (!MMI8Flag("GEMMA_MM_I8_HEAD_SPEC") || qbatch.Size() != 1 ||
      qbatch.PrefixEnd(0) != 0 || runtime.layers_output ||
      runtime.activations_observer || runtime.image_tokens != nullptr ||
      (config.model != Model::GEMMA3_270M))
    return false;
  // These hooks observe every body invocation, including rejected drafts.
  if (GCPP_TENSOR_STATS || MMI8CalibrationCaptureEnabled() ||
      !HeadSpecNativeBodySettled(env))
    return false;
  const MatPtr& head = weights.lm_head.HasPtr()
                           ? weights.lm_head
                           : weights.embedder_input_embedding;
  auto& cache = MMI8WeightCache::Get();
  if (!cache.Enabled() || cache.ScalingEnabled() || !cache.Eligible(head) ||
      head.Rows() % 8 != 0)
    return false;
  MMConfig native;
  // Waiting for the native head to settle also ensures final-norm folding and
  // lazy quantized head preparation occurred in their original order.
  if (!HeadSpecNativeConfig(head, env, native)) return false;
  const MMI8B* B = CallUpcasted(
      &head, [&](const auto* typed) { return cache.Lookup(*typed, env); });
  if (B == nullptr || B->block_size == 0 || !B->packed_micro) return false;

  if (!MMI8Flag("GEMMA_MM_I8_HEAD_SPEC_ANN")) return false;
  const auto ann = HeadSpecAnnFind(head.RowBytes(0), HeadSpecAnnBinding(*B));
  if (!ann || !ann->index) return false;
  HeadSpecAnnQueryStats ann_stats;
  std::vector<float> ann_query;
  bool ann_failed = false;

  const size_t horizon = HWY_MIN(
      size_t{16},
      HWY_MAX(size_t{2}, MMI8EnvSize("GEMMA_MM_I8_HEAD_SPEC_HORIZON", 4)));
  const size_t k = head.Cols(), n = head.Rows();
  const std::vector<IndexRange> full_range{IndexRange(0, n)};
  MatStorageT<BF16> saved("head_spec_a", Extents2D(horizon, k),
                          env.ctx.allocator, MatPadding::kOdd);
  MatStorageT<float> draft_logits("head_spec_draft", Extents2D(1, n),
                                  env.ctx.allocator, MatPadding::kOdd);
  MatStorageT<float> verified("head_spec_full", Extents2D(horizon, n),
                              env.ctx.allocator, MatPadding::kOdd);
  MMI8AStorage storage(horizon, k, env.ctx.allocator);
  std::vector<int> drafts(horizon);
  std::vector<uint8_t> has_proposal(horizon);
  const bool skip_last_head = MMI8Flag("GEMMA_MM_I8_SPEC_SKIP_LAST_HEAD", true);
  RuntimeConfig private_runtime = runtime;
  private_runtime.layers_output = {};
  private_runtime.activations_observer = {};
  const size_t wrap = HeadSpecFirstWrap(qbatch.KV(0));
  size_t generated = 0, drafted = 0, accepted = 0, bonus = 0;
  while (generated < max_steps && non_eos.Any()) {
    const size_t start = qbatch.Pos(0);
    const size_t safe = start < wrap ? wrap - start : 0;
    size_t count = HWY_MIN(horizon, HWY_MIN(max_steps - generated, safe));
    if (count < 2 || ann_failed) {
      // This preserves native cache writes at/after wrap and the final row.
      Transformer(config, runtime, weights, activations, qbatch, env);
      SampleAndStream(config, runtime, weights, sample, activations, qbatch,
                      env, non_eos, timing);
      ++generated;
      continue;
    }
    const int previous = qbatch.PrevToken(0);
    activations.SetBatchSize(1);
    for (size_t row = 0; row < count; ++row) {
      // The last verified row is a bonus output: no later input depends on a
      // proposal for it. Its body and saved norm still feed exact verification.
      has_proposal[row] = !skip_last_head || row + 1 < count;
      Transformer(config, private_runtime, weights, activations, qbatch, env);
      FinalNormBatched(config, weights, activations, env);
      hwy::CopyBytes(activations.x_bf.Row(0), saved.Row(row), k * sizeof(BF16));
      saved.SetScale(activations.x_bf.Scale());
      if (has_proposal[row]) {
        if (!HeadSpecAnnPropose(*ann, activations.x_bf, *B, native,
                                draft_logits, storage, env, ann_query,
                                ann_stats, drafts[row])) {
          // Finish the current row exactly, verify the completed prefix,
          // then use native generation after an ANN failure.
          HeadSpecProject(activations.x_bf, *B, native, full_range,
                          draft_logits, storage, env);
          drafts[row] = HeadSpecDraftArgmax(draft_logits, full_range);
          count = row + 1;
          ann_failed = true;
        }
        qbatch.PrevToken(0) = drafts[row];
        ++drafted;
      }
      ++qbatch.MutablePos(0);
    }
    saved.OverrideRows(count);
    verified.OverrideRows(count);
    HeadSpecProject(saved, *B, native, full_range, verified, storage, env);
    hwy::BitSet4096<> verify_rows;
    for (size_t row = 0; row < count; ++row) verify_rows.Set(row);
    MaybeLogitsSoftCapBatched(config.final_cap, verified, verify_rows, env.ctx);
    qbatch.MutablePos(0) = start;
    qbatch.PrevToken(0) = previous;
    for (size_t row = 0; row < count && non_eos.Any(); ++row) {
      const size_t pos = qbatch.Pos(0) + 1;
      timing.NotifyGenerated(1);
      TokenAndProb token;
      ParallelFor(Parallelism::kFlat, size_t{1}, env.ctx, 0,
                  Callers::kSampleAndStream, [&](size_t, size_t worker) {
                    // Calling a custom sampler here also exposes the exact full
                    // logit row during validation. Never call it on unaccepted
                    // future contexts.
                    token = sample(0, pos, verified.RowSpan(row), worker);
                  });
      StreamAndUpdateEOS(0, pos, token.token, token.prob, config, runtime,
                         qbatch, true, non_eos);
      ++generated;
      if (!has_proposal[row]) {
        // This is the final row and every preceding proposal was accepted.
        // Read no draft ID for a row whose proposal head was skipped.
        ++bonus;
        break;
      }
      if (token.token != drafts[row]) break;
      ++accepted;
    }
    // All rejected KV rows are beyond MutablePos and are overwritten before
    // they can participate in a later causal M1 attention call.
  }
  activations.SetBatchSize(1);
  if (runtime.verbosity >= 1 || MMI8Flag("GEMMA_MM_I8_HEAD_SPEC_STATS")) {
    fprintf(stderr, "Head speculation: accepted %zu / %zu ANN drafts\n",
            accepted, drafted);
    fprintf(stderr, "Head speculation bonus: verified %zu outputs\n", bonus);
    fprintf(stderr,
            "Head ANN queries: count=%zu failures=%zu candidate_ids=%zu "
            "reconstruction_seconds=%.6f search_seconds=%.6f "
            "rerank_seconds=%.6f\n",
            ann_stats.queries, ann_stats.failures, ann_stats.candidate_ids,
            ann_stats.reconstruction_seconds, ann_stats.search_seconds,
            ann_stats.rerank_seconds);
  }
  return true;
#endif
}
#endif
