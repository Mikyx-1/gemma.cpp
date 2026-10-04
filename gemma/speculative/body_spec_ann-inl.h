// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0
// Optional270M W8+ANN draft for the unchanged exact1B verifier. Include inside
// the target namespace after BodySpecPrefillDraft. No public model ABI changes.
#if GEMMA_HEAD_SPEC_HNSW
#include "gemma/speculative/body_spec_ann_stopping-inl.h"
#include "gemma/speculative/body_spec_defer-inl.h"
#include "gemma/speculative/body_spec_pair-inl.h"
#include "gemma/speculative/body_spec_stopped_leaf-inl.h"

static std::shared_ptr<const HeadSpecAnnEntry> BodySpecAnnPrepareIndex(
    const MatPtr& head, const MMI8B* B) {
  if (B == nullptr || !HeadSpecAnnAdapter::Supported(*B)) return {};
  const auto binding = HeadSpecAnnBinding(*B);
  if (auto existing = HeadSpecAnnFind(head.RowBytes(0), binding))
    return existing;
  const auto parameters = HeadSpecAnnParameters();
  const size_t dimensions = HeadSpecAnnAdapter::FeatureDim(*B);
  auto entry = HeadSpecAnnPrepareOnce(
      head.RowBytes(0), binding, [&](HeadSpecAnnEntry& state) {
        const double hash_start = HeadSpecAnnNow();
        state.fingerprint = HeadSpecAnnFingerprint(*B);
        state.hash_seconds = HeadSpecAnnNow() - hash_start;
        if (state.fingerprint.empty()) {
          state.error = "SHA256 of head features failed";
          return;
        }
        const char* configured = getenv("GEMMA_MM_I8_HEAD_SPEC_ANN_CACHE_DIR");
        const std::string directory =
            configured != nullptr ? configured : ".temp/hnsw-head-i8-indexes";
        if (!directory.empty()) {
          state.index_path =
              (std::filesystem::path(directory) / (state.fingerprint + ".bin"))
                  .string();
          if (HeadSpecAnnLoad(state, dimensions, B->Rows(), parameters)) return;
        }
        struct BuildProgress {
          double start;
          double last;
        };
        BuildProgress progress{HeadSpecAnnNow(), HeadSpecAnnNow()};
        const auto notify = [](size_t done, size_t total, void* opaque) {
          auto& state = *static_cast<BuildProgress*>(opaque);
          const double now = HeadSpecAnnNow();
          if (done == total || now - state.last >= 10.0) {
            fprintf(stderr, "Head ANN build: %zu/%zu weights, %.3f seconds\n",
                    done, total, now - state.start);
            state.last = now;
          }
          return true;
        };
        const auto reader = [](const void* opaque, size_t token, float* row,
                               size_t dims) {
          return HeadSpecAnnAdapter::FillWeightRow(
              *static_cast<const MMI8B*>(opaque), token, row, dims);
        };
        const double build_start = HeadSpecAnnNow();
        state.index =
            HeadSpecAnnIndex::Build(dimensions, B->Rows(), parameters, B,
                                    reader, &state.error, notify, &progress);
        state.build_seconds = HeadSpecAnnNow() - build_start;
        if (!state.index) return;
        state.mode = "built";
        if (!state.index_path.empty() &&
            !HeadSpecAnnSave(state, dimensions, B->Rows(), parameters))
          state.error = "Index ready in memory; persistence failed";
      });
  return entry;
}

static bool TryGenerateBodySpecAnn(const ModelConfig& config,
                                   const RuntimeConfig& runtime,
                                   const WeightsPtrs& weights,
                                   Activations& activations, QBatch& qbatch,
                                   MatMulEnv& env, TimingInfo& timing,
                                   hwy::BitSet4096<>& non_eos, size_t max_steps,
                                   const SampleFunc& sample) {
  const double setup_start = hwy::platform::Now();
  BodySpecAnnDraftStopping stopping(
      getenv("GEMMA_MM_I8_BODY_SPEC_ANN_STOP_MARGIN"));
  if (!stopping.Valid() || (stopping.Enabled() &&
                            !MMI8Flag("GEMMA_MM_I8_SPEC_SKIP_LAST_HEAD", true)))
    return false;
  if (!MMI8Flag("GEMMA_MM_I8_BODY_SPEC") ||
      !MMI8Flag("GEMMA_MM_I8_BODY_SPEC_ANN") ||
      config.model != Model::GEMMA3_1B || qbatch.Size() != 1 ||
      qbatch.InitialPos(0) != 0 || qbatch.PrefixEnd(0) != 0 ||
      qbatch.Prompt(0).size() == 0 ||
      qbatch.Pos(0) != qbatch.Prompt(0).size() - 1 ||
      runtime.attention_impl != AttentionImpl::kFlash ||
      activations.attention_impl != AttentionImpl::kFlash ||
      !BodySpecHasFlatFlashCache(qbatch.KV(0)) ||
      env.ctx.pools.MaxWorkers() != 6 || runtime.top_k != 1 ||
      runtime.use_mtp || runtime.use_continuous_batching ||
      runtime.layers_output || runtime.activations_observer ||
      runtime.image_tokens != nullptr || GCPP_TENSOR_STATS ||
      MMI8CalibrationCaptureEnabled() || !HeadSpecNativeBodySettled(env) ||
      max_steps == 0 || !non_eos.Any()) {
    return false;
  }
  // Bound speculative writes by the actual flat allocation, not the rounded
  // length of the separately allocated tiled cache.
  const size_t seq_len = qbatch.KV(0).kv_cache.Rows();
  // Before wrap, rejected future KV rows cannot overwrite committed history.
  // Requests reaching wrap use the native path from the outset.
  if (qbatch.Pos(0) >= seq_len || max_steps > seq_len - qbatch.Pos(0))
    return false;
  auto& cache = MMI8WeightCache::Get();
  if (!cache.Enabled() || cache.ScalingEnabled()) return false;
  const MatPtr& target_head = weights.lm_head.HasPtr()
                                  ? weights.lm_head
                                  : weights.embedder_input_embedding;
  if (!target_head.HasPtr() || !cache.Eligible(target_head)) return false;
  const auto draft = FindBodySpecDraft(target_head.RowBytes(0));
  if (!draft || draft->Config().model != Model::GEMMA3_270M ||
      draft->Config().ple_dim != 0 ||
      draft->Config().vocab_size != config.vocab_size) {
    return false;
  }
  MMConfig native_head;
  if (!HeadSpecNativeConfig(target_head, env, native_head)) return false;
  const MMI8B* target_b = CallUpcasted(&target_head, [&](const auto* typed) {
    return cache.Lookup(*typed, env);
  });
  if (target_b == nullptr || !ExactI8PackedProjectionSupported(*target_b))
    return false;

  const size_t horizon = HWY_MIN(
      size_t{16},
      HWY_MAX(size_t{2}, MMI8EnvSize("GEMMA_MM_I8_BODY_SPEC_HORIZON", 4)));
  ExactDenseBody verifier(config, runtime, horizon, seq_len, env);
  if (!verifier.Prepare(config, weights, activations, env)) return false;

  // Draft state is request-local and shares only CPU workers with the target.
  // Its shape keys and fixed schedules cannot modify the target's MatMulEnv.
  MatMulEnv draft_env(env.ctx);
  draft_env.autotune = false;
  RuntimeConfig draft_runtime = runtime;
  draft_runtime.stream_token = {};
  draft_runtime.batch_stream_token = {};
  draft_runtime.accept_token = {};
  draft_runtime.sample_func = {};
  draft_runtime.layers_output = {};
  draft_runtime.activations_observer = {};
  draft_runtime.image_tokens = nullptr;
  draft_runtime.use_mtp = false;
  draft_runtime.use_continuous_batching = false;
  draft_runtime.verbosity = 0;
  draft_runtime.temperature = 0.0f;
  draft_runtime.top_k = 1;
  draft_runtime.decode_qbatch_size = 1;
  draft_runtime.attention_impl = AttentionImpl::kFlash;
  draft_runtime.kv_cache_type = Type::kBF16;
  const size_t draft_batch = HWY_MAX(
      size_t{1},
      HWY_MIN(runtime.prefill_tbatch_size, qbatch.Prompt(0).size() - 1));
  draft_runtime.prefill_tbatch_size = draft_batch;
  InferenceArgs draft_inference = draft->Inference();
  draft_inference.seq_len = seq_len;
  const auto& draft_config = draft->Config();
  const auto& draft_weights = draft->Weights();
  KVCache draft_kv(draft_config, draft_inference, draft_runtime,
                   env.ctx.allocator);
  if (!BodySpecHasFlatFlashCache(draft_kv.ToPtr()) ||
      draft_kv.kv_cache.Rows() != seq_len)
    return false;
  AllQueries draft_queries;
  draft_queries.Append(PerQuery{.prompt = qbatch.Prompt(0),
                                .mutable_pos = 0,
                                .initial_pos = 0,
                                .prefix_end = 0,
                                .kv_cache = draft_kv.ToPtr()});
  QBatch draft_query(0, 1, draft_queries);
  Activations draft_activations(draft_runtime, draft_config, draft_batch,
                                seq_len, env.ctx, draft_env.row_ptrs);
  const double prefill_start = hwy::platform::Now();
  BodySpecPrefillDraft(draft_config, draft_weights, draft_batch,
                       draft_activations, draft_query, draft_env);
  const double draft_prefill_seconds = hwy::platform::Now() - prefill_start;
  HWY_ASSERT(draft_query.Pos(0) == qbatch.Pos(0) &&
             draft_query.PrevToken(0) == qbatch.PrevToken(0));

  const MatPtr& draft_head = draft_weights.lm_head.HasPtr()
                                 ? draft_weights.lm_head
                                 : draft_weights.embedder_input_embedding;
  if (!draft_head.HasPtr() || !cache.Eligible(draft_head) ||
      draft_head.Rows() != target_head.Rows()) {
    return false;
  }
  const double head_prepare_start = hwy::platform::Now();
  // Fold/load the draft norm through the same ordering as FinalNormBatched,
  // before caching its quantized head. No target weights are changed here.
  (void)cache.NormWeights(draft_weights.final_norm_scale, {&draft_head},
                          draft_env);
  const MMI8B* draft_b = CallUpcasted(&draft_head, [&](const auto* typed) {
    return cache.Lookup(*typed, draft_env);
  });
  if (draft_b == nullptr || !ExactI8PackedProjectionSupported(*draft_b))
    return false;
  const size_t draft_k = draft_head.Cols(), vocab = target_head.Rows();
  // The proposal may choose any arithmetic partition. Only target verification
  // must retain its original M1 KC boundaries and sampled probabilities.
  const MMConfig draft_head_config(1, draft_k, vocab, 1, 1, draft_k, vocab,
                                   HWY_MIN(size_t{32}, draft_k), 32,
                                   MMOrder::kNT, 4);
  // Every request reaches this point only after fresh private prompt prefill.
  // Only the index/weight buffers are shared across requests. In particular,
  // the draft group inherits the target process (128 for the original 1B run)
  // and cannot reuse the ordinary 270M group64 index under another fingerprint.
  const double draft_head_prepare_seconds =
      hwy::platform::Now() - head_prepare_start;
  const auto binding = HeadSpecAnnBinding(*draft_b);
  const bool index_reused =
      bool(HeadSpecAnnFind(draft_head.RowBytes(0), binding));
  const double index_start = hwy::platform::Now();
  const auto ann = BodySpecAnnPrepareIndex(draft_head, draft_b);
  const double index_seconds = hwy::platform::Now() - index_start;
  const auto parameters = HeadSpecAnnParameters();
  const bool ready = ann && ann->index;
  if (!ready) {
    fprintf(
        stderr,
        "Body ANN setup: ready=%d model=270m group=%zu dim=%zu "
        "vocab=%zu ef=%zu candidates=%zu storage=%s mode=%s index_bytes=%zu "
        "fingerprint=%s index_seconds=%.6f draft_prefill_seconds=%.6f "
        "head_prepare_seconds=%.6f index_hash_seconds=%.6f "
        "index_load_seconds=%.6f index_build_seconds=%.6f "
        "index_save_seconds=%.6f "
        "setup_seconds=%.6f\n",
        int(ready), draft_b->block_size, draft_k, vocab, parameters.search_ef,
        parameters.candidates, "i8",
        index_reused ? "reused"
        : ann        ? ann->mode.c_str()
                     : "failed",
        ann ? ann->index_bytes : 0, ann ? ann->fingerprint.c_str() : "none",
        index_seconds, draft_prefill_seconds, draft_head_prepare_seconds,
        ann && !index_reused ? ann->hash_seconds : 0.0,
        ann && !index_reused ? ann->load_seconds : 0.0,
        ann && !index_reused ? ann->build_seconds : 0.0,
        ann && !index_reused ? ann->save_seconds : 0.0,
        hwy::platform::Now() - setup_start);
    return false;
  }
  const std::vector<IndexRange> full_range{IndexRange(0, vocab)};
  MatStorageT<float> proposal_logits("body_draft", Extents2D(1, vocab),
                                     env.ctx.allocator, MatPadding::kOdd);
  MatStorageT<float> verified("body_verify", Extents2D(horizon, vocab),
                              env.ctx.allocator, MatPadding::kOdd);
  MMI8AStorage draft_quantized(1, draft_k, env.ctx.allocator);
  MMI8AStorage target_quantized(horizon, target_head.Cols(), env.ctx.allocator);
  std::vector<int> drafts(horizon), inputs(horizon);
  std::vector<bool> has_proposal(horizon, false);
  HeadSpecAnnQueryStats ann_stats;
  std::vector<float> ann_query;
  bool ann_failed = false;
  size_t draft_rows = 0, verified_rows = 0, verified_spine_rows = 0;
  double draft_body_seconds = 0, draft_head_seconds = 0;
  double verify_body_seconds = 0, verify_head_seconds = 0;
  double sample_seconds = 0, native_seconds = 0;
  const bool skip_last_head = MMI8Flag("GEMMA_MM_I8_SPEC_SKIP_LAST_HEAD", true);
  const bool defer_last_body =
      MMI8Flag("GEMMA_MM_I8_BODY_SPEC_DEFER_LAST_BODY", false);
  const bool pair_enabled =
      MMI8Flag("GEMMA_MM_I8_BODY_SPEC_PAIR_CATCHUP", false);
  BodySpecPairedDraft paired_draft;
  size_t deferred_rows = 0, catchup_rows = 0;
  double catchup_seconds = 0;
  size_t generated = 0, proposed = 0, accepted = 0, corrected = 0;
  size_t rounds = 0, native_steps = 0, bonus = 0;
  const bool leaf_enabled =
      MMI8Flag("GEMMA_MM_I8_BODY_SPEC_STOPPED_LEAF", false);
  const bool leaf_policy_supported = leaf_enabled && horizon == 4 &&
                                     stopping.Enabled() && skip_last_head &&
                                     defer_last_body;
  ExactStoppedLeafKV leaf_snapshot;
  size_t leaf_offers = 0, leaf_computed = 0, leaf_selected = 0, leaf_bonus = 0;
  size_t leaf_restores = 0, leaf_catchup_rows = 0, leaf_snapshot_bytes = 0;
  size_t leaf_spine_batch_layers = 0;
  double leaf_restore_seconds = 0;
  fprintf(stderr,
          "Body ANN setup: ready=%d model=270m group=%zu dim=%zu "
          "vocab=%zu ef=%zu candidates=%zu storage=%s mode=%s index_bytes=%zu "
          "fingerprint=%s index_seconds=%.6f draft_prefill_seconds=%.6f "
          "head_prepare_seconds=%.6f index_hash_seconds=%.6f "
          "index_load_seconds=%.6f index_build_seconds=%.6f "
          "index_save_seconds=%.6f "
          "setup_seconds=%.6f\n",
          int(ready), draft_b->block_size, draft_k, vocab, parameters.search_ef,
          parameters.candidates, "i8",
          index_reused ? "reused"
          : ann        ? ann->mode.c_str()
                       : "failed",
          ann ? ann->index_bytes : 0, ann ? ann->fingerprint.c_str() : "none",
          index_seconds, draft_prefill_seconds, draft_head_prepare_seconds,
          ann && !index_reused ? ann->hash_seconds : 0.0,
          ann && !index_reused ? ann->load_seconds : 0.0,
          ann && !index_reused ? ann->build_seconds : 0.0,
          ann && !index_reused ? ann->save_seconds : 0.0,
          hwy::platform::Now() - setup_start);
  while (generated < max_steps && non_eos.Any()) {
    size_t count = HWY_MIN(horizon, max_steps - generated);
    if (count == 1 || ann_failed) {
      const double native_start = hwy::platform::Now();
      Transformer(config, runtime, weights, activations, qbatch, env);
      SampleAndStream(config, runtime, weights, sample, activations, qbatch,
                      env, non_eos, timing);
      ++generated;
      ++native_steps;
      native_seconds += hwy::platform::Now() - native_start;
      continue;
    }
    const size_t start = qbatch.Pos(0);
    const int previous = qbatch.PrevToken(0);
    HWY_ASSERT(draft_query.Pos(0) == start &&
               draft_query.PrevToken(0) == previous);
    inputs[0] = previous;
    BodySpecDeferredDraftBody deferred;
    BodySpecStoppedLeafRound leaf;
    for (size_t row = 0; row < count; ++row) {
      has_proposal[row] = !skip_last_head || row + 1 < count;
      if (deferred.TryDefer(defer_last_body, skip_last_head, row, count,
                            draft_query.Pos(0), inputs[row])) {
        HWY_ASSERT(!has_proposal[row]);
        // This row supplies no proposal. The exact verifier already has its
        // input; postpone private KV work until this bonus actually commits.
        ++deferred_rows;
        ++draft_query.MutablePos(0);
        break;
      }
      const double body_start = hwy::platform::Now();
      const bool reused_pair =
          pair_enabled && row == 0 &&
          paired_draft.Consume(draft_query.Pos(0), inputs[row],
                               draft_activations);
      if (!reused_pair) {
        Transformer(draft_config, draft_runtime, draft_weights,
                    draft_activations, draft_query, draft_env);
        ++draft_rows;
      }
      draft_body_seconds += hwy::platform::Now() - body_start;
      // The default path eagerly maintains private KV through the headless
      // input. The optional deferred path catches it up only after commitment.
      if (has_proposal[row]) {
        const double head_start = hwy::platform::Now();
        FinalNormBatched(draft_config, draft_weights, draft_activations,
                         draft_env);
        int second_token = -1;
        const bool ok = HeadSpecAnnPropose(
            *ann, draft_activations.x_bf, *draft_b, draft_head_config,
            proposal_logits, draft_quantized, draft_env, ann_query, ann_stats,
            drafts[row], stopping.Enabled() ? &second_token : nullptr);
        draft_head_seconds += hwy::platform::Now() - head_start;
        if (!ok) {
          // This body's input is valid, but no draft ID is available. Its exact
          // target row supplies a bonus; remaining tokens in this invocation
          // use the unchanged native1B loop. Never expose an uninitialized
          // proposal.
          has_proposal[row] = false;
          count = row + 1;
          ann_failed = true;
          ++draft_query.MutablePos(0);
          break;
        }
        if (row + 1 < count) inputs[row + 1] = drafts[row];
        draft_query.PrevToken(0) = drafts[row];
        ++proposed;
        if (stopping.Enabled()) {
          const bool valid_second = second_token >= 0 &&
                                    static_cast<size_t>(second_token) < vocab &&
                                    second_token != drafts[row];
          const uint8_t* attached = proposal_logits.GetRowPtrs() != nullptr
                                        ? proposal_logits.GetRowPtrs()[0]
                                        : nullptr;
          const float* scores = attached != nullptr
                                    ? reinterpret_cast<const float*>(attached)
                                    : proposal_logits.Row(0);
          const float best_score = scores[drafts[row]];
          const float second_score = valid_second ? scores[second_token] : 0.0f;
          if (stopping.Stop(best_score, second_score, valid_second,
                            draft_config.final_cap, row, count)) {
            if (leaf_policy_supported) {
              HWY_ASSERT(leaf.Offer(row, count, drafts[row], second_token));
              ++leaf_offers;
            }
            // Preserve the already initialized next input as a headless bonus.
            // The next iteration performs its private body or defers that body
            // through the same commitment/catch-up policy as a normal last row.
            count = row + 2;
          }
        }
      }
      ++draft_query.MutablePos(0);
    }
    ++rounds;
    const double verify_body_start = hwy::platform::Now();
    if (leaf.Matches(count) && !ann_failed &&
        deferred.Position() == start + count - 1 &&
        deferred.Input() == inputs[count - 1]) {
      leaf.SetComputed(verifier.RunStoppedSecondLeaf(
          inputs, count, start, leaf.Second(), config, runtime, weights,
          activations, qbatch, env, leaf_snapshot));
    }
    if (!leaf.Computed())
      verifier.Run(inputs, count, start, config, runtime, weights, activations,
                   qbatch, env);
    verify_body_seconds += hwy::platform::Now() - verify_body_start;
    verified_spine_rows += count;
    const size_t head_count = count + static_cast<size_t>(leaf.Computed());
    verified_rows += head_count;
    if (leaf.Computed()) {
      ++leaf_computed;
      leaf_snapshot_bytes = leaf_snapshot.SnapshotBytes();
      leaf_spine_batch_layers += leaf_snapshot.BatchAttentionLayers();
    }
    HWY_ASSERT(qbatch.Pos(0) == start && qbatch.PrevToken(0) == previous);
    verified.OverrideRows(head_count);
    const double verify_head_start = hwy::platform::Now();
    HeadSpecProject(verifier.Result().x_bf, *target_b, native_head, full_range,
                    verified, target_quantized, env);
    hwy::BitSet4096<> verify_rows;
    for (size_t row = 0; row < head_count; ++row) verify_rows.Set(row);
    MaybeLogitsSoftCapBatched(config.final_cap, verified, verify_rows, env.ctx);
    verify_head_seconds += hwy::platform::Now() - verify_head_start;
    const double sample_start = hwy::platform::Now();
    for (size_t row = 0; row < count && non_eos.Any(); ++row) {
      const size_t pos = qbatch.Pos(0) + 1;
      timing.NotifyGenerated(1);
      TokenAndProb token;
      ParallelFor(Parallelism::kFlat, size_t{1}, env.ctx, 0,
                  Callers::kSampleAndStream, [&](size_t, size_t worker) {
                    // Sample only a causal verified row. This also lets the
                    // untimed custom sampler capture every committed full-logit
                    // row for byte comparison.
                    token = sample(0, pos, verified.RowSpan(row), worker);
                  });
      StreamAndUpdateEOS(0, pos, token.token, token.prob, config, runtime,
                         qbatch, true, non_eos);
      ++generated;
      if (!has_proposal[row]) {
        // Normal last row or the truncated failed-ANN row has no draft ID.
        ++bonus;
        break;
      }
      if (token.token != drafts[row]) {
        ++corrected;
        if (leaf.IsSecondHit(row, token.token)) {
          ++leaf_selected;
          if (leaf.ShouldSampleLeaf(row, token.token, non_eos.Any(), generated,
                                    max_steps)) {
            // Storage rowcount is a sibling of spine rowcount-1, and consumes
            // second at start+count-1. Sample it only after its parent selected
            // that exact input; callbacks never see an off-path leaf.
            const size_t leaf_pos = qbatch.Pos(0) + 1;
            HWY_ASSERT(qbatch.Pos(0) == start + count - 1);
            timing.NotifyGenerated(1);
            TokenAndProb leaf_token;
            ParallelFor(Parallelism::kFlat, size_t{1}, env.ctx, 0,
                        Callers::kSampleAndStream, [&](size_t, size_t worker) {
                          leaf_token = sample(0, leaf_pos,
                                              verified.RowSpan(count), worker);
                        });
            StreamAndUpdateEOS(0, leaf_pos, leaf_token.token, leaf_token.prob,
                               config, runtime, qbatch, true, non_eos);
            ++generated;
            ++leaf_bonus;
            leaf.MarkCommitted();
          }
        }
        break;
      }
      ++accepted;
    }
    sample_seconds += hwy::platform::Now() - sample_start;
    if (leaf.Computed() && !leaf.Committed()) {
      const double restore_start = hwy::platform::Now();
      HWY_ASSERT(leaf_snapshot.Restore(qbatch.KV(0)));
      const double elapsed = hwy::platform::Now() - restore_start;
      verify_body_seconds += elapsed;
      leaf_restore_seconds += elapsed;
      ++leaf_restores;
    }
    if (deferred.NeedsCatchUp(qbatch.Pos(0), generated, max_steps,
                              non_eos.Any(), ann_failed)) {
      // The selected catch-up input and next first input are both already
      // committed. Optionally share their 270M projection passes, retaining the
      // next raw hidden row for one use at exactly its position and input ID.
      // No1B KV copy, ANN query, target operation or sampling occurs here.
      draft_query.MutablePos(0) = deferred.Position();
      draft_query.PrevToken(0) = leaf.CatchUpInput(deferred.Input());
      const double body_start = hwy::platform::Now();
      const bool paired =
          pair_enabled &&
          paired_draft.Try(non_eos.Any(), ann_failed, generated, max_steps,
                           deferred.Position(), draft_query.PrevToken(0),
                           qbatch.Pos(0), qbatch.PrevToken(0), draft_config,
                           draft_runtime, draft_weights, draft_activations,
                           draft_query, draft_env);
      if (!paired) {
        Transformer(draft_config, draft_runtime, draft_weights,
                    draft_activations, draft_query, draft_env);
      }
      const double elapsed = hwy::platform::Now() - body_start;
      draft_body_seconds += elapsed;
      // Paired time is reported separately as a subset of draft body time;
      // the legacy catch-up timer covers only unpaired eager calls.
      if (!paired) catchup_seconds += elapsed;
      draft_rows += paired ? 2 : 1;
      ++catchup_rows;
      if (leaf.Committed()) ++leaf_catchup_rows;
    }
    // The private cache contains the committed prefix; a paired call also
    // prepared the next known input. Its matching raw hidden row is consumed
    // once at the next round's first row. Other future rows remain disposable.
    // The target cache and the selected next input are unchanged.
    draft_query.MutablePos(0) = qbatch.Pos(0);
    draft_query.PrevToken(0) = qbatch.PrevToken(0);
  }
  activations.SetBatchSize(1);
  if (pair_enabled) {
    HWY_ASSERT(!paired_draft.Pending() &&
               paired_draft.consumed == paired_draft.batches &&
               paired_draft.batches <= paired_draft.attempts &&
               paired_draft.attempts <= catchup_rows);
    fprintf(stderr,
            "Body ANN paired draft: enabled=%d attempts=%zu "
            "batches=%zu fallback=%zu consumed=%zu paired_rows=%zu "
            "transformer_calls_avoided=%zu body_calls_saved=%zu "
            "preparation_seconds=%.6f batch_seconds=%.6f "
            "copy_seconds=%.6f\n",
            int(pair_enabled), paired_draft.attempts, paired_draft.batches,
            paired_draft.attempts - paired_draft.batches, paired_draft.consumed,
            2 * paired_draft.batches, 2 * paired_draft.batches,
            paired_draft.batches, paired_draft.preparation_seconds,
            paired_draft.batch_seconds, paired_draft.copy_seconds);
  }
  if (leaf_enabled) {
    HWY_ASSERT(leaf_computed <= leaf_offers && leaf_offers <= stopping.stops &&
               leaf_selected <= corrected && leaf_bonus <= leaf_selected &&
               leaf_restores + leaf_bonus == leaf_computed &&
               leaf_catchup_rows <= leaf_bonus &&
               verified_rows == verified_spine_rows + leaf_computed);
    fprintf(stderr,
            "Body ANN stopped leaf: enabled=1 policy_supported=%d "
            "offers=%zu computed=%zu declined=%zu selected=%zu "
            "bonus=%zu restores=%zu catchup_rows=%zu "
            "verified_spine_rows=%zu verified_leaf_rows=%zu "
            "snapshot_bytes=%zu spine_batch_layers=%zu "
            "restore_seconds=%.6f\n",
            int(leaf_policy_supported), leaf_offers, leaf_computed,
            leaf_offers - leaf_computed, leaf_selected, leaf_bonus,
            leaf_restores, leaf_catchup_rows, verified_spine_rows,
            leaf_computed, leaf_snapshot_bytes, leaf_spine_batch_layers,
            leaf_restore_seconds);
  }
  if (stopping.Enabled()) {
    fprintf(stderr,
            "Body ANN stopping: margin=%.9g checked=%zu guarded=%zu "
            "stops=%zu avoided_rows=%zu horizon=%zu\n",
            stopping.Margin(), stopping.checked, stopping.guarded,
            stopping.stops, stopping.avoided_rows, horizon);
  }
  if (defer_last_body) {
    HWY_ASSERT(catchup_rows <= deferred_rows &&
               catchup_rows <= bonus + leaf_bonus);
    HWY_ASSERT(draft_rows + deferred_rows - catchup_rows ==
               verified_spine_rows);
    fprintf(stderr,
            "Body ANN deferred body: enabled=1 deferred_rows=%zu "
            "catchup_rows=%zu omitted_rows=%zu catchup_seconds=%.6f\n",
            deferred_rows, catchup_rows, deferred_rows - catchup_rows,
            catchup_seconds);
  }
  // This opt-in path always reports actual execution and all timed work.
  {
    fprintf(stderr,
            "Body speculation: accepted %zu / %zu 270M drafts; rounds %zu; "
            "corrected %zu; native %zu\n",
            accepted, proposed, rounds, corrected, native_steps);
    fprintf(stderr, "Body speculation bonus: verified %zu outputs\n", bonus);
    fprintf(
        stderr,
        "Body ANN work: draft_rows=%zu verified_rows=%zu "
        "proposed=%zu queries=%zu failures=%zu candidate_ids=%zu rounds=%zu "
        "generated=%zu skip_last_head=%d\n",
        draft_rows, verified_rows, proposed, ann_stats.queries,
        ann_stats.failures, ann_stats.candidate_ids, rounds, generated,
        int(skip_last_head));
    fprintf(
        stderr,
        "Body ANN timing: draft_body_seconds=%.6f "
        "draft_head_seconds=%.6f verify_body_seconds=%.6f "
        "verify_head_seconds=%.6f sample_seconds=%.6f native_seconds=%.6f "
        "reconstruction_seconds=%.6f search_seconds=%.6f rerank_seconds=%.6f\n",
        draft_body_seconds, draft_head_seconds, verify_body_seconds,
        verify_head_seconds, sample_seconds, native_seconds,
        ann_stats.reconstruction_seconds, ann_stats.search_seconds,
        ann_stats.rerank_seconds);
  }
  return true;
}
#endif  // GEMMA_HEAD_SPEC_HNSW
