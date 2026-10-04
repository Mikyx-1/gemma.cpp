// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0
// Included inside the target namespace after native head eligibility helpers.
// Requires optional head_spec_ann_registry.h outside the target namespace.
#if defined(THIRD_PARTY_GEMMA_CPP_GEMMA_HEAD_SPEC_ANN_TOGGLE) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef THIRD_PARTY_GEMMA_CPP_GEMMA_HEAD_SPEC_ANN_TOGGLE
#undef THIRD_PARTY_GEMMA_CPP_GEMMA_HEAD_SPEC_ANN_TOGGLE
#else
#define THIRD_PARTY_GEMMA_CPP_GEMMA_HEAD_SPEC_ANN_TOGGLE
#endif

#include "gemma/speculative/head_spec_ann_adapter-inl.h"

static speculative::HnswHeadParameters HeadSpecAnnParameters() {
  speculative::HnswHeadParameters p;
  p.links = MMI8EnvSize("GEMMA_MM_I8_HEAD_SPEC_ANN_M", 16);
  p.construction_ef = MMI8EnvSize("GEMMA_MM_I8_HEAD_SPEC_ANN_EFC", 100);
  p.search_ef = MMI8EnvSize("GEMMA_MM_I8_HEAD_SPEC_ANN_EF", 128);
  p.candidates = MMI8EnvSize("GEMMA_MM_I8_HEAD_SPEC_ANN_CANDIDATES", 32);
  p.seed = MMI8EnvSize("GEMMA_MM_I8_HEAD_SPEC_ANN_SEED", 100);
  return p;
}

static std::string HeadSpecAnnBinding(const MMI8B& B) {
  const auto p = HeadSpecAnnParameters();
  std::ostringstream text;
  text << B.data << ':' << static_cast<const void*>(B.data->Row(0)) << ':'
       << B.scale << ':' << B.bias << ':' << B.a_pre_scale << ':' << B.Rows()
       << ':' << B.Cols() << ':' << B.block_size << ':' << B.dual_a << ':'
       << B.data->Stride() << ':' << HWY_TARGET << ':' << GEMMA_MM_I8_BIASED_B
       << ':' << MMI8RotateBlockSize() << ':' << MMI8HashBits() << ':'
       << MMI8Flag("GEMMA_MM_I8_MATCH_BF16_A") << ':' << p.links << ':'
       << p.construction_ef << ':' << p.search_ef << ':' << p.candidates << ':'
       << p.seed << ':' << true;
  return text.str();
}

static std::string HeadSpecAnnFingerprint(const MMI8B& B) {
  const auto p = HeadSpecAnnParameters();
  // Query ef/candidate count do not alter the stored graph. They are part of
  // the in-memory binding, but share the disk graph when only search changes.
  const nlohmann::json schema = {
      {"format", HeadSpecAnnIndex::ManifestFormat()},
      {"hnsw_commit", "d9b3608c83d83b46c96e25088cb1d729b29dcfe9"},
      {"features", HeadSpecAnnIndex::FeatureFormat()},
      {"rows", B.Rows()},
      {"cols", B.Cols()},
      {"block", B.block_size},
      {"biased_b", GEMMA_MM_I8_BIASED_B},
      {"dual_a", B.dual_a},
      {"has_bias", B.bias != nullptr},
      {"has_prescale", B.a_pre_scale != nullptr},
      {"rotate_block", MMI8RotateBlockSize()},
      {"hash_bits", MMI8HashBits()},
      {"match_bf16_a", MMI8Flag("GEMMA_MM_I8_MATCH_BF16_A")},
      {"M", p.links},
      {"ef_construction", p.construction_ef},
      {"seed", p.seed},
      {"size_t_bytes", sizeof(size_t)},
      {"float_bytes", sizeof(float)},
      {"hwy_target", HWY_TARGET}};
  const std::string header = schema.dump();
  HeadSpecAnnSha256 hash;
  hash.Add(header.data(), header.size());
  // Nominal row pointers within a packed tile do not describe ordinary rows.
  // Hash its contiguous eight-row payload and skip allocation padding.
  for (size_t n = 0; n < B.Rows(); n += 8)
    hash.Add(B.data->Row(n), 8 * B.Cols());
  hash.Add(B.scale, B.Rows() * (B.Cols() / B.block_size) * sizeof(float));
  if (B.bias != nullptr) hash.Add(B.bias, B.Rows() * sizeof(float));
  if (B.a_pre_scale != nullptr)
    hash.Add(B.a_pre_scale, B.Cols() * sizeof(float));
  return hash.Finish();
}

// Invoked after prefill and before generate_start. There are no request-derived
// index features: qbatch/runtime are consulted only to avoid preparing an index
// for a request that cannot use the exact speculative driver.
static void HeadSpecAnnPrepareForGeneration(const ModelConfig& config,
                                            const RuntimeConfig& runtime,
                                            const WeightsPtrs& weights,
                                            const QBatch& qbatch,
                                            MatMulEnv& env) {
  if (!MMI8Flag("GEMMA_MM_I8_HEAD_SPEC") ||
      !MMI8Flag("GEMMA_MM_I8_HEAD_SPEC_ANN") || qbatch.Size() != 1 ||
      qbatch.PrefixEnd(0) != 0 || runtime.layers_output ||
      runtime.activations_observer || runtime.image_tokens != nullptr ||
      (config.model != Model::GEMMA3_270M) || GCPP_TENSOR_STATS ||
      MMI8CalibrationCaptureEnabled() || !HeadSpecNativeBodySettled(env))
    return;
  const MatPtr& head = weights.lm_head.HasPtr()
                           ? weights.lm_head
                           : weights.embedder_input_embedding;
  auto& cache = MMI8WeightCache::Get();
  if (!head.HasPtr() || !cache.Enabled() || cache.ScalingEnabled() ||
      !cache.Eligible(head) || head.Rows() % 8 != 0)
    return;
  MMConfig native;
  if (!HeadSpecNativeConfig(head, env, native)) return;
  const double quant_start = HeadSpecAnnNow();
  const MMI8B* B = CallUpcasted(
      &head, [&](const auto* typed) { return cache.Lookup(*typed, env); });
  const double quant_seconds = HeadSpecAnnNow() - quant_start;
  if (B == nullptr || !HeadSpecAnnAdapter::Supported(*B)) return;
  const auto binding = HeadSpecAnnBinding(*B);
  if (HeadSpecAnnFind(head.RowBytes(0), binding)) return;
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
  fprintf(stderr,
          "Head ANN prepare: status=%s mode=%s rows=%zu dim=%zu M=%zu efc=%zu "
          "ef=%zu candidates=%zu quant_seconds=%.6f hash_seconds=%.6f "
          "load_seconds=%.6f build_seconds=%.6f save_seconds=%.6f "
          "prepare_seconds=%.6f index_bytes=%zu rss_before_bytes=%zu "
          "rss_after_bytes=%zu process_peak_rss_bytes=%zu fingerprint=%s "
          "storage=%s vector_bytes=%zu kernel=%s error=%s\n",
          entry->index ? "ready" : "fallback", entry->mode.c_str(), B->Rows(),
          dimensions, parameters.links, parameters.construction_ef,
          parameters.search_ef, parameters.candidates, quant_seconds,
          entry->hash_seconds, entry->load_seconds, entry->build_seconds,
          entry->save_seconds, entry->prepare_seconds, entry->index_bytes,
          entry->rss_before_bytes, entry->rss_after_bytes,
          entry->process_peak_rss_bytes, entry->fingerprint.c_str(), "i8",
          entry->index ? entry->index->VectorBytes() : 0,
          entry->index ? entry->index->Kernel() : "none", entry->error.c_str());
}

struct HeadSpecAnnQueryStats {
  size_t queries = 0;
  size_t failures = 0;
  size_t candidate_ids = 0;
  double reconstruction_seconds = 0;
  double search_seconds = 0;
  double rerank_seconds = 0;
};
static bool HeadSpecAnnPropose(const HeadSpecAnnEntry& entry,
                               const MatPtrT<BF16>& input, const MMI8B& head,
                               const MMConfig& native, MatPtrT<float>& logits,
                               MMI8AStorage& storage, MatMulEnv& env,
                               std::vector<float>& query,
                               HeadSpecAnnQueryStats& stats, int& proposed,
                               int* second_token = nullptr) {
  if (second_token != nullptr) *second_token = -1;
  ++stats.queries;
  // Explicit failure injection exercises the partial-draft fallback in tests.
  const size_t fail_after =
      MMI8EnvSize("GEMMA_MM_I8_HEAD_SPEC_ANN_FAIL_AFTER", 0);
  if (fail_after != 0 && stats.queries > fail_after) {
    ++stats.failures;
    return false;
  }
  const double before = HeadSpecAnnNow();
  query.resize(entry.index->FeatureDim());
  const bool filled = HeadSpecAnnAdapter::FillQuery(
      input, 0, head, storage, env.ctx, query.data(), query.size());
  stats.reconstruction_seconds += HeadSpecAnnNow() - before;
  if (!filled) {
    ++stats.failures;
    return false;
  }
  std::vector<int> candidates;
  std::string error;
  const double search_start = HeadSpecAnnNow();
  const bool found =
      entry.index->Candidates(query.data(), query.size(), candidates, &error);
  stats.search_seconds += HeadSpecAnnNow() - search_start;
  if (!found) {
    ++stats.failures;
    return false;
  }
  stats.candidate_ids += candidates.size();
  const double rerank_start = HeadSpecAnnNow();
  const bool ok =
      HeadSpecAnnRerankCandidates(input, head, native, candidates, logits,
                                  storage, env, proposed, second_token);
  stats.rerank_seconds += HeadSpecAnnNow() - rerank_start;
  if (!ok) ++stats.failures;
  return ok;
}
#endif
