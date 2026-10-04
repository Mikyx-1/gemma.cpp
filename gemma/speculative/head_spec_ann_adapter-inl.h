// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Include inside the target namespace after exact_i8_project-inl.h. This file
// has no ANN-library dependency. Float features are proposals only; candidate
// reranking below uses the original packed head and native KC schedule.
#if defined(THIRD_PARTY_GEMMA_CPP_GEMMA_HEAD_SPEC_ANN_ADAPTER_TOGGLE) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef THIRD_PARTY_GEMMA_CPP_GEMMA_HEAD_SPEC_ANN_ADAPTER_TOGGLE
#undef THIRD_PARTY_GEMMA_CPP_GEMMA_HEAD_SPEC_ANN_ADAPTER_TOGGLE
#else
#define THIRD_PARTY_GEMMA_CPP_GEMMA_HEAD_SPEC_ANN_ADAPTER_TOGGLE
#endif

struct HeadSpecAnnAdapter {
  static bool Supported(const MMI8B& head) {
    return ExactI8PackedProjectionSupported(head);
  }

  // A constant query coordinate represents the optional per-channel bias.
  // The ANN wrapper may pad further with zeros for its SIMD distance kernel.
  static size_t FeatureDim(const MMI8B& head) {
    return head.Cols() + (head.bias != nullptr ? 1 : 0);
  }

  // Stream this row directly into the ANN index. Do not allocate an extra
  // full-vocabulary F32 matrix. Reads the immutable packed N8 x K4 payload.
  static bool FillWeightRow(const MMI8B& head, size_t token_id, float* out,
                            size_t output_dim) {
    if (!Supported(head) || token_id >= head.Rows() || out == nullptr ||
        output_dim < FeatureDim(head))
      return false;
    const size_t k = head.Cols();
    const auto* tile =
        reinterpret_cast<const MMI8BT*>(head.data->Row(token_id & ~size_t{7}));
    const size_t lane = (token_id % 8) * 4;
    for (size_t c = 0; c < k; ++c) {
      const size_t offset = (c & ~size_t{3}) * 8 + lane + c % 4;
      const int32_t q =
          static_cast<int32_t>(tile[offset]) - (GEMMA_MM_I8_BIASED_B ? 128 : 0);
      out[c] = static_cast<float>(q) *
               head.scale[(c / head.block_size) * head.Rows() + token_id];
    }
    if (head.bias != nullptr) out[k] = head.bias[token_id];
    for (size_t c = FeatureDim(head); c < output_dim; ++c) out[c] = 0.0f;
    return true;
  }

  // Reconstruct the query in the same rotated basis as the packed weights.
  // Quantize one row with the original M1 dual policy and MATCH_BF16_A
  // behavior. Its F32 dot with FillWeightRow is approximate: native group/KC
  // rounding is deliberately retained only by the exact packed reranking step.
  template <typename TA>
  static bool FillQuery(const MatPtrT<TA>& input, size_t row, const MMI8B& head,
                        MMI8AStorage& storage, ThreadingContext& ctx,
                        float* out, size_t output_dim, size_t cluster_idx = 0) {
    static_assert(IsBF16<TA>() || IsF32<TA>());
    if (!Supported(head) || row >= input.Rows() ||
        input.Cols() != head.Cols() || out == nullptr ||
        output_dim < FeatureDim(head))
      return false;
    MatPtrT<TA> one("head_ann_query", Extents2D(1, input.Cols()));
    one.SetPtr(const_cast<TA*>(input.Row(row)), input.Stride());
    one.SetScale(input.Scale());
    const bool dual = MMI8UseDualA(head, 1);
    MMI8AView residual;
    const auto av = QuantizeA(one, storage, ctx, cluster_idx, head.a_pre_scale,
                              head.block_size, dual ? &residual : nullptr);
    const size_t k = head.Cols();
    for (size_t c = 0; c < k; ++c) {
      const size_t group = c / head.block_size;
      float value = static_cast<float>(av.data.Row(0)[c]) *
                    av.scale[group * av.scale_stride];
      if (dual) {
        value += static_cast<float>(residual.data.Row(0)[c]) *
                 residual.scale[group * residual.scale_stride];
      }
      out[c] = value;
    }
    if (head.bias != nullptr) out[k] = 1.0f;
    for (size_t c = FeatureDim(head); c < output_dim; ++c) out[c] = 0.0f;
    return true;
  }
};

// Group candidate IDs into existing N8 tiles. Including each tile's neighbors
// avoids copying weights on every query and can improve the draft proposal.
static bool HeadSpecAnnCandidateRanges(const std::vector<int>& candidates,
                                       size_t vocab_size,
                                       std::vector<IndexRange>& ranges) {
  if (candidates.empty() || vocab_size == 0 || vocab_size % 8 != 0)
    return false;
  std::vector<size_t> tiles;
  tiles.reserve(candidates.size());
  for (int id : candidates) {
    if (id < 0 || static_cast<size_t>(id) >= vocab_size) return false;
    tiles.push_back(static_cast<size_t>(id) & ~size_t{7});
  }
  std::sort(tiles.begin(), tiles.end());
  tiles.erase(std::unique(tiles.begin(), tiles.end()), tiles.end());
  ranges.clear();
  for (size_t i = 0; i < tiles.size();) {
    const size_t begin = tiles[i];
    size_t end = begin + 8;
    while (++i < tiles.size() && tiles[i] == end) end += 8;
    ranges.emplace_back(begin, end);
  }
  return true;
}

// Diagnostic only: scan the same exact-reranked ranges, excluding the original
// winner. Ascending ranges plus strict comparison preserve lowest-ID ties.
// No additional ANN query or projection is performed.
static int HeadSpecAnnSecondToken(const float* scores,
                                  const std::vector<IndexRange>& ranges,
                                  size_t first) {
  int second = -1;
  float value = 0.0f;
  for (const auto& range : ranges)
    for (size_t id : range) {
      if (id == first) continue;
      if (second < 0 || scores[id] > value) {
        second = static_cast<int>(id);
        value = scores[id];
      }
    }
  return second;
}

// The caller supplies the target head's settled M1 config, never a config
// derived from ANN feature dimensions or the number of returned candidates.
// logits has one full-vocabulary row; unselected entries remain untouched.
// The returned token remains a proposal requiring full target verification.
template <typename TA>
static bool HeadSpecAnnRerankCandidates(
    const MatPtrT<TA>& input, const MMI8B& head, const MMConfig& native,
    const std::vector<int>& candidates, MatPtrT<float>& logits,
    MMI8AStorage& storage, MatMulEnv& env, int& proposed_token,
    int* second_token = nullptr) {
  if (second_token != nullptr) *second_token = -1;
  if (!HeadSpecAnnAdapter::Supported(head) || input.Rows() != 1 ||
      logits.Rows() != 1 || logits.Cols() != head.Rows())
    return false;
  std::vector<IndexRange> ranges;
  if (!HeadSpecAnnCandidateRanges(candidates, head.Rows(), ranges))
    return false;
  if (!ExactI8ProjectWithConfig<false>(input, head, nullptr, native, ranges,
                                       logits, storage, env, nullptr,
                                       MMOptions()))
    return false;
  const uint8_t* const attached =
      logits.GetRowPtrs() != nullptr ? logits.GetRowPtrs()[0] : nullptr;
  const float* const scores = attached != nullptr
                                  ? reinterpret_cast<const float*>(attached)
                                  : logits.Row(0);
  size_t best = ranges.front().begin();
  float value = scores[best];
  for (const auto& rn : ranges) {
    for (size_t id : rn) {
      if (scores[id] > value) {
        best = id;
        value = scores[id];
      }
    }
  }
  proposed_token = static_cast<int>(best);
  if (second_token != nullptr)
    *second_token = HeadSpecAnnSecondToken(scores, ranges, best);
  return true;
}

#endif
