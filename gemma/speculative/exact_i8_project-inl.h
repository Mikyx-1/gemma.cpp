// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Include inside the target namespace after matmul_i8-inl.h. These helpers
// batch independent rows while retaining the settled native M1 quantization
// policy, group arithmetic, and every KC output store. They neither create
// tuning keys nor advance existing tuning trials.
#if defined(THIRD_PARTY_GEMMA_CPP_GEMMA_EXACT_I8_PROJECT_TOGGLE) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef THIRD_PARTY_GEMMA_CPP_GEMMA_EXACT_I8_PROJECT_TOGGLE
#undef THIRD_PARTY_GEMMA_CPP_GEMMA_EXACT_I8_PROJECT_TOGGLE
#else
#define THIRD_PARTY_GEMMA_CPP_GEMMA_EXACT_I8_PROJECT_TOGGLE
#endif

// num_B must match the native operation: fused gate/up uses the TwoMatMul
// schedule, even though each branch is evaluated separately below.
static bool ExactI8CopyNativeConfig(const MMI8B& B, size_t num_B,
                                    const MatMulEnv& env, MMConfig& config,
                                    size_t cluster_idx = 0) {
  if (B.data == nullptr || (num_B != 1 && num_B != 2) ||
      cluster_idx >= env.per_cluster.size()) {
    return false;
  }
  const auto activation =
      B.block_size ? MMActivation::kI8Block : MMActivation::kI8;
  const auto wanted =
      MMKeys::KeyFromDims(1, B.Cols(), B.Rows(), num_B, activation);
  const auto& cluster = env.per_cluster[cluster_idx];
  const auto keys = cluster.keys.Keys();
  for (size_t i = 0; i < keys.size(); ++i) {
    if (keys[i] != wanted) continue;
    const auto* best = cluster.per_key[i].autotune.Best();
    if (best == nullptr) return false;
    config = *best;
    return true;
  }
  return false;
}

static bool ExactI8PackedProjectionSupported(const MMI8B& B) {
  return B.data != nullptr && B.scale != nullptr && B.packed_micro &&
         (B.block_size == 32 || B.block_size == 64 || B.block_size == 128) &&
         B.Cols() != 0 && B.Cols() % B.block_size == 0 && B.Rows() != 0 &&
         B.Rows() % 8 == 0;
}

// Four-column endpoints are supported by the existing packed N8 half-tile
// path. Sorted, disjoint ranges ensure tasks cannot race on output lanes.
// Sparse ranges retain the original B and full B.Rows() scale-group stride.
static bool ExactI8ProjectionTasks(const std::vector<IndexRange>& columns,
                                   size_t n, const MatMulEnv& env,
                                   std::vector<IndexRange>& tasks) {
  size_t total = 0, previous_end = 0;
  for (const auto& range : columns) {
    if (range.begin() < previous_end || range.begin() >= range.end() ||
        range.end() > n || range.begin() % 4 != 0 || range.end() % 4 != 0) {
      return false;
    }
    previous_end = range.end();
    total += range.Num();
  }
  const size_t workers = HWY_MAX(size_t{1}, env.ctx.pools.MaxWorkers());
  // The fused second branch uses the existing per-worker C tile, whose width
  // is bounded by kMaxNC. The same bound also keeps single-branch tasks small.
  const size_t grain = HWY_MIN(
      size_t{kMaxNC},
      HWY_MAX(size_t{8}, hwy::RoundUpTo(hwy::DivCeil(total, workers * 4), 8)));
  for (const auto& range : columns) {
    for (size_t begin = range.begin(); begin < range.end(); begin += grain) {
      tasks.emplace_back(
          begin, HWY_MIN(begin + grain, static_cast<size_t>(range.end())));
    }
  }
  return true;
}

// Internal entry point: native must be copied from the matching M1 key. Public
// wrappers below perform that lookup before any quantization or output write.
template <bool kTwo, typename TA, typename TC>
static HWY_NOINLINE bool ExactI8ProjectWithConfig(
    const MatPtrT<TA>& input, const MMI8B& B, const MMI8B* B2,
    const MMConfig& native, const std::vector<IndexRange>& columns,
    MatPtrT<TC>& output, MMI8AStorage& storage, MatMulEnv& env,
    const float* add, MMOptions options) {
  static_assert(IsBF16<TA>() || IsF32<TA>());
  static_assert(IsBF16<TC>() || IsF32<TC>());
  static_assert(!kTwo || IsBF16<TC>());
  if (!ExactI8PackedProjectionSupported(B)) return false;
  const size_t m = input.Rows(), k = input.Cols(), n = B.Rows();
  if (m == 0 || m > kMaxBatchSize || k != B.Cols() || output.Rows() != m ||
      output.Cols() != n || options.cluster_idx >= env.per_cluster.size()) {
    return false;
  }
  if constexpr (kTwo) {
    if (B2 == nullptr || !ExactI8PackedProjectionSupported(*B2) ||
        B2->Cols() != k || B2->Rows() != n || B.block_size != B2->block_size ||
        B.a_pre_scale != nullptr || B2->a_pre_scale != nullptr ||
        add != nullptr || options.func == nullptr) {
      return false;
    }
  } else {
    // Match MaybeMatMulI8's single-branch contract.
    if (B2 != nullptr || options.func != nullptr) return false;
  }

  std::vector<IndexRange> tasks;
  if (!ExactI8ProjectionTasks(columns, n, env, tasks)) return false;
  if (tasks.empty()) return true;

  // Batching rows must not disable native M1 dual-A, including when the
  // optional DUAL_A_M1_ONLY policy is enabled. QuantizeA also retains the
  // MATCH_BF16_A roundtrip for F32 inputs, as used by SumHeads.
  bool dual = MMI8UseDualA(B, 1);
  if constexpr (kTwo) dual = dual || MMI8UseDualA(*B2, 1);
  MMI8AView residual;
  const auto av =
      QuantizeA(input, storage, env.ctx, options.cluster_idx, B.a_pre_scale,
                B.block_size, dual ? &residual : nullptr);

  // A private fixed tuner satisfies MMArgs without touching MatMulEnv state.
  MMAutoTune<MMConfig> tuner;
  tuner.SetCandidates({native}, false);
  const MMArgs args(env, m, k, n, 1.0f, add, options, tuner, native);
  const auto ranges_kc = native.RangesOfKC(k);

  // Local row pointers support strided tensors and noncontiguous Q/K/V cache
  // destinations without overwriting env.row_ptrs or attaching new pointers.
  std::vector<uint8_t*> pointers(m);
  uint8_t** const attached = output.GetRowPtrs();
  for (size_t r = 0; r < m; ++r) {
    pointers[r] = attached != nullptr
                      ? attached[r]
                      : reinterpret_cast<uint8_t*>(output.Row(r));
  }
  const RowPtrs<TC> rows(pointers.data());
  ParallelFor(
      Parallelism::kFlat, tasks.size(), env.ctx, options.cluster_idx,
      Callers::kMMClusterForN, [&](size_t task, size_t worker) HWY_ATTR {
        const auto rn = tasks[task];
        const IndexRange rm(0, m);
        auto c1 = rows;
        auto out = c1.View(0, rn.begin(), rn.Num());
        for (size_t ki = 0; ki < ranges_kc.NumTasks(); ++ki) {
          const auto rk = ranges_kc.Range(ki);
          if (ki == 0) {
            MMI8Kernel::B3A2C0(av, B, rm, rk, rn, args, MMSetC(), out);
          } else {
            MMI8Kernel::B3A2C0(av, B, rm, rk, rn, args, MMAddC(), out);
          }
        }
        if constexpr (kTwo) {
          const StridedViewBF c2 =
              env.C_tiles.C(Extents2D(m, rn.Num()), worker);
          for (size_t ki = 0; ki < ranges_kc.NumTasks(); ++ki) {
            const auto rk = ranges_kc.Range(ki);
            if (ki == 0) {
              MMI8Kernel::B3A2C0(av, *B2, rm, rk, rn, args, MMSetC(), c2);
            } else {
              MMI8Kernel::B3A2C0(av, *B2, rm, rk, rn, args, MMAddC(), c2);
            }
          }
          // Both branches have completed every native BF16 KC store.
          // This is the same existing gate/up activation callback as
          // TwoMatMul uses.
          options.MaybeCallFunc(c1, rm, rn, c2, worker);
        }
      });
  return true;
}

// BF16/F32 input -> BF16/F32 output, with optional sparse output columns.
// False means unsupported or no settled matching M1 config; output is
// untouched. The caller owns storage with capacity for input.Rows() x
// input.Cols().
template <typename TA, typename TC>
static bool ExactI8Project(const MatPtrT<TA>& input, const MMI8B& B,
                           const std::vector<IndexRange>& columns,
                           MatPtrT<TC>& output, MMI8AStorage& storage,
                           MatMulEnv& env, const float* add = nullptr,
                           MMOptions options = MMOptions()) {
  MMConfig native;
  if (!ExactI8CopyNativeConfig(B, 1, env, native, options.cluster_idx)) {
    return false;
  }
  return ExactI8ProjectWithConfig<false>(input, B, nullptr, native, columns,
                                         output, storage, env, add, options);
}

template <typename TA, typename TC>
static bool ExactI8Project(const MatPtrT<TA>& input, const MMI8B& B,
                           MatPtrT<TC>& output, MMI8AStorage& storage,
                           MatMulEnv& env, const float* add = nullptr,
                           MMOptions options = MMOptions()) {
  if (B.data == nullptr) return false;
  return ExactI8Project(input, B, {IndexRange(0, B.Rows())}, output, storage,
                        env, add, options);
}

// Fused BF16 branches: output is the first branch, the second lives in the
// existing worker tile, and options.func combines them after both KC loops.
template <typename TA>
static bool ExactI8ProjectPair(const MatPtrT<TA>& input, const MMI8B& B1,
                               const MMI8B& B2,
                               const std::vector<IndexRange>& columns,
                               MatPtrT<BF16>& output, MMI8AStorage& storage,
                               MatMulEnv& env, MMOptions options) {
  MMConfig native;
  if (!ExactI8CopyNativeConfig(B1, 2, env, native, options.cluster_idx)) {
    return false;
  }
  return ExactI8ProjectWithConfig<true>(input, B1, &B2, native, columns, output,
                                        storage, env, nullptr, options);
}

template <typename TA>
static bool ExactI8ProjectPair(const MatPtrT<TA>& input, const MMI8B& B1,
                               const MMI8B& B2, MatPtrT<BF16>& output,
                               MMI8AStorage& storage, MatMulEnv& env,
                               MMOptions options) {
  if (B1.data == nullptr) return false;
  return ExactI8ProjectPair(input, B1, B2, {IndexRange(0, B1.Rows())}, output,
                            storage, env, options);
}

#endif  // include guard
