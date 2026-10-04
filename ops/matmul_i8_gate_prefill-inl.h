// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0
// Gate/up prefill kernels; include after quantizers and compact-prefix helpers.
#if defined(THIRD_PARTY_GEMMA_CPP_MATMUL_I8_GATE_PREFILL_TOGGLE) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef THIRD_PARTY_GEMMA_CPP_MATMUL_I8_GATE_PREFILL_TOGGLE
#undef THIRD_PARTY_GEMMA_CPP_MATMUL_I8_GATE_PREFILL_TOGGLE
#else
#define THIRD_PARTY_GEMMA_CPP_MATMUL_I8_GATE_PREFILL_TOGGLE
#endif
// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0
// Gate/up blocking with unchanged grouped arithmetic and KC stores.
// Include within target namespace after native and private compact-prefix
// helpers.
#if HWY_TARGET == HWY_AVX2 && GEMMA_MM_I8_BIASED_B && defined(__GNUC__) && \
    !defined(__clang__) && HWY_ARCH_X86_64
namespace mmi8_gate_blocked512 {
namespace cp = mmi8_compact_prefix;
template <size_t kGroup, size_t kStep, class BT, class Tag, class CView>
static void ReferenceRange(const MMI8AView& A, const BT& B,
                           const IndexRange& rm, const IndexRange& kc,
                           const IndexRange& rn, const float* add, Tag tag,
                           CView C) {
  if constexpr (kStep == 1) {
    MMI8MicroPrefillRange48<kGroup>(A, B, rm, kc, rn, add, tag, C);
  } else {
    size_t n = rn.begin();
    for (; n + 48 <= rn.end(); n += 48)
      for (size_t r = 0; r < rm.Num(); ++r)
        cp::CompactTile48<kGroup>(A, B, rm.begin() + r, n, kc,
                                  MMI8Bias(B, add, n), tag,
                                  C.View(r, n - rn.begin(), 48));
    for (; n + 32 <= rn.end(); n += 32)
      for (size_t r = 0; r < rm.Num(); ++r)
        cp::CompactDecodeTile<32, kGroup>(A, B, rm.begin() + r, n, kc,
                                          MMI8Bias(B, add, n), tag,
                                          C.View(r, n - rn.begin(), 32));
    if (n + 16 <= rn.end()) {
      for (size_t r = 0; r < rm.Num(); ++r)
        cp::CompactDecodeTile<16, kGroup>(A, B, rm.begin() + r, n, kc,
                                          MMI8Bias(B, add, n), tag,
                                          C.View(r, n - rn.begin(), 16));
      n += 16;
    }
    if (n < rn.end()) {
      HWY_ASSERT(rn.end() - n == 8);
      for (size_t r = 0; r < rm.Num(); ++r)
        cp::CompactNativeRowTail<kGroup, false, true>(
            A, B, IndexRange(rm.begin() + r, rm.begin() + r + 1), kc,
            IndexRange(n, rn.end()), add, tag, C.View(r, n - rn.begin(), 8));
    }
  }
}

template <size_t kBlock, size_t kPrefixStep, bool kFirst, class BT>
static HWY_NOINLINE void AccumulateBlock(const MMI8AView& A, const BT& B,
                                         size_t row_a, size_t row_b,
                                         const IndexRange& range_kc,
                                         float* scratch) {
  static_assert(kBlock == 32 || kBlock == 64 || kBlock == 128);
  const hn::ScalableTag<int32_t> di;
  const hn::ScalableTag<float> df;
  using VI = hn::Vec<decltype(di)>;
  using VF = hn::Vec<decltype(df)>;
  const auto* ar = A.data.Row(row_a);
  const auto& residual = *A.residual;
  const auto* er = residual.data.Row(row_a);
  const auto* prefix = A.prefix + row_a * A.prefix_stride;
  const auto* prefix1 = residual.prefix + row_a * residual.prefix_stride;
  const auto* b0 = reinterpret_cast<const uint8_t*>(B.data->Row(row_b + 0));
  const auto* b1 = reinterpret_cast<const uint8_t*>(B.data->Row(row_b + 8));
  const auto* b2 = reinterpret_cast<const uint8_t*>(B.data->Row(row_b + 16));
  const auto* b3 = reinterpret_cast<const uint8_t*>(B.data->Row(row_b + 24));
  const auto* b4 = reinterpret_cast<const uint8_t*>(B.data->Row(row_b + 32));
  const auto* b5 = reinterpret_cast<const uint8_t*>(B.data->Row(row_b + 40));
  VF f0 = hn::Zero(df), f1 = f0, f2 = f0, f3 = f0, f4 = f0, f5 = f0;
  HWY_DASSERT(range_kc.Num() != 0);
  if constexpr (!kFirst) {
    f0 = hn::LoadU(df, scratch + 0);
    f1 = hn::LoadU(df, scratch + 8);
    f2 = hn::LoadU(df, scratch + 16);
    f3 = hn::LoadU(df, scratch + 24);
    f4 = hn::LoadU(df, scratch + 32);
    f5 = hn::LoadU(df, scratch + 40);
  }
  size_t group = range_kc.begin() / kBlock;
  for (size_t c = range_kc.begin(); c < range_kc.end(); ++group) {
    const size_t num_k =
        HWY_MIN(kBlock - c % kBlock, static_cast<size_t>(range_kc.end()) - c);
    HWY_DASSERT(num_k >= 4 && num_k % 4 == 0);
    VI p0, p1, p2, p3, p4, p5, q0, q1, q2, q3, q4, q5;
    VI a, e, w;
    size_t offset;
    // Twelve integer chains + two activation broadcasts + one B scratch use
    // fifteen YMM registers. Six live F32 values may spill at group boundaries;
    // the compiler cannot insert stack traffic into this complete K4 loop.
    asm("vpxor %[p0], %[p0], %[p0]\n\t"
        "vpxor %[q0], %[q0], %[q0]\n\t"
        "vpxor %[p1], %[p1], %[p1]\n\t"
        "vpxor %[q1], %[q1], %[q1]\n\t"
        "vpxor %[p2], %[p2], %[p2]\n\t"
        "vpxor %[q2], %[q2], %[q2]\n\t"
        "vpxor %[p3], %[p3], %[p3]\n\t"
        "vpxor %[q3], %[q3], %[q3]\n\t"
        "vpxor %[p4], %[p4], %[p4]\n\t"
        "vpxor %[q4], %[q4], %[q4]\n\t"
        "vpxor %[p5], %[p5], %[p5]\n\t"
        "vpxor %[q5], %[q5], %[q5]\n\t"
        "xor %[offset], %[offset]\n\t"
        ".Lgemma_i8_n48_group_%=:\n\t"
        "vpbroadcastd (%[ar], %[offset]), %[a]\n\t"
        "vpbroadcastd (%[er], %[offset]), %[e]\n\t"
        "vmovdqu (%[b0], %[offset], 8), %[w]\n\t"
        "%{vex%} vpdpbusd %[a], %[w], %[p0]\n\t"
        "%{vex%} vpdpbusd %[e], %[w], %[q0]\n\t"
        "vmovdqu (%[b1], %[offset], 8), %[w]\n\t"
        "%{vex%} vpdpbusd %[a], %[w], %[p1]\n\t"
        "%{vex%} vpdpbusd %[e], %[w], %[q1]\n\t"
        "vmovdqu (%[b2], %[offset], 8), %[w]\n\t"
        "%{vex%} vpdpbusd %[a], %[w], %[p2]\n\t"
        "%{vex%} vpdpbusd %[e], %[w], %[q2]\n\t"
        "vmovdqu (%[b3], %[offset], 8), %[w]\n\t"
        "%{vex%} vpdpbusd %[a], %[w], %[p3]\n\t"
        "%{vex%} vpdpbusd %[e], %[w], %[q3]\n\t"
        "vmovdqu (%[b4], %[offset], 8), %[w]\n\t"
        "%{vex%} vpdpbusd %[a], %[w], %[p4]\n\t"
        "%{vex%} vpdpbusd %[e], %[w], %[q4]\n\t"
        "vmovdqu (%[b5], %[offset], 8), %[w]\n\t"
        "%{vex%} vpdpbusd %[a], %[w], %[p5]\n\t"
        "%{vex%} vpdpbusd %[e], %[w], %[q5]\n\t"
        "add $4, %[offset]\n\t"
        "cmp %[num_k], %[offset]\n\t"
        "jb .Lgemma_i8_n48_group_%="
        : [p0] "=&x"(p0.raw), [q0] "=&x"(q0.raw), [p1] "=&x"(p1.raw),
          [q1] "=&x"(q1.raw), [p2] "=&x"(p2.raw), [q2] "=&x"(q2.raw),
          [p3] "=&x"(p3.raw), [q3] "=&x"(q3.raw), [p4] "=&x"(p4.raw),
          [q4] "=&x"(q4.raw), [p5] "=&x"(p5.raw), [q5] "=&x"(q5.raw),
          [a] "=&x"(a.raw), [e] "=&x"(e.raw), [w] "=&x"(w.raw),
          [offset] "=&r"(offset)
        : [b0] "r"(b0 + c * 8), [b1] "r"(b1 + c * 8), [b2] "r"(b2 + c * 8),
          [b3] "r"(b3 + c * 8), [b4] "r"(b4 + c * 8), [b5] "r"(b5 + c * 8),
          [ar] "r"(ar + c), [er] "r"(er + c), [num_k] "r"(num_k)
        : "cc", "memory");
    const int32_t correction =
        (prefix[(c + num_k) / kPrefixStep] - prefix[c / kPrefixStep]) * 128;
    const int32_t correction1 =
        (prefix1[(c + num_k) / kPrefixStep] - prefix1[c / kPrefixStep]) * 128;
    const auto as = hn::Set(df, A.scale[row_a + group * A.scale_stride]);
    const auto as1 =
        hn::Set(df, residual.scale[row_a + group * residual.scale_stride]);
    const auto accumulate = [&](VI primary, VI extra, size_t n,
                                VF& value) HWY_ATTR {
      const auto bs = hn::LoadU(df, B.scale + group * B.Rows() + row_b + n);
      const auto scale = hn::Mul(bs, as);
      primary = hn::Sub(primary, hn::Set(di, correction));
      value = hn::MulAdd(hn::ConvertTo(df, primary), scale, value);
      const auto scale1 = hn::Mul(bs, as1);
      extra = hn::Sub(extra, hn::Set(di, correction1));
      value = hn::MulAdd(hn::ConvertTo(df, extra), scale1, value);
    };
    accumulate(p0, q0, 0, f0);
    accumulate(p1, q1, 8, f1);
    accumulate(p2, q2, 16, f2);
    accumulate(p3, q3, 24, f3);
    accumulate(p4, q4, 32, f4);
    accumulate(p5, q5, 40, f5);
    c += num_k;
  }
  hn::StoreU(f0, df, scratch + 0);
  hn::StoreU(f1, df, scratch + 8);
  hn::StoreU(f2, df, scratch + 16);
  hn::StoreU(f3, df, scratch + 24);
  hn::StoreU(f4, df, scratch + 32);
  hn::StoreU(f5, df, scratch + 40);
}

template <size_t kBlock, size_t kPrefixStep, class BT, class Tag, class CView>
static HWY_NOINLINE void RangeImpl(const MMI8AView& A, const BT& B,
                                   const IndexRange& range_mc,
                                   const IndexRange& range_kc,
                                   const IndexRange& range_nc, const float* add,
                                   Tag tag, CView C, float* scratch,
                                   size_t scratch_rows) {
  constexpr size_t kInner = 512;
  static_assert(kPrefixStep == 1 || kPrefixStep == 32);
  static_assert(kInner >= kBlock && kInner % kBlock == 0);
  HWY_DASSERT(scratch_rows >= range_mc.Num() && range_kc.Num() != 0);
  HWY_DASSERT(range_nc.begin() % 8 == 0 && range_nc.Num() % 4 == 0);
  HWY_DASSERT(range_kc.begin() % 4 == 0 && range_kc.end() % 4 == 0);
  const hn::ScalableTag<float> df;
  using TC = hwy::RemoveCvRef<decltype(C.Row(0)[0])>;
  const hn::Rebind<TC, decltype(df)> dc;
  size_t nc = range_nc.begin();
  for (; nc + 48 <= range_nc.end(); nc += 48) {
    const size_t begin = range_kc.begin();
    const size_t first_end = HWY_MIN(static_cast<size_t>(range_kc.end()),
                                     begin + kInner - begin % kInner);
    for (size_t r = 0; r < range_mc.Num(); ++r)
      AccumulateBlock<kBlock, kPrefixStep, true>(A, B, range_mc.begin() + r, nc,
                                                 IndexRange(begin, first_end),
                                                 scratch + r * 48);
    for (size_t c = first_end; c < range_kc.end(); c += kInner) {
      const size_t end =
          HWY_MIN(static_cast<size_t>(range_kc.end()), c + kInner);
      for (size_t r = 0; r < range_mc.Num(); ++r)
        AccumulateBlock<kBlock, kPrefixStep, false>(A, B, range_mc.begin() + r,
                                                    nc, IndexRange(c, end),
                                                    scratch + r * 48);
    }
    // C/bias and BF16 narrowing occur only at the original KC boundary.
    const float* bias = MMI8Bias(B, add, nc);
    for (size_t r = 0; r < range_mc.Num(); ++r) {
      for (size_t n = 0; n < 48; n += 8) {
        auto value = hn::LoadU(df, scratch + r * 48 + n);
        TC* const pos = C.Row(r) + nc - range_nc.begin() + n;
        if constexpr (hwy::IsSame<Tag, MMAddC>()) {
          value = hn::Add(value, F32FromTC(dc, hn::LoadU(dc, pos)));
        } else {
          static_assert(hwy::IsSame<Tag, MMSetC>());
          if (bias != nullptr) value = hn::Add(value, hn::LoadU(df, bias + n));
        }
        hn::StoreU(TCFromF32(dc, value), dc, pos);
      }
    }
  }
  if (nc < range_nc.end()) {
    const size_t tail = static_cast<size_t>(range_nc.end()) - nc;
    ReferenceRange<kBlock, kPrefixStep>(
        A, B, range_mc, range_kc, IndexRange(nc, range_nc.end()), add, tag,
        C.View(0, nc - range_nc.begin(), tail));
  }
}

template <size_t kGroup, size_t kStep, class BT, class Tag, class CView>
static bool Range(const MMI8AView& A, const BT& B, const IndexRange& rm,
                  const IndexRange& kc, const IndexRange& rn, const float* add,
                  Tag tag, CView C, float* scratch, size_t scratch_rows) {
  static_assert(kStep == 1 || kStep == 32);
  if (scratch == nullptr || scratch_rows < rm.Num()) return false;
  const size_t alignment = kStep == 1 ? 4 : 32;
  if (!MMI8NativeVNNI() || B.data == nullptr || B.data->Row(0) == nullptr ||
      !B.packed_micro || !B.dual_a || B.block_size != kGroup ||
      B.scale == nullptr || A.block_size != kGroup || A.residual == nullptr ||
      A.residual->block_size != kGroup || A.data.Row(0) == nullptr ||
      A.residual->data.Row(0) == nullptr || A.scale == nullptr ||
      A.residual->scale == nullptr || A.prefix == nullptr ||
      A.residual->prefix == nullptr || A.data.Cols() != B.Cols() ||
      A.residual->data.Cols() != B.Cols() || B.Cols() % kGroup ||
      rm.Num() == 0 || kc.Num() == 0 || kc.begin() % alignment ||
      kc.end() % alignment || kc.end() > B.Cols() || rn.Num() == 0 ||
      rn.begin() % 8 || rn.Num() % 8 || rn.end() > B.Rows())
    return false;
  if constexpr (kStep == 32) {
    if (A.prefix_stride != cp::PrefixStride(B.Cols()) ||
        A.residual->prefix_stride != cp::PrefixStride(B.Cols()))
      return false;
  } else {
    if (A.prefix_stride < B.Cols() + 1 ||
        A.residual->prefix_stride < B.Cols() + 1)
      return false;
  }
  RangeImpl<kGroup, kStep>(A, B, rm, kc, rn, add, tag, C, scratch,
                           scratch_rows);
  return true;
}

// Scratch holds at least scratch_rows*48 floats and must be disjoint from A, B,
// C and metadata. First512 block overwrites every active scratch row before any
// later block reload. Allocation capacity may be reused; contents are never
// cached. Full caller retains ordinary row bounds and complete-prefix contract.
template <size_t kGroup, class BT, class Tag, class CView>
static bool FullRange(const MMI8AView& A, const BT& B, const IndexRange& rm,
                      const IndexRange& kc, const IndexRange& rn,
                      const float* add, Tag tag, CView C, float* scratch,
                      size_t scratch_rows) {
  return Range<kGroup, 1>(A, B, rm, kc, rn, add, tag, C, scratch, scratch_rows);
}
// Compact caller carries an explicit wrapper with row/K extents; its view can
// never enter any full-prefix tail. Unsupported calls leave C untouched.
template <size_t kGroup, class BT, class Tag, class CView>
static bool CompactRange(const cp::Activation& A, const BT& B,
                         const IndexRange& rm, const IndexRange& kc,
                         const IndexRange& rn, const float* add, Tag tag,
                         CView C, float* scratch, size_t scratch_rows) {
  if (!A.ready || A.group != kGroup || rm.end() > A.rows || B.data == nullptr ||
      A.k != B.Cols())
    return false;
  return Range<kGroup, 32>(A.view, B, rm, kc, rn, add, tag, C, scratch,
                           scratch_rows);
}
}  // namespace mmi8_gate_blocked512
#endif

#if HWY_TARGET == HWY_AVX2 && GEMMA_MM_I8_BIASED_B && defined(__GNUC__) && \
    !defined(__clang__) && HWY_ARCH_X86_64
namespace mmi8_gate_prefill {
namespace cp = mmi8_compact_prefix;
namespace blocked = mmi8_gate_blocked512;

// This first route is intentionally limited to the measured selected1B gate
// schedule. A constructor comparison also checks its private partition
// multiples without exposing them or changing MMConfig's public layout.
static HWY_INLINE bool WeightSupported(const MMI8B& b, size_t m) {
  if (b.data == nullptr || b.Rows() != 6912 || b.Cols() != 1152) return false;
  return b.data->Row(0) != nullptr && b.scale != nullptr &&
         b.block_size == 128 && b.packed_micro && b.dual_a &&
         MMI8UseDualA(b, m) && b.a_pre_scale == nullptr;
}
static HWY_INLINE bool Supported(size_t m, size_t k, size_t n,
                                 const MMI8B& first, const MMI8B& second,
                                 const MMConfig* settled, MMOptions options) {
  if (m <= 16 || m > 77 || k != 1152 || n != 6912 || settled == nullptr ||
      options.func == nullptr || !MMI8FastMicro() || !MMI8NativeVNNI() ||
      !WeightSupported(first, m) || !WeightSupported(second, m) ||
      k % MMI8RotateBlockSize())
    return false;
  static const bool n48 = MMI8Flag("GEMMA_MM_I8_MICRO_PREFILL_N48", false);
  if (!n48 || MMI8Flag("GEMMA_MM_I8_MICRO_DECODE", false)) return false;
  const MMConfig expected(m, k, n, 1, 77, 1152, 96, 32, 32, MMOrder::kNT_MT, 1);
  if (memcmp(settled, &expected, sizeof(expected)) != 0) return false;

  // Validate the same actual ranges that MMLoops will consume before changing
  // prefix storage. There is no fallback after private preparation begins.
  const auto rows = settled->RangesOfMC(m);
  const auto columns = settled->RangesOfNC(n);
  const auto kc = settled->RangesOfKC(k);
  if (rows.NumTasks() != 1 || columns.NumTasks() != 72 || kc.NumTasks() != 1)
    return false;
  if (size_t(rows.Range(0).begin()) != 0 || size_t(rows.Range(0).end()) != m ||
      size_t(kc.Range(0).begin()) != 0 || size_t(kc.Range(0).end()) != k)
    return false;
  size_t end = 0;
  for (size_t i = 0; i < columns.NumTasks(); ++i) {
    const auto range = columns.Range(i);
    if (size_t(range.begin()) != end || range.Num() != 96 ||
        size_t(range.begin()) % 8 || range.Num() % 16 ||
        size_t(range.end()) > n)
      return false;
    end = size_t(range.end());
  }
  return end == n;
}

// One non-template per-target TLS instance is shared by B1 RowPtrs and B2
// StridedView instantiations. It never aliases env.C_tiles, which retains B2
// for the unchanged callback. Every originalKC's first block overwrites its
// carry.
inline float* Scratch(size_t rows) {
  HWY_ASSERT(rows <= 77);
  thread_local hwy::AlignedVector<float> carry;
  if (carry.size() < rows * 48) carry.resize(rows * 48);
  return carry.data();
}

template <size_t kPrefixStep>
struct BlockedView {
  const MMI8AView* view;
};

template <size_t kPrefixStep>
class BlockedKernel {
 public:
  using AView = BlockedView<kPrefixStep>;

  template <class BT, class Tag, class CView>
  static void B3A2C0(const AView a, const BT& b, const IndexRange& rows,
                     const IndexRange& kc, const IndexRange& columns,
                     const MMArgs& args, Tag tag, CView output) {
    const bool handled = blocked::Range<128, kPrefixStep>(
        *a.view, b, rows, kc, columns, args.add, tag, output,
        Scratch(rows.Num()), rows.Num());
    if (!handled)
      HWY_ABORT("Validated blocked gate range unexpectedly declined");
  }

  // Define our own dispatch so a compact prefix can never enter an inherited
  // native static ForeachKC. Initial guard admits only one original full-KC.
  template <class BT, class CView>
  static void ForeachKC(const AView a, const BT& b, const IndexRange& rows,
                        const IndexRangePartition& kc,
                        const IndexRange& columns, const MMArgs& args,
                        CView output) {
    kc.VisitFirst([&](const IndexRange& range) {
      B3A2C0(a, b, rows, range, columns, args, MMSetC(), output);
    });
    kc.VisitRemaining([&](const IndexRange& range) {
      B3A2C0(a, b, rows, range, columns, args, MMAddC(), output);
    });
  }
};

}  // namespace mmi8_gate_prefill
#endif

static HWY_INLINE bool MMI8TryGatePrefill(
    const MatPtrT<BF16>& input, const MMI8B& first, const MMI8B& second,
    MatMulEnv& env, MatPtrT<BF16>& output, RowPtrs<BF16> output_rows,
    MMI8AStorage& storage, MMOptions options, MMAutoTune<MMConfig>& tuner) {
#if HWY_TARGET == HWY_AVX2 && GEMMA_MM_I8_BIASED_B && defined(__GNUC__) && \
    !defined(__clang__) && HWY_ARCH_X86_64
  static const bool compact =
      MMI8Flag("GEMMA_MM_I8_GATE_COMPACT_PREFIX", false);
  static const bool blocked512 = MMI8Flag("GEMMA_MM_I8_GATE_BLOCKED512", false);
  if (!compact && !blocked512) return false;
  if (first.data == nullptr || second.data == nullptr ||
      options.cluster_idx >= env.per_cluster.size() ||
      options.cluster_idx >= env.row_ptrs.size())
    return false;
  const size_t m = input.Rows(), k = input.Cols(), n = first.Rows();
  const MMConfig* settled = tuner.Best();
  if (output.Rows() != m || output.Cols() != n ||
      !mmi8_gate_prefill::Supported(m, k, n, first, second, settled, options) ||
      input.Row(0) == nullptr)
    return false;
  for (size_t r = 0; r < m; ++r)
    if (output_rows.Row(r) == nullptr) return false;

  // Every decline above precedes private prefix writes. Reuse native MMLoops so
  // B2 retains its real per-worker BF16 tile and
  // options.func/opaque/parallelism run unchanged after the two projections. No
  // public layout or key is added.
  const MMArgs args(env, m, k, n, 1.0f, nullptr, options, tuner, *settled);
  if (compact) {
    mmi8_compact_prefix::Activation activation;
    if (!mmi8_compact_prefix::Prepare(input, storage, env.ctx,
                                      options.cluster_idx, 128, nullptr,
                                      activation))
      HWY_ABORT("Validated compact gate preparation unexpectedly declined");
    if (blocked512) {
      const mmi8_gate_prefill::BlockedView<32> view{&activation.view};
      MMLoops::Dispatch<mmi8_gate_prefill::BlockedKernel<32>>(
          view, first, &second, output_rows, args);
    } else {
      const mmi8_compact_prefix::KernelView view{&activation, 48};
      MMLoops::Dispatch<mmi8_compact_prefix::Kernel>(view, first, &second,
                                                     output_rows, args);
    }
  } else {
    // Ordinary full-stride prefixes (possibly sparse endpoint writes) keep the
    // exact original producer selection and public prefix-mode policy.
    const size_t prefix_mode = MMI8PrefixModeFor(first, &second, settled, k);
    MMI8AView residual;
    const MMI8AView activation =
        QuantizeA(input, storage, env.ctx, options.cluster_idx, nullptr, 128,
                  &residual, prefix_mode);
    const mmi8_gate_prefill::BlockedView<1> view{&activation};
    MMLoops::Dispatch<mmi8_gate_prefill::BlockedKernel<1>>(view, first, &second,
                                                           output_rows, args);
  }
#ifdef GEMMA_MM_I8_GATE_PREFILL_TEST_HOOK
  GEMMA_MM_I8_GATE_PREFILL_TEST_HOOK(compact, blocked512, m);
#endif
  return true;
#else
  (void)input;
  (void)first;
  (void)second;
  (void)env;
  (void)output;
  (void)output_rows;
  (void)storage;
  (void)options;
  (void)tuner;
  return false;
#endif
}

#endif  // target toggle
