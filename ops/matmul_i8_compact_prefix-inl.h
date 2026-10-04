// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0
// Included inside the Highway target namespace after the original quantizers.
#if defined(THIRD_PARTY_GEMMA_CPP_MATMUL_I8_COMPACT_PREFIX_TOGGLE) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef THIRD_PARTY_GEMMA_CPP_MATMUL_I8_COMPACT_PREFIX_TOGGLE
#undef THIRD_PARTY_GEMMA_CPP_MATMUL_I8_COMPACT_PREFIX_TOGGLE
#else
#define THIRD_PARTY_GEMMA_CPP_MATMUL_I8_COMPACT_PREFIX_TOGGLE
#endif
// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0
// Isolated experiment: include inside the target namespace after
// matmul_i8-inl.h. All compact MMI8AView values remain private to this
// namespace/wrapper. They MUST NOT be passed to any original kernel, ViewGroup,
// or RowSum method.
#if HWY_TARGET == HWY_AVX2 && GEMMA_MM_I8_BIASED_B && defined(__GNUC__) && \
    !defined(__clang__) && HWY_ARCH_X86_64
namespace mmi8_compact_prefix {
static HWY_INLINE bool EndpointSums(const int8_t* input, size_t count,
                                    int32_t base, int32_t* prefix) {
  if (input == nullptr || prefix == nullptr || count == 0 || count % 32 != 0)
    return false;
  const __m256i sign = _mm256_set1_epi8(-128);
  const __m256i zero = _mm256_setzero_si256();
  int32_t sum = base;
  prefix[0] = base;
  for (size_t c = 0; c < count; c += 32) {
    const auto q =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input + c));
    // Unsigned(q xor128) = q+128; four SAD lanes sum32 bytes exactly.
    const auto four = _mm256_sad_epu8(_mm256_xor_si256(q, sign), zero);
    auto two = _mm_add_epi64(_mm256_castsi256_si128(four),
                             _mm256_extracti128_si256(four, 1));
    two = _mm_add_epi64(two, _mm_unpackhi_epi64(two, two));
    sum += _mm_cvtsi128_si32(two) - 4096;
    prefix[c / 32 + 1] = sum;
  }
  return true;
}

static HWY_INLINE void BuildPrefix(const int8_t* values, size_t count,
                                   int32_t base, int32_t* prefix, size_t) {
  const bool handled = EndpointSums(values, count, base, prefix);
  HWY_ASSERT(handled);
}
static HWY_INLINE size_t PrefixStride(size_t k) {
  return hwy::RoundUpTo(k / 32 + 1, HWY_ALIGNMENT / sizeof(int32_t));
}
struct GroupView {
  const float* scale;
  const int32_t* prefix;
  size_t stride;
  int32_t RowSum(size_t r, size_t cols) const {
    const auto* p = prefix + r * stride;
    return p[cols / 32] - p[0];
  }
};
static HWY_INLINE GroupView CompactGroupView(const MMI8AView& a, size_t r,
                                             size_t c, size_t, size_t group) {
  return {a.scale + r + group * a.scale_stride,
          a.prefix + r * a.prefix_stride + c / 32, a.prefix_stride};
}

template <typename TA>
static HWY_INLINE float CompactQuantizeRowA(
    const TA* HWY_RESTRICT in, size_t k, MMI8AT* HWY_RESTRICT out,
    int32_t* HWY_RESTRICT prefix, size_t padded_k, int32_t prefix_base = 0,
    size_t prefix_scan_mode = MMI8PrefixScanMode()) {
  const hn::ScalableTag<float> df;
  const hn::Rebind<int32_t, decltype(df)> di32;
  const hn::Rebind<MMI8AT, decltype(df)> d8;
  using VF = hn::Vec<decltype(df)>;
  const size_t NF = hn::Lanes(df);

  VF vmax = hn::Zero(df);
  size_t i = 0;
  if (k >= NF) {
    for (; i <= k - NF; i += NF) {
      vmax = hn::Max(vmax, hn::Abs(LoadF32(df, in + i)));
    }
  }
  if (i != k) {
    vmax = hn::Max(vmax, hn::Abs(LoadNF32(df, in + i, k - i)));
  }
  const float amax = hn::ReduceMax(df, vmax);

  const float scale = (amax == 0.0f) ? 1.0f : amax / kMMI8Max;
  const float inv_scale = (amax == 0.0f) ? 0.0f : kMMI8Max / amax;
  const VF vinv = hn::Set(df, inv_scale);

  i = 0;
  if (k >= NF) {
    for (; i <= k - NF; i += NF) {
      const auto q = hn::NearestInt(hn::Mul(LoadF32(df, in + i), vinv));
      hn::StoreU(hn::DemoteTo(d8, q), d8, out + i);
    }
  }
  for (; i < k; ++i) {
    const float in_f = hwy::ConvertScalarTo<float>(in[i]);
    out[i] = static_cast<MMI8AT>(std::lroundf(in_f * inv_scale));
  }
  for (; i < padded_k; ++i) {
    out[i] = static_cast<MMI8AT>(0);
  }

  if constexpr (GEMMA_MM_I8_BIASED_B) {
    BuildPrefix(out, k, prefix_base, prefix, prefix_scan_mode);
  }
  return scale;
}

template <size_t kGroup>
static HWY_INLINE MMI8DualRawScales CompactDualQuantizeGroupFused(
    float* HWY_RESTRICT values, MMI8AT* HWY_RESTRICT primary,
    int32_t* HWY_RESTRICT prefix, MMI8AT* HWY_RESTRICT residual,
    int32_t* HWY_RESTRICT residual_prefix, int32_t prefix_base,
    int32_t residual_base, size_t prefix_mode) {
  static_assert(kGroup == 32 || kGroup == 64 || kGroup == 128);
  const hn::ScalableTag<float> df;
  const hn::Rebind<int32_t, decltype(df)> di;
  const hn::Rebind<MMI8AT, decltype(df)> d8;
  const size_t nf = hn::Lanes(df);
  using VF = hn::Vec<decltype(df)>;
  VF maximum = hn::Zero(df);
  for (size_t j = 0; j < kGroup; j += nf)
    maximum = hn::Max(maximum, hn::Abs(hn::LoadU(df, values + j)));
  const float amax = hn::ReduceMax(df, maximum);
  const float raw_scale = amax == 0.0f ? 1.0f : amax / kMMI8Max;
  const float inverse = amax == 0.0f ? 0.0f : kMMI8Max / amax;
  const VF scale = hn::Set(df, raw_scale), inv = hn::Set(df, inverse);

  VF error_maximum = hn::Zero(df);
  for (size_t j = 0; j < kGroup; j += nf) {
    const auto value = hn::LoadU(df, values + j);
    const auto rounded = hn::NearestInt(hn::Mul(value, inv));
    const auto bytes = hn::DemoteTo(d8, rounded);
    hn::StoreU(bytes, d8, primary + j);
    // The original path reloads the stored signed byte before dequantization.
    // Preserve that saturating I32->I8->I32 round trip in registers.
    const auto q = hn::ConvertTo(df, hn::PromoteTo(di, bytes));
    const auto error = hn::NegMulAdd(q, scale, value);
    hn::StoreU(error, df, values + j);
    // Same per-lane maximum sequence and ReduceMax as QuantizeRowA, now
    // consuming each error vector before it has to be reloaded from scratch.
    error_maximum = hn::Max(error_maximum, hn::Abs(error));
  }
  if constexpr (GEMMA_MM_I8_BIASED_B)
    BuildPrefix(primary, kGroup, prefix_base, prefix, prefix_mode);

  const float error_amax = hn::ReduceMax(df, error_maximum);
  const float error_scale = error_amax == 0.0f ? 1.0f : error_amax / kMMI8Max;
  const float error_inverse = error_amax == 0.0f ? 0.0f : kMMI8Max / error_amax;
  const VF error_inv = hn::Set(df, error_inverse);
  for (size_t j = 0; j < kGroup; j += nf) {
    const auto q =
        hn::NearestInt(hn::Mul(hn::LoadU(df, values + j), error_inv));
    hn::StoreU(hn::DemoteTo(d8, q), d8, residual + j);
  }
  if constexpr (GEMMA_MM_I8_BIASED_B)
    BuildPrefix(residual, kGroup, residual_base, residual_prefix, prefix_mode);
  return {raw_scale, error_scale};
}

static HWY_INLINE bool CompactTryDualQuantizeGroupFused(
    float* values, size_t group, MMI8AT* primary, int32_t* prefix,
    MMI8AT* residual, int32_t* residual_prefix, int32_t prefix_base,
    int32_t residual_base, size_t prefix_mode, MMI8DualRawScales& scales) {
#if HWY_TARGET == HWY_AVX2 && HWY_ARCH_X86
  switch (group) {
    case 32:
      scales = CompactDualQuantizeGroupFused<32>(
          values, primary, prefix, residual, residual_prefix, prefix_base,
          residual_base, prefix_mode);
      return true;
    case 64:
      scales = CompactDualQuantizeGroupFused<64>(
          values, primary, prefix, residual, residual_prefix, prefix_base,
          residual_base, prefix_mode);
      return true;
    case 128:
      scales = CompactDualQuantizeGroupFused<128>(
          values, primary, prefix, residual, residual_prefix, prefix_base,
          residual_base, prefix_mode);
      return true;
  }
#endif
  return false;
}

template <typename TA>
static HWY_NOINLINE MMI8AView CompactQuantizeAFusedDual(
    const MatPtrT<TA>& input, MMI8AStorage& storage, ThreadingContext& ctx,
    size_t cluster_idx, const float* pre_scale, size_t block_size,
    MMI8AView* residual, size_t prefix_mode) {
  HWY_DASSERT(MMI8FusedDualQuantSupported(input.Rows(), input.Cols(),
                                          block_size, residual != nullptr));
  MMI8AView view = storage.View(input.Extents(), block_size);
  *residual = storage.ResidualView(input.Extents(), block_size);
  view.residual = residual;
  const size_t k = input.Cols();
  view.prefix_stride = PrefixStride(k);
  if (residual != nullptr) residual->prefix_stride = view.prefix_stride;
  const size_t padded_k =
      hwy::RoundUpTo(k, hn::Lanes(hn::ScalableTag<int8_t>()));
  const float input_scale = input.Scale();
  static const bool match_bf16 = MMI8Flag("GEMMA_MM_I8_MATCH_BF16_A");
  const bool typed_rotate = MMI8TypedRotateEnabled();
  HWY_DASSERT(k % MMI8RotateBlockSize() == 0);
  ParallelFor(
      Parallelism::kFlat, input.Rows(), ctx, cluster_idx, Callers::kMMQuantizeA,
      [&](size_t row, size_t) HWY_ATTR {
        thread_local hwy::AlignedVector<float> rotated;
        if (rotated.size() < padded_k) rotated.resize(padded_k);
        if (typed_rotate &&
            MMI8TryTypedPrepareRotate(input.Row(row), k, pre_scale, match_bf16,
                                      rotated.data(), MMI8RotateBlockSize(),
                                      MMI8HashBits(), MMI8FastRotate())) {
        } else {
          MMI8PrepareInputRow(input.Row(row), k, pre_scale, match_bf16,
                              rotated.data());
          MMI8Rotate(rotated.data(), k);
        }
        int32_t* prefix = storage.prefix(0) + row * PrefixStride(k);
        int32_t* extra_prefix =
            storage.residual_prefix(0) + row * PrefixStride(k);
        for (size_t c = 0; c < k; c += block_size) {
          const int32_t base = GEMMA_MM_I8_BIASED_B && c ? prefix[c / 32] : 0;
          const int32_t extra_base =
              GEMMA_MM_I8_BIASED_B && c ? extra_prefix[c / 32] : 0;
          MMI8DualRawScales raw;
          const bool handled = CompactTryDualQuantizeGroupFused(
              rotated.data() + c, block_size, view.data.Row(row) + c,
              prefix + c / 32, residual->data.Row(row) + c,
              extra_prefix + c / 32, base, extra_base, prefix_mode, raw);
          HWY_ASSERT(handled);
          storage.scale()[(c / block_size) * view.scale_stride + row] =
              input_scale * raw.primary;
          storage.residual_scale()[(c / block_size) * residual->scale_stride +
                                   row] = input_scale * raw.residual;
        }
        for (size_t c = k; c < padded_k; ++c) {
          view.data.Row(row)[c] = 0;
          residual->data.Row(row)[c] = 0;
        }
      });
  return view;
}

template <typename TA>
static HWY_NOINLINE MMI8AView CompactQuantizeA(
    const MatPtrT<TA>& A, MMI8AStorage& storage, ThreadingContext& ctx,
    size_t cluster_idx, const float* a_pre_scale = nullptr,
    size_t block_size = 0, MMI8AView* residual = nullptr,
    size_t prefix_scan_mode = MMI8PrefixScanMode()) {
#if HWY_TARGET == HWY_AVX2 && HWY_ARCH_X86
  if (MMI8FusedDualQuantEnabled() &&
      MMI8FusedDualQuantSupported(A.Rows(), A.Cols(), block_size,
                                  residual != nullptr)) {
    const auto fused =
        CompactQuantizeAFusedDual(A, storage, ctx, cluster_idx, a_pre_scale,
                                  block_size, residual, prefix_scan_mode);
    return fused;
  }
#endif
  MMI8AView view = storage.View(A.Extents(), block_size);
  if (residual != nullptr) {
    *residual = storage.ResidualView(A.Extents(), block_size);
    view.residual = residual;
  }
  const size_t k = A.Cols();
  view.prefix_stride = PrefixStride(k);
  if (residual != nullptr) residual->prefix_stride = view.prefix_stride;
  HWY_ASSERT(block_size == 0 ||
             ((block_size == 32 || block_size == 64 || block_size == 128) &&
              k % block_size == 0));
  const size_t padded_k =
      hwy::RoundUpTo(k, hn::Lanes(hn::ScalableTag<int8_t>()));
  float* HWY_RESTRICT scale = storage.scale();
  const float a_scale = A.Scale();
  static const bool match_bf16 = MMI8Flag("GEMMA_MM_I8_MATCH_BF16_A");
  const bool typed_rotate = MMI8TypedRotateEnabled();
  HWY_DASSERT((k % MMI8RotateBlockSize()) == 0);

  ParallelFor(
      Parallelism::kFlat, A.Rows(), ctx, cluster_idx, Callers::kMMQuantizeA,
      [&](size_t r, size_t /*worker*/) HWY_ATTR {
        thread_local hwy::AlignedVector<float> rotated;
        if (rotated.size() < padded_k) rotated.resize(padded_k);
        if (typed_rotate &&
            MMI8TryTypedPrepareRotate(A.Row(r), k, a_pre_scale, match_bf16,
                                      rotated.data(), MMI8RotateBlockSize(),
                                      MMI8HashBits(), MMI8FastRotate())) {
        } else {
          MMI8PrepareInputRow(A.Row(r), k, a_pre_scale, match_bf16,
                              rotated.data());
          MMI8Rotate(rotated.data(), k);
        }
        const size_t group_size = block_size ? block_size : k;
        int32_t* prefix = storage.prefix(0) + r * PrefixStride(k);
        for (size_t c = 0; c < k; c += group_size) {
          const int32_t base = GEMMA_MM_I8_BIASED_B && c ? prefix[c / 32] : 0;
          const float raw_scale = CompactQuantizeRowA(
              rotated.data() + c, group_size, view.data.Row(r) + c,
              prefix + c / 32, group_size, base, prefix_scan_mode);
          scale[(c / group_size) * view.scale_stride + r] = a_scale * raw_scale;
          if (residual != nullptr) {
            const hn::CappedTag<float, 32> df;
            const hn::Rebind<int32_t, decltype(df)> di;
            const hn::Rebind<int8_t, decltype(df)> d8;
            for (size_t j = 0; j < group_size; j += hn::Lanes(df)) {
              const auto q = hn::ConvertTo(
                  df,
                  hn::PromoteTo(di, hn::LoadU(d8, view.data.Row(r) + c + j)));
              const auto error =
                  hn::NegMulAdd(q, hn::Set(df, raw_scale),
                                hn::LoadU(df, rotated.data() + c + j));
              hn::StoreU(error, df, rotated.data() + c + j);
            }
            int32_t* rp = storage.residual_prefix(0) + r * PrefixStride(k);
            const int32_t residual_base =
                GEMMA_MM_I8_BIASED_B && c ? rp[c / 32] : 0;
            storage.residual_scale()[(c / group_size) * residual->scale_stride +
                                     r] =
                a_scale * CompactQuantizeRowA(rotated.data() + c, group_size,
                                              residual->data.Row(r) + c,
                                              rp + c / 32, group_size,
                                              residual_base, prefix_scan_mode);
          }
        }
        for (size_t c = k; c < padded_k; ++c) {
          view.data.Row(r)[c] = 0;
          if (residual != nullptr) residual->data.Row(r)[c] = 0;
        }
      });
  return view;
}

template <size_t kOutputs, size_t kBlock, class BT, class Tag, class CView>
static HWY_NOINLINE void CompactDecodeTile(const MMI8AView& A, const BT& B,
                                           size_t row_a, size_t row_b,
                                           const IndexRange& range_kc,
                                           const float* add, Tag, CView C) {
  static_assert(kOutputs == 16 || kOutputs == 32);
  static_assert(kBlock == 32 || kBlock == 64 || kBlock == 128);
  constexpr size_t kUnroll = 32 / kOutputs;
  const hn::ScalableTag<int8_t> da;
  const hn::ScalableTag<uint8_t> db;
  const hn::ScalableTag<int32_t> di;
  const hn::ScalableTag<uint32_t> du;
  const hn::ScalableTag<float> df;
  using VI = hn::Vec<decltype(di)>;
  using VF = hn::Vec<decltype(df)>;
  const auto* ar = A.data.Row(row_a);
  const auto& residual = *A.residual;
  const auto* ar1 = residual.data.Row(row_a);
  const auto* prefix = A.prefix + row_a * A.prefix_stride;
  const auto* prefix1 = residual.prefix + row_a * residual.prefix_stride;
  const auto* b0 = reinterpret_cast<const uint8_t*>(B.data->Row(row_b));
  const auto* b1 = reinterpret_cast<const uint8_t*>(B.data->Row(row_b + 8));
  const auto* b2 =
      kOutputs == 32 ? reinterpret_cast<const uint8_t*>(B.data->Row(row_b + 16))
                     : nullptr;
  const auto* b3 =
      kOutputs == 32 ? reinterpret_cast<const uint8_t*>(B.data->Row(row_b + 24))
                     : nullptr;
  VF f0 = hn::Zero(df), f1 = f0, f2 = f0, f3 = f0;
  const auto activation = [&](const int8_t* source) HWY_ATTR {
    uint32_t bits;
    hwy::CopyBytes<4>(source, &bits);
    return hn::BitCast(da, hn::Set(du, bits));
  };
  const auto dot = [&](const auto a, const auto a1, const uint8_t* weights,
                       VI& sum, VI& residual_sum) HWY_ATTR {
    const auto b = hn::LoadU(db, weights);
    asm("%{vex%} vpdpbusd %[a], %[b], %[sum]\n\t"
        "%{vex%} vpdpbusd %[a1], %[b], %[residual_sum]"
        : [sum] "+&x"(sum.raw), [residual_sum] "+&x"(residual_sum.raw)
        : [a] "x"(a.raw), [a1] "x"(a1.raw), [b] "x"(b.raw));
  };
  size_t group = range_kc.begin() / kBlock;
  for (size_t c = range_kc.begin(); c < range_kc.end(); ++group) {
    const size_t num_k = HWY_MIN(kBlock - c % kBlock, range_kc.end() - c);
    VI d0 = hn::Zero(di), d1 = d0, d2 = d0, d3 = d0;
    VI e0 = d0, e1 = d0, e2 = d0, e3 = d0;
    size_t k = 0;
    for (; k + 4 * kUnroll <= num_k; k += 4 * kUnroll) {
      const size_t pos = c + k;
      const auto a = activation(ar + pos);
      const auto a1 = activation(ar1 + pos);
      dot(a, a1, b0 + pos * 8, d0, e0);
      dot(a, a1, b1 + pos * 8, d1, e1);
      if constexpr (kOutputs == 32) {
        dot(a, a1, b2 + pos * 8, d2, e2);
        dot(a, a1, b3 + pos * 8, d3, e3);
      } else {
        const auto next = activation(ar + pos + 4);
        const auto next1 = activation(ar1 + pos + 4);
        dot(next, next1, b0 + (pos + 4) * 8, d2, e2);
        dot(next, next1, b1 + (pos + 4) * 8, d3, e3);
      }
    }
    if constexpr (kOutputs == 16) {
      for (; k < num_k; k += 4) {
        const size_t pos = c + k;
        const auto a = activation(ar + pos);
        const auto a1 = activation(ar1 + pos);
        dot(a, a1, b0 + pos * 8, d0, e0);
        dot(a, a1, b1 + pos * 8, d1, e1);
      }
      d0 = hn::Add(d0, d2);
      d1 = hn::Add(d1, d3);
      e0 = hn::Add(e0, e2);
      e1 = hn::Add(e1, e3);
    }
    const int32_t correction =
        (prefix[(c + num_k) / 32] - prefix[c / 32]) * 128;
    const int32_t correction1 =
        (prefix1[(c + num_k) / 32] - prefix1[c / 32]) * 128;
    const auto as = hn::Set(df, A.scale[row_a + group * A.scale_stride]);
    const auto as1 =
        hn::Set(df, residual.scale[row_a + group * residual.scale_stride]);
    const auto accumulate = [&](VI sum, VI residual_sum, size_t n,
                                VF& value) HWY_ATTR {
      const auto bs = hn::LoadU(df, B.scale + group * B.Rows() + row_b + n);
      const auto scale = hn::Mul(bs, as);
      sum = hn::Sub(sum, hn::Set(di, correction));
      value = hn::MulAdd(hn::ConvertTo(df, sum), scale, value);
      const auto scale1 = hn::Mul(bs, as1);
      residual_sum = hn::Sub(residual_sum, hn::Set(di, correction1));
      value = hn::MulAdd(hn::ConvertTo(df, residual_sum), scale1, value);
    };
    accumulate(d0, e0, 0, f0);
    accumulate(d1, e1, 8, f1);
    if constexpr (kOutputs == 32) {
      accumulate(d2, e2, 16, f2);
      accumulate(d3, e3, 24, f3);
    }
    c += num_k;
  }
  using TC = hwy::RemoveCvRef<decltype(C.Row(0)[0])>;
  const hn::Rebind<TC, decltype(df)> dc;
  const auto store = [&](VF value, size_t n) HWY_ATTR {
    TC* const pos = C.Row(0) + n;
    if constexpr (hwy::IsSame<Tag, MMAddC>()) {
      value = hn::Add(value, F32FromTC(dc, hn::LoadU(dc, pos)));
    } else {
      static_assert(hwy::IsSame<Tag, MMSetC>());
      if (add != nullptr) value = hn::Add(value, hn::LoadU(df, add + n));
    }
    hn::StoreU(TCFromF32(dc, value), dc, pos);
  };
  store(f0, 0);
  store(f1, 8);
  if constexpr (kOutputs == 32) {
    store(f2, 16);
    store(f3, 24);
  }
}

template <size_t kBlock, class BT, class Tag, class CView>
static HWY_NOINLINE void CompactTile48(const MMI8AView& A, const BT& B,
                                       size_t row_a, size_t row_b,
                                       const IndexRange& range_kc,
                                       const float* add, Tag, CView C) {
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
        (prefix[(c + num_k) / 32] - prefix[c / 32]) * 128;
    const int32_t correction1 =
        (prefix1[(c + num_k) / 32] - prefix1[c / 32]) * 128;
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
  using TC = hwy::RemoveCvRef<decltype(C.Row(0)[0])>;
  const hn::Rebind<TC, decltype(df)> dc;
  const auto store = [&](VF value, size_t n) HWY_ATTR {
    TC* const pos = C.Row(0) + n;
    if constexpr (hwy::IsSame<Tag, MMAddC>()) {
      value = hn::Add(value, F32FromTC(dc, hn::LoadU(dc, pos)));
    } else {
      static_assert(hwy::IsSame<Tag, MMSetC>());
      if (add != nullptr) value = hn::Add(value, hn::LoadU(df, add + n));
    }
    hn::StoreU(TCFromF32(dc, value), dc, pos);
  };
  store(f0, 0);
  store(f1, 8);
  store(f2, 16);
  store(f3, 24);
  store(f4, 32);
  store(f5, 40);
}

template <size_t kBlock, bool kAlignedGroups = true, bool kDual = false,
          typename BT, class Tag, class CView>
static HWY_NOINLINE void CompactNativeRowTail(const MMI8AView& A, const BT& B,
                                              const IndexRange& range_mc,
                                              const IndexRange& range_kc,
                                              const IndexRange& range_nc,
                                              const float* add_base, Tag tag,
                                              CView C) {
  // Caller has exactly one row: original multi-row branch is unreachable.
  const hn::ScalableTag<int8_t> da;
  const hn::ScalableTag<uint8_t> db;
  const hn::Repartition<int32_t, decltype(da)> di;
  const hn::Repartition<uint32_t, decltype(da)> du;
  const hn::ScalableTag<float> df;
  const hn::Full128<float> d4f;
  for (size_t nc = range_nc.begin(); nc < range_nc.end();) {
    const size_t row_b = nc & ~size_t{7};
    const size_t lane = nc % 8;
    const size_t count = HWY_MIN(size_t{8} - lane, range_nc.end() - nc);
    const size_t inc = nc - range_nc.begin();
    const auto* packed = reinterpret_cast<const uint8_t*>(B.data->Row(row_b));
    const float* add = MMI8Bias(B, add_base, nc);
    for (size_t r = 0; r < range_mc.Num(); ++r) {
      const auto* ar = A.data.Row(range_mc.begin() + r);
      const auto* ar1 =
          kDual ? A.residual->data.Row(range_mc.begin() + r) : nullptr;
      auto accum = hn::Zero(df);
      size_t group = range_kc.begin() / kBlock;
      for (size_t c = range_kc.begin(); c < range_kc.end(); ++group) {
        const size_t num_k =
            kAlignedGroups ? kBlock
                           : HWY_MIN(kBlock - c % kBlock, range_kc.end() - c);
        const auto* br = packed + c * 8;
        auto d0 = hn::Zero(di), d1 = hn::Zero(di);
        auto d2 = hn::Zero(di), d3 = hn::Zero(di);
        auto e0 = hn::Zero(di), e1 = hn::Zero(di);
        auto e2 = hn::Zero(di), e3 = hn::Zero(di);
        const auto dot = [&](size_t offset, auto& sum,
                             auto& residual_sum) HWY_ATTR {
          uint32_t bits;
          hwy::CopyBytes<4>(ar + c + offset, &bits);
          const auto a = hn::BitCast(da, hn::Set(du, bits));
          const auto b = hn::LoadU(db, br + offset * 8);
          if constexpr (kDual) {
            uint32_t residual_bits;
            hwy::CopyBytes<4>(ar1 + c + offset, &residual_bits);
            const auto a1 = hn::BitCast(da, hn::Set(du, residual_bits));
            // Keep B in a register for both dots. The dual specialization
            // requires more than the eight SIMD registers of x86-32.
            asm("%{vex%} vpdpbusd %[a], %[b], %[sum]\n\t"
                "%{vex%} vpdpbusd %[a1], %[b], %[residual_sum]"
                : [sum] "+&x"(sum.raw), [residual_sum] "+&x"(residual_sum.raw)
                : [a] "x"(a.raw), [a1] "x"(a1.raw), [b] "x"(b.raw));
          } else {
            asm("%{vex%} vpdpbusd %[a], %[b], %[sum]"
                : [sum] "+x"(sum.raw)
                : [a] "x"(a.raw), [b] "x"(b.raw));
          }
        };
        size_t k = 0;
        for (; k + 16 <= num_k; k += 16) {
          dot(k, d0, e0);
          dot(k + 4, d1, e1);
          dot(k + 8, d2, e2);
          dot(k + 12, d3, e3);
        }
        if constexpr (!kAlignedGroups) {
          for (; k < num_k; k += 4) dot(k, d0, e0);
        }
        auto sum = hn::Add(hn::Add(d0, d1), hn::Add(d2, d3));
        const auto av =
            CompactGroupView(A, range_mc.begin() + r, c, num_k, group);
        sum = hn::Sub(sum, hn::Set(di, av.RowSum(0, num_k) * 128));
        const auto bs = hn::LoadU(df, B.scale + group * B.Rows() + row_b);
        const auto scale = hn::Mul(bs, hn::Set(df, av.scale[0]));
        accum = hn::MulAdd(hn::ConvertTo(df, sum), scale, accum);
        if constexpr (kDual) {
          auto residual_sum = hn::Add(hn::Add(e0, e1), hn::Add(e2, e3));
          const auto rv = CompactGroupView(*A.residual, range_mc.begin() + r, c,
                                           num_k, group);
          residual_sum =
              hn::Sub(residual_sum, hn::Set(di, rv.RowSum(0, num_k) * 128));
          const auto residual_scale = hn::Mul(bs, hn::Set(df, rv.scale[0]));
          accum = hn::MulAdd(hn::ConvertTo(df, residual_sum), residual_scale,
                             accum);
        }
        c += num_k;
      }
      if (count == 8) {
        using TC = hwy::RemoveCvRef<decltype(C.Row(0)[0])>;
        const hn::Rebind<TC, decltype(df)> dc;
        TC* pos = C.Row(r) + inc;
        if constexpr (hwy::IsSame<Tag, MMAddC>()) {
          accum = hn::Add(accum, F32FromTC(dc, hn::LoadU(dc, pos)));
        } else if (add != nullptr) {
          accum = hn::Add(accum, hn::LoadU(df, add));
        }
        hn::StoreU(TCFromF32(dc, accum), dc, pos);
      } else {
        // Generic scheduling may split N at four-channel boundaries.
        HWY_DASSERT(count == 4);
        const auto half =
            lane ? hn::UpperHalf(d4f, accum) : hn::LowerHalf(d4f, accum);
        MMI8MicroPrefillStore(d4f, half, r, 0, add, tag, C.View(0, inc, 4));
      }
    }
    nc += count;
  }
}
// Public experimental wrapper deliberately has no MMI8AView conversion. Its
// residual pointer refers to this object, so copying/moving is forbidden.
struct Activation {
  MMI8AView view{};
  MMI8AView residual{};
  size_t rows = 0, k = 0, group = 0;
  bool ready = false;
  Activation() = default;
  Activation(const Activation&) = delete;
  Activation& operator=(const Activation&) = delete;
};

template <typename TA>
static bool Prepare(const MatPtrT<TA>& input, MMI8AStorage& storage,
                    ThreadingContext& ctx, size_t cluster, size_t group,
                    const float* pre_scale, Activation& result) {
  result.ready = false;
  const size_t k = input.Cols();
  if (!MMI8NativeVNNI() || input.Rows() == 0 || k == 0 ||
      input.Row(0) == nullptr || (group != 32 && group != 64 && group != 128) ||
      k % group != 0 || k % MMI8RotateBlockSize() != 0)
    return false;
  result.view = CompactQuantizeA(input, storage, ctx, cluster, pre_scale, group,
                                 &result.residual, 32);
  result.rows = input.Rows();
  result.k = k;
  result.group = group;
  result.ready = true;
  return true;
}

template <size_t kTile, size_t kGroup, class BT, class Tag, class CView>
static bool Range(const Activation& packed, const BT& B, const IndexRange& rows,
                  const IndexRange& kc, const IndexRange& columns,
                  const float* add, Tag tag, CView output) {
  static_assert(kTile == 32 || kTile == 48);
  static_assert(kGroup == 32 || kGroup == 64 || kGroup == 128);
  if (!packed.ready || packed.group != kGroup || B.block_size != kGroup ||
      !B.packed_micro || !B.dual_a || B.data == nullptr || B.scale == nullptr ||
      B.Cols() != packed.k || rows.Num() == 0 || rows.end() > packed.rows ||
      kc.Num() == 0 || kc.begin() % 32 || kc.end() % 32 ||
      kc.end() > packed.k || columns.Num() == 0 || columns.begin() % 8 ||
      columns.Num() % 16 || columns.end() > B.Rows())
    return false;
  const auto& A = packed.view;
  if (rows.Num() == 1) {
    if (kc.begin() % kGroup == 0 && kc.end() % kGroup == 0)
      CompactNativeRowTail<kGroup, true, true>(A, B, rows, kc, columns, add,
                                               tag, output);
    else
      CompactNativeRowTail<kGroup, false, true>(A, B, rows, kc, columns, add,
                                                tag, output);
    return true;
  }
  size_t n = columns.begin();
  if constexpr (kTile == 48) {
    for (; n + 48 <= columns.end(); n += 48) {
      const float* bias = MMI8Bias(B, add, n);
      for (size_t r = 0; r < rows.Num(); ++r)
        CompactTile48<kGroup>(A, B, rows.begin() + r, n, kc, bias, tag,
                              output.View(r, n - columns.begin(), 48));
    }
  }
  for (; n + 32 <= columns.end(); n += 32) {
    const float* bias = MMI8Bias(B, add, n);
    for (size_t r = 0; r < rows.Num(); ++r)
      CompactDecodeTile<32, kGroup>(A, B, rows.begin() + r, n, kc, bias, tag,
                                    output.View(r, n - columns.begin(), 32));
  }
  if (n < columns.end()) {
    HWY_ASSERT(columns.end() - n == 16);
    const float* bias = MMI8Bias(B, add, n);
    for (size_t r = 0; r < rows.Num(); ++r)
      CompactDecodeTile<16, kGroup>(A, B, rows.begin() + r, n, kc, bias, tag,
                                    output.View(r, n - columns.begin(), 16));
  }
  return true;
}
}  // namespace mmi8_compact_prefix
#endif

#if HWY_TARGET == HWY_AVX2 && GEMMA_MM_I8_BIASED_B && defined(__GNUC__) && \
    !defined(__clang__) && HWY_ARCH_X86_64
namespace mmi8_compact_prefix {

// Guard the entire operation before overwriting even the first prefix word.
// The initial production scope covers the measured down-like projections.
static HWY_INLINE bool Supported(size_t m, size_t k, size_t n, const MMI8B& b,
                                 const MMConfig* settled, MMOptions options) {
  if (m <= 16 || m > kMaxBatchSize || k < 2048 || n == 0 || n % 16 ||
      settled == nullptr || options.func != nullptr || !MMI8FastMicro() ||
      !MMI8NativeVNNI() || b.data == nullptr || b.data->Row(0) == nullptr ||
      b.scale == nullptr || !b.packed_micro || !MMI8UseDualA(b, m) ||
      (b.block_size != 32 && b.block_size != 64 && b.block_size != 128) ||
      k % b.block_size || b.Cols() != k || b.Rows() != n ||
      k % MMI8RotateBlockSize() ||
      (settled->Order() != MMOrder::kNT_MT &&
       settled->Order() != MMOrder::kNT_MT_K))
    return false;
  const auto rows = settled->RangesOfMC(m);
  const auto columns = settled->RangesOfNC(n);
  const auto kc = settled->RangesOfKC(k);
  if (rows.NumTasks() == 0 || columns.NumTasks() == 0 || kc.NumTasks() == 0)
    return false;
  for (size_t i = 0; i < rows.NumTasks(); ++i) {
    const auto r = rows.Range(i);
    if (r.Num() == 0 || r.end() > m) return false;
  }
  for (size_t i = 0; i < columns.NumTasks(); ++i) {
    const auto r = columns.Range(i);
    if (r.Num() == 0 || r.begin() % 8 || r.Num() % 16 || r.end() > n)
      return false;
  }
  for (size_t i = 0; i < kc.NumTasks(); ++i) {
    const auto r = kc.Range(i);
    if (r.Num() == 0 || r.begin() % 32 || r.end() % 32 || r.end() > k)
      return false;
  }
  return true;
}

// Distinct from MMI8AView: MMLoops cannot accidentally invoke a native kernel
// with compact prefixes. Activation and its residual stay alive until Dispatch
// joins all workers. No view or borrowed tuner pointer is cached across calls.
struct KernelView {
  const Activation* activation;
  size_t tile;
};
class Kernel {
 public:
  using AView = KernelView;

  template <size_t kGroup, class BT, class Tag, class CView>
  static void Group(const AView a, const BT& b, const IndexRange& rows,
                    const IndexRange& kc, const IndexRange& columns,
                    const MMArgs& args, Tag tag, CView output) {
    const bool handled =
        a.tile == 48 ? Range<48, kGroup>(*a.activation, b, rows, kc, columns,
                                         args.add, tag, output)
                     : Range<32, kGroup>(*a.activation, b, rows, kc, columns,
                                         args.add, tag, output);
    // All guards have already passed. Never forward a compact view to a native
    // fallback, even if future scheduling changes violate this internal
    // contract.
    HWY_ASSERT(handled);
  }

  template <class BT, class Tag, class CView>
  static void B3A2C0(const AView a, const BT& b, const IndexRange& rows,
                     const IndexRange& kc, const IndexRange& columns,
                     const MMArgs& args, Tag tag, CView output) {
    switch (b.block_size) {
      case 32:
        return Group<32>(a, b, rows, kc, columns, args, tag, output);
      case 64:
        return Group<64>(a, b, rows, kc, columns, args, tag, output);
      case 128:
        return Group<128>(a, b, rows, kc, columns, args, tag, output);
      default:
        HWY_ABORT("Unsupported compact prefix group %zu", b.block_size);
    }
  }

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

}  // namespace mmi8_compact_prefix
#endif

template <typename TA, typename TC>
static HWY_INLINE bool MMI8TryCompactPrefixMatMul(
    const MatPtrT<TA>& input, const MMI8B& weights, const float* add,
    MatMulEnv& env, MatPtrT<TC>& output, RowPtrs<TC> output_rows,
    MMI8AStorage& storage, MMOptions options, MMAutoTune<MMConfig>& tuner) {
#if HWY_TARGET == HWY_AVX2 && GEMMA_MM_I8_BIASED_B && defined(__GNUC__) && \
    !defined(__clang__) && HWY_ARCH_X86_64
  if constexpr ((IsBF16<TA>() || IsF32<TA>()) &&
                (IsBF16<TC>() || IsF32<TC>())) {
    static const bool enabled = MMI8Flag("GEMMA_MM_I8_COMPACT_PREFIX", false);
    if (!enabled || weights.data == nullptr) return false;
    static const bool n48 = MMI8Flag("GEMMA_MM_I8_MICRO_PREFILL_N48", false);
    static const bool n32 = MMI8Flag("GEMMA_MM_I8_MICRO_PREFILL_N32", false);
    if ((!n48 && !n32) || MMI8Flag("GEMMA_MM_I8_MICRO_DECODE", false))
      return false;
    const size_t m = input.Rows(), k = input.Cols(), n = weights.Rows();
    const MMConfig* settled = tuner.Best();
    if (output.Rows() != m || output.Cols() != n || input.Row(0) == nullptr ||
        !mmi8_compact_prefix::Supported(m, k, n, weights, settled, options))
      return false;
    // After this point the whole operation is supported. There is no declining
    // path after compact writes; private dispatch retains original KC stores.
    mmi8_compact_prefix::Activation compact;
    if (!mmi8_compact_prefix::Prepare(input, storage, env.ctx,
                                      options.cluster_idx, weights.block_size,
                                      weights.a_pre_scale, compact))
      HWY_ABORT("Validated compact prefix preparation unexpectedly declined");
    const size_t tile = n48 ? 48 : 32;
    const MMArgs args(env, m, k, n, 1.0f, add, options, tuner, *settled);
    const MMI8B* second = nullptr;
    const mmi8_compact_prefix::KernelView view{&compact, tile};
    MMLoops::Dispatch<mmi8_compact_prefix::Kernel>(view, weights, second,
                                                   output_rows, args);
    return true;
  }
#endif
  (void)input;
  (void)weights;
  (void)add;
  (void)env;
  (void)output;
  (void)output_rows;
  (void)storage;
  (void)options;
  (void)tuner;
  return false;
}
#endif  // target toggle
