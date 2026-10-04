// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0
// Included inside the Highway target namespace after MMI8PrepareInputRow.
// Exact AVX2 dual-A quantization for complete original groups only.
// The disabled and unsupported paths remain in the original QuantizeA body.

static inline bool MMI8FusedDualQuantEnabled() {
  static const bool enabled = MMI8Flag("GEMMA_MM_I8_FUSED_DUAL_QUANT");
  return enabled;
}

static HWY_INLINE bool MMI8FusedDualQuantSupported(size_t rows, size_t k,
                                                   size_t group, bool dual) {
#if HWY_TARGET == HWY_AVX2 && HWY_ARCH_X86
  return dual && rows != 0 && k != 0 &&
         (group == 32 || group == 64 || group == 128) && k % group == 0 &&
         k % MMI8RotateBlockSize() == 0;
#else
  (void)rows;
  (void)k;
  (void)group;
  (void)dual;
  return false;
#endif
}

struct MMI8DualRawScales {
  float primary = 0.0f;
  float residual = 0.0f;
};

#if HWY_TARGET == HWY_AVX2 && HWY_ARCH_X86
// Full groups only. `values` initially contains the already prepared/rotated
// input, and ends with the same residual-error F32 bits as the original path.
template <size_t kGroup>
static HWY_INLINE MMI8DualRawScales MMI8DualQuantizeGroupFused(
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
    MMI8BuildPrefix(primary, kGroup, prefix_base, prefix, prefix_mode);

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
    MMI8BuildPrefix(residual, kGroup, residual_base, residual_prefix,
                    prefix_mode);
  return {raw_scale, error_scale};
}
#endif

// Unsupported raw group sizes leave every output untouched. QuantizeA
// selects this path only after validating the complete matrix operation.
static HWY_INLINE bool MMI8TryDualQuantizeGroupFused(
    float* values, size_t group, MMI8AT* primary, int32_t* prefix,
    MMI8AT* residual, int32_t* residual_prefix, int32_t prefix_base,
    int32_t residual_base, size_t prefix_mode, MMI8DualRawScales& scales) {
#if HWY_TARGET == HWY_AVX2 && HWY_ARCH_X86
  switch (group) {
    case 32:
      scales = MMI8DualQuantizeGroupFused<32>(values, primary, prefix, residual,
                                              residual_prefix, prefix_base,
                                              residual_base, prefix_mode);
      return true;
    case 64:
      scales = MMI8DualQuantizeGroupFused<64>(values, primary, prefix, residual,
                                              residual_prefix, prefix_base,
                                              residual_base, prefix_mode);
      return true;
    case 128:
      scales = MMI8DualQuantizeGroupFused<128>(
          values, primary, prefix, residual, residual_prefix, prefix_base,
          residual_base, prefix_mode);
      return true;
  }
#endif
  return false;
}

#if HWY_TARGET == HWY_AVX2 && HWY_ARCH_X86
template <typename TA>
static HWY_NOINLINE MMI8AView MMI8QuantizeAFusedDual(
    const MatPtrT<TA>& input, MMI8AStorage& storage, ThreadingContext& ctx,
    size_t cluster_idx, const float* pre_scale, size_t block_size,
    MMI8AView* residual, size_t prefix_mode) {
  HWY_DASSERT(MMI8FusedDualQuantSupported(input.Rows(), input.Cols(),
                                          block_size, residual != nullptr));
  MMI8AView view = storage.View(input.Extents(), block_size);
  *residual = storage.ResidualView(input.Extents(), block_size);
  view.residual = residual;
  const size_t k = input.Cols();
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
        int32_t* prefix = storage.prefix(row);
        int32_t* extra_prefix = storage.residual_prefix(row);
        for (size_t c = 0; c < k; c += block_size) {
          const int32_t base = GEMMA_MM_I8_BIASED_B && c ? prefix[c] : 0;
          const int32_t extra_base =
              GEMMA_MM_I8_BIASED_B && c ? extra_prefix[c] : 0;
          MMI8DualRawScales raw;
          const bool handled = MMI8TryDualQuantizeGroupFused(
              rotated.data() + c, block_size, view.data.Row(row) + c,
              prefix + c, residual->data.Row(row) + c, extra_prefix + c, base,
              extra_base, prefix_mode, raw);
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
#endif
