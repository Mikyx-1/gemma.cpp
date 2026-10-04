// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0
// Isolated typed input -> first rotation-stage fusion. Quantizers unchanged.
// Include after matmul_i8-inl.h inside its target namespace.
#if HWY_TARGET == HWY_AVX2 && HWY_ARCH_X86
static const uint32_t* MMI8TypedRotateSigns(size_t k, size_t hash_bits) {
  thread_local hwy::AlignedVector<uint32_t> signs;
  thread_local size_t cached_hash = 0;
  if (signs.size() < k || cached_hash != hash_bits) {
    signs.resize(k);
    for (size_t c = 0; c < k; ++c)
      signs[c] = MMI8NegativeSign(c, hash_bits) ? 0x80000000u : 0u;
    cached_hash = hash_bits;
  }
  return signs.data();
}

// GCC may commute Add operands when fusing preparation with the butterfly;
// that changes which NaN sign/payload propagates. Reject unsafe inputs before
// any output write. Bounds also leave a factor >=4 below overflow through all
// R128 additions. This deliberately conservative pass is inside timed calls.
template <class TA>
static HWY_INLINE bool MMI8TypedRotateSafe(const TA* input, size_t k,
                                           const float* pre_scale) {
  // With no prescale, |input| <=2^119. With prescale, each factor <=2^59.
  const uint32_t bound = pre_scale ? 0x5d000000u : 0x7b000000u;
  if constexpr (IsBF16<TA>()) {
    const hn::CappedTag<BF16, 16> dbf;
    const hn::Rebind<uint16_t, decltype(dbf)> du;
    auto largest = hn::Zero(du);
    const auto mask = hn::Set(du, uint16_t{0x7fff});
    for (size_t c = 0; c < k; c += hn::Lanes(dbf))
      largest = hn::Max(
          largest, hn::And(hn::BitCast(du, hn::LoadU(dbf, input + c)), mask));
    if (hn::ReduceMax(du, largest) > uint16_t(bound >> 16)) return false;
  } else {
    const hn::CappedTag<float, 8> df;
    const hn::Rebind<uint32_t, decltype(df)> du;
    auto largest = hn::Zero(du);
    const auto mask = hn::Set(du, 0x7fffffffu);
    for (size_t c = 0; c < k; c += hn::Lanes(df))
      largest = hn::Max(
          largest, hn::And(hn::BitCast(du, hn::LoadU(df, input + c)), mask));
    if (hn::ReduceMax(du, largest) > bound) return false;
  }
  if (pre_scale) {
    const hn::CappedTag<float, 8> df;
    const hn::Rebind<uint32_t, decltype(df)> du;
    auto largest = hn::Zero(du);
    const auto mask = hn::Set(du, 0x7fffffffu);
    for (size_t c = 0; c < k; c += hn::Lanes(df))
      largest =
          hn::Max(largest,
                  hn::And(hn::BitCast(du, hn::LoadU(df, pre_scale + c)), mask));
    if (hn::ReduceMax(du, largest) > bound) return false;
  }
  return true;
}

template <class DF, class VF>
static HWY_INLINE VF MMI8TypedRotateButterfly(DF df, VF v) {
  const hn::Rebind<uint32_t, DF> du;
  const auto lane = hn::Iota(du, 0);
  for (size_t width = 1; width < hn::Lanes(df); width *= 2) {
    const auto bit = hn::Set(du, static_cast<uint32_t>(width));
    const auto perm = hn::IndicesFromVec(df, hn::Xor(lane, bit));
    const auto other = hn::TableLookupLanes(v, perm);
    const auto upper =
        hn::RebindMask(df, hn::Ne(hn::And(lane, bit), hn::Zero(du)));
    const auto left = hn::IfThenElse(upper, other, v);
    const auto right = hn::IfThenElse(upper, v, other);
    v = hn::IfThenElse(upper, hn::Sub(left, right), hn::Add(left, right));
  }
  return v;
}

template <size_t kRotate, class TA, bool kMatch, bool kPreScale>
static HWY_NOINLINE void MMI8TypedRotateImpl(
    const TA* HWY_RESTRICT input, size_t k, const float* HWY_RESTRICT pre_scale,
    size_t hash_bits, float* HWY_RESTRICT output) {
  static_assert(kRotate == 64 || kRotate == 128);
  static_assert(IsBF16<TA>() || IsF32<TA>());
  static_assert(!kMatch || IsF32<TA>());
  const hn::CappedTag<float, 8> df;
  const hn::Rebind<uint32_t, decltype(df)> du;
  const hn::Repartition<BF16, decltype(df)> dbf;
  const hn::Half<decltype(dbf)> dhbf;
  using VF = hn::Vec<decltype(df)>;
  const size_t lanes = hn::Lanes(df);
  const uint32_t* signs = MMI8TypedRotateSigns(k, hash_bits);
  for (size_t block = 0; block < k; block += kRotate) {
    float* HWY_RESTRICT x = output + block;
    // Two vectors preserve the exact OrderedDemote2To grouping used by
    // CompressTraits<float>::DecompressAndZeroPad. No rounded-BF16 scratch.
    for (size_t i = 0; i < kRotate; i += 2 * lanes) {
      VF v0, v1;
      if constexpr (IsBF16<TA>()) {
        const auto bf = hn::LoadU(dbf, input + block + i);
        v0 = hn::PromoteTo(df, hn::LowerHalf(dhbf, bf));
        v1 = hn::PromoteTo(df, hn::UpperHalf(dhbf, bf));
      } else {
        v0 = hn::LoadU(df, input + block + i);
        v1 = hn::LoadU(df, input + block + i + lanes);
        if constexpr (kMatch) {
          const auto bf = hn::OrderedDemote2To(dbf, v0, v1);
          v0 = hn::PromoteTo(df, hn::LowerHalf(dhbf, bf));
          v1 = hn::PromoteTo(df, hn::UpperHalf(dhbf, bf));
        }
      }
      if constexpr (kPreScale) {
        v0 = hn::Mul(v0, hn::LoadU(df, pre_scale + block + i));
        v1 = hn::Mul(v1, hn::LoadU(df, pre_scale + block + i + lanes));
      }
      v0 = hn::BitCast(
          df, hn::Xor(hn::BitCast(du, v0), hn::LoadU(du, signs + block + i)));
      v1 = hn::BitCast(df, hn::Xor(hn::BitCast(du, v1),
                                   hn::LoadU(du, signs + block + i + lanes)));
      hn::StoreU(MMI8TypedRotateButterfly(df, v0), df, x + i);
      hn::StoreU(MMI8TypedRotateButterfly(df, v1), df, x + i + lanes);
    }
    // Identical remaining butterflies. Normalize only after the last add/sub.
    for (size_t width = lanes; width < kRotate / 2; width *= 2) {
      for (size_t start = 0; start < kRotate; start += 2 * width) {
        for (size_t i = 0; i < width; i += lanes) {
          const auto left = hn::LoadU(df, x + start + i);
          const auto right = hn::LoadU(df, x + start + width + i);
          hn::StoreU(hn::Add(left, right), df, x + start + i);
          hn::StoreU(hn::Sub(left, right), df, x + start + width + i);
        }
      }
    }
    constexpr float normalize = kRotate == 64 ? 0.125f : 0.08838834764831845f;
    const auto norm = hn::Set(df, normalize);
    for (size_t i = 0; i < kRotate / 2; i += lanes) {
      const auto left = hn::LoadU(df, x + i);
      const auto right = hn::LoadU(df, x + kRotate / 2 + i);
      hn::StoreU(hn::Mul(hn::Add(left, right), norm), df, x + i);
      hn::StoreU(hn::Mul(hn::Sub(left, right), norm), df, x + kRotate / 2 + i);
    }
  }
}

template <size_t kRotate, class TA>
static HWY_INLINE void MMI8TypedRotateDispatch(const TA* input, size_t k,
                                               const float* pre_scale,
                                               bool match_bf16,
                                               size_t hash_bits,
                                               float* output) {
  if constexpr (IsF32<TA>()) {
    if (match_bf16) {
      if (pre_scale)
        MMI8TypedRotateImpl<kRotate, TA, true, true>(input, k, pre_scale,
                                                     hash_bits, output);
      else
        MMI8TypedRotateImpl<kRotate, TA, true, false>(input, k, nullptr,
                                                      hash_bits, output);
      return;
    }
  }
  if (pre_scale)
    MMI8TypedRotateImpl<kRotate, TA, false, true>(input, k, pre_scale,
                                                  hash_bits, output);
  else
    MMI8TypedRotateImpl<kRotate, TA, false, false>(input, k, nullptr, hash_bits,
                                                   output);
}
#endif

// Valid native rows only. Declined inputs leave output and input untouched, so
// callers retain their original generic preparation/rotation path. Input.Scale
// is deliberately absent: native QuantizeA applies it later to stored scales.
template <class TA>
static bool MMI8TypedPrepareRotateSupported(const TA* input, size_t k,
                                            const float* pre_scale,
                                            bool match_bf16, float* output,
                                            size_t rotate_block,
                                            size_t hash_bits, bool fast) {
  static_assert(IsBF16<TA>() || IsF32<TA>());
#if HWY_TARGET == HWY_AVX2 && HWY_ARCH_X86
  if (!fast || !input || !output || k == 0 ||
      (rotate_block != 64 && rotate_block != 128) ||
      (hash_bits != 16 && hash_bits != 32) || k % rotate_block != 0)
    return false;
  if (!MMI8TypedRotateSafe(input, k, pre_scale)) return false;
  if (rotate_block == 64)
    MMI8TypedRotateDispatch<64>(input, k, pre_scale, match_bf16, hash_bits,
                                output);
  else
    MMI8TypedRotateDispatch<128>(input, k, pre_scale, match_bf16, hash_bits,
                                 output);
  return true;
#else
  (void)input;
  (void)k;
  (void)pre_scale;
  (void)match_bf16;
  (void)output;
  (void)rotate_block;
  (void)hash_bits;
  (void)fast;
  return false;
#endif
}

// Preserve the generic native preparation fallback for other input types.
template <class TA>
static HWY_INLINE bool MMI8TryTypedPrepareRotate(const TA* input, size_t k,
                                                 const float* pre_scale,
                                                 bool match_bf16, float* output,
                                                 size_t rotate_block,
                                                 size_t hash_bits, bool fast) {
  if constexpr (IsBF16<TA>() || IsF32<TA>()) {
    return MMI8TypedPrepareRotateSupported(
        input, k, pre_scale, match_bf16, output, rotate_block, hash_bits, fast);
  } else {
    (void)input;
    (void)k;
    (void)pre_scale;
    (void)match_bf16;
    (void)output;
    (void)rotate_block;
    (void)hash_bits;
    (void)fast;
    return false;
  }
}

// Default-off optional preparation/rotation fusion. Capture once per matrix.
static inline bool MMI8TypedRotateEnabled() {
  static const bool enabled = MMI8Flag("GEMMA_MM_I8_TYPED_ROTATE");
  return enabled;
}
