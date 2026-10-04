// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0
// Included inside the Highway target namespace before MMI8BuildPrefix.
// No layout change: only proved32-aligned prefix entries are written.
static constexpr size_t kMMI8PrefixEndpointsMode = 32;

static inline bool MMI8PrefixEndpointsEnabled() {
  static const bool enabled = MMI8Flag("GEMMA_MM_I8_PREFIX_ENDPOINTS");
  return enabled;
}

// Do not retain `settled`: MMAutoTune may invalidate its Best() pointer.
// This check and the following dispatch use the same completed configuration.
static HWY_INLINE bool MMI8PrefixEndpointsSupported(const MMI8B& B,
                                                    const MMI8B* B2,
                                                    const MMConfig* settled,
                                                    size_t k) {
#if HWY_TARGET == HWY_AVX2 && HWY_ARCH_X86 && GEMMA_MM_I8_BIASED_B
  if (settled == nullptr || k == 0) return false;
  const auto packed = [&](const MMI8B& weight) {
    return weight.data != nullptr && weight.scale != nullptr &&
           weight.packed_micro &&
           (weight.block_size == 32 || weight.block_size == 64 ||
            weight.block_size == 128) &&
           k % weight.block_size == 0 && weight.Cols() == k;
  };
  if (!packed(B) ||
      (B2 != nullptr && (!packed(*B2) || B.block_size != B2->block_size)))
    return false;
  const auto ranges = settled->RangesOfKC(k);
  if (ranges.NumTasks() == 0) return false;
  for (size_t i = 0; i < ranges.NumTasks(); ++i) {
    const auto range = ranges.Range(i);
    if (range.begin() % 32 != 0 || range.end() % 32 != 0) return false;
  }
  return true;
#else
  (void)B;
  (void)B2;
  (void)settled;
  (void)k;
  return false;
#endif
}

static HWY_INLINE size_t MMI8PrefixModeFor(const MMI8B& B, const MMI8B* B2,
                                           const MMConfig* settled, size_t k) {
  return MMI8PrefixEndpointsEnabled() &&
                 MMI8PrefixEndpointsSupported(B, B2, settled, k)
             ? kMMI8PrefixEndpointsMode
             : MMI8PrefixScanMode();
}

#if HWY_TARGET == HWY_AVX2 && HWY_ARCH_X86 && GEMMA_MM_I8_BIASED_B
static HWY_INLINE bool MMI8PrefixEndpoints32(const int8_t* input, size_t count,
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
    prefix[c + 32] = sum;
  }
  return true;
}
#endif
