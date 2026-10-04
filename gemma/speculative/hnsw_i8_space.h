// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0
// Proposal-only maximum-inner-product space. All stored rows use the same
// symmetric byte quantization, so hnswlib can compare query/weight or two
// weight rows without an external float matrix. This does not define target
// logits.
#ifndef THIRD_PARTY_GEMMA_CPP_SPECULATIVE_HNSW_I8_SPACE_H_
#define THIRD_PARTY_GEMMA_CPP_SPECULATIVE_HNSW_I8_SPACE_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include "hnswlib/hnswlib.h"

// The optional kernel uses AVX-VNNI (VEX), not AVX512-VNNI. GCC 13+ exposes
// _mm256_dpbusd_avx_epi32 and the avxvnni target. Other compilers use scalar
// code.
#if (defined(__x86_64__) || defined(__i386__)) && defined(__GNUC__) && \
    !defined(__clang__) && __GNUC__ >= 13
#include <immintrin.h>
#define GEMMA_HNSW_I8_HAS_VNNI 1
#else
#define GEMMA_HNSW_I8_HAS_VNNI 0
#endif

namespace gcpp {
namespace speculative {

class HnswI8InnerProductSpace final : public hnswlib::SpaceInterface<float> {
 public:
  // Byte format v1: IEEE F32 scale, I32 signed row sum, then padded biased U8.
  // q is in [-127, 127]; stored byte is q+128. Padding represents q=0, not
  // U8=0.
  static constexpr size_t kHeaderBytes = sizeof(float) + sizeof(int32_t);
  static constexpr size_t kMaxDimensions = 65536;
  static constexpr const char* kFeatureFormat = "gemma-head-ann-row-i8-v1";
  static_assert(sizeof(float) == 4 && sizeof(int32_t) == 4,
                "Byte proposal format requires 32-bit fields");

  explicit HnswI8InnerProductSpace(size_t dimensions, bool force_scalar = false)
      : dimensions_(dimensions), padded_((dimensions + 31) & ~size_t{31}) {
    if (dimensions == 0 || dimensions > kMaxDimensions)
      throw std::invalid_argument("Unsupported I8 proposal dimension");
    distance_ = DistanceScalar;
#if GEMMA_HNSW_I8_HAS_VNNI
    if (!force_scalar && NativeAvailable()) {
      distance_ = DistanceVnni;
      uses_vnni_ = true;
    }
#else
    (void)force_scalar;
#endif
  }

  static size_t EncodedBytes(size_t dimensions) {
    return kHeaderBytes + ((dimensions + 31) & ~size_t{31});
  }
  size_t get_data_size() override { return kHeaderBytes + padded_; }
  hnswlib::DISTFUNC<float> get_dist_func() override { return distance_; }
  void* get_dist_func_param() override { return &padded_; }
  size_t Dimensions() const { return dimensions_; }
  size_t PaddedDimensions() const { return padded_; }
  bool UsesVnni() const { return uses_vnni_; }

  static bool NativeAvailable() {
#if GEMMA_HNSW_I8_HAS_VNNI
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("avxvnni");
#else
    return false;
#endif
  }

  bool Encode(const float* input, size_t dimensions,
              std::vector<uint8_t>& encoded) const {
    if (input == nullptr || dimensions != dimensions_) return false;
    float maximum = 0.0f;
    for (size_t c = 0; c < dimensions; ++c) {
      if (!std::isfinite(input[c])) return false;
      maximum = std::max(maximum, std::abs(input[c]));
    }
    const float scale = maximum == 0.0f ? 1.0f : maximum / 127.0f;
    // Reject extreme underflow rather than store a zero scale. A rejected query
    // uses exact native fallback; ordinary model rows are far from this case.
    if (!(scale > 0.0f) || !std::isfinite(scale)) return false;
    encoded.assign(kHeaderBytes + padded_, uint8_t{128});
    int32_t sum = 0;
    for (size_t c = 0; c < dimensions; ++c) {
      // Double prevents overflow of 1/scale for tiny finite rows. The format
      // chooses round-to-nearest, ties away from zero, independently of fenv.
      const double normalized = static_cast<double>(input[c]) / scale;
      const int q = std::max(
          -127, std::min(127, static_cast<int>(std::round(normalized))));
      encoded[kHeaderBytes + c] = static_cast<uint8_t>(q + 128);
      sum += q;
    }
    std::memcpy(encoded.data(), &scale, sizeof(scale));
    std::memcpy(encoded.data() + sizeof(scale), &sum, sizeof(sum));
    return true;
  }

  static float Scale(const void* encoded) {
    float value;
    std::memcpy(&value, encoded, sizeof(value));
    return value;
  }
  static int32_t SignedSum(const void* encoded) {
    int32_t value;
    std::memcpy(&value, static_cast<const uint8_t*>(encoded) + sizeof(float),
                sizeof(value));
    return value;
  }
  static const uint8_t* Payload(const void* encoded) {
    return static_cast<const uint8_t*>(encoded) + kHeaderBytes;
  }

  // Independent scalar integer path, also exposed for synthetic validation.
  static int32_t DotScalar(const void* a, const void* b, size_t padded) {
    const uint8_t* pa = Payload(a);
    const uint8_t* pb = Payload(b);
    int32_t dot = 0;
    for (size_t c = 0; c < padded; ++c)
      dot += (static_cast<int32_t>(pa[c]) - 128) *
             (static_cast<int32_t>(pb[c]) - 128);
    return dot;
  }

  int32_t DotForTesting(const void* a, const void* b) const {
#if GEMMA_HNSW_I8_HAS_VNNI
    if (uses_vnni_) return DotVnni(a, b, padded_);
#endif
    return DotScalar(a, b, padded_);
  }

 private:
  static float FinishDistance(const void* a, const void* b, int32_t dot) {
    const float scale = Scale(a) * Scale(b);
    // One explicit rounding prevents target-specific FMA contraction changing
    // the proposal distance between scalar and VNNI dispatch functions.
    const volatile float score = static_cast<float>(dot) * scale;
    return 1.0f - score;
  }
  static float DistanceScalar(const void* a, const void* b, const void* param) {
    return FinishDistance(a, b,
                          DotScalar(a, b, *static_cast<const size_t*>(param)));
  }

#if GEMMA_HNSW_I8_HAS_VNNI
  __attribute__((target("avx2,avxvnni"))) static int32_t DotVnni(
      const void* a, const void* b, size_t padded) {
    const uint8_t* pa = Payload(a);
    const uint8_t* pb = Payload(b);
    const __m256i sign = _mm256_set1_epi8(static_cast<char>(0x80));
    __m256i c0 = _mm256_setzero_si256();
    __m256i c1 = _mm256_setzero_si256();
    __m256i c2 = _mm256_setzero_si256();
    __m256i c3 = _mm256_setzero_si256();
    size_t c = 0;
    for (; c + 128 <= padded; c += 128) {
      const __m256i a0 = _mm256_xor_si256(
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pa + c)), sign);
      const __m256i a1 = _mm256_xor_si256(
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pa + c + 32)),
          sign);
      const __m256i a2 = _mm256_xor_si256(
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pa + c + 64)),
          sign);
      const __m256i a3 = _mm256_xor_si256(
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pa + c + 96)),
          sign);
      c0 = _mm256_dpbusd_avx_epi32(
          c0, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pb + c)), a0);
      c1 = _mm256_dpbusd_avx_epi32(
          c1, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pb + c + 32)),
          a1);
      c2 = _mm256_dpbusd_avx_epi32(
          c2, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pb + c + 64)),
          a2);
      c3 = _mm256_dpbusd_avx_epi32(
          c3, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pb + c + 96)),
          a3);
    }
    for (; c < padded; c += 32) {
      const __m256i av = _mm256_xor_si256(
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pa + c)), sign);
      c0 = _mm256_dpbusd_avx_epi32(
          c0, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pb + c)), av);
    }
    const __m256i combined =
        _mm256_add_epi32(_mm256_add_epi32(c0, c1), _mm256_add_epi32(c2, c3));
    __m128i sum = _mm_add_epi32(_mm256_castsi256_si128(combined),
                                _mm256_extracti128_si256(combined, 1));
    sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 8));
    sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 4));
    // dot(U8_B, S8_A) = dot(S8_B, S8_A) + 128 * sum(S8_A).
    // 65536 dimensions bound every raw/corrected sum within signed I32.
    return _mm_cvtsi128_si32(sum) - 128 * SignedSum(a);
  }
  __attribute__((target("avx2,avxvnni"))) static float DistanceVnni(
      const void* a, const void* b, const void* param) {
    return FinishDistance(a, b,
                          DotVnni(a, b, *static_cast<const size_t*>(param)));
  }
#endif

  size_t dimensions_;
  size_t padded_;
  bool uses_vnni_ = false;
  hnswlib::DISTFUNC<float> distance_;
};

}  // namespace speculative
}  // namespace gcpp
#endif
