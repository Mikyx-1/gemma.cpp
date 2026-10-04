// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Included inside the target namespace after the shared INT8 store helpers.
// Uses the existing N8 x K4 layout and preserves both per-group FMAs and every
// KC output store. A larger N tile shares activation work across weight
// streams.

#if defined(THIRD_PARTY_GEMMA_CPP_MATMUL_I8_MICRO_TILE_TOGGLE) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef THIRD_PARTY_GEMMA_CPP_MATMUL_I8_MICRO_TILE_TOGGLE
#undef THIRD_PARTY_GEMMA_CPP_MATMUL_I8_MICRO_TILE_TOGGLE
#else
#define THIRD_PARTY_GEMMA_CPP_MATMUL_I8_MICRO_TILE_TOGGLE
#endif

#if HWY_TARGET == HWY_AVX2 && GEMMA_MM_I8_BIASED_B && defined(__GNUC__) && \
    !defined(__clang__) && HWY_ARCH_X86_64

template <size_t kOutputs, size_t kBlock, class BT, class Tag, class CView>
static HWY_NOINLINE void MMI8MicroTile(const MMI8AView& A, const BT& B,
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
    const int32_t correction = (prefix[c + num_k] - prefix[c]) * 128;
    const int32_t correction1 = (prefix1[c + num_k] - prefix1[c]) * 128;
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

#endif  // native x86-64 AVX-VNNI
#endif  // include guard
