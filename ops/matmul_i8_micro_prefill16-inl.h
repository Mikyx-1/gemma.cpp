// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Included inside the target namespace after matmul_i8_micro_prefill-inl.h.
// Optional MR2xN16 variant for the existing packed N8 microscale weight layout.

#if defined(THIRD_PARTY_GEMMA_CPP_MATMUL_I8_MICRO_PREFILL16_TOGGLE) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef THIRD_PARTY_GEMMA_CPP_MATMUL_I8_MICRO_PREFILL16_TOGGLE
#undef THIRD_PARTY_GEMMA_CPP_MATMUL_I8_MICRO_PREFILL16_TOGGLE
#else
#define THIRD_PARTY_GEMMA_CPP_MATMUL_I8_MICRO_PREFILL16_TOGGLE
#endif

#if HWY_TARGET == HWY_AVX2 && GEMMA_MM_I8_BIASED_B && defined(__GNUC__) && \
    !defined(__clang__) && HWY_ARCH_X86_64

// Two rows x sixteen columns reuse activation broadcasts across two packed
// N8 tiles. Eight integer sums, four float sums and three operand registers
// fit in the sixteen AVX2 registers.
template <size_t kBlock, bool kAligned, bool kDual, class BT, class Tag,
          class CView>
static HWY_NOINLINE void MMI8MicroPrefillTile16(const MMI8AView& A, const BT& B,
                                                size_t row_a, size_t imc,
                                                const IndexRange& range_kc,
                                                size_t row_b, size_t inc,
                                                const float* add, Tag tag,
                                                CView C) {
  static_assert(kBlock == 32 || kBlock == 64 || kBlock == 128);
  const hn::ScalableTag<int32_t> di;
  const hn::ScalableTag<float> df;
  using VI = hn::Vec<decltype(di)>;
  auto f00 = hn::Zero(df), f01 = f00, f10 = f00, f11 = f00;
  const auto* b0 = reinterpret_cast<const uint8_t*>(B.data->Row(row_b));
  const auto* b1 = reinterpret_cast<const uint8_t*>(B.data->Row(row_b + 8));
  const auto* a0 = A.data.Row(row_a);
  const auto* a1 = A.data.Row(row_a + 1);
  const auto* e0 = kDual ? A.residual->data.Row(row_a) : nullptr;
  const auto* e1 = kDual ? A.residual->data.Row(row_a + 1) : nullptr;
  size_t group = range_kc.begin() / kBlock;
  for (size_t c = range_kc.begin(); c < range_kc.end(); ++group) {
    const size_t num_k = kAligned
                             ? kBlock
                             : HWY_MIN(kBlock - c % kBlock,
                                       static_cast<size_t>(range_kc.end()) - c);
    VI p00 = hn::Zero(di), p01 = p00, p10 = p00, p11 = p00;
    VI q00 = p00, q01 = p00, q10 = p00, q11 = p00;
    for (size_t k = 0; k < num_k; k += 4) {
      // One statement fixes the register lifetime of both weight vectors and
      // the shared activation scratch. The dual path has eight integer sums,
      // four floating sums, two weights and one broadcast live (15 registers).
      // Four broadcasts serve eight dots; MR4xN8 needs eight broadcasts.
      using Bytes4 = const uint8_t[4];
      using Bytes32 = const uint8_t[32];
      const size_t pos = c + k;
      VI weight0, weight1, activation;
      if constexpr (kDual) {
        asm("vmovdqu %[b0], %[w0]\n\t"
            "vmovdqu %[b1], %[w1]\n\t"
            "vpbroadcastd %[a0], %[a]\n\t"
            "%{vex%} vpdpbusd %[a], %[w0], %[p00]\n\t"
            "%{vex%} vpdpbusd %[a], %[w1], %[p01]\n\t"
            "vpbroadcastd %[e0], %[a]\n\t"
            "%{vex%} vpdpbusd %[a], %[w0], %[q00]\n\t"
            "%{vex%} vpdpbusd %[a], %[w1], %[q01]\n\t"
            "vpbroadcastd %[a1], %[a]\n\t"
            "%{vex%} vpdpbusd %[a], %[w0], %[p10]\n\t"
            "%{vex%} vpdpbusd %[a], %[w1], %[p11]\n\t"
            "vpbroadcastd %[e1], %[a]\n\t"
            "%{vex%} vpdpbusd %[a], %[w0], %[q10]\n\t"
            "%{vex%} vpdpbusd %[a], %[w1], %[q11]"
            : [p00] "+&x"(p00.raw), [p01] "+&x"(p01.raw), [p10] "+&x"(p10.raw),
              [p11] "+&x"(p11.raw), [q00] "+&x"(q00.raw), [q01] "+&x"(q01.raw),
              [q10] "+&x"(q10.raw), [q11] "+&x"(q11.raw),
              [w0] "=&x"(weight0.raw), [w1] "=&x"(weight1.raw),
              [a] "=&x"(activation.raw)
            : [b0] "m"(*reinterpret_cast<Bytes32*>(b0 + pos * 8)),
              [b1] "m"(*reinterpret_cast<Bytes32*>(b1 + pos * 8)),
              [a0] "m"(*reinterpret_cast<Bytes4*>(a0 + pos)),
              [e0] "m"(*reinterpret_cast<Bytes4*>(e0 + pos)),
              [a1] "m"(*reinterpret_cast<Bytes4*>(a1 + pos)),
              [e1] "m"(*reinterpret_cast<Bytes4*>(e1 + pos)));
      } else {
        asm("vmovdqu %[b0], %[w0]\n\t"
            "vmovdqu %[b1], %[w1]\n\t"
            "vpbroadcastd %[a0], %[a]\n\t"
            "%{vex%} vpdpbusd %[a], %[w0], %[p00]\n\t"
            "%{vex%} vpdpbusd %[a], %[w1], %[p01]\n\t"
            "vpbroadcastd %[a1], %[a]\n\t"
            "%{vex%} vpdpbusd %[a], %[w0], %[p10]\n\t"
            "%{vex%} vpdpbusd %[a], %[w1], %[p11]"
            : [p00] "+&x"(p00.raw), [p01] "+&x"(p01.raw), [p10] "+&x"(p10.raw),
              [p11] "+&x"(p11.raw), [w0] "=&x"(weight0.raw),
              [w1] "=&x"(weight1.raw), [a] "=&x"(activation.raw)
            : [b0] "m"(*reinterpret_cast<Bytes32*>(b0 + pos * 8)),
              [b1] "m"(*reinterpret_cast<Bytes32*>(b1 + pos * 8)),
              [a0] "m"(*reinterpret_cast<Bytes4*>(a0 + pos)),
              [a1] "m"(*reinterpret_cast<Bytes4*>(a1 + pos)));
      }
    }
    const auto accumulate = [&](VI primary, VI residual, size_t r, size_t n,
                                auto& result) HWY_ATTR {
      const auto bs = hn::LoadU(df, B.scale + group * B.Rows() + row_b + n);
      const auto av = A.ViewGroup(row_a + r, c, num_k, group);
      primary = hn::Sub(primary, hn::Set(di, av.RowSum(0, num_k) * 128));
      const auto scale = hn::Mul(bs, hn::Set(df, av.scale[0]));
      result = hn::MulAdd(hn::ConvertTo(df, primary), scale, result);
      if constexpr (kDual) {
        const auto rv = A.residual->ViewGroup(row_a + r, c, num_k, group);
        residual = hn::Sub(residual, hn::Set(di, rv.RowSum(0, num_k) * 128));
        const auto residual_scale = hn::Mul(bs, hn::Set(df, rv.scale[0]));
        result =
            hn::MulAdd(hn::ConvertTo(df, residual), residual_scale, result);
      }
    };
    accumulate(p00, q00, 0, 0, f00);
    accumulate(p01, q01, 0, 8, f01);
    accumulate(p10, q10, 1, 0, f10);
    accumulate(p11, q11, 1, 8, f11);
    c += num_k;
  }
  MMI8MicroPrefillStore(df, f00, imc, inc, add, tag, C);
  MMI8MicroPrefillStore(df, f01, imc, inc + 8, add ? add + 8 : nullptr, tag, C);
  MMI8MicroPrefillStore(df, f10, imc + 1, inc, add, tag, C);
  MMI8MicroPrefillStore(df, f11, imc + 1, inc + 8, add ? add + 8 : nullptr, tag,
                        C);
}

template <size_t kBlock, bool kAligned, bool kDual, class BT, class Tag,
          class CView>
static HWY_NOINLINE void MMI8PackedMicroPrefill16(
    const MMI8AView& A, const BT& B, const IndexRange& range_mc,
    const IndexRange& range_kc, const IndexRange& range_nc, const MMArgs& args,
    Tag tag, CView C) {
  for (size_t nc = range_nc.begin(); nc < range_nc.end();) {
    const size_t inc = nc - range_nc.begin();
    const size_t remaining = static_cast<size_t>(range_nc.end()) - nc;
    if (nc % 8 == 0 && remaining >= 16) {
      const float* add = MMI8Bias(B, args.add, nc);
      size_t r = 0;
      for (; r + 2 <= range_mc.Num(); r += 2) {
        MMI8MicroPrefillTile16<kBlock, kAligned, kDual>(
            A, B, range_mc.begin() + r, r, range_kc, nc, inc, add, tag, C);
      }
      if (r < range_mc.Num()) {
        for (size_t n = 0; n < 16; n += 8) {
          MMI8MicroPrefillTile8<1, kBlock, kAligned, kDual>(
              A, B, range_mc.begin() + r, r, range_kc, nc + n, 0, 8, inc + n,
              add ? add + n : nullptr, tag, C);
        }
      }
      nc += 16;
    } else {
      const size_t count = HWY_MIN(size_t{8} - nc % 8, remaining);
      MMI8PackedMicroPrefill<kBlock, kAligned, kDual>(
          A, B, range_mc, range_kc, IndexRange(nc, nc + count), args, tag,
          C.View(0, inc, count));
      nc += count;
    }
  }
}

#endif
#endif
