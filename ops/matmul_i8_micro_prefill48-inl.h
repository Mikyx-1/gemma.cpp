// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Optional MR1 x N48 prefill kernel. Include inside the target namespace
// after the existing micro-prefill and shared tile helpers. Uses unchanged N8
// packed weights and preserves primary/residual FMA order at every group.
#if defined(THIRD_PARTY_GEMMA_CPP_MATMUL_I8_MICRO_PREFILL48_TOGGLE) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef THIRD_PARTY_GEMMA_CPP_MATMUL_I8_MICRO_PREFILL48_TOGGLE
#undef THIRD_PARTY_GEMMA_CPP_MATMUL_I8_MICRO_PREFILL48_TOGGLE
#else
#define THIRD_PARTY_GEMMA_CPP_MATMUL_I8_MICRO_PREFILL48_TOGGLE
#endif

#if HWY_TARGET == HWY_AVX2 && GEMMA_MM_I8_BIASED_B && defined(__GNUC__) && \
    !defined(__clang__) && HWY_ARCH_X86_64

template <size_t kBlock, class BT, class Tag, class CView>
static HWY_NOINLINE void MMI8MicroPrefillTile48(const MMI8AView& A, const BT& B,
                                                size_t row_a, size_t row_b,
                                                const IndexRange& range_kc,
                                                const float* add, Tag,
                                                CView C) {
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
    const int32_t correction = (prefix[c + num_k] - prefix[c]) * 128;
    const int32_t correction1 = (prefix1[c + num_k] - prefix1[c]) * 128;
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

template <size_t kBlock, class BT, class Tag, class CView>
static HWY_NOINLINE void MMI8MicroPrefillRange48(
    const MMI8AView& A, const BT& B, const IndexRange& range_mc,
    const IndexRange& range_kc, const IndexRange& range_nc, const float* add,
    Tag tag, CView C) {
  size_t nc = range_nc.begin();
  for (; nc + 48 <= range_nc.end(); nc += 48) {
    const float* bias = MMI8Bias(B, add, nc);
    for (size_t r = 0; r < range_mc.Num(); ++r)
      MMI8MicroPrefillTile48<kBlock>(A, B, range_mc.begin() + r, nc, range_kc,
                                     bias, tag,
                                     C.View(r, nc - range_nc.begin(), 48));
  }
  if (nc + 32 <= range_nc.end()) {
    const float* bias = MMI8Bias(B, add, nc);
    for (size_t r = 0; r < range_mc.Num(); ++r)
      MMI8MicroTile<32, kBlock>(A, B, range_mc.begin() + r, nc, range_kc, bias,
                                tag, C.View(r, nc - range_nc.begin(), 32));
    nc += 32;
  }
  if (nc + 16 <= range_nc.end()) {
    const float* bias = MMI8Bias(B, add, nc);
    for (size_t r = 0; r < range_mc.Num(); ++r)
      MMI8MicroTile<16, kBlock>(A, B, range_mc.begin() + r, nc, range_kc, bias,
                                tag, C.View(r, nc - range_nc.begin(), 16));
    nc += 16;
  }
  for (; nc < range_nc.end(); nc += 8) {
    const size_t count =
        HWY_MIN(size_t{8}, static_cast<size_t>(range_nc.end()) - nc);
    const float* bias = MMI8Bias(B, add, nc);
    for (size_t r = 0; r < range_mc.Num(); ++r)
      MMI8MicroPrefillTile8<1, kBlock, false, true>(
          A, B, range_mc.begin() + r, 0, range_kc, nc, 0, count, 0, bias, tag,
          C.View(r, nc - range_nc.begin(), count));
  }
}

template <class BT, class Tag, class CView>
static HWY_INLINE bool MMI8TryMicroPrefill48(const MMI8AView& A, const BT& B,
                                             const IndexRange& range_mc,
                                             const IndexRange& range_kc,
                                             const IndexRange& range_nc,
                                             const MMArgs& args, Tag tag,
                                             CView C) {
  static const bool enabled = MMI8Flag("GEMMA_MM_I8_MICRO_PREFILL_N48", false);
  if (!enabled || !MMI8FastMicro() || !MMI8NativeVNNI() || !B.packed_micro ||
      !B.dual_a || A.residual == nullptr || range_mc.Num() < 2 ||
      range_nc.begin() % 8 != 0 || range_nc.Num() % 4 != 0 ||
      range_kc.begin() % 4 != 0 || range_kc.end() % 4 != 0)
    return false;
  if (B.block_size == 32)
    MMI8MicroPrefillRange48<32>(A, B, range_mc, range_kc, range_nc, args.add,
                                tag, C);
  else if (B.block_size == 64)
    MMI8MicroPrefillRange48<64>(A, B, range_mc, range_kc, range_nc, args.add,
                                tag, C);
  else if (B.block_size == 128)
    MMI8MicroPrefillRange48<128>(A, B, range_mc, range_kc, range_nc, args.add,
                                 tag, C);
  else
    return false;
  return true;
}
#endif
#endif
