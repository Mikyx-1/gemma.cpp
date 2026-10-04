// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Included inside the target namespace, after MMI8AView and MMI8B.
// Multi-row packed microscale kernels preserve every group/KC rounding point.

#if defined(THIRD_PARTY_GEMMA_CPP_MATMUL_I8_MICRO_PREFILL_TOGGLE) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef THIRD_PARTY_GEMMA_CPP_MATMUL_I8_MICRO_PREFILL_TOGGLE
#undef THIRD_PARTY_GEMMA_CPP_MATMUL_I8_MICRO_PREFILL_TOGGLE
#else
#define THIRD_PARTY_GEMMA_CPP_MATMUL_I8_MICRO_PREFILL_TOGGLE
#endif

#if HWY_TARGET == HWY_AVX2 && GEMMA_MM_I8_BIASED_B && defined(__GNUC__) && \
    !defined(__clang__) && HWY_ARCH_X86_64

template <class DF, class VF, class Tag, class CView>
static HWY_INLINE void MMI8MicroPrefillStore(DF df, VF accum, size_t r,
                                             size_t inc, const float* add, Tag,
                                             CView C) {
  using TC = hwy::RemoveCvRef<decltype(C.Row(0)[0])>;
  const hn::Rebind<TC, DF> dc;
  TC* const pos = C.Row(r) + inc;
  if constexpr (hwy::IsSame<Tag, MMAddC>()) {
    accum = hn::Add(accum, F32FromTC(dc, hn::LoadU(dc, pos)));
  } else {
    static_assert(hwy::IsSame<Tag, MMSetC>());
    if (add != nullptr) accum = hn::Add(accum, hn::LoadU(df, add));
  }
  hn::StoreU(TCFromF32(dc, accum), dc, pos);
}

// Four rows share each weight vector and group scale. Four primary and four
// residual sums plus four float accumulators leave room for B and both A
// broadcasts in sixteen AVX2 registers. Smaller row tails use extra K chains.
template <size_t kRows, size_t kBlock, bool kAligned, bool kDual, class BT,
          class Tag, class CView>
static HWY_NOINLINE void MMI8MicroPrefillTile8(
    const MMI8AView& A, const BT& B, size_t row_a, size_t imc,
    const IndexRange& range_kc, size_t row_b, size_t lane, size_t count,
    size_t inc, const float* add, Tag tag, CView C) {
  static_assert(kRows == 1 || kRows == 2 || kRows == 4);
  static_assert(kBlock == 32 || kBlock == 64 || kBlock == 128);
  constexpr size_t kUnroll = 4 / kRows;
  const bool use_group_asm = []() {
    if constexpr (kRows == 4 && kDual) {
      static const bool enabled =
          MMI8Flag("GEMMA_MM_I8_MICRO_PREFILL_GROUP_ASM", false);
      return enabled;
    }
    return false;
  }();
  const hn::ScalableTag<int8_t> da;
  const hn::ScalableTag<uint8_t> db;
  const hn::ScalableTag<int32_t> di;
  const hn::ScalableTag<uint32_t> du;
  const hn::ScalableTag<float> df;
  const hn::Full128<float> d4f;
  using VI = hn::Vec<decltype(di)>;
  auto f0 = hn::Zero(df), f1 = f0, f2 = f0, f3 = f0;
  const auto* packed = reinterpret_cast<const uint8_t*>(B.data->Row(row_b));
  const int8_t* a0 = kRows > 0 ? A.data.Row(row_a + 0) : nullptr;
  const int8_t* e0 =
      kDual && kRows > 0 ? A.residual->data.Row(row_a + 0) : nullptr;
  const int8_t* a1 = kRows > 1 ? A.data.Row(row_a + 1) : nullptr;
  const int8_t* e1 =
      kDual && kRows > 1 ? A.residual->data.Row(row_a + 1) : nullptr;
  const int8_t* a2 = kRows > 2 ? A.data.Row(row_a + 2) : nullptr;
  const int8_t* e2 =
      kDual && kRows > 2 ? A.residual->data.Row(row_a + 2) : nullptr;
  const int8_t* a3 = kRows > 3 ? A.data.Row(row_a + 3) : nullptr;
  const int8_t* e3 =
      kDual && kRows > 3 ? A.residual->data.Row(row_a + 3) : nullptr;
  size_t group = range_kc.begin() / kBlock;
  for (size_t c = range_kc.begin(); c < range_kc.end(); ++group) {
    const size_t num_k = kAligned
                             ? kBlock
                             : HWY_MIN(kBlock - c % kBlock,
                                       static_cast<size_t>(range_kc.end()) - c);
    VI p0 = hn::Zero(di), p1 = p0, p2 = p0, p3 = p0;
    VI q0 = p0, q1 = p0, q2 = p0, q3 = p0;
    const auto dot = [&](const int8_t* ap, const int8_t* ar, size_t offset,
                         auto b, VI& primary, VI& residual) HWY_ATTR {
      uint32_t bits;
      hwy::CopyBytes<4>(ap + c + offset, &bits);
      const auto a = hn::BitCast(da, hn::Set(du, bits));
      if constexpr (kDual) {
        uint32_t residual_bits;
        hwy::CopyBytes<4>(ar + c + offset, &residual_bits);
        const auto a1 = hn::BitCast(da, hn::Set(du, residual_bits));
        asm("%{vex%} vpdpbusd %[a], %[b], %[primary]\n\t"
            "%{vex%} vpdpbusd %[a1], %[b], %[residual]"
            : [primary] "+&x"(primary.raw), [residual] "+&x"(residual.raw)
            : [a] "x"(a.raw), [a1] "x"(a1.raw), [b] "x"(b.raw));
      } else {
        asm("%{vex%} vpdpbusd %[a], %[b], %[primary]"
            : [primary] "+x"(primary.raw)
            : [a] "x"(a.raw), [b] "x"(b.raw));
      }
    };
    size_t k = 0;
    if constexpr (kRows == 4 && kDual) {
      if (use_group_asm && num_k >= 4 && num_k % 4 == 0) {
        // Keep all K iterations in the same asm statement. This avoids the
        // sixteen register moves GCC otherwise emits for loop-carried sums.
        // There are eight integer sums, two scratch vectors, and four float
        // accumulators live across this loop: fourteen AVX2 registers total.
        // num_k may be a partial group; neither its endpoint nor the following
        // group correction/FMA/store order changes.
        VI weights, activation;
        size_t offset;
        asm("vpxor %[p0], %[p0], %[p0]\n\t"
            "vpxor %[q0], %[q0], %[q0]\n\t"
            "vpxor %[p1], %[p1], %[p1]\n\t"
            "vpxor %[q1], %[q1], %[q1]\n\t"
            "vpxor %[p2], %[p2], %[p2]\n\t"
            "vpxor %[q2], %[q2], %[q2]\n\t"
            "vpxor %[p3], %[p3], %[p3]\n\t"
            "vpxor %[q3], %[q3], %[q3]\n\t"
            "xor %[offset], %[offset]\n\t"
            ".Lgemma_i8_mr4_group_%=:\n\t"
            "vmovdqu (%[b], %[offset], 8), %[w]\n\t"
            "vpbroadcastd (%[a0], %[offset]), %[a]\n\t"
            "%{vex%} vpdpbusd %[a], %[w], %[p0]\n\t"
            "vpbroadcastd (%[e0], %[offset]), %[a]\n\t"
            "%{vex%} vpdpbusd %[a], %[w], %[q0]\n\t"
            "vpbroadcastd (%[a1], %[offset]), %[a]\n\t"
            "%{vex%} vpdpbusd %[a], %[w], %[p1]\n\t"
            "vpbroadcastd (%[e1], %[offset]), %[a]\n\t"
            "%{vex%} vpdpbusd %[a], %[w], %[q1]\n\t"
            "vpbroadcastd (%[a2], %[offset]), %[a]\n\t"
            "%{vex%} vpdpbusd %[a], %[w], %[p2]\n\t"
            "vpbroadcastd (%[e2], %[offset]), %[a]\n\t"
            "%{vex%} vpdpbusd %[a], %[w], %[q2]\n\t"
            "vpbroadcastd (%[a3], %[offset]), %[a]\n\t"
            "%{vex%} vpdpbusd %[a], %[w], %[p3]\n\t"
            "vpbroadcastd (%[e3], %[offset]), %[a]\n\t"
            "%{vex%} vpdpbusd %[a], %[w], %[q3]\n\t"
            "add $4, %[offset]\n\t"
            "cmp %[num_k], %[offset]\n\t"
            "jb .Lgemma_i8_mr4_group_%="
            : [p0] "=&x"(p0.raw), [q0] "=&x"(q0.raw), [p1] "=&x"(p1.raw),
              [q1] "=&x"(q1.raw), [p2] "=&x"(p2.raw), [q2] "=&x"(q2.raw),
              [p3] "=&x"(p3.raw), [q3] "=&x"(q3.raw), [w] "=&x"(weights.raw),
              [a] "=&x"(activation.raw), [offset] "=&r"(offset)
            : [b] "r"(packed + c * 8), [a0] "r"(a0 + c), [e0] "r"(e0 + c),
              [a1] "r"(a1 + c), [e1] "r"(e1 + c), [a2] "r"(a2 + c),
              [e2] "r"(e2 + c), [a3] "r"(a3 + c), [e3] "r"(e3 + c),
              [num_k] "r"(num_k)
            : "cc", "memory");
        // The asm read each group byte once and completed the integer sums.
        // Skip the existing K loop while retaining its fallback when disabled.
        k = num_k;
      }
    }
    for (; k + 4 * kUnroll <= num_k; k += 4 * kUnroll) {
      const auto b0 = hn::LoadU(db, packed + (c + k) * 8);
      if constexpr (kRows == 4 && kDual) {
        // Keep B shared and broadcast one activation at a time. Separate dot
        // asm statements let GCC hoist all eight broadcasts and reload B,
        // spilling integer accumulators despite the fourteen-register live set.
        // The four-byte array operands permit unaligned accesses without an
        // aliased uint32_t dereference and describe the exact memory reads.
        using Bytes4 = const uint8_t[4];
        const size_t pos = c + k;
        VI activation;
        asm("vpbroadcastd %[a0], %[a]\n\t"
            "%{vex%} vpdpbusd %[a], %[b], %[p0]\n\t"
            "vpbroadcastd %[e0], %[a]\n\t"
            "%{vex%} vpdpbusd %[a], %[b], %[q0]\n\t"
            "vpbroadcastd %[a1], %[a]\n\t"
            "%{vex%} vpdpbusd %[a], %[b], %[p1]\n\t"
            "vpbroadcastd %[e1], %[a]\n\t"
            "%{vex%} vpdpbusd %[a], %[b], %[q1]\n\t"
            "vpbroadcastd %[a2], %[a]\n\t"
            "%{vex%} vpdpbusd %[a], %[b], %[p2]\n\t"
            "vpbroadcastd %[e2], %[a]\n\t"
            "%{vex%} vpdpbusd %[a], %[b], %[q2]\n\t"
            "vpbroadcastd %[a3], %[a]\n\t"
            "%{vex%} vpdpbusd %[a], %[b], %[p3]\n\t"
            "vpbroadcastd %[e3], %[a]\n\t"
            "%{vex%} vpdpbusd %[a], %[b], %[q3]"
            : [p0] "+&x"(p0.raw), [q0] "+&x"(q0.raw), [p1] "+&x"(p1.raw),
              [q1] "+&x"(q1.raw), [p2] "+&x"(p2.raw), [q2] "+&x"(q2.raw),
              [p3] "+&x"(p3.raw), [q3] "+&x"(q3.raw), [a] "=&x"(activation.raw)
            : [b] "x"(b0.raw), [a0] "m"(*reinterpret_cast<Bytes4*>(a0 + pos)),
              [e0] "m"(*reinterpret_cast<Bytes4*>(e0 + pos)),
              [a1] "m"(*reinterpret_cast<Bytes4*>(a1 + pos)),
              [e1] "m"(*reinterpret_cast<Bytes4*>(e1 + pos)),
              [a2] "m"(*reinterpret_cast<Bytes4*>(a2 + pos)),
              [e2] "m"(*reinterpret_cast<Bytes4*>(e2 + pos)),
              [a3] "m"(*reinterpret_cast<Bytes4*>(a3 + pos)),
              [e3] "m"(*reinterpret_cast<Bytes4*>(e3 + pos)));
      } else {
        dot(a0, e0, k, b0, p0, q0);
        if constexpr (kRows > 1) dot(a1, e1, k, b0, p1, q1);
        if constexpr (kRows > 2) {
          dot(a2, e2, k, b0, p2, q2);
          dot(a3, e3, k, b0, p3, q3);
        }
      }
      if constexpr (kUnroll > 1) {
        const auto b1 = hn::LoadU(db, packed + (c + k + 4) * 8);
        if constexpr (kRows == 2) {
          dot(a0, e0, k + 4, b1, p2, q2);
          dot(a1, e1, k + 4, b1, p3, q3);
        } else {
          dot(a0, e0, k + 4, b1, p1, q1);
          const auto b2 = hn::LoadU(db, packed + (c + k + 8) * 8);
          const auto b3 = hn::LoadU(db, packed + (c + k + 12) * 8);
          dot(a0, e0, k + 8, b2, p2, q2);
          dot(a0, e0, k + 12, b3, p3, q3);
        }
      }
    }
    if constexpr (!kAligned) {
      for (; k < num_k; k += 4) {
        const auto b0 = hn::LoadU(db, packed + (c + k) * 8);
        dot(a0, e0, k, b0, p0, q0);
        if constexpr (kRows > 1) dot(a1, e1, k, b0, p1, q1);
        if constexpr (kRows > 2) {
          dot(a2, e2, k, b0, p2, q2);
          dot(a3, e3, k, b0, p3, q3);
        }
      }
    }
    if constexpr (kRows == 2) {
      p0 = hn::Add(p0, p2);
      p1 = hn::Add(p1, p3);
      if constexpr (kDual) {
        q0 = hn::Add(q0, q2);
        q1 = hn::Add(q1, q3);
      }
    } else if constexpr (kRows == 1) {
      p0 = hn::Add(hn::Add(p0, p1), hn::Add(p2, p3));
      if constexpr (kDual) q0 = hn::Add(hn::Add(q0, q1), hn::Add(q2, q3));
    }
    const auto bs = hn::LoadU(df, B.scale + group * B.Rows() + row_b);
    const auto accumulate = [&](VI primary, VI residual, size_t r,
                                auto& result) HWY_ATTR {
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
    accumulate(p0, q0, 0, f0);
    if constexpr (kRows > 1) accumulate(p1, q1, 1, f1);
    if constexpr (kRows > 2) {
      accumulate(p2, q2, 2, f2);
      accumulate(p3, q3, 3, f3);
    }
    c += num_k;
  }
  const auto store = [&](auto value, size_t r) HWY_ATTR {
    if (count == 8) {
      MMI8MicroPrefillStore(df, value, imc + r, inc, add, tag, C);
    } else {
      HWY_DASSERT(count == 4);
      const auto half =
          lane ? hn::UpperHalf(d4f, value) : hn::LowerHalf(d4f, value);
      MMI8MicroPrefillStore(d4f, half, imc + r, inc, add, tag, C);
    }
  };
  store(f0, 0);
  if constexpr (kRows > 1) store(f1, 1);
  if constexpr (kRows > 2) {
    store(f2, 2);
    store(f3, 3);
  }
}

template <size_t kBlock, bool kAligned, bool kDual, class BT, class Tag,
          class CView>
static HWY_NOINLINE void MMI8PackedMicroPrefill(const MMI8AView& A, const BT& B,
                                                const IndexRange& range_mc,
                                                const IndexRange& range_kc,
                                                const IndexRange& range_nc,
                                                const MMArgs& args, Tag tag,
                                                CView C) {
  for (size_t nc = range_nc.begin(); nc < range_nc.end();) {
    const size_t row_b = nc & ~size_t{7};
    const size_t lane = nc % 8;
    const size_t count =
        HWY_MIN(size_t{8} - lane, static_cast<size_t>(range_nc.end()) - nc);
    const size_t inc = nc - range_nc.begin();
    const float* add = MMI8Bias(B, args.add, nc);
    size_t r = 0;
    for (; r + 4 <= range_mc.Num(); r += 4) {
      MMI8MicroPrefillTile8<4, kBlock, kAligned, kDual>(
          A, B, range_mc.begin() + r, r, range_kc, row_b, lane, count, inc, add,
          tag, C);
    }
    if (r + 2 <= range_mc.Num()) {
      MMI8MicroPrefillTile8<2, kBlock, kAligned, kDual>(
          A, B, range_mc.begin() + r, r, range_kc, row_b, lane, count, inc, add,
          tag, C);
      r += 2;
    }
    if (r < range_mc.Num()) {
      MMI8MicroPrefillTile8<1, kBlock, kAligned, kDual>(
          A, B, range_mc.begin() + r, r, range_kc, row_b, lane, count, inc, add,
          tag, C);
    }
    nc += count;
  }
}
#endif
#endif
