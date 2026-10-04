// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Included inside the target namespace after matmul_i8_micro_tile-inl.h.
// Reuses the native dual-A N32/N16 tile for independent prefill rows.

#if defined(THIRD_PARTY_GEMMA_CPP_MATMUL_I8_MICRO_PREFILL32_TOGGLE) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef THIRD_PARTY_GEMMA_CPP_MATMUL_I8_MICRO_PREFILL32_TOGGLE
#undef THIRD_PARTY_GEMMA_CPP_MATMUL_I8_MICRO_PREFILL32_TOGGLE
#else
#define THIRD_PARTY_GEMMA_CPP_MATMUL_I8_MICRO_PREFILL32_TOGGLE
#endif

#if HWY_TARGET == HWY_AVX2 && GEMMA_MM_I8_BIASED_B && defined(__GNUC__) && \
    !defined(__clang__) && HWY_ARCH_X86_64

template <size_t kBlock, class BT, class Tag, class CView>
static HWY_NOINLINE void MMI8MicroPrefillRange32(
    const MMI8AView& A, const BT& B, const IndexRange& range_mc,
    const IndexRange& range_kc, const IndexRange& range_nc, const float* add,
    Tag tag, CView C) {
  // N is outermost so each group of rows reuses the same small weight tile.
  // Unlike MR4xN8, each row shares its broadcasts and group metadata across
  // 32 channels. The underlying kernel retains native per-group FMA order.
  size_t nc = range_nc.begin();
  for (; nc + 32 <= range_nc.end(); nc += 32) {
    const float* bias = MMI8Bias(B, add, nc);
    for (size_t r = 0; r < range_mc.Num(); ++r) {
      MMI8MicroTile<32, kBlock>(A, B, range_mc.begin() + r, nc, range_kc, bias,
                                tag, C.View(r, nc - range_nc.begin(), 32));
    }
  }
  if (nc < range_nc.end()) {
    HWY_DASSERT(range_nc.end() - nc == 16);
    const float* bias = MMI8Bias(B, add, nc);
    for (size_t r = 0; r < range_mc.Num(); ++r) {
      MMI8MicroTile<16, kBlock>(A, B, range_mc.begin() + r, nc, range_kc, bias,
                                tag, C.View(r, nc - range_nc.begin(), 16));
    }
  }
}

template <class BT, class Tag, class CView>
static HWY_INLINE bool MMI8TryMicroPrefill32(const MMI8AView& A, const BT& B,
                                             const IndexRange& range_mc,
                                             const IndexRange& range_kc,
                                             const IndexRange& range_nc,
                                             const MMArgs& args, Tag tag,
                                             CView C) {
  static const bool enabled = MMI8Flag("GEMMA_MM_I8_MICRO_PREFILL_N32", false);
  if (!enabled || !MMI8FastMicro() || !MMI8NativeVNNI() || !B.packed_micro ||
      !B.dual_a || A.residual == nullptr || range_mc.Num() < 2 ||
      range_nc.begin() % 8 != 0 || range_nc.Num() % 16 != 0 ||
      range_kc.begin() % 4 != 0 || range_kc.end() % 4 != 0) {
    return false;
  }
  if (B.block_size == 32) {
    MMI8MicroPrefillRange32<32>(A, B, range_mc, range_kc, range_nc, args.add,
                                tag, C);
  } else if (B.block_size == 64) {
    MMI8MicroPrefillRange32<64>(A, B, range_mc, range_kc, range_nc, args.add,
                                tag, C);
  } else if (B.block_size == 128) {
    MMI8MicroPrefillRange32<128>(A, B, range_mc, range_kc, range_nc, args.add,
                                 tag, C);
  } else {
    return false;
  }
  return true;
}

#endif
#endif
