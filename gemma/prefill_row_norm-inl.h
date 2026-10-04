// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0
// Opt-in scheduling helper; include inside the target namespace after ops.
// Calls the original row helpers, preserving every reduction and output store.
template <typename T>
static bool PrivatePrefillNormResidualPreNorm(
    const MatPtr& post_weight, MatPtrT<T>& other, MatPtrT<float>& x,
    const MatPtr& pre_weight, MatPtrT<BF16>& out, ThreadingContext& ctx) {
  if (other.Rows() <= 1 || other.Cols() == 0 || !other.HasPtr() ||
      !x.HasPtr() || !out.HasPtr() || !other.SameShape(x) ||
      !x.SameShape(out) || !post_weight.HasPtr() || !pre_weight.HasPtr() ||
      post_weight.Rows() != 1 || pre_weight.Rows() != 1 ||
      post_weight.Cols() != x.Cols() || pre_weight.Cols() != x.Cols() ||
      post_weight.GetType() != pre_weight.GetType())
    return false;
  CallUpcastedSame(
      &post_weight, &pre_weight, [&](const auto* post, const auto* pre) {
        ParallelFor(
            Parallelism::kFlat, x.Rows(), ctx, 0, Callers::kOpsRMSNormBatched,
            [&](size_t row, size_t worker) HWY_ATTR {
              RMSNormInplace(post->PackedScale1(), 0, other.Row(row),
                             other.Cols(), ctx, worker);
              AddFrom(other.Row(row), x.Row(row), x.Cols(), ctx, worker);
              RMSNorm(x.Row(row), pre->PackedScale1(), 0, out.Row(row),
                      x.Cols(), ctx, worker);
            });
      });
  return true;
}

template <typename T>
static bool PrivatePrefillNormResidual(const MatPtr& post_weight,
                                       MatPtrT<T>& other, MatPtrT<float>& x,
                                       ThreadingContext& ctx) {
  if (other.Rows() <= 1 || other.Cols() == 0 || !other.HasPtr() ||
      !x.HasPtr() || !other.SameShape(x) || !post_weight.HasPtr() ||
      post_weight.Rows() != 1 || post_weight.Cols() != x.Cols())
    return false;
  CallUpcasted(&post_weight, [&](const auto* post) {
    ParallelFor(Parallelism::kFlat, x.Rows(), ctx, 0,
                Callers::kOpsRMSNormInplaceBatched,
                [&](size_t row, size_t worker) HWY_ATTR {
                  RMSNormInplace(post->PackedScale1(), 0, other.Row(row),
                                 other.Cols(), ctx, worker);
                  AddFrom(other.Row(row), x.Row(row), x.Cols(), ctx, worker);
                });
  });
  return true;
}
