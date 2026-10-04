// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0
// Request-local control for one terminal sibling. No model/cache storage.
#if defined(THIRD_PARTY_GEMMA_CPP_BODY_SPEC_STOPPED_LEAF_TOGGLE) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef THIRD_PARTY_GEMMA_CPP_BODY_SPEC_STOPPED_LEAF_TOGGLE
#undef THIRD_PARTY_GEMMA_CPP_BODY_SPEC_STOPPED_LEAF_TOGGLE
#else
#define THIRD_PARTY_GEMMA_CPP_BODY_SPEC_STOPPED_LEAF_TOGGLE
#endif

class BodySpecStoppedLeafRound {
 public:
  // Called only inside the actual successful stopping branch, before count is
  // shortened. A requested count below4 is not evidence of a stopping event.
  bool Offer(size_t row, size_t original_count, int top1, int second) {
    if (offered_ || original_count < 3 || original_count > 4 ||
        row >= original_count - 2 || top1 < 0 || second < 0 || top1 == second)
      return false;
    parent_ = row;
    count_ = row + 2;
    second_ = second;
    offered_ = true;
    return true;
  }
  bool Matches(size_t spine_count) const {
    return offered_ && spine_count == count_;
  }
  void SetComputed(bool computed) {
    HWY_ASSERT(!computed || offered_);
    computed_ = computed;
  }
  bool Computed() const { return computed_; }
  bool IsSecondHit(size_t sampled_row, int token) const {
    return computed_ && sampled_row == parent_ && token == second_;
  }
  bool ShouldSampleLeaf(size_t sampled_row, int token, bool active,
                        size_t generated, size_t max_steps) const {
    return !committed_ && IsSecondHit(sampled_row, token) && active &&
           generated < max_steps;
  }
  void MarkCommitted() {
    HWY_ASSERT(computed_ && !committed_);
    committed_ = true;
  }
  bool Committed() const { return committed_; }
  int Second() const { return second_; }
  // A sampled leaf consumes second at the same position as the old deferred
  // top1 sibling. Catch up exactly one of these inputs, never both.
  int CatchUpInput(int deferred_top1) const {
    return committed_ ? second_ : deferred_top1;
  }

 private:
  size_t parent_ = 0;
  size_t count_ = 0;
  int second_ = -1;
  bool offered_ = false;
  bool computed_ = false;
  bool committed_ = false;
};
#endif
