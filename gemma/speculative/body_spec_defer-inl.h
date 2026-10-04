// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0
// Private control state for one speculative round. Include in the target
// namespace; no tensor, model or cache layout changes.
#if defined(THIRD_PARTY_GEMMA_CPP_GEMMA_BODY_SPEC_DEFER_TOGGLE) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef THIRD_PARTY_GEMMA_CPP_GEMMA_BODY_SPEC_DEFER_TOGGLE
#undef THIRD_PARTY_GEMMA_CPP_GEMMA_BODY_SPEC_DEFER_TOGGLE
#else
#define THIRD_PARTY_GEMMA_CPP_GEMMA_BODY_SPEC_DEFER_TOGGLE
#endif

class BodySpecDeferredDraftBody {
 public:
  bool TryDefer(bool enabled, bool skip_last_head, size_t row, size_t count,
                size_t position, int input) {
    if (!enabled || !skip_last_head || pending_ || count < 2 || row >= count ||
        count - row != 1)
      return false;
    pending_ = true;
    position_ = position;
    input_ = input;
    return true;
  }

  // The missing input is committed only when the target advances past it.
  // No private state will be read if generation ended, ANN failed, or only
  // the final native token remains. Subtractions avoid size_t wraparound.
  bool NeedsCatchUp(size_t verified_position, size_t generated,
                    size_t max_steps, bool active, bool ann_failed) const {
    return pending_ && active && !ann_failed && verified_position > position_ &&
           verified_position - position_ == 1 && generated < max_steps &&
           max_steps - generated > 1;
  }

  size_t Position() const { return position_; }
  int Input() const { return input_; }

 private:
  bool pending_ = false;
  size_t position_ = 0;
  int input_ = 0;
};
#endif
