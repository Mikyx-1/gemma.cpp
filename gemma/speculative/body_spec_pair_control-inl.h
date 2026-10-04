// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0
// Request-local, single-use identity for a known next private body row.
#if defined(THIRD_PARTY_GEMMA_CPP_BODY_SPEC_PAIR_CONTROL_TOGGLE) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef THIRD_PARTY_GEMMA_CPP_BODY_SPEC_PAIR_CONTROL_TOGGLE
#undef THIRD_PARTY_GEMMA_CPP_BODY_SPEC_PAIR_CONTROL_TOGGLE
#else
#define THIRD_PARTY_GEMMA_CPP_BODY_SPEC_PAIR_CONTROL_TOGGLE
#endif

class BodySpecPairedRowState {
 public:
  static bool Eligible(bool enabled, bool active, bool failed, size_t generated,
                       size_t max_steps, size_t catchup_position,
                       int catchup_input, size_t next_position, int next_input,
                       size_t vocab) {
    return enabled && active && !failed && generated < max_steps &&
           max_steps - generated >= 2 && next_position > catchup_position &&
           next_position - catchup_position == 1 && catchup_input >= 0 &&
           next_input >= 0 && static_cast<size_t>(catchup_input) < vocab &&
           static_cast<size_t>(next_input) < vocab;
  }
  bool Pending() const { return pending_; }
  bool Publish(size_t position, int input) {
    if (pending_) return false;
    position_ = position;
    input_ = input;
    pending_ = true;
    return true;
  }
  bool Matches(size_t position, int input) const {
    return pending_ && position == position_ && input == input_;
  }
  bool Consume(size_t position, int input) {
    if (!Matches(position, input)) return false;
    pending_ = false;
    return true;
  }

 private:
  bool pending_ = false;
  size_t position_ = 0;
  int input_ = 0;
};
#endif
