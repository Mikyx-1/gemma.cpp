// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0
// Request-local stopping policy for optional 270M ANN proposals. Include inside
// the target namespace with <cmath>, <cstdlib>, and <cstring> already
// available. Only draft length changes; the target verifier still commits every
// output.
class BodySpecAnnDraftStopping {
 public:
  explicit BodySpecAnnDraftStopping(const char* text) {
    if (text == nullptr || *text == '\0') return;
    char* end = nullptr;
    margin_ = std::strtof(text, &end);
    valid_ = end != text && *end == '\0' && std::isfinite(margin_) &&
             margin_ >= 0.0f;
  }
  bool Valid() const { return valid_; }
  bool Enabled() const { return valid_ && margin_ > 0.0f; }
  float Margin() const { return margin_; }
  // An accepted last proposal is followed by one exact bonus row. This row
  // needs a private body eagerly or a deferred catch-up after commitment.
  // The policy can only shorten a round by
  // at least one row; it cannot change its current proposal or grow capacity.
  bool Stop(float best, float second, bool has_second, float final_cap,
            size_t row, size_t count) {
    if (!Enabled() || count < 3 || row >= count - 2) return false;
    if (!has_second || !std::isfinite(best) || !std::isfinite(second) ||
        best < second || !std::isfinite(final_cap) || final_cap < 0.0f) {
      ++guarded;
      return false;
    }
    ++checked;
    if (final_cap > 0.0f) {
      best = final_cap * std::tanh(best / final_cap);
      second = final_cap * std::tanh(second / final_cap);
    }
    // A top-two margin is a heuristic, not a full-vocabulary probability.
    // A strict boundary gives deterministic behavior for exact ties to margin.
    if (best - second >= margin_) return false;
    ++stops;
    avoided_rows += count - row - 2;
    return true;
  }
  size_t checked = 0, guarded = 0, stops = 0, avoided_rows = 0;

 private:
  float margin_ = 0.0f;
  bool valid_ = true;
};
