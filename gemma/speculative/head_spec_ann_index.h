// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0
#ifndef THIRD_PARTY_GEMMA_CPP_HEAD_SPEC_ANN_INDEX_H_
#define THIRD_PARTY_GEMMA_CPP_HEAD_SPEC_ANN_INDEX_H_
#include "gemma/speculative/hnsw_head_i8_proposal.h"
namespace gcpp {
class HeadSpecAnnIndex {
 public:
  using Parameters = speculative::HnswHeadParameters;
  using ReadWeightRow = speculative::HnswHeadI8Proposal::ReadWeightRow;
  using Progress = speculative::HnswHeadI8Proposal::Progress;
  static const char* ManifestFormat() { return "gemma-head-ann-i8-v1"; }
  static const char* FeatureFormat() {
    return speculative::HnswHeadI8Proposal::kFeatureFormat;
  }
  static std::shared_ptr<HeadSpecAnnIndex> Build(
      size_t dim, size_t vocabulary, Parameters parameters, const void* weights,
      ReadWeightRow reader, std::string* error, Progress progress = nullptr,
      void* opaque = nullptr) {
    auto result = std::make_shared<HeadSpecAnnIndex>();
    result->index_ = speculative::HnswHeadI8Proposal::Build(
        dim, vocabulary, parameters, weights, reader, error, progress, opaque);
    return result->index_ ? result : nullptr;
  }
  static std::shared_ptr<HeadSpecAnnIndex> Load(size_t dim, size_t vocabulary,
                                                Parameters parameters,
                                                const std::string& path,
                                                std::string* error) {
    auto result = std::make_shared<HeadSpecAnnIndex>();
    result->index_ = speculative::HnswHeadI8Proposal::Load(
        dim, vocabulary, parameters, path, error);
    return result->index_ ? result : nullptr;
  }
  size_t FeatureDim() const { return index_->FeatureDim(); }
  size_t VectorBytes() const { return index_->VectorBytes(); }
  const char* Storage() const { return "i8"; }
  const char* Kernel() const {
    return index_->UsesVnni() ? "avx-vnni" : "scalar";
  }
  bool Candidates(const float* query, size_t dim, std::vector<int>& tokens,
                  std::string* error) const {
    return index_->Candidates(query, dim, tokens, error);
  }
  bool SaveIndex(const std::string& path, std::string* error) const {
    return index_->SaveIndex(path, error);
  }

 private:
  std::unique_ptr<speculative::HnswHeadI8Proposal> index_;
};
}  // namespace gcpp
#endif
