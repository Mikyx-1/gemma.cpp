// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// INT8 proposal index. Uses a distinct
// feature/cache format from the F32 proposal index. Exact reranking and target
// full-vocabulary verification remain mandatory outside this wrapper.
// Requires the pinned hnswlib dependency enabled by CMake.
// All index rows come from immutable model weights through the supplied reader.
// Returned IDs must be reranked with the original head kernel and verified by
// the original complete-vocabulary target computation before being committed.
#ifndef THIRD_PARTY_GEMMA_CPP_SPECULATIVE_HNSW_HEAD_I8_PROPOSAL_H_
#define THIRD_PARTY_GEMMA_CPP_SPECULATIVE_HNSW_HEAD_I8_PROPOSAL_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "gemma/speculative/hnsw_i8_space.h"

namespace gcpp {
namespace speculative {

struct HnswHeadParameters {
  size_t links = 16;
  size_t construction_ef = 100;
  size_t search_ef = 128;
  size_t candidates = 32;
  size_t seed = 100;
};
class HnswHeadI8Proposal {
 public:
  static constexpr const char* kFeatureFormat =
      HnswI8InnerProductSpace::kFeatureFormat;

  // Writes exactly feature_dim floats. An optional final bias coordinate is
  // supplied by the adapter. No request tokens, logits or responses are inputs.
  using ReadWeightRow = bool (*)(const void* weights, size_t token,
                                 float* destination, size_t feature_dim);
  using Progress = bool (*)(size_t completed, size_t total, void* opaque);

  // Build explicitly during model preparation, with construction serialized.
  // Token-ID insertion order and seed are fixed. The reader streams one row,
  // so no second vocabulary-sized float matrix is allocated by this wrapper.
  static std::unique_ptr<HnswHeadI8Proposal> Build(
      size_t feature_dim, size_t vocabulary, HnswHeadParameters parameters,
      const void* weights, ReadWeightRow read, std::string* error,
      Progress progress = nullptr, void* progress_opaque = nullptr) {
    if (error != nullptr) error->clear();
    if (read == nullptr ||
        !ValidParameters(feature_dim, vocabulary, parameters)) {
      Fail(error, "Invalid HNSW proposal parameters");
      return nullptr;
    }
    try {
      std::unique_ptr<HnswHeadI8Proposal> result(
          new HnswHeadI8Proposal(feature_dim, vocabulary, parameters));
      std::vector<float> row(feature_dim, 0.0f);
      std::vector<uint8_t> encoded;
      for (size_t token = 0; token < vocabulary; ++token) {
        if (!read(weights, token, row.data(), feature_dim)) {
          Fail(error, "Weight reader rejected an index row");
          return nullptr;
        }
        if (!result->space_->Encode(row.data(), feature_dim, encoded)) {
          Fail(error, "I8 proposal weight encoding failed");
          return nullptr;
        }
        result->index_->addPoint(encoded.data(), token);
        if (progress != nullptr &&
            (token + 1 == vocabulary || (token + 1) % 4096 == 0) &&
            !progress(token + 1, vocabulary, progress_opaque)) {
          Fail(error, "HNSW proposal construction cancelled");
          return nullptr;
        }
      }
      // Never mutate ef while queries are running. It is fixed at publication.
      result->index_->setEf(parameters.search_ef);
      return result;
    } catch (const std::exception& e) {
      Fail(error, e.what());
      return nullptr;
    }
  }

  // The registry must authenticate the model/config fingerprint before Load.
  // Read and validate the small serialized header before hnswlib allocates the
  // graph. Construct directly from the file; do not allocate an empty full
  // index first. The optional disk cache is tied to the pinned hnswlib ABI.
  static std::unique_ptr<HnswHeadI8Proposal> Load(size_t feature_dim,
                                                  size_t vocabulary,
                                                  HnswHeadParameters parameters,
                                                  const std::string& path,
                                                  std::string* error) {
    if (error != nullptr) error->clear();
    if (!ValidParameters(feature_dim, vocabulary, parameters)) {
      Fail(error, "Invalid HNSW proposal parameters");
      return nullptr;
    }
    try {
      SerializedHeader header;
      if (!ReadHeader(path, header, error) ||
          !HeaderMatches(header, feature_dim, vocabulary, parameters)) {
        if (error != nullptr && error->empty())
          Fail(error, "HNSW index header does not match proposal parameters");
        return nullptr;
      }
      std::unique_ptr<HnswHeadI8Proposal> result(new HnswHeadI8Proposal(
          feature_dim, vocabulary, parameters, path, LoadTag{}));
      auto& index = *result->index_;
      if (index.getCurrentElementCount() != vocabulary ||
          index.getMaxElements() != vocabulary ||
          index.getDeletedCount() != 0 || index.M_ != parameters.links ||
          index.maxM_ != parameters.links ||
          index.maxM0_ != 2 * parameters.links ||
          index.ef_construction_ != parameters.construction_ef ||
          index.offsetLevel0_ != header.level0_offset ||
          index.offsetData_ != header.data_offset ||
          index.label_offset_ != header.label_offset ||
          index.size_data_per_element_ != header.stride ||
          index.data_size_ != result->space_->get_data_size()) {
        Fail(error,
             "Loaded HNSW index layout does not match proposal parameters");
        return nullptr;
      }
      result->index_->setEf(parameters.search_ef);
      return result;
    } catch (const std::exception& e) {
      Fail(error, e.what());
      return nullptr;
    }
  }

  // Write the pinned hnswlib format with checked streams. hnswlib::saveIndex
  // does not propagate stream errors, so emit its same fields here and verify
  // the closed file size. Atomic rename/manifest publication belongs to the
  // caller; on failure this method may leave a partial temporary file.
  bool SaveIndex(const std::string& path, std::string* error) const {
    if (error != nullptr) error->clear();
    try {
      auto& index = *index_;
      const size_t count = index.getCurrentElementCount();
      static_assert(sizeof(index.cur_element_count) == sizeof(count),
                    "Pinned HNSW atomic count serialization changed");
      std::ofstream output;
      output.exceptions(std::ios::failbit | std::ios::badbit);
      output.open(path, std::ios::binary | std::ios::trunc);
      WriteField(output, index.offsetLevel0_);
      WriteField(output, index.max_elements_);
      WriteField(output, count);
      WriteField(output, index.size_data_per_element_);
      WriteField(output, index.label_offset_);
      WriteField(output, index.offsetData_);
      WriteField(output, index.maxlevel_);
      WriteField(output, index.enterpoint_node_);
      WriteField(output, index.maxM_);
      WriteField(output, index.maxM0_);
      WriteField(output, index.M_);
      WriteField(output, index.mult_);
      WriteField(output, index.ef_construction_);
      output.write(
          index.data_level0_memory_,
          static_cast<std::streamsize>(count * index.size_data_per_element_));
      for (size_t i = 0; i < count; ++i) {
        const unsigned int bytes =
            index.element_levels_[i] > 0
                ? static_cast<unsigned int>(index.size_links_per_element_ *
                                            index.element_levels_[i])
                : 0;
        WriteField(output, bytes);
        if (bytes != 0)
          output.write(index.linkLists_[i],
                       static_cast<std::streamsize>(bytes));
      }
      output.close();
      std::ifstream saved(path, std::ios::binary | std::ios::ate);
      const auto size = saved.tellg();
      if (!saved || size < std::streampos{0} ||
          static_cast<uint64_t>(size) != index.indexFileSize()) {
        Fail(error, "Saved HNSW index has an unexpected file size");
        return false;
      }
      return true;
    } catch (const std::exception& e) {
      Fail(error, e.what());
      return false;
    }
  }

  // The caller reconstructs the native rotated primary+residual float query.
  // No normalization: cosine distance would change maximum-inner-product rank.
  // This does not alter the graph, learn from requests or retain the query.
  bool Candidates(const float* query, size_t feature_dim,
                  std::vector<int>& tokens, std::string* error) const {
    tokens.clear();
    if (error != nullptr) error->clear();
    if (query == nullptr || feature_dim != feature_dim_) {
      Fail(error, "Invalid HNSW proposal query shape");
      return false;
    }
    for (size_t c = 0; c < feature_dim_; ++c) {
      if (!std::isfinite(query[c])) {
        Fail(error, "Non-finite HNSW proposal query");
        return false;
      }
    }
    try {
      std::vector<uint8_t> encoded;
      if (!space_->Encode(query, feature_dim_, encoded)) {
        Fail(error, "I8 proposal query encoding failed");
        return false;
      }
      auto found = index_->searchKnn(encoded.data(), parameters_.candidates);
      if (found.size() != parameters_.candidates) {
        Fail(error, "HNSW proposal returned too few candidates");
        return false;
      }
      tokens.reserve(found.size());
      while (!found.empty()) {
        const size_t token = found.top().second;
        if (token >= vocabulary_) {
          Fail(error, "HNSW proposal label outside vocabulary");
          tokens.clear();
          return false;
        }
        tokens.push_back(static_cast<int>(token));
        found.pop();
      }
      // Ascending IDs make packed-N8 expansion and exact reranking
      // deterministic.
      std::sort(tokens.begin(), tokens.end());
      tokens.erase(std::unique(tokens.begin(), tokens.end()), tokens.end());
      return !tokens.empty();
    } catch (const std::exception& e) {
      Fail(error, e.what());
      return false;
    }
  }

  size_t FeatureDim() const { return feature_dim_; }
  size_t PaddedDim() const { return padded_dim_; }
  size_t Vocabulary() const { return vocabulary_; }
  size_t VectorBytes() const { return space_->get_data_size(); }
  bool UsesVnni() const { return space_->UsesVnni(); }

 private:
  struct LoadTag {};
  struct SerializedHeader {
    size_t level0_offset = 0, capacity = 0, count = 0, stride = 0;
    size_t label_offset = 0, data_offset = 0;
    int max_level = 0;
    hnswlib::tableint entry = 0;
    size_t max_m = 0, max_m0 = 0, m = 0;
    double multiplier = 0;
    size_t construction_ef = 0;
    uint64_t file_bytes = 0;
  };

  static bool ValidParameters(size_t feature_dim, size_t vocabulary,
                              const HnswHeadParameters& parameters) {
    if (feature_dim == 0 || vocabulary == 0 || parameters.links < 2 ||
        parameters.links > 64 ||
        parameters.construction_ef < parameters.links ||
        parameters.candidates == 0 || parameters.candidates > vocabulary ||
        parameters.search_ef < parameters.candidates ||
        feature_dim > HnswI8InnerProductSpace::kMaxDimensions ||
        vocabulary > static_cast<size_t>(std::numeric_limits<int>::max()))
      return false;
    const size_t vector_bytes =
        HnswI8InnerProductSpace::EncodedBytes(feature_dim);
    const size_t overhead = 2 * parameters.links * sizeof(hnswlib::tableint) +
                            sizeof(hnswlib::linklistsizeint) +
                            sizeof(hnswlib::labeltype);
    if (vector_bytes > SIZE_MAX - overhead) return false;
    const size_t stride = vector_bytes + overhead;
    return vocabulary <=
           static_cast<size_t>(std::numeric_limits<std::streamsize>::max()) /
               stride;
  }

  template <class T>
  static bool ReadField(std::istream& input, T& field) {
    return static_cast<bool>(
        input.read(reinterpret_cast<char*>(&field), sizeof(field)));
  }
  template <class T>
  static void WriteField(std::ostream& output, const T& field) {
    output.write(reinterpret_cast<const char*>(&field), sizeof(field));
  }

  static bool ReadHeader(const std::string& path, SerializedHeader& header,
                         std::string* error) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    const auto bytes = input.tellg();
    if (!input || bytes < std::streampos{0}) {
      Fail(error, "Cannot open HNSW proposal index");
      return false;
    }
    header.file_bytes = static_cast<uint64_t>(bytes);
    input.seekg(0, std::ios::beg);
    const bool ok =
        ReadField(input, header.level0_offset) &&
        ReadField(input, header.capacity) && ReadField(input, header.count) &&
        ReadField(input, header.stride) &&
        ReadField(input, header.label_offset) &&
        ReadField(input, header.data_offset) &&
        ReadField(input, header.max_level) && ReadField(input, header.entry) &&
        ReadField(input, header.max_m) && ReadField(input, header.max_m0) &&
        ReadField(input, header.m) && ReadField(input, header.multiplier) &&
        ReadField(input, header.construction_ef);
    if (!ok) Fail(error, "Truncated HNSW proposal index header");
    return ok;
  }

  static bool HeaderMatches(const SerializedHeader& header, size_t feature_dim,
                            size_t vocabulary,
                            const HnswHeadParameters& parameters) {
    const size_t vector_bytes =
        HnswI8InnerProductSpace::EncodedBytes(feature_dim);
    const size_t links = 2 * parameters.links * sizeof(hnswlib::tableint) +
                         sizeof(hnswlib::linklistsizeint);
    const size_t label = links + vector_bytes;
    const size_t stride = label + sizeof(hnswlib::labeltype);
    constexpr size_t header_bytes = 10 * sizeof(size_t) + sizeof(int) +
                                    sizeof(hnswlib::tableint) + sizeof(double);
    const uint64_t minimum_bytes =
        header_bytes + uint64_t{vocabulary} * (stride + sizeof(unsigned int));
    return header.level0_offset == 0 && header.capacity == vocabulary &&
           header.count == vocabulary && header.data_offset == links &&
           header.label_offset == label && header.stride == stride &&
           header.max_m == parameters.links &&
           header.max_m0 == 2 * parameters.links &&
           header.m == parameters.links &&
           header.construction_ef == parameters.construction_ef &&
           header.max_level >= 0 && header.max_level <= 64 &&
           header.entry < vocabulary && std::isfinite(header.multiplier) &&
           header.multiplier > 0 &&
           std::abs(header.multiplier -
                    1.0 / std::log(double(parameters.links))) < 1e-12 &&
           header.file_bytes >= minimum_bytes;
  }

  HnswHeadI8Proposal(size_t feature_dim, size_t vocabulary,
                     HnswHeadParameters parameters, const std::string& path,
                     LoadTag)
      : feature_dim_(feature_dim),
        padded_dim_((feature_dim + 31) & ~size_t{31}),
        vocabulary_(vocabulary),
        parameters_(parameters),
        space_(new HnswI8InnerProductSpace(feature_dim_)),
        index_(new hnswlib::HierarchicalNSW<float>(space_.get(), path, false,
                                                   vocabulary_)) {}

  HnswHeadI8Proposal(size_t feature_dim, size_t vocabulary,
                     HnswHeadParameters parameters)
      : feature_dim_(feature_dim),
        padded_dim_((feature_dim + 31) & ~size_t{31}),
        vocabulary_(vocabulary),
        parameters_(parameters),
        space_(new HnswI8InnerProductSpace(feature_dim_)),
        index_(new hnswlib::HierarchicalNSW<float>(
            space_.get(), vocabulary_, parameters.links,
            parameters.construction_ef, parameters.seed)) {}

  static void Fail(std::string* error, const char* message) {
    if (error != nullptr) *error = message;
  }

  size_t feature_dim_;
  size_t padded_dim_;
  size_t vocabulary_;
  HnswHeadParameters parameters_;
  // Destruction order matters: the graph stores pointers to space metadata.
  std::unique_ptr<HnswI8InnerProductSpace> space_;
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> index_;
};

}  // namespace speculative
}  // namespace gcpp

#endif
