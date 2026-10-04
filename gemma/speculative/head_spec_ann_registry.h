// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0
// Optional support for immutable HNSW head indexes. Included only by gemma.cc
// when GEMMA_HEAD_SPEC_HNSW is enabled; no public object layout changes.
#ifndef THIRD_PARTY_GEMMA_CPP_GEMMA_HEAD_SPEC_ANN_REGISTRY_H_
#define THIRD_PARTY_GEMMA_CPP_GEMMA_HEAD_SPEC_ANN_REGISTRY_H_

#include <openssl/evp.h>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "gemma/speculative/head_spec_ann_index.h"

namespace gcpp {

inline double HeadSpecAnnNow() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

class HeadSpecAnnSha256 {
 public:
  HeadSpecAnnSha256() : ctx_(EVP_MD_CTX_new()) {
    ok_ =
        ctx_ != nullptr && EVP_DigestInit_ex(ctx_, EVP_sha256(), nullptr) == 1;
  }
  ~HeadSpecAnnSha256() { EVP_MD_CTX_free(ctx_); }
  HeadSpecAnnSha256(const HeadSpecAnnSha256&) = delete;
  HeadSpecAnnSha256& operator=(const HeadSpecAnnSha256&) = delete;
  void Add(const void* bytes, size_t count) {
    if (ok_ && count != 0) ok_ = EVP_DigestUpdate(ctx_, bytes, count) == 1;
  }
  std::string Finish() {
    unsigned char bytes[EVP_MAX_MD_SIZE];
    unsigned count = 0;
    if (!ok_ || EVP_DigestFinal_ex(ctx_, bytes, &count) != 1 || count != 32)
      return {};
    std::string out;
    out.reserve(64);
    constexpr char hex[] = "0123456789abcdef";
    for (unsigned i = 0; i < count; ++i) {
      out += hex[bytes[i] >> 4];
      out += hex[bytes[i] & 15];
    }
    return out;
  }

 private:
  EVP_MD_CTX* ctx_;
  bool ok_ = false;
};

inline std::string HeadSpecAnnFileHash(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return {};
  HeadSpecAnnSha256 hash;
  char bytes[64 * 1024];
  while (in) {
    in.read(bytes, sizeof(bytes));
    const auto count = in.gcount();
    if (count > 0) hash.Add(bytes, static_cast<size_t>(count));
  }
  return in.eof() ? hash.Finish() : std::string();
}

struct HeadSpecAnnScopedSeconds {
  explicit HeadSpecAnnScopedSeconds(double& seconds)
      : seconds(seconds), start(HeadSpecAnnNow()) {}
  ~HeadSpecAnnScopedSeconds() { seconds += HeadSpecAnnNow() - start; }
  double& seconds;
  double start;
};

// Linux reports these values in KiB. Zero means the platform has no /proc
// source. Peak RSS is process-wide and includes the model, not just the index.
inline size_t HeadSpecAnnMemoryBytes(const char* field) {
  std::ifstream status("/proc/self/status");
  std::string line;
  while (std::getline(status, line)) {
    if (line.compare(0, std::char_traits<char>::length(field), field) != 0)
      continue;
    std::istringstream value(
        line.substr(std::char_traits<char>::length(field)));
    size_t kib = 0;
    if (value >> kib) return kib * 1024;
  }
  return 0;
}

struct HeadSpecAnnEntry {
  std::string binding;
  std::string fingerprint;
  std::string error;
  std::string mode;
  std::string index_path;
  std::shared_ptr<HeadSpecAnnIndex> index;
  double hash_seconds = 0;
  double load_seconds = 0;
  double build_seconds = 0;
  double save_seconds = 0;
  double prepare_seconds = 0;
  size_t index_bytes = 0;
  size_t rss_before_bytes = 0;
  size_t rss_after_bytes = 0;
  size_t process_peak_rss_bytes = 0;
};

struct HeadSpecAnnRegistryState {
  std::mutex mutex;
  std::unordered_map<const void*, std::shared_ptr<HeadSpecAnnEntry>> entries;
};
inline HeadSpecAnnRegistryState& HeadSpecAnnRegistry() {
  // A static Gemma may unregister during shutdown after ordinary statics have
  // been destroyed. Keep this small registry alive; unregister frees indexes.
  static auto* registry = new HeadSpecAnnRegistryState;
  return *registry;
}
inline std::shared_ptr<const HeadSpecAnnEntry> HeadSpecAnnFind(
    const void* head_key, const std::string& binding) {
  auto& registry = HeadSpecAnnRegistry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  const auto found = registry.entries.find(head_key);
  if (found == registry.entries.end() || found->second->binding != binding)
    return nullptr;
  return found->second;
}
// Holding this lock serializes index construction and prevents duplicate work.
// Queries hold their own shared_ptr after lookup; they do not retain this lock.
// Failed preparation is also remembered for this binding, avoiding a rebuild on
// every timed repetition. A changed binding or new model lifetime can retry.
template <class Prepare>
inline std::shared_ptr<const HeadSpecAnnEntry> HeadSpecAnnPrepareOnce(
    const void* head_key, const std::string& binding, const Prepare& prepare) {
  auto& registry = HeadSpecAnnRegistry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  const auto found = registry.entries.find(head_key);
  if (found != registry.entries.end() && found->second->binding == binding)
    return found->second;
  auto entry = std::make_shared<HeadSpecAnnEntry>();
  entry->binding = binding;
  entry->rss_before_bytes = HeadSpecAnnMemoryBytes("VmRSS:");
  const double start = HeadSpecAnnNow();
  try {
    prepare(*entry);
  } catch (const std::exception& e) {
    entry->error = e.what();
  }
  entry->prepare_seconds = HeadSpecAnnNow() - start;
  entry->rss_after_bytes = HeadSpecAnnMemoryBytes("VmRSS:");
  entry->process_peak_rss_bytes = HeadSpecAnnMemoryBytes("VmHWM:");
  registry.entries[head_key] = entry;
  return entry;
}
inline void UnregisterHeadSpecAnn(const void* head_key) {
  auto& registry = HeadSpecAnnRegistry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  registry.entries.erase(head_key);
}

// A disk index is used only after its model/config and file SHA256 match.
// These functions run exclusively in the pre-generation preparation hook.
inline bool HeadSpecAnnLoad(HeadSpecAnnEntry& entry, size_t feature_dim,
                            size_t vocabulary,
                            speculative::HnswHeadParameters parameters) {
  HeadSpecAnnScopedSeconds timer(entry.load_seconds);
  try {
    std::ifstream file(entry.index_path + ".json");
    if (!file) return false;
    nlohmann::json metadata;
    file >> metadata;
    if (metadata.value("format", "") != HeadSpecAnnIndex::ManifestFormat() ||
        metadata.value("fingerprint", "") != entry.fingerprint ||
        metadata.value("hnsw_commit", "") !=
            "d9b3608c83d83b46c96e25088cb1d729b29dcfe9" ||
        metadata.value("feature_dim", size_t{0}) != feature_dim ||
        metadata.value("vocabulary", size_t{0}) != vocabulary ||
        metadata.value("size_t_bytes", size_t{0}) != sizeof(size_t) ||
        metadata.value("index_bytes", uintmax_t{0}) !=
            std::filesystem::file_size(entry.index_path) ||
        metadata.value("index_sha256", "").size() != 64 ||
        metadata.value("index_sha256", "") !=
            HeadSpecAnnFileHash(entry.index_path))
      return false;
    std::string error;
    entry.index = HeadSpecAnnIndex::Load(feature_dim, vocabulary, parameters,
                                         entry.index_path, &error);
    if (!entry.index) return false;
    entry.index_bytes = std::filesystem::file_size(entry.index_path);
    entry.mode = "loaded";
    return true;
  } catch (const std::exception&) {
    return false;
  }
}
inline bool HeadSpecAnnSave(HeadSpecAnnEntry& entry, size_t feature_dim,
                            size_t vocabulary,
                            speculative::HnswHeadParameters parameters) {
  HeadSpecAnnScopedSeconds timer(entry.save_seconds);
  const double start = HeadSpecAnnNow();
  // The registry address distinguishes concurrent processes with ASLR, and the
  // monotonic timestamp distinguishes repeated attempts. Publish metadata last.
  const std::string temporary =
      entry.index_path + ".tmp-" +
      std::to_string(reinterpret_cast<uintptr_t>(&entry)) + "-" +
      std::to_string(static_cast<uint64_t>(start * 1E6));
  struct Cleanup {
    const std::string& path;
    ~Cleanup() {
      std::error_code ignored;
      std::filesystem::remove(path, ignored);
      std::filesystem::remove(path + ".json", ignored);
    }
  } cleanup{temporary};
  try {
    const auto directory =
        std::filesystem::path(entry.index_path).parent_path();
    if (!directory.empty()) std::filesystem::create_directories(directory);
    std::string error;
    if (!entry.index->SaveIndex(temporary, &error)) return false;
    const std::string file_hash = HeadSpecAnnFileHash(temporary);
    if (file_hash.empty()) return false;
    const uintmax_t bytes = std::filesystem::file_size(temporary);
    nlohmann::json metadata = {
        {"format", HeadSpecAnnIndex::ManifestFormat()},
        {"storage", entry.index->Storage()},
        {"vector_bytes", entry.index->VectorBytes()},
        {"kernel", entry.index->Kernel()},
        {"fingerprint", entry.fingerprint},
        {"hnsw_commit", "d9b3608c83d83b46c96e25088cb1d729b29dcfe9"},
        {"feature_dim", feature_dim},
        {"vocabulary", vocabulary},
        {"size_t_bytes", sizeof(size_t)},
        {"index_sha256", file_hash},
        {"index_bytes", bytes},
        {"M", parameters.links},
        {"ef_construction", parameters.construction_ef},
        {"ef_search", parameters.search_ef},
        {"candidates", parameters.candidates},
        {"seed", parameters.seed},
        {"build_seconds", entry.build_seconds}};
    std::ofstream out(temporary + ".json", std::ios::trunc);
    out << metadata.dump(2) << '\n';
    out.close();
    if (!out) return false;
    std::filesystem::rename(temporary, entry.index_path);
    std::filesystem::rename(temporary + ".json", entry.index_path + ".json");
    entry.index_bytes = bytes;
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

}  // namespace gcpp
#endif
