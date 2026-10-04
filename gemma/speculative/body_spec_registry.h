// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0
#ifndef THIRD_PARTY_GEMMA_CPP_GEMMA_BODY_SPEC_REGISTRY_H_
#define THIRD_PARTY_GEMMA_CPP_GEMMA_BODY_SPEC_REGISTRY_H_

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_map>
#include <utility>

#include "gemma/gemma.h"
#include "gemma/model_store.h"
#include "io/blob_store.h"

namespace gcpp {

// Owns only immutable proposal models. Query tokens, responses, KV caches,
// activations and matmul/autotuner state belong to the future verifier.
struct BodySpecDraftRegistry {
  std::mutex mutex;
  std::unordered_map<const void*, std::shared_ptr<const Gemma>> models;
};

inline BodySpecDraftRegistry& BodySpecDrafts() {
  // Gemma destruction calls back into this registry. Keeping the small registry
  // alive avoids static-destruction-order recursion; target destruction still
  // releases every registered draft via UnregisterBodySpecDraft.
  static BodySpecDraftRegistry* const registry = new BodySpecDraftRegistry;
  return *registry;
}

inline bool& BodySpecDraftLoading() {
  static thread_local bool loading = false;
  return loading;
}

class BodySpecDraftLoadScope {
 public:
  BodySpecDraftLoadScope() : previous_(BodySpecDraftLoading()) {
    BodySpecDraftLoading() = true;
  }
  ~BodySpecDraftLoadScope() { BodySpecDraftLoading() = previous_; }
  BodySpecDraftLoadScope(const BodySpecDraftLoadScope&) = delete;
  BodySpecDraftLoadScope& operator=(const BodySpecDraftLoadScope&) = delete;

 private:
  bool previous_;
};

inline std::shared_ptr<const Gemma> FindBodySpecDraft(const void* target_head) {
  auto& registry = BodySpecDrafts();
  std::lock_guard<std::mutex> lock(registry.mutex);
  const auto it = registry.models.find(target_head);
  return it == registry.models.end() ? nullptr : it->second;
}

inline void UnregisterBodySpecDraft(const void* target_head) {
  std::shared_ptr<const Gemma> retired;
  {
    auto& registry = BodySpecDrafts();
    std::lock_guard<std::mutex> lock(registry.mutex);
    const auto it = registry.models.find(target_head);
    if (it == registry.models.end()) return;
    retired = std::move(it->second);
    registry.models.erase(it);
  }
  // Destroy outside the mutex: the draft's Gemma destructor also unregisters.
  // Existing lookup holders keep it alive until their own references expire.
  retired.reset();
}

inline void ValidateBodySpecDraft(const ModelConfig& draft_config,
                                  const GemmaTokenizer& draft_tokenizer,
                                  const Gemma& target) {
  if (draft_config.model != Model::GEMMA3_270M) {
    throw std::runtime_error("BODY_SPEC requires a Gemma3 270M draft model");
  }
  if (draft_config.vocab_size != target.Config().vocab_size ||
      draft_tokenizer.Serialize() != target.Tokenizer().Serialize()) {
    throw std::runtime_error(
        "BODY_SPEC draft and target tokenizer bytes must be identical");
  }
}

// Constructor-only hook: all file I/O and model loading finish before timed
// inference. No new context/pool is created. The future verifier must use its
// own MatMulEnv and per-query KV cache, referencing this same CPU context.
inline void RegisterBodySpecDraft(const void* target_head, const Gemma& target,
                                  const GemmaArgs& target_args,
                                  ThreadingContext& ctx) {
  const char* enabled = std::getenv("GEMMA_MM_I8_BODY_SPEC");
  if (enabled == nullptr || std::atoi(enabled) == 0 || BodySpecDraftLoading() ||
      target.Config().model != Model::GEMMA3_1B) {
    return;
  }
  if (target_head == nullptr || ctx.pools.MaxWorkers() != 6) {
    throw std::runtime_error(
        "BODY_SPEC requires a valid 1B target head and six CPU workers");
  }
  const char* raw_path = std::getenv("GEMMA_MM_I8_DRAFT_WEIGHTS");
  if (raw_path == nullptr || raw_path[0] == '\0') {
    throw std::runtime_error("BODY_SPEC requires GEMMA_MM_I8_DRAFT_WEIGHTS");
  }
  const std::string path(raw_path);
  std::error_code error;
  if (path.find("://") != std::string::npos ||
      !std::filesystem::is_regular_file(std::filesystem::path(path), error) ||
      error) {
    throw std::runtime_error(
        "BODY_SPEC draft weights must be a local regular file");
  }

  // Initialize the registry before recursively constructing a Gemma, without
  // holding its mutex during model construction or validation.
  auto& registry = BodySpecDrafts();
  if (FindBodySpecDraft(target_head)) {
    throw std::runtime_error("BODY_SPEC target head is already registered");
  }
  BodySpecDraftLoadScope loading;
  GemmaArgs draft_args = target_args;
  draft_args.loader.weights = Path(path);
  // Validate the tokenizer embedded in the draft file itself. A target-side
  // tokenizer override must not hide incompatible draft token IDs.
  draft_args.loader.tokenizer = Path();
  draft_args.loader.wrapping = Tristate::kDefault;
  {
    BlobReader reader(draft_args.loader.weights);
    ModelStore metadata(reader);
    ValidateBodySpecDraft(metadata.Config(), metadata.Tokenizer(), target);
    reader.CloseFile();
  }
  std::shared_ptr<const Gemma> draft = std::make_shared<Gemma>(draft_args, ctx);
  // Recheck the loaded object in case the local file changed after preflight.
  ValidateBodySpecDraft(draft->Config(), draft->Tokenizer(), target);
  bool inserted;
  {
    std::lock_guard<std::mutex> lock(registry.mutex);
    inserted = registry.models.emplace(target_head, draft).second;
  }
  if (!inserted) {
    throw std::runtime_error(
        "BODY_SPEC target head was concurrently registered");
  }
}

}  // namespace gcpp
#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_BODY_SPEC_REGISTRY_H_
