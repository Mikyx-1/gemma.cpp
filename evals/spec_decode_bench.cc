// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0
// Native/speculative decoding throughput and complete-output validation.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "evals/benchmark_helper.h"
#include "nlohmann/json.hpp"
#include "ops/matmul.h"

namespace gcpp {
namespace {
using json = nlohmann::json;
constexpr const char* kPrompt =
    "Write a long, detailed story about a young engineer who repairs an "
    "old lighthouse on a remote island. Describe the journey, the island, "
    "the broken machinery, and how the engineer works with local people "
    "to solve the problem. Include dialogue, technical details, and "
    "several scenes. The story should be at least one thousand words. "
    "Begin the story now.";
void Require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
void SetEnv(const char* name, const std::string& value) {
#ifdef _WIN32
  Require(_putenv_s(name, value.c_str()) == 0,
          "Cannot set benchmark environment");
#else
  Require(setenv(name, value.c_str(), 1) == 0,
          "Cannot set benchmark environment");
#endif
}
struct BenchArgs : ArgsBase<BenchArgs> {
  BenchArgs(int argc, char** argv, ConsumedArgs& consumed) {
    InitAndParse(argc, argv, consumed);
  }
  std::string mode, model, prompt;
  size_t warmup, max_warmup, reps, generated, stop_after;
  Path logits, schedule_in, schedule_out;
  template <class Visitor>
  void ForEach(const Visitor& visit) {
    visit(mode, "bench_mode", std::string("native"),
          "native or spec; run each in a separate process.");
    visit(model, "bench_model", std::string("270m"),
          "270m or 1b; selects the matched W8A8 policy.");
    visit(prompt, "bench_prompt", std::string(kPrompt),
          "User prompt to wrap and tokenize.");
    visit(warmup, "bench_warmup", size_t{2},
          "Untimed warmups, including index preparation.");
    visit(max_warmup, "bench_max_warmup", size_t{128},
          "Maximum short native warmups to settle fresh autotuning.");
    visit(reps, "bench_reps", size_t{3}, "Timed repetitions.");
    visit(generated, "bench_generated", size_t{128},
          "Maximum generated tokens; EOS may stop earlier.");
    visit(stop_after, "bench_stop_after", size_t{0},
          "Callback-stop validation; zero disables.");
    visit(schedule_in, "bench_schedule_in", Path(),
          "Same-machine frozen native schedules.");
    visit(schedule_out, "bench_schedule_out", Path(),
          "Save settled native schedules for comparison.");
    visit(logits, "bench_logits", Path(),
          "Complete F32 logits from an untimed replay.");
  }
};
std::string HexBytes(const void* data, size_t size) {
  const auto* bytes = static_cast<const unsigned char*>(data);
  constexpr char digits[] = "0123456789abcdef";
  std::string out;
  out.reserve(size * 2);
  for (size_t i = 0; i < size; ++i) {
    out.push_back(digits[bytes[i] >> 4]);
    out.push_back(digits[bytes[i] & 15]);
  }
  return out;
}
json ScheduleState(const MatMulEnv& env) {
  json result = json::array();
  for (const auto& cluster : env.per_cluster) {
    const auto keys = cluster.keys.Keys();
    Require(keys.size() == cluster.per_key.size(), "Matmul key count differs");
    json rows = json::array();
    for (size_t i = 0; i < keys.size(); ++i) {
      const auto& key = cluster.per_key[i];
      const auto* best = key.autotune.Best();
      Require(best != nullptr, "Matmul schedule is unsettled");
      json kc = json::array();
      const size_t k = (keys[i] >> 16) & ((uint64_t{1} << 20) - 1);
      const auto ranges = best->RangesOfKC(k);
      for (size_t j = 0; j < ranges.NumTasks(); ++j)
        kc.push_back(
            {size_t(ranges.Range(j).begin()), size_t(ranges.Range(j).end())});
      const auto* par = key.autotune_par_a.Best();
      Require(par != nullptr || !key.autotune_par_a.HasCandidates(),
              "Decompression schedule is unsettled");
      rows.push_back({{"key", keys[i]},
                      {"config", HexBytes(best, sizeof(*best))},
                      {"kc", kc},
                      {"par_a", par ? json(unsigned(*par)) : json(nullptr)}});
    }
    result.push_back(rows);
  }
  return result;
}
bool SchedulesSettled(const MatMulEnv& env) {
  for (const auto& cluster : env.per_cluster) {
    for (const auto& key : cluster.per_key) {
      if (key.autotune.Best() == nullptr ||
          (key.autotune_par_a.Best() == nullptr &&
           key.autotune_par_a.HasCandidates()))
        return false;
    }
  }
  return true;
}
struct RunResult {
  TimingInfo timing;
  std::vector<int> tokens;
  std::vector<uint32_t> probabilities;
};
RunResult Generate(GemmaEnv& env, const std::vector<int>& prompt,
                   size_t stop_after) {
  RunResult result;
  size_t streamed = 0;
  auto& runtime = env.MutableConfig();
  runtime.batch_stream_token = [&](size_t query, size_t pos, int token,
                                   float probability) {
    Require(query == 0 && pos == streamed++, "Unexpected streamed position");
    Require(token >= 0 && unsigned(token) < env.GetGemma()->Config().vocab_size,
            "Invalid output token");
    std::string text;
    Require(env.GetGemma()->Tokenizer().Decode(std::vector<int>{token}, &text),
            "Cannot decode streamed token");
    if (pos >= prompt.size()) {
      result.tokens.push_back(token);
      uint32_t bits;
      memcpy(&bits, &probability, sizeof(bits));
      result.probabilities.push_back(bits);
    }
    return stop_after == 0 || result.tokens.size() < stop_after;
  };
  const PromptTokens view(prompt.data(), prompt.size());
  const QueriesPromptTokens prompts(&view, 1);
  auto& cache = env.MutableKVCache();
  AllQueries queries(prompts, hwy::Span<KVCache>(&cache, 1));
  env.GetGemma()->GenerateBatch(runtime, queries, env.MutableEnv(),
                                result.timing);
  runtime.batch_stream_token = {};
  Require(result.tokens.size() == result.timing.tokens_generated,
          "Generated-token counter differs");
  return result;
}
int Run(int argc, char** argv) {
  InternalInit();
  ConsumedArgs consumed(argc, argv);
  GemmaArgs args(argc, argv, consumed);
  BenchArgs bench(argc, argv, consumed);
  if (HasHelp(argc, argv)) {
    args.Help();
    bench.Help();
    consumed.ClearUnconsumed();
    return 0;
  }
  consumed.AbortIfUnconsumed();
  Require(bench.mode == "native" || bench.mode == "spec",
          "Invalid benchmark mode");
  Require(bench.model == "270m" || bench.model == "1b",
          "Invalid benchmark model");
  Require(bench.warmup >= 2 && bench.max_warmup >= 2 && bench.reps > 0 &&
              bench.generated > 0,
          "Need two warmups and positive repetitions/tokens");
  Require(args.threading.max_threads == 6, "Use six workers");
  Require(getenv("GEMMA_MM_I8_IMPORT_DIR") == nullptr,
          "GPTQ is excluded from this benchmark");
  const bool small = bench.model == "270m", spec = bench.mode == "spec";
  for (const auto& item : std::vector<std::pair<const char*, const char*>>{
           {"GEMMA_MM_I8", "1"},
           {"GEMMA_MM_I8_MICROSCALE", "1"},
           {"GEMMA_MM_I8_L2_SCALE", "0"},
           {"GEMMA_MM_I8_BLOCK_SIZE", "128"},
           {"GEMMA_MM_I8_HASH_BITS", "32"},
           {"GEMMA_MM_I8_PACKED_HEAD", "1"},
           {"GEMMA_MM_I8_PACKED_HEAD_FULL_K", "1"},
           {"GEMMA_MM_I8_DUAL_A_BODY", "1"},
           {"GEMMA_MM_I8_DUAL_A_HEAD", "1"},
           {"GEMMA_MM_I8_MATCH_BF16_A", "1"}})
    SetEnv(item.first, item.second);
  SetEnv("GEMMA_MM_I8_QUANT_BLOCK_SIZE", small ? "64" : "128");
  SetEnv("GEMMA_MM_I8_MIN_K_SPLITS", small ? "1" : "0");
  SetEnv("GEMMA_MM_I8_HEAD_SPEC", spec && small ? "1" : "0");
  SetEnv("GEMMA_MM_I8_HEAD_SPEC_ANN", spec && small ? "1" : "0");
  SetEnv("GEMMA_MM_I8_BODY_SPEC", spec && !small ? "1" : "0");
  SetEnv("GEMMA_MM_I8_BODY_SPEC_ANN", spec && !small ? "1" : "0");
  SetEnv("GEMMA_MM_I8_HEAD_SPEC_HORIZON", "8");
  SetEnv("GEMMA_MM_I8_BODY_SPEC_HORIZON", "4");
  SetEnv("GEMMA_MM_I8_HEAD_SPEC_ANN_EF", "512");
  SetEnv("GEMMA_MM_I8_HEAD_SPEC_ANN_CANDIDATES", "32");
  SetEnv("GEMMA_MM_I8_HEAD_SPEC_ANN_SEED", small ? "100" : "17");
  SetEnv("GEMMA_MM_I8_BODY_SPEC_DEFER_LAST_BODY", "1");
  SetEnv("GEMMA_MM_I8_BODY_SPEC_STOPPED_LEAF", "1");
  SetEnv("GEMMA_MM_I8_BODY_SPEC_PAIR_CATCHUP", "1");
  SetEnv("GEMMA_MM_I8_BODY_SPEC_ANN_STOP_MARGIN", "1.0");
  SetEnv("GEMMA_MM_I8_EXACT_BATCH_ATTENTION", "1");
  SetEnv("GEMMA_MM_I8_HEAD_SPEC_STATS", "1");
  SetEnv("GEMMA_MM_I8_BODY_SPEC_STATS", "1");
  GemmaEnv env(args);
  env.MutableEnv().autotune = bench.schedule_in.Empty();
  Require(env.GetGemma()->Config().model ==
              (small ? Model::GEMMA3_270M : Model::GEMMA3_1B),
          "Weights differ from --bench_model");
  auto& runtime = env.MutableConfig();
  runtime.max_generated_tokens = bench.generated;
  runtime.temperature = 0.0f;
  runtime.verbosity = 0;
  Require(runtime.top_k == 1, "Use top_k=1 for exact greedy comparison");
  const auto prompt = env.WrapAndTokenize(bench.prompt);
  Require(!prompt.empty(), "Empty prompt");
  json policy = json::object();
  for (const char* name :
       {"GEMMA_MM_I8", "GEMMA_MM_I8_MICROSCALE", "GEMMA_MM_I8_L2_SCALE",
        "GEMMA_MM_I8_QUANT_BLOCK_SIZE", "GEMMA_MM_I8_MIN_K_SPLITS",
        "GEMMA_MM_I8_BLOCK_SIZE", "GEMMA_MM_I8_HASH_BITS",
        "GEMMA_MM_I8_PACKED_HEAD", "GEMMA_MM_I8_PACKED_HEAD_FULL_K",
        "GEMMA_MM_I8_DUAL_A_BODY", "GEMMA_MM_I8_DUAL_A_HEAD",
        "GEMMA_MM_I8_MATCH_BF16_A"})
    policy[name] = getenv(name);
  const json identity = {
      {"model", env.GetGemma()->Config().Specifier()},
      {"threads", args.threading.max_threads},
      {"vector_bytes", env.MutableEnv().ctx.cache_info.VectorBytes()},
      {"config_bytes", sizeof(MMConfig)},
      {"policy", policy}};
  if (!bench.schedule_in.Empty()) {
    std::ifstream file(bench.schedule_in.path);
    Require(bool(file), "Cannot open schedule snapshot");
    json saved;
    file >> saved;
    Require(saved.at("identity") == identity, "Schedule identity differs");
    const auto& clusters = saved.at("schedules");
    auto& mm = env.MutableEnv();
    Require(clusters.size() == mm.per_cluster.size(), "Cluster count differs");
    for (size_t ci = 0; ci < clusters.size(); ++ci) {
      auto& cluster = mm.per_cluster[ci];
      Require(cluster.keys.Keys().size() == 0, "Import before inference");
      cluster.per_key.resize(clusters[ci].size());
      for (size_t ki = 0; ki < clusters[ci].size(); ++ki) {
        const auto& row = clusters[ci][ki];
        const uint64_t key = row.at("key");
        Require(key != 0, "Zero schedule key");
        for (const auto seen : cluster.keys.Keys())
          Require(seen != key, "Duplicate schedule key");
        const auto hex = row.at("config").get<std::string>();
        Require(hex.size() == sizeof(MMConfig) * 2, "Config byte size differs");
        unsigned char bytes[sizeof(MMConfig)];
        const auto digit = [](char c) -> unsigned {
          if (c >= '0' && c <= '9') return c - '0';
          if (c >= 'a' && c <= 'f') return c - 'a' + 10;
          throw std::runtime_error("Invalid config hex");
        };
        for (size_t i = 0; i < sizeof(bytes); ++i)
          bytes[i] = 16 * digit(hex[2 * i]) + digit(hex[2 * i + 1]);
        MMConfig config;
        memcpy(&config, bytes, sizeof(config));
        Require((config.MR() == 1 || config.MR() == 2 || config.MR() == 4) &&
                    config.MC() && config.KC() && config.NC() &&
                    StringFromOrder(config.Order()) != nullptr &&
                    config.InnerTasks() >= 1 && config.InnerTasks() <= 4,
                "Invalid imported config");
        cluster.keys.Append(key, mm.ctx.cache_info.VectorBytes());
        cluster.per_key[ki].autotune.SetCandidates({config}, false);
        if (!row.at("par_a").is_null()) {
          const unsigned value = row.at("par_a");
          Require(value == 0 || value == 1 || value == 2 || value == 4 ||
                      value == 5,
                  "Invalid decompression schedule");
          cluster.per_key[ki].autotune_par_a.SetCandidates(
              {static_cast<MMParA>(value)}, false);
        }
      }
    }
    Require(ScheduleState(mm) == clusters, "Schedule import differs");
  }
  if (bench.schedule_in.Empty()) {
    // Settle the actual native schedules before enabling proposals. Two
    // output tokens exercise the same M1 matmuls without long trial decodes.
    SetEnv("GEMMA_MM_I8_HEAD_SPEC", "0");
    SetEnv("GEMMA_MM_I8_BODY_SPEC", "0");
    runtime.max_generated_tokens = std::min(size_t{2}, bench.generated);
    size_t trials = 0;
    do {
      Generate(env, prompt, bench.stop_after);
      ++trials;
    } while (trials < bench.max_warmup &&
             (trials < 2 || !SchedulesSettled(env.MutableEnv())));
    Require(SchedulesSettled(env.MutableEnv()),
            "Native autotuning did not settle; increase --bench_max_warmup");
    env.MutableEnv().autotune = false;
    runtime.max_generated_tokens = bench.generated;
    SetEnv("GEMMA_MM_I8_HEAD_SPEC", spec && small ? "1" : "0");
    SetEnv("GEMMA_MM_I8_BODY_SPEC", spec && !small ? "1" : "0");
  }
  for (size_t i = 0; i < bench.warmup; ++i)
    Generate(env, prompt, bench.stop_after);
  const json schedules = ScheduleState(env.MutableEnv());
  if (!bench.schedule_out.Empty()) {
    std::ofstream file(bench.schedule_out.path);
    Require(bool(file), "Cannot open schedule output");
    file << json({{"identity", identity}, {"schedules", schedules}}).dump();
    file.close();
    Require(bool(file), "Cannot save schedules");
  }
  json samples = json::array();
  RunResult reference;
  for (size_t i = 0; i < bench.reps; ++i) {
    RunResult run = Generate(env, prompt, bench.stop_after);
    Require(ScheduleState(env.MutableEnv()) == schedules,
            "Schedules changed during measurement");
    if (i == 0) reference = run;
    Require(run.tokens == reference.tokens &&
                run.probabilities == reference.probabilities,
            "Output bits changed between repetitions");
    Require(run.timing.generate_duration > 0, "Invalid generation duration");
    samples.push_back(
        {{"prefill_tps",
          run.timing.prefill_tokens / run.timing.prefill_duration},
         {"decode_tps", run.tokens.size() / run.timing.generate_duration}});
  }
  std::ofstream dump;
  if (!bench.logits.Empty()) {
    dump.open(bench.logits.path, std::ios::binary | std::ios::trunc);
    Require(bool(dump), "Cannot open logits dump");
  }
  size_t step = 0;
  runtime.sample_func = [&](size_t qi, size_t pos, Logits logits, size_t) {
    Require(qi == 0 && pos == prompt.size() + step &&
                step < reference.tokens.size(),
            "Unexpected replay position");
    Require(logits.size() == size_t(env.GetGemma()->Config().vocab_size),
            "Incomplete vocabulary logits");
    if (dump.is_open()) {
      dump.write(reinterpret_cast<const char*>(logits.data()),
                 logits.size() * sizeof(float));
      Require(bool(dump), "Cannot write logits dump");
    }
    float probability;
    memcpy(&probability, &reference.probabilities[step], sizeof(probability));
    return TokenAndProb{reference.tokens[step++], probability};
  };
  const auto replay = Generate(env, prompt, bench.stop_after);
  runtime.sample_func = {};
  Require(step == reference.tokens.size() &&
              replay.tokens == reference.tokens &&
              replay.probabilities == reference.probabilities,
          "Full-logit replay output differs");
  Require(ScheduleState(env.MutableEnv()) == schedules,
          "Schedule changed during replay");
  if (dump.is_open()) {
    dump.close();
    Require(bool(dump), "Cannot close logits dump");
  }
  printf("SPEC_BENCH_JSON %s\n",
         json({{"schema", "gemma-spec-bench-v1"},
               {"model", bench.model},
               {"mode", bench.mode},
               {"prompt", prompt},
               {"samples", samples},
               {"tokens", reference.tokens},
               {"probability_bits", reference.probabilities},
               {"vocab_size", env.GetGemma()->Config().vocab_size},
               {"schedules", schedules},
               {"full_logits_replay", true}})
             .dump()
             .c_str());
  return 0;
}
}  // namespace
}  // namespace gcpp
int main(int argc, char** argv) {
  try {
    return gcpp::Run(argc, argv);
  } catch (const std::exception& error) {
    fprintf(stderr, "spec_decode_bench: %s\n", error.what());
    return 1;
  }
}
