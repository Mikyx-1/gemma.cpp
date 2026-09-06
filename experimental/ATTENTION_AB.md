# Single-row attention A/B evaluation

`attention_ab.py` compares two release builds of the existing `gemma` and
`single_benchmark` executables. It runs one process at a time, alternates
baseline/candidate order in each pair, and retains every measured trial.

The [2026-09-06 report](results/single-attention-20260906/REPORT.md) evaluates
Gemma 3 270M IT, 1B IT, and 4B IT SFP on an Intel Core i5-12400F.

## Reproduce

Build the baseline and candidate revisions in separate checkouts, with the same
compiler, dependency revisions, and CMake settings. Copy or retain both `gemma`
and `single_benchmark` in each binary directory. The measured revisions,
compiler flags, dependency hashes, and binary hashes are in the report's
`manifest.json` and `build.json`. Do not reuse binaries whose provenance is
unknown or compare different cached dependency revisions.

Typical release configuration:

```sh
cmake -S <checkout> -B <build> \
  -DCMAKE_BUILD_TYPE=Release -DGEMMA_ENABLE_TESTS=OFF \
  -DHWY_ENABLE_TESTS=OFF -DSPM_ENABLE_SHARED=OFF -DSPM_ABSL_PROVIDER=module
cmake --build <build> --target gemma single_benchmark -j2
```

Run from the repository containing the weights and external tokenizer:

```sh
python3 experimental/attention_ab.py \
  --base <baseline-binary-directory> \
  --candidate <candidate-binary-directory> \
  --base-revision <baseline-commit> \
  --candidate-revision <candidate-commit> \
  --model 270m=270m-sfp-it.sbs \
  --model 1b=1b-it-sfp.sbs \
  --model 4b=4b-it-sfp.sbs \
  --threads 12 --ram-reserve-gib 1.5 \
  --output <new-results-directory>
python3 experimental/summarize_attention_ab.py <new-results-directory>
```

The `--threads 12` argument passes the runtime’s maximum-thread setting. On
this host, its topology selection used six physical cores, with SMT siblings
skipped. This effective setting is recorded in `build.json`.

Python 3.11+ and Linux `/proc` are required. The output directory must be new;
resuming or overwriting a partial run is intentionally unsupported. A failed
process, incomplete likelihood trace, timeout, or RAM guard stop aborts the
matrix instead of silently dropping or replacing a trial.

The RAM guard samples system `MemAvailable` and process `VmHWM` every 250 ms. It
kills only the active benchmark process group if available RAM falls below the
configured reserve. Memory telemetry and swap-free values are retained per run.
This is a sampled guard, not a kernel-enforced memory limit. Close unrelated
heavy applications when comparing timings.

## Workloads and interpretation

- **Prefill and time to first token:** five alternating pairs per model, after
  one explicitly marked warmup per binary/model. The prompt is the entire
  `testdata/frankenstein_13.txt`; generation is limited to one token. Reported
  inference timing excludes model loading, while process wall time includes it.
- **Free-running decode:** three alternating pairs on the long prompt and one
  pair each on an explanation and a Python coding prompt. Top-k is explicitly
  one, deterministic sampling is enabled, and generation is capped at 128
  tokens. Early EOS is retained. Compare byte equality as well as throughput;
  different token histories or lengths prevent a strict decode-speed claim.
  The saved stdout includes the identical CLI prompt-progress dots; equality
  therefore detects output-text differences, but is not a token-ID guarantee.
- **Reference likelihood:** three alternating pairs on each of three samples.
  Two samples are 1,024-character slices of the repository's public-domain
  Frankenstein text (`frankenstein_chap1.txt[0:1024]` and
  `frankenstein_chap4.txt[8192:9216]`). The third is a synthetic technical
  paragraph. All exact inputs are saved and hashed.

Reference scoring uses `single_benchmark --cross_entropy <text> --wrapping 0
--verbosity 3`. The existing evaluator tokenizes raw text without prepending
BOS or chat wrapping, then forces each reference token as the next input. Both
binaries therefore see the same token history, even if greedy output differs.
The 4B loader warns that `--wrapping` is ignored for VLMs, but this evaluator
still calls `GemmaEnv::Tokenize` directly and does not wrap the reference text.
Per-token probabilities and negative log probabilities are retained. This is a
small numerical/language-model sanity check, not a standardized task benchmark
or a claim about absolute model quality.

`summary.json` reports conditional perplexity as `2 ** mean(token_bits)`, pooled
over all three samples. It excludes the evaluator's artificial uniform
first-token cost for each sample. It also retains the evaluator's original
bits-per-byte statistic per sample. Lower is better for both metrics. Compare
the candidate change with the min/max spread of repeated baseline runs; a
small point estimate alone does not establish a quality improvement or loss.

## Numerical regression tests

`TestSingleAttentionReference` calls the actual single-row kernel and compares
it with independently materialized double-precision softmax. It covers lengths
1, 2, 17, 129, 1,024, and 8,192; contiguous and circular windows; soft caps 0 and
50; tied, increasing, decreasing, alternating extreme, and mixed scores; and
signed, large positive ramp, and large positive constant value vectors.

The existing integration tests keep their original `1e-5` relative tolerance.
The new oracle uses `2e-5 + 2e-5 * abs(reference)` and also reports the maximum
`abs(error) / max(1, abs(reference))` across cases. Results are saved under
`validation/` in the report directory.

The repository's Gemma and Highway CMake test targets have overlapping names.
For this run, Highway's tests were disabled and the already-cached GoogleTest
source was added through `CMAKE_PROJECT_gemma_INCLUDE`, with `GTest::Main` as an
alias of `gtest_main`. Then `GEMMA_ENABLE_TESTS=ON` enabled the unchanged Gemma
integration tests and the new direct tests. This local build workaround does
not alter the inference implementation or dependency sources.

For a fresh checkout without that cache, populate the source location used by
the saved prelude, using the exact GoogleTest revision from this run:

```sh
git clone https://github.com/google/googletest.git \
  build/_deps/highway-build/googletest-src
git -C build/_deps/highway-build/googletest-src checkout 52eb8108c5bdec04579160ae17225d66034bd723
cmake -S . -B build -DHWY_ENABLE_TESTS=OFF -DGEMMA_ENABLE_TESTS=ON \
  -DCMAKE_PROJECT_gemma_INCLUDE="$PWD/experimental/results/single-attention-20260906/validation/gtest-prelude.cmake"
cmake --build build --target flash_attention_test -j2
build/flash_attention_test
```
