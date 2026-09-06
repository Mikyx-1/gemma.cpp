# Single-row attention: speed, numerical accuracy, and output quality

This report compares baseline `3ed403e28707c0e3eb5f48b0e487b63a446d8e2e` with
candidate `24a62021cecd67443589e408162731d742daed4f`. The candidate keeps the online
softmax numerator unnormalized, computes one exponential per subsequent history
position, and normalizes once at the end. Partial sums are folded every 128
positions to reduce rounding error on long, same-sign value sequences.

## Setup

- Intel Core i5-12400F, 6 cores / 12 threads; Linux x86_64; GCC 13.3.0.
- Matching release flags: `-O3 -DNDEBUG -std=gnu++17 -fPIC` for the library.
- `--num_threads 12` is a maximum; the runtime used six physical cores in
  one cluster, skipping SMT siblings. Pinning and spin waits were enabled.
- Sequence length 8,192; AVX2 SIMD on this host.
- Default prefill batch size 256 and decode query batch size 16.
- `intel_pstate` with the `powersave` governor; CPU clocks were not locked.
- Fresh processes, serial execution, alternating baseline/candidate order.
- One explicitly marked prefill warmup per model and variant.
- All planned measurements are retained; the warmups are excluded from medians.
- The 4B weights use the external `tokenizer.spm`; all invocations specify it.
- All three models used the default `Read` loader mode. Effective settings are
  saved in [runtime-config.json](validation/runtime-config.json).
- The inference implementation is identical between the tested candidate and
  the final PR head; later commits add evaluation tooling/results/documentation.

The [method](../../ATTENTION_AB.md), [build metadata](build.json), and
[manifest](manifest.json) provide commands, versions, dependencies, input
fixtures, artifact sizes, SHA-256 hashes, timestamps, and RAM guard settings.
Every actual command and measured value is retained in [results.jsonl](results.jsonl).

## Prompt processing

Five measured A/B pairs per model use the same 3,366-token prompt and generate
one token. Prefill and time-to-first-token exclude model loading. Percentages
compare medians; these are measurements on this host, not portable guarantees.

| Model | Prefill tok/s base → candidate | Change | TTFT ms base → candidate | Change |
|---|---:|---:|---:|---:|
| 270m | 1001.70 → 1158.94 | +15.70% | 3390 → 2933 | -13.48% |
| 1b | 228.10 → 239.24 | +4.88% | 14831 → 14143 | -4.64% |
| 4b | 47.64 → 50.20 | +5.37% | 70938 → 67326 | -5.09% |

All raw prefill pairs:

| Model | Pair/order | Base tok/s | Candidate tok/s | Base TTFT ms | Candidate TTFT ms |
|---|---|---:|---:|---:|---:|
| 270m | 1 / B→C | 1001.70 | 1139.26 | 3390 | 2983 |
| 270m | 2 / C→B | 995.37 | 1158.91 | 3412 | 2933 |
| 270m | 3 / B→C | 1005.44 | 1158.94 | 3378 | 2933 |
| 270m | 4 / C→B | 1000.69 | 1163.91 | 3394 | 2921 |
| 270m | 5 / B→C | 1010.78 | 1182.48 | 3360 | 2875 |
| 1b | 1 / B→C | 226.66 | 236.87 | 14925 | 14285 |
| 1b | 2 / C→B | 229.61 | 241.17 | 14734 | 14031 |
| 1b | 3 / B→C | 230.76 | 239.07 | 14661 | 14153 |
| 1b | 4 / C→B | 228.10 | 239.24 | 14831 | 14143 |
| 1b | 5 / B→C | 227.53 | 241.38 | 14868 | 14019 |
| 4b | 1 / B→C | 48.60 | 50.63 | 69556 | 66750 |
| 4b | 2 / C→B | 47.64 | 50.20 | 70938 | 67326 |
| 4b | 3 / B→C | 47.00 | 43.04 | 71902 | 78479 |
| 4b | 4 / C→B | 48.08 | 50.35 | 70292 | 67123 |
| 4b | 5 / B→C | 45.22 | 39.79 | 74847 | 85743 |

Paired prefill consistency:

- 270m: candidate faster in 5/5 paired runs.
- 1b: candidate faster in 5/5 paired runs.
- 4b: candidate faster in 3/5 paired runs.

The 4B candidate has two conspicuously slow trials. Its favorable median
is weaker evidence of a repeatable improvement than the smaller-model
results; the raw spread and paired outcomes should accompany the headline.


## Reference likelihood: a small quality sanity check

Each of three exact text samples is scored in three alternating A/B pairs.
The evaluator forces reference tokens as the continuation, so both versions
see identical histories. The samples are two short public-domain literary
excerpts and one synthetic technical paragraph; they are not a held-out
benchmark suite. The existing evaluator uses raw text without BOS/chat wrapping.
Absolute scores depend on this raw-text format and the weights; the analysis
compares the two revisions for each model. No task-accuracy, multilingual,
or broad answer-quality claim follows.

Conditional perplexity pools predicted-token negative log probabilities across
the three texts, excluding the evaluator’s artificial uniform first-token cost.
Lower is better. Ranges show the three fresh-process repeats, not confidence intervals.

| Model | Scored tokens/repeat | Base PPL median [min, max] | Candidate PPL median [min, max] | Change |
|---|---:|---:|---:|---:|
| 270m | 615 | 1675.643077 [1654.683298, 1682.456665] | 1666.371528 [1651.754614, 1684.054233] | -0.55331% |
| 1b | 615 | 145.244302 [145.238078, 145.246945] | 143.844733 [143.833982, 143.857048] | -0.96360% |
| 4b | 615 | 293.909418 [293.844367, 296.933555] | 295.901739 [295.801236, 297.117550] | +0.67787% |

The 1B point estimate is favorable on this sample, with separated repeat ranges.
The 4B candidate has **0.68% higher (worse) median conditional perplexity**.
Its repeat range overlaps the baseline range, as does the 270M comparison.
These observations do not prove quality equivalence or rule out a regression;
a broader, standard evaluation would be needed for that conclusion.

Per-sample median bits per byte (includes the evaluator’s first-token cost):

| Model | Sample | Input tokens | Base [min, max] | Candidate [min, max] |
|---|---|---:|---:|---:|
| 270m | frankenstein_chap1 | 227 | 2.303823 [2.292045, 2.303823] | 2.294469 [2.294333, 2.300506] |
| 270m | frankenstein_chap4 | 215 | 2.340673 [2.340384, 2.340675] | 2.346343 [2.338587, 2.349071] |
| 270m | technical | 176 | 1.778785 [1.778199, 1.781332] | 1.776967 [1.776967, 1.777188] |
| 1b | frankenstein_chap1 | 227 | 1.551771 [1.551765, 1.551819] | 1.545654 [1.545654, 1.545654] |
| 1b | frankenstein_chap4 | 215 | 1.551419 [1.551409, 1.551479] | 1.549037 [1.548973, 1.549037] |
| 1b | technical | 176 | 1.220120 [1.220120, 1.220120] | 1.220277 [1.220277, 1.220350] |
| 4b | frankenstein_chap1 | 227 | 1.605800 [1.604189, 1.605800] | 1.606108 [1.605778, 1.606456] |
| 4b | frankenstein_chap4 | 215 | 1.881088 [1.879977, 1.889400] | 1.885245 [1.885191, 1.889857] |
| 4b | technical | 176 | 1.440870 [1.440316, 1.441171] | 1.440844 [1.440413, 1.440844] |

The [machine-readable summary](summary.json) includes all three pooled PPL
repeats and mean/maximum absolute per-token log-probability changes. The raw
results contain every reference token ID and likelihood. A change comparable
to baseline variation does not establish an improvement or degradation.

## Free-running generation and byte identity

Greedy generation uses `--top_k 1 --deterministic 1`, capped at 128 tokens.
There are three repeats of the long prompt plus one explanation and one coding
prompt per variant. Byte equality is an observed text-output check, not a
mathematical guarantee. Different outputs are not themselves evidence of worse
answers. Repeated baseline outputs also check reproducibility of the whole runtime.

| Model | Byte-identical paired runs | Distinct baseline long outputs / 3 repeats | Long decode tok/s base → candidate (medians) |
|---|---:|---:|---:|
| 270m | 4/5 | 2/3 | 39.95 → 39.23 |
| 1b | 4/5 | 1/3 | 15.05 → 15.06 |
| 4b | 2/5 | 3/3 | 3.83 → 3.87 |

All decode measurements, including divergent outputs and early EOS:

| Model | Prompt | Pair | Base tokens / tok/s | Candidate tokens / tok/s | Byte identical |
|---|---|---:|---:|---:|---|
| 270m | long | 1 | 128 / 40.68 | 128 / 39.23 | yes |
| 270m | long | 2 | 128 / 39.95 | 128 / 34.73 | no |
| 270m | long | 3 | 128 / 39.79 | 128 / 41.47 | yes |
| 270m | explanation | 1 | 128 / 46.78 | 128 / 48.00 | yes |
| 270m | code | 1 | 128 / 46.31 | 128 / 42.02 | yes |
| 1b | long | 1 | 128 / 15.05 | 128 / 14.92 | yes |
| 1b | long | 2 | 128 / 14.94 | 128 / 15.19 | yes |
| 1b | long | 3 | 128 / 15.09 | 128 / 15.06 | yes |
| 1b | explanation | 1 | 128 / 16.46 | 128 / 16.64 | no |
| 1b | code | 1 | 128 / 16.50 | 128 / 16.64 | yes |
| 4b | long | 1 | 128 / 2.27 | 128 / 3.28 | no |
| 4b | long | 2 | 128 / 3.83 | 128 / 3.87 | no |
| 4b | long | 3 | 128 / 3.86 | 128 / 3.91 | no |
| 4b | explanation | 1 | 128 / 4.47 | 128 / 4.43 | yes |
| 4b | code | 1 | 128 / 4.46 | 128 / 4.46 | yes |

Decode rates are descriptive when histories differ. Even equal-output
comparisons need to be assessed against their raw timing spread; these short
runs do not establish a general decode speedup. Full outputs are in results.jsonl.

## Numerical regression and its fix

The original PR implementation passed the new small signed-value oracle but
failed the existing integration test, reaching roughly `1.153e-5` relative
difference against its `1e-5` tolerance. The unmodified baseline passed that
existing test. Long unnormalized float sums introduced extra rounding error
for large positive values. The candidate now folds blocks of 128 terms into
the numerator and rescales both partial sums when the maximum changes.

The existing integration tolerance is unchanged. All four runnable tests pass
(integration and direct-reference tests on both AVX2 and EMU128). The direct
oracle covers 168 cases per SIMD target through 8,192 positions, including
circular windows, soft caps, extreme scores, and large positive values.
Maximum `abs(error) / max(1, abs(reference))` is `2.86101204e-6`.
Maximum absolute error is `0.00732421875`, occurring on large-magnitude values.
See [test output](validation/attention-tests.txt) and build.json for reproduction.

This adds a fixed 4 KiB stack buffer per active single-row call, without heap
allocation or KV-cache layout changes. It preserves the one-position fast path.
Floating-point reassociation remains: generated bytes are not guaranteed identical.

## CI status for the measured code revision

Linux CMake, macOS CMake, and Bazel builds passed at `24a6202`.
These workflows build the executable; they do not run the numerical tests above.
Windows fails before compilation because CMake receives policy version `3`
instead of `3.5`, and the configured Visual Studio 2022 generator is unavailable.
The same failure was present before this update. See the
[CI record](validation/ci-code-revision.json) and
[Windows error excerpt](validation/windows-config-failure.txt).

## RAM monitoring

A 250 ms monitor records process VmHWM, system MemAvailable, and swap-free
values. It stops the active benchmark process group below a 1.5 GiB available
RAM reserve. These are sampled observations, not kernel-enforced limits.

| Model | Maximum sampled process RSS GiB | Minimum available system RAM GiB | Guard stops |
|---|---:|---:|---:|
| 270m | 0.630 | 10.716 | 0 |
| 1b | 1.538 | 9.462 | 0 |
| 4b | 6.319 | 4.282 | 0 |

All 120 processes completed successfully, including six marked warmups.
See per-run memory telemetry in results.jsonl; existing swap usage is not
treated as zero merely because the guard did not fire.
