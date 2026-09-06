#!/usr/bin/env python3
"""Summarize attention_ab.py data without discarding trials or conflating metrics."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean, median


def span(values):
    return {"median": median(values), "min": min(values), "max": max(values)}


def summarize(folder):
    rows = [json.loads(line) for line in (folder / "results.jsonl").read_text().splitlines()]
    failed = [r for r in rows if r["returncode"]]
    if failed:
        raise ValueError(f"{len(failed)} failed runs: inspect raw results")
    rows = [r for r in rows if not r["warmup"]]
    report = {}
    for model in dict.fromkeys(r["model"] for r in rows):
        selected = [r for r in rows if r["model"] == model]
        result = {"speed": {}, "quality": {}, "outputs": []}
        for phase in ("prefill", "decode"):
            runs = [r for r in selected if r["phase"] == phase and r["case"] == "long"]
            variants = {v: [r for r in runs if r["variant"] == v] for v in ("base", "candidate")}
            keys = ("prefill_tok_s", "ttft_ms") if phase == "prefill" else ("generate_tok_s",)
            result["speed"][phase] = {
                key: {v: span([r[key] for r in rs]) for v, rs in variants.items()} for key in keys}
            result["speed"][phase]["raw"] = [{k: r[k] for k in ("pair", "variant", *keys, "generated_tokens")} for r in runs]
        for r in selected:
            if r["phase"] != "decode" or r["variant"] != "base":
                continue
            c = next(x for x in selected if x["phase"] == "decode" and x["variant"] == "candidate" and x["case"] == r["case"] and x["pair"] == r["pair"])
            btext, ctext = r["output"].encode(), c["output"].encode()
            prefix = next((i for i, (b, c) in enumerate(zip(btext, ctext)) if b != c), min(len(btext), len(ctext)))
            result["outputs"].append({"case": r["case"], "pair": r["pair"],
                "byte_identical": btext == ctext, "matching_stdout_prefix_bytes": prefix,
                "base_generated_tokens": r["generated_tokens"], "candidate_generated_tokens": c["generated_tokens"]})
        result["baseline_long_unique_outputs"] = len({r["output"] for r in selected if r["phase"] == "decode" and r["case"] == "long" and r["variant"] == "base"})
        qruns = [r for r in selected if r["phase"] == "quality"]
        cases = list(dict.fromkeys(r["case"] for r in qruns))
        pairs = sorted({r["pair"] for r in qruns})
        raw_quality = []
        for pair in pairs:
            qpair = {}
            for variant in ("base", "candidate"):
                rs = [r for r in qruns if r["pair"] == pair and r["variant"] == variant]
                assert len(rs) == len(cases)
                # Conditional perplexity excludes the evaluator's artificial
                # uniform-vocabulary cost for the first token in each excerpt.
                bits = sum(t["bits"] for r in rs for t in r["per_token"])
                tokens = sum(len(r["per_token"]) for r in rs)
                qpair[variant] = {"conditional_bits_per_token": bits / tokens,
                                  "conditional_perplexity": 2 ** (bits / tokens),
                                  "scored_tokens": tokens}
            deltas = []
            for case in cases:
                b, c = [next(r for r in qruns if r["pair"] == pair and r["case"] == case and r["variant"] == v) for v in ("base", "candidate")]
                assert [(t["pos"], t["token"]) for t in b["per_token"]] == [(t["pos"], t["token"]) for t in c["per_token"]]
                deltas.extend(abs(bt["bits"] - ct["bits"]) for bt, ct in zip(b["per_token"], c["per_token"]))
            qpair["mean_absolute_token_bits_delta"] = mean(deltas)
            qpair["max_absolute_token_bits_delta"] = max(deltas)
            qpair["pair"] = pair
            raw_quality.append(qpair)
        result["quality"]["aggregate"] = {v: span([r[v]["conditional_perplexity"] for r in raw_quality]) for v in ("base", "candidate")}
        result["quality"]["raw"] = raw_quality
        result["quality"]["cases"] = {
            case: {v: {"bits_per_byte": span([r["bits_per_byte"] for r in qruns if r["case"] == case and r["variant"] == v]),
                       "input_tokens": next(r["input_tokens"] for r in qruns if r["case"] == case)} for v in ("base", "candidate")} for case in cases}
        report[model] = result
    (folder / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["| Model | Prefill tok/s base → candidate | Change | TTFT ms base → candidate | Change |", "|---|---:|---:|---:|---:|"]
    for model, r in report.items():
        p = r["speed"]["prefill"]
        b, c = (p["prefill_tok_s"][v]["median"] for v in ("base", "candidate"))
        tb, tc = (p["ttft_ms"][v]["median"] for v in ("base", "candidate"))
        lines.append(f"| {model} | {b:.2f} → {c:.2f} | {(c/b-1)*100:+.2f}% | {tb:.0f} → {tc:.0f} | {(tc/tb-1)*100:+.2f}% |")
    lines += ["", "| Model | Conditional perplexity base → candidate | Change | Greedy byte-identical pairs |", "|---|---:|---:|---:|"]
    for model, r in report.items():
        q = r["quality"]["aggregate"]
        b, c = (q[v]["median"] for v in ("base", "candidate"))
        same = sum(o["byte_identical"] for o in r["outputs"])
        lines.append(f"| {model} | {b:.6f} → {c:.6f} | {(c/b-1)*100:+.5f}% | {same}/{len(r['outputs'])} |")
    lines += ["", "Decode rates are descriptive when generated outputs differ; inspect summary.json before making a speed claim.", ""]
    (folder / "tables.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    summarize(parser.parse_args().folder)
