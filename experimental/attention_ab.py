#!/usr/bin/env python3
"""Compare unchanged gemma/single_benchmark executables; retain every trial.

Build the two revisions with identical release flags into separate binary dirs.
This runner alternates A/B order, runs processes serially, and records commands,
artifact hashes, outputs, timing, and teacher-forced reference likelihoods.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import time


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def memory_kib():
    return {line.split(":")[0]: int(line.split()[1])
            for line in Path("/proc/meminfo").read_text().splitlines()
            if line.startswith(("MemAvailable:", "SwapFree:"))}


def guarded_run(command, stdout, stderr, reserve_kib):
    before = memory_kib()
    if before["MemAvailable"] < reserve_kib:
        raise RuntimeError("Insufficient available RAM to start the next trial")
    proc = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True)
    minimum = before["MemAvailable"]
    peak_rss = 0
    reason = None
    deadline = time.monotonic() + 1800
    try:
        while proc.poll() is None:
            mem = memory_kib()
            minimum = min(minimum, mem["MemAvailable"])
            try:
                status = Path(f"/proc/{proc.pid}/status").read_text()
                match = re.search(r"VmHWM:\s+(\d+)", status)
                if match:
                    peak_rss = max(peak_rss, int(match[1]))
            except FileNotFoundError:
                pass
            if minimum < reserve_kib:
                reason = "available RAM below reserve"
            elif time.monotonic() > deadline:
                reason = "1800 second timeout"
            if reason:
                os.killpg(proc.pid, signal.SIGKILL)
                break
            time.sleep(0.25)
        proc.wait()
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    return proc.returncode, {
        "minimum_available_kib": minimum, "sampled_peak_rss_kib": peak_rss,
        "swap_free_before_kib": before["SwapFree"],
        "swap_free_after_kib": memory_kib()["SwapFree"], "stop_reason": reason,
    }


def capture(command):
    return subprocess.check_output(command, text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--candidate-revision", required=True)
    parser.add_argument("--model", action="append", required=True,
                        help="Label=weights path; repeat for multiple models")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--ram-reserve-gib", type=float, default=1.5)
    parser.add_argument("--prefill-pairs", type=int, default=5)
    parser.add_argument("--decode-pairs", type=int, default=3)
    parser.add_argument("--quality-pairs", type=int, default=3)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    args.output.mkdir(parents=True, exist_ok=False)
    logs = args.output / "logs"
    logs.mkdir()
    inputs = args.output / "inputs"
    inputs.mkdir()
    # Keep fixtures in the results so future repository text changes do not
    # change the workload. Character slicing is explicit, not token slicing.
    references = {}
    for chapter in (1, 4):
        name = f"frankenstein_chap{chapter}"
        offset = 0 if chapter == 1 else 8192
        text = (root / "testdata" / f"{name}.txt").read_text()[offset:offset + 1024]
        references[name] = text
    references["technical"] = (
        "A computer represents many real numbers using floating-point arithmetic. "
        "Each operation rounds its result to a finite set of representable values. "
        "Consequently, adding the same numbers in a different order can produce a "
        "slightly different result. This matters when a program compares two "
        "nearly equal scores and chooses the larger one. A small numerical change "
        "can change a discrete decision even if the underlying computation is "
        "mathematically equivalent.\n\n"
        "To compare two inference implementations, use the same model weights, "
        "input text, tokenizer, compiler flags, and thread settings. Measure prompt "
        "processing separately from generating the response. Repeat measurements "
        "and alternate execution order to reduce systematic bias. For a numerical "
        "quality check, score an identical reference continuation under both "
        "implementations. This keeps the token history fixed even when greedy "
        "generation would choose different words. Report the distribution of "
        "results as well as their average. A small sample cannot establish "
        "accuracy on every language or task.\n"
    )
    prompts = {
        "long": (root / "testdata/frankenstein_13.txt").read_text(),
        "explanation": "Explain why floating-point addition is not associative. Give a numerical example and explain the consequences for reproducible programs.",
        "code": "Write a Python function that merges two sorted lists of integers, preserving duplicates. Explain its time and space complexity.",
    }
    for name, value in {**references, **prompts}.items():
        (inputs / f"{name}.txt").write_text(value)
    binaries = {"base": args.base.resolve(), "candidate": args.candidate.resolve()}
    models = dict(item.split("=", 1) for item in args.model)
    manifest = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runner_command": list(__import__("sys").argv),
        "base_revision": args.base_revision,
        "candidate_revision": args.candidate_revision,
        "platform": platform.platform(), "cpu": capture(["lscpu"]),
        "compiler": capture(["c++", "--version"]),
        "cmake": capture(["cmake", "--version"]),
        "threads": args.threads, "seq_len": 8192,
        "ram_reserve_gib": args.ram_reserve_gib, "ram_sample_interval_seconds": 0.25,
        "prefill_pairs": args.prefill_pairs, "decode_pairs": args.decode_pairs,
        "quality_pairs": args.quality_pairs,
        "artifacts": {},
    }
    artifacts = [Path(p) for p in models.values()] + [root / "tokenizer.spm"]
    artifacts += [p / name for p in binaries.values() for name in ("gemma", "single_benchmark")]
    artifacts += list(inputs.glob("*.txt")) + [Path(__file__)]
    for path in artifacts:
        manifest["artifacts"][str(path)] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def run(model, variant, phase, case, pair, warmup=False):
        name = f"{model}-{phase}-{case}-{pair:02d}-{variant}"
        tool = "single_benchmark" if phase == "quality" else "gemma"
        command = [str(binaries[variant] / tool), "--weights", str(Path(models[model]).resolve()),
                   "--tokenizer", str(root / "tokenizer.spm"),
                   "--seq_len", "8192", "--num_threads", str(args.threads),
                   "--pin", "1", "--spin", "1", "--top_k", "1", "--deterministic", "1"]
        if phase == "quality":
            command += ["--cross_entropy", str(inputs / f"{case}.txt"), "--wrapping", "0", "--verbosity", "3"]
        else:
            command += ["--prompt_file", str(inputs / f"{case}.txt"),
                        "--max_generated_tokens", "1" if phase == "prefill" else "128", "--verbosity", "1"]
        print(f"START {name}", flush=True)
        start = time.monotonic()
        with (logs / f"{name}.stdout").open("w") as stdout, (logs / f"{name}.stderr").open("w") as stderr:
            returncode, memory = guarded_run(command, stdout, stderr, int(args.ram_reserve_gib * 1024 ** 2))
        elapsed = time.monotonic() - start
        out = (logs / f"{name}.stdout").read_text()
        err = (logs / f"{name}.stderr").read_text()
        row = dict(model=model, variant=variant, phase=phase, case=case, pair=pair,
                   warmup=warmup, command=command, returncode=returncode, memory=memory,
                   wall_seconds=elapsed, stdout_sha256=sha256(logs / f"{name}.stdout"),
                   stderr_sha256=sha256(logs / f"{name}.stderr"))
        if returncode == 0:
            if phase == "quality":
                row["input_tokens"] = int(re.search(r"Number of input tokens: (\d+)", out)[1])
                row["cross_entropy_bits"] = float(re.search(r"Total cross entropy: ([\d.eE+-]+)", out)[1])
                row["bits_per_byte"] = float(re.search(r"Cross entropy per byte: ([\d.eE+-]+)", out)[1])
                row["per_token"] = [dict(pos=int(m[0]), token=int(m[1]), probability=float(m[2]), bits=float(m[3]))
                    for m in re.findall(r"pos\s+(\d+) token\s+(\d+) = .*?\s+([\d.eE+-]+)\s+([\d.eE+-]+) bits", out)]
                if len(row["per_token"]) != row["input_tokens"] - 1:
                    raise RuntimeError(f"Incomplete reference scoring: {name}")
            else:
                m = re.search(r"Prefill: (\d+) ms for (\d+) prompt tokens \(([\d.]+) tokens / sec\); Time to first token: (\d+) ms", err)
                row.update(prefill_ms=int(m[1]), prompt_tokens=int(m[2]), prefill_tok_s=float(m[3]), ttft_ms=int(m[4]))
                m = re.search(r"Generate: (\d+) ms for (\d+) tokens \(([\d.]+) tokens / sec\)", err)
                row.update(generate_ms=int(m[1]), generated_tokens=int(m[2]), generate_tok_s=float(m[3]))
                row["output"] = out
        with (args.output / "results.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(f"DONE {name} {elapsed:.1f}s rc={returncode}", flush=True)
        if returncode:
            raise RuntimeError(f"Failed {name}: {err[-2000:]}")

    def pairs(model, phase, case, count):
        for pair in range(1, count + 1):
            order = ("base", "candidate") if pair % 2 else ("candidate", "base")
            for variant in order:
                run(model, variant, phase, case, pair)

    for model in models:
        for variant in binaries:
            run(model, variant, "prefill", "long", 0, warmup=True)
        pairs(model, "prefill", "long", args.prefill_pairs)
        pairs(model, "decode", "long", args.decode_pairs)
        for case in ("explanation", "code"):
            pairs(model, "decode", case, 1)
        for case in references:
            pairs(model, "quality", case, args.quality_pairs)
    manifest["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
