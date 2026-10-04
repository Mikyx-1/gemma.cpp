#!/usr/bin/env python3
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Compare native and speculative greedy decoding on six pinned CPU workers.

Build spec_decode_bench with GEMMA_HEAD_SPEC_HNSW=ON, then run:
  python3 evals/spec_decode_compare.py --binary build/spec_decode_bench \\
      --model 270m --weights 270m-sfp-it.sbs
  python3 evals/spec_decode_compare.py --binary build/spec_decode_bench \\
      --model 1b --weights 1b-it-sfp.sbs --draft-weights 270m-sfp-it.sbs

Warmups and index creation are excluded from throughput. Each timed process
also replays sampling through the actual selected driver, dumping all F32
logits. Dumps and schedules live in a temporary directory and are removed.
The INT8 HNSW cache is retained for subsequent runs.
"""
import argparse
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import tempfile


def same_bytes(left, right):
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open('rb') as a, right.open('rb') as b:
        while True:
            x, y = a.read(1024 * 1024), b.read(1024 * 1024)
            if x != y:
                return False
            if not x:
                return True


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--model', choices=['270m', '1b'], required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--draft-weights', type=Path)
    parser.add_argument('--cache-dir', type=Path,
                        default=Path('.temp/hnsw-head-i8-indexes'))
    parser.add_argument('--cpus', default='0,2,4,6,8,10')
    parser.add_argument('--processes', type=int, default=4,
                        help='Processes per mode; sequential alternating order.')
    parser.add_argument('--reps', type=int, default=3)
    parser.add_argument('--generated', type=int, default=128)
    parser.add_argument('--stop-after', type=int, default=0)
    parser.add_argument('--prompt')
    parser.add_argument('--fail-after', type=int, default=0,
                        help='Inject an ANN failure to check native fallback.')
    parser.add_argument('--schedule-in', type=Path,
                        help='Optional same-machine native benchmark snapshot.')
    args = parser.parse_args()
    if args.model == '1b' and args.draft_weights is None:
        parser.error('1b requires --draft-weights')
    if args.processes < 1 or args.reps < 1 or args.generated < 1:
        parser.error('Process/repetition/token counts must be positive')
    if args.stop_after < 0 or args.fail_after < 0:
        parser.error('Stop/failure counts cannot be negative')
    for path in (args.binary, args.weights, args.draft_weights, args.schedule_in):
        if path is not None and not path.is_file():
            parser.error(f'Missing file: {path}')
    env = {k: v for k, v in os.environ.items() if not k.startswith('GEMMA_MM_I8')}
    env['GEMMA_MM_I8_HEAD_SPEC_ANN_CACHE_DIR'] = str(args.cache_dir.resolve())
    env['GEMMA_MM_I8_HEAD_SPEC_ANN_FAIL_AFTER'] = str(args.fail_after)
    if args.draft_weights:
        env['GEMMA_MM_I8_DRAFT_WEIGHTS'] = str(args.draft_weights.resolve())
    samples = {'native': [], 'spec': []}
    proof_fields = ('prompt', 'tokens', 'probability_bits', 'vocab_size', 'schedules')
    # Reverse every other pair to reduce bias from thermal/frequency drift.
    order = [mode for i in range(args.processes)
             for mode in (('native', 'spec') if i % 2 == 0 else ('spec', 'native'))]
    counts = {'native': 0, 'spec': 0}
    with tempfile.TemporaryDirectory(prefix='gemma-spec-') as scratch:
        scratch = Path(scratch)
        schedule = args.schedule_in.resolve() if args.schedule_in else scratch / 'schedule.json'
        reference, reference_dump = None, None
        for mode in order:
            index = counts[mode]
            counts[mode] += 1
            dump = scratch / f'{mode}-{index}.bin'
            command = ['taskset', '-c', args.cpus, str(args.binary.resolve()),
                       '--weights', str(args.weights.resolve()), '--num_threads', '6',
                       '--bench_mode', mode, '--bench_model', args.model,
                       '--bench_reps', str(args.reps), '--bench_generated', str(args.generated),
                       '--bench_stop_after', str(args.stop_after), '--bench_logits', str(dump)]
            if args.prompt is not None:
                command += ['--bench_prompt', args.prompt]
            if schedule.is_file():
                command += ['--bench_schedule_in', str(schedule)]
            else:
                command += ['--bench_schedule_out', str(schedule)]
            print(f'{mode} process {index + 1}/{args.processes}', flush=True)
            run = subprocess.run(command, env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True)
            if run.returncode:
                raise RuntimeError(f'{mode} exited {run.returncode}\n{run.stderr[-12000:]}')
            records = [line[len('SPEC_BENCH_JSON '):] for line in run.stdout.splitlines()
                       if line.startswith('SPEC_BENCH_JSON ')]
            if len(records) != 1:
                raise RuntimeError('Missing benchmark result\n' + run.stderr[-4000:])
            result = json.loads(records[0])
            if not result['tokens']:
                raise RuntimeError('No generated tokens')
            expected_size = len(result['tokens']) * result['vocab_size'] * 4
            if dump.stat().st_size != expected_size:
                raise RuntimeError('Incomplete full-vocabulary logits dump')
            if reference is None:
                reference, reference_dump = result, dump
            else:
                for field in proof_fields:
                    if reference[field] != result[field]:
                        raise RuntimeError(f'{mode} {field} differs from native')
                if not same_bytes(reference_dump, dump):
                    raise RuntimeError(f'{mode} complete F32 logit bytes differ from native')
                dump.unlink()
            if mode == 'spec' and args.generated > 1 and args.stop_after != 1:
                pattern = r'(Head|Body) speculation: accepted (\d+) / (\d+)'
                counters = re.findall(pattern, run.stderr)
                if not counters and (args.fail_after or args.generated <= 3):
                    print('  native fallback for this short/failure case', flush=True)
                elif not counters:
                    raise RuntimeError('Speculative driver did not run\n' + run.stderr[-4000:])
                if counters:
                    _, accepted, proposed = counters[-1]
                    print(f'  accepted {accepted}/{proposed} proposals', flush=True)
            samples[mode].extend(result['samples'])
        metrics = {}
        for stage in ('prefill_tps', 'decode_tps'):
            before = statistics.median(x[stage] for x in samples['native'])
            after = statistics.median(x[stage] for x in samples['spec'])
            metrics[stage] = {'native': before, 'spec': after,
                              'improvement_percent': 100 * (after / before - 1)}
        print(json.dumps({'model': args.model, 'metrics': metrics,
                          'processes_per_mode': args.processes, 'reps': args.reps,
                          'generated_tokens': len(reference['tokens']),
                          'full_logit_bytes_per_run': reference_dump.stat().st_size,
                          'bit_identical': True}, indent=2))


if __name__ == '__main__':
    main()
