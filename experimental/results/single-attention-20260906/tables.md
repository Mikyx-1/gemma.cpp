| Model | Prefill tok/s base → candidate | Change | TTFT ms base → candidate | Change |
|---|---:|---:|---:|---:|
| 270m | 1001.70 → 1158.94 | +15.70% | 3390 → 2933 | -13.48% |
| 1b | 228.10 → 239.24 | +4.88% | 14831 → 14143 | -4.64% |
| 4b | 47.64 → 50.20 | +5.37% | 70938 → 67326 | -5.09% |

| Model | Conditional perplexity base → candidate | Change | Greedy byte-identical pairs |
|---|---:|---:|---:|
| 270m | 1675.643077 → 1666.371528 | -0.55331% | 4/5 |
| 1b | 145.244302 → 143.844733 | -0.96360% | 4/5 |
| 4b | 293.909418 → 295.901739 | +0.67787% | 2/5 |

Decode rates are descriptive when generated outputs differ; inspect summary.json before making a speed claim.
