# BigCode MinHash baseline vs core A/B/W/C

Baseline: BigCode `text-dedup` MinHash estimator (256 permutations, SHA1-32, `optimal_param`, no exact verification), run on the same core shingles and scored against the same exact truth C. Core numbers: `core/main_decision_table.json` (seed means, min-id representative) and `core/runs.jsonl` (pair recall).

## C4 τ=0.75

| config | deleted | unsupported | unsup/deleted | unsup/input | pair recall | residual | witness | t_pair+union (s) | peak RSS (MB) |
|---|---|---|---|---|---|---|---|---|---|
| 8x32 | 1682 | 32 | 0.019 | 0.00005 | 0.045 | 65644 | 0.981 | 5.5 | 1168 |
| 13x19 | 3650 | 171 | 0.047 | 0.00024 | 0.250 | 8630 | 0.945 | 8.0 | 1168 |
| 17x15 | 5207 | 760 | 0.146 | 0.00109 | 0.499 | 1179 | 0.801 | 10.0 | 1168 |
| 21x12 | 7319 | 2198 | 0.300 | 0.00314 | 0.760 | 224 | 0.612 | 11.5 | 1168 |
| 25x10 | 9484 | 4101 | 0.432 | 0.00586 | 0.911 | 56 | 0.470 | 12.8 | 1168 |
| 14x8 | 10460 | 5061 | 0.484 | 0.00723 | 0.914 | 60 | 0.425 | 7.0 | 1168 |
| 32x8 | 12959 | 7452 | 0.575 | 0.01065 | 0.990 | 4 | 0.341 | 16.2 | 1168 |
| 40x6 | 17972 | 12399 | 0.690 | 0.01771 | 1.000 | 0 | 0.240 | 20.4 | 1168 |

**Core references (min-id, seed means).**

| method | deleted | unsup/deleted | unsup/input | pair recall | witness | residual |
|---|---|---|---|---|---|---|
| A | 10406 | 0.483 | 0.00719 | 0.088 | 0.427 | 68.0 |
| B | 5095 | 0.000 | 0.00000 | 0.927 | 0.947 | 102.7 |
| W | 4817 | 0.000 | 0.00000 | 0.927 | 1.000 | 231.0 |
| C | 5222 | 0.000 | 0.00000 | — | 0.939 | 0.0 |

**Matched deletion budget** (baseline interpolated on `deleted`):

| target | baseline deleted | baseline unsup/deleted | baseline pair recall | core unsup/deleted | core pair recall |
|---|---|---|---|---|---|
| A (10406) | 10406 | 0.481 | 0.914 | 0.483 | 0.088 |
| B (5095) | 5095 | 0.139 | 0.482 | 0.000 | 0.927 |
| C (5222) | 5222 | 0.147 | 0.501 | 0.000 | — |
| W (4817) | 4817 | 0.121 | 0.437 | 0.000 | 0.927 |

**Matched pair recall** (baseline interpolated on `pair_recall`):

| target | baseline deleted | baseline unsup/deleted | core deleted | core unsup/deleted |
|---|---|---|---|---|
| B (0.927) | 10888 | 0.499 | 5095 | 0.000 |
| W (0.927) | 10888 | 0.499 | 4817 | 0.000 |

## C4 τ=0.8

| config | deleted | unsupported | unsup/deleted | unsup/input | pair recall | residual | witness | t_pair+union (s) | peak RSS (MB) |
|---|---|---|---|---|---|---|---|---|---|
| 8x32 | 1682 | 36 | 0.021 | 0.00005 | 0.064 | 39012 | 0.978 | 5.4 | 2795 |
| 13x19 | 3650 | 420 | 0.115 | 0.00060 | 0.325 | 2861 | 0.863 | 8.1 | 2795 |
| 17x15 | 5207 | 1443 | 0.277 | 0.00206 | 0.603 | 205 | 0.660 | 10.1 | 2795 |
| 21x12 | 7319 | 3310 | 0.452 | 0.00473 | 0.848 | 32 | 0.457 | 11.7 | 2795 |
| 25x10 | 9484 | 5396 | 0.569 | 0.00771 | 0.961 | 3 | 0.348 | 13.4 | 2795 |
| 14x8 | 10460 | 6366 | 0.609 | 0.00909 | 0.953 | 10 | 0.314 | 7.4 | 2795 |
| 32x8 | 12959 | 8824 | 0.681 | 0.01261 | 0.998 | 0 | 0.251 | 16.3 | 2795 |
| 40x6 | 17972 | 13795 | 0.768 | 0.01971 | 1.000 | 0 | 0.177 | 20.7 | 2795 |

**Core references (min-id, seed means).**

| method | deleted | unsup/deleted | unsup/input | pair recall | witness | residual |
|---|---|---|---|---|---|---|
| A | 10397 | 0.607 | 0.00902 | 0.096 | 0.317 | 10.2 |
| B | 3887 | 0.000 | 0.00000 | 0.957 | 0.916 | 18.6 |
| W | 3674 | 0.000 | 0.00000 | 0.957 | 1.000 | 43.6 |
| C | 3910 | 0.000 | 0.00000 | — | 0.916 | 0.0 |

**Matched deletion budget** (baseline interpolated on `deleted`):

| target | baseline deleted | baseline unsup/deleted | baseline pair recall | core unsup/deleted | core pair recall |
|---|---|---|---|---|---|
| A (10397) | 10397 | 0.606 | 0.954 | 0.607 | 0.096 |
| B (3887) | 3887 | 0.140 | 0.368 | 0.000 | 0.957 |
| C (3910) | 3910 | 0.142 | 0.372 | 0.000 | — |
| W (3674) | 3674 | 0.118 | 0.330 | 0.000 | 0.957 |

**Matched pair recall** (baseline interpolated on `pair_recall`):

| target | baseline deleted | baseline unsup/deleted | core deleted | core unsup/deleted |
|---|---|---|---|---|
| B (0.957) | 9996 | 0.590 | 3887 | 0.000 |
| W (0.957) | 9996 | 0.590 | 3674 | 0.000 |

## FW τ=0.75

| config | deleted | unsupported | unsup/deleted | unsup/input | pair recall | residual | witness | t_pair+union (s) | peak RSS (MB) |
|---|---|---|---|---|---|---|---|---|---|
| 8x32 | 226 | 0 | 0.000 | 0.00000 | 0.462 | 205 | 1.000 | 7.9 | 1588 |
| 13x19 | 284 | 4 | 0.014 | 0.00000 | 0.674 | 100 | 0.979 | 11.8 | 1588 |
| 17x15 | 375 | 47 | 0.125 | 0.00005 | 0.811 | 41 | 0.859 | 14.6 | 1588 |
| 21x12 | 539 | 176 | 0.327 | 0.00018 | 0.933 | 14 | 0.627 | 16.6 | 1588 |
| 25x10 | 835 | 453 | 0.543 | 0.00045 | 0.969 | 2 | 0.408 | 19.1 | 1588 |
| 14x8 | 1110 | 726 | 0.654 | 0.00073 | 0.981 | 1 | 0.311 | 10.0 | 1588 |
| 32x8 | 1725 | 1333 | 0.773 | 0.00133 | 1.000 | 0 | 0.194 | 22.7 | 1588 |
| 40x6 | 4659 | 4258 | 0.914 | 0.00426 | 1.000 | 0 | 0.068 | 27.6 | 1588 |

**Core references (min-id, seed means).**

| method | deleted | unsup/deleted | unsup/input | pair recall | witness | residual |
|---|---|---|---|---|---|---|
| A | 1098 | 0.652 | 0.00072 | 0.782 | 0.306 | 5.0 |
| B | 366 | 0.000 | 0.00000 | 0.971 | 0.962 | 6.7 |
| W | 356 | 0.000 | 0.00000 | 0.971 | 1.000 | 9.0 |
| C | 375 | 0.000 | 0.00000 | — | 0.955 | 0.0 |

**Matched deletion budget** (baseline interpolated on `deleted`):

| target | baseline deleted | baseline unsup/deleted | baseline pair recall | core unsup/deleted | core pair recall |
|---|---|---|---|---|---|
| A (1098) | 1098 | 0.649 | 0.980 | 0.652 | 0.782 |
| B (366) | 366 | 0.115 | 0.798 | 0.000 | 0.971 |
| C (375) | 375 | 0.125 | 0.811 | 0.000 | — |
| W (356) | 356 | 0.102 | 0.782 | 0.000 | 0.971 |

**Matched pair recall** (baseline interpolated on `pair_recall`):

| target | baseline deleted | baseline unsup/deleted | core deleted | core unsup/deleted |
|---|---|---|---|---|
| B (0.971) | 881 | 0.561 | 366 | 0.000 |
| W (0.971) | 881 | 0.561 | 356 | 0.000 |

## FW τ=0.8

| config | deleted | unsupported | unsup/deleted | unsup/input | pair recall | residual | witness | t_pair+union (s) | peak RSS (MB) |
|---|---|---|---|---|---|---|---|---|---|
| 8x32 | 226 | 0 | 0.000 | 0.00000 | 0.571 | 120 | 1.000 | 7.8 | 5112 |
| 13x19 | 284 | 7 | 0.025 | 0.00001 | 0.811 | 36 | 0.961 | 11.9 | 5112 |
| 17x15 | 375 | 68 | 0.181 | 0.00007 | 0.925 | 9 | 0.784 | 15.0 | 5112 |
| 21x12 | 539 | 223 | 0.414 | 0.00022 | 0.993 | 0 | 0.562 | 17.1 | 5112 |
| 25x10 | 835 | 517 | 0.619 | 0.00052 | 0.998 | 0 | 0.360 | 19.6 | 5112 |
| 14x8 | 1110 | 790 | 0.712 | 0.00079 | 0.998 | 0 | 0.268 | 10.2 | 5112 |
| 32x8 | 1725 | 1403 | 0.813 | 0.00140 | 1.000 | 0 | 0.172 | 23.2 | 5112 |
| 40x6 | 4659 | 4334 | 0.930 | 0.00433 | 1.000 | 0 | 0.062 | 28.3 | 5112 |

**Core references (min-id, seed means).**

| method | deleted | unsup/deleted | unsup/input | pair recall | witness | residual |
|---|---|---|---|---|---|---|
| A | 1093 | 0.707 | 0.00077 | 0.819 | 0.271 | 0.6 |
| B | 313 | 0.000 | 0.00000 | 0.996 | 0.982 | 0.6 |
| W | 308 | 0.000 | 0.00000 | 0.996 | 1.000 | 0.8 |
| C | 314 | 0.000 | 0.00000 | — | 0.981 | 0.0 |

**Matched deletion budget** (baseline interpolated on `deleted`):

| target | baseline deleted | baseline unsup/deleted | baseline pair recall | core unsup/deleted | core pair recall |
|---|---|---|---|---|---|
| A (1093) | 1093 | 0.706 | 0.998 | 0.707 | 0.819 |
| B (313) | 313 | 0.075 | 0.848 | 0.000 | 0.996 |
| C (314) | 314 | 0.076 | 0.849 | 0.000 | — |
| W (308) | 308 | 0.067 | 0.842 | 0.000 | 0.996 |

**Matched pair recall** (baseline interpolated on `pair_recall`):

| target | baseline deleted | baseline unsup/deleted | core deleted | core unsup/deleted |
|---|---|---|---|---|
| B (0.996) | 717 | 0.537 | 313 | 0.000 |
| W (0.996) | 717 | 0.537 | 308 | 0.000 |

