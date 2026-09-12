# Length-aware GRPO reward simulation

## Decision

Keep both reward functions so they can be compared in GRPO experiments:

- `multiplicative` is the original and remains the default:
  `R = C(0.70 + 0.15S + 0.15A)`.
- `relative` is the experimental reward described below.

For each completion `i` in the candidates for one prompt, let:

- `Cᵢ` be correctness in `{0, 0.25, 0.5, 0.75, 1}`.
- `Sᵢ` be semantic simplicity in `[0, 1]`.
- `Aᵢ` be ASD-STE100 compliance in `[0, 1]`.
- `Qᵢ = SᵢAᵢ` be joint style quality.
- `qᵢ = (Qᵢ - min(Q)) / max(max(Q) - min(Q), 0.25)` across useful
  completions for that prompt. The minimum denominator prevents tiny judge
  differences from expanding into the complete bonus.

The reward is:

```text
Rᵢ = 0                                      if Cᵢ = 0
Rᵢ = (Cᵢ + 0.24qᵢ) / 1.24                  otherwise
```

Prompt-local scaling makes the complete `0.24` style signal available at all
nonzero correctness levels. The old multiplicative reward reduced the maximum
style signal to `0.075` at correctness `0.25`, which caused correctness and its
length correlation to dominate early optimization.

`Q` is a product because simplicity must not compensate for poor STE
compliance, or the reverse. If all useful candidates have the same `Q`, their
style bonus is zero and only correctness distinguishes them.

## Correctness guarantee

Adjacent correctness grades differ by `0.25`, while the largest possible style
bonus is `0.24`. Thus, before the common normalization by `1.24`:

```text
maximum reward at lower grade = C + 0.24
minimum reward at next grade  = C + 0.25
```

The higher correctness grade always wins by at least `0.01`. A completion with
zero correctness always receives zero. An exhaustive grid test covers all
6,250 pairs of correctness grades and style scores.

## Dynamic offline simulation

The replay joins 387 stored answers for 100 prompts across four model runs. It
uses:

- technical adequacy as correctness;
- semantic simplicity as simplicity;
- the mean of seven deterministic STE checks as the ASD-STE100 proxy;
- stored output tokens, average sentence length, and long-sentence fraction as
  independent behavior measurements.

For each prompt, the simulation standardizes candidate rewards as GRPO does.
It then applies `P(i) ∝ exp(t × advantageᵢ)`. Increasing `t` approximates a
policy that moves from no reward preference toward stronger preference for
high-reward completions. This directly tests the early, middle, and late
behavior that the first replay omitted.

All values below are changes from the uniform candidate policy:

| Strength | Old Δ tokens | New Δ tokens | New Δ correctness | New Δ simplicity | New Δ sentence length |
| -------: | -----------: | -----------: | ----------------: | ---------------: | --------------------: |
|     0.10 |        +1.67 |        -2.81 |           +0.0105 |          +0.0075 |               -0.0323 |
|     0.25 |        +3.74 |        -7.32 |           +0.0258 |          +0.0182 |               -0.0820 |
|     0.50 |        +6.12 |       -15.30 |           +0.0497 |          +0.0343 |               -0.1645 |
|     1.00 |        +7.87 |       -30.75 |           +0.0882 |          +0.0592 |               -0.3069 |
|     2.00 |        +4.72 |       -51.25 |           +0.1280 |          +0.0830 |               -0.4470 |
|     4.00 |        -8.52 |       -64.42 |           +0.1478 |          +0.0925 |               -0.4957 |
|     8.00 |       -30.58 |       -67.11 |           +0.1548 |          +0.0935 |               -0.5238 |

The old reward reproduces the reported failure: it increases total answer
length through most of the simulated optimization path. The new reward reduces
total length, average sentence length, and long-sentence fraction from the
first nonzero step. Correctness, simplicity, and STE compliance improve at
every step.

The replay also removes candidates above successive correctness caps to model
earlier policy stages. At the first nonzero strength:

| Maximum available correctness | Prompts | Δ correctness | Δ tokens | Δ sentence length |
| ----------------------------: | ------: | ------------: | -------: | ----------------: |
|                          0.25 |       4 |        0.0000 |    -1.95 |           -0.0779 |
|                          0.50 |      50 |       +0.0042 |    -1.81 |           -0.0054 |
|                          0.75 |      92 |       +0.0068 |    -3.09 |           -0.0477 |

Thus, the simplicity signal operates even when the available answers are weak;
it does not wait for correctness to approach `1.0`.

The early-step result also survives 100 deterministic noise trials that add
Gaussian noise with standard deviation `0.05` to both style scores. Every trial
reduces output length; the token change ranges from `-3.88` to `-0.97`, with a
mean of `-2.35`. This check exercises the minimum style-spread safeguard.

Run the reward invariants with:

```bash
.venv/bin/pytest -p no:rerunfailures -q tests/test_grpo_rewards.py
```

Select the reward for a training run with `--reward-function multiplicative`
or `--reward-function relative`. In both cases, TensorBoard logs the
combined reward and its correctness, simplicity, and ASD-STE100 inputs. The
combined reward has weight `1.0`; the three input rewards have weight `0.0`, so
only the selected combined reward trains the policy.

## Boundary

This is an offline policy simulation, not a claim about the exact trajectory of
a neural-network optimizer. It tests observed tradeoffs in the available model
outputs and catches the verbosity failure that the first score-grid replay
missed.

The simulation validates reward behavior, not judge accuracy. ASD-STE100 has 53
writing rules and a controlled dictionary. Production claims of full compliance
still require an approved checker or expert review.

References: [ASD-STE100 Issue 9](https://www.asd-ste100.org/assets/files/ASD-STE100_ISSUE9.pdf),
[official overview](https://www.asd-ste100.org/about_STE.html).
