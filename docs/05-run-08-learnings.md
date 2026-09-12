# Run 08 learnings: reward correctness without rewarding verbosity

## Summary

Run 08 did not produce the expected offline improvement from GRPO. The most
important learning is not that correctness received too much weight. It is that
the reward did not express the intended meaning of correctness.

The training judge could raise correctness when an answer added accurate but
optional detail. GRPO could therefore prefer a longer answer even when a shorter
answer already fulfilled the request. The reward also continuously traded
correctness against style, although the intended policy is closer to a gate:
substantive correctness determines whether an answer is acceptable; simplicity,
clarity, and ASD-STE100 compliance determine how well an acceptable answer is
written.

This led to a design decision for the next run:

1. Replace the weighted reward with a substantive-correctness gate and a
   separate writing-quality score.

This change makes the optimization target match the desired behavior more
closely. It does not, by itself, prove that reward design caused run 08's
regression. The branch does not contain the run's rollouts, component rewards,
or trainer logs, so the causal diagnosis remains limited.

## What the original reward optimized

Run 08 used the multiplicative reward

```text
R = C(0.70 + 0.15S + 0.15A),
```

where:

- `C` is correctness, graded in steps from 0 to 1;
- `S` is semantic simplicity in `[0, 1]`; and
- `A` is ASD-STE100-style compliance in `[0, 1]`.

Multiplying by `C` has a sensible purpose: an unusable answer should not earn a
high reward merely because it is easy to read. The problem lies in what `C`
measured and in the continuous exchange rate between `C`, `S`, and `A`.

For example:

| Answer                                      |  `C` |  `S` |  `A` | Reward |
| ------------------------------------------- | ---: | ---: | ---: | -----: |
| Mostly correct and very simple              | 0.75 | 0.90 | 0.90 | 0.7275 |
| Fully correct but substantially less simple | 1.00 | 0.50 | 0.50 | 0.8500 |

The second answer wins comfortably. This is appropriate if the first answer has
a material factual defect. It is counterproductive if the only difference is
that the second answer includes more optional background and the judge treats
that extra completeness as greater correctness.

The formula does not even impose a strict correctness ordering across its full
range. An answer with `C=0.75` and perfect style scores receives `0.75`, while an
answer with `C=1` and zero style scores receives `0.70`. The practical preference
therefore depends on the particular score combination rather than a clear rule
about substantive correctness.

## Correctness must mean necessary correctness

The desired distinction is between substantive defects and completeness
differences:

- A wrong central claim, missing requested component, unsafe instruction, or
  omitted necessary caveat is a correctness failure.
- Optional background, extra examples, peripheral exceptions, and further
  elaboration are not required for full correctness unless the prompt asks for
  them.
- Adding optional detail must not increase correctness. Any factual claims in
  that detail must still be accurate.
- Wording, organization, verbosity, and controlled-language violations belong
  in the writing scores unless they change the meaning or prevent task
  fulfillment.

This definition lets a concise answer receive the highest correctness grade.
It also prevents the reward from treating verbosity as a low-risk way to collect
correctness points.

The judge should first identify a short checklist of the facts, constraints,
caveats, requested parts, and format requirements necessary for the prompt. It
should apply the same checklist to every completion in the group. The grades
then describe the effect of the most consequential defect:

|  Grade | Meaning                                          |
| -----: | ------------------------------------------------ |
|  `1.0` | Pass: no substantive error or necessary omission |
| `0.75` | Localized failure requiring a limited correction |
|  `0.5` | Major failure affecting the central answer       |
|  `0.0` | Fundamentally wrong, nonresponsive, or unusable  |

These are severity categories, not probabilities. Only `1.0` passes the gate.
A short reason should accompany each correctness judgment during calibration so
that false failures, especially demands for optional detail, are auditable.

## The new gated reward

For completion `y`, define:

- `C(y)` as the substantive-correctness grade;
- `S(y)` as semantic simplicity;
- `H(y)` as clarity and coherence;
- `A(y)` as selected ASD-STE100 compliance; and
- `D(y) = 1 - C(y)` as failure severity.

Writing quality is

```text
Q(y) = H(y)(0.6S(y) + 0.4A(y)).
```

The proposed reward is

```text
         Q(y)                 if C(y) = 1
R(y) = {
        -lambda D(y)          otherwise.
```

Use `lambda = 1` as the initial calibration value. Passing answers then receive
a reward in `[0, 1]`, while localized, major, and fundamental failures receive
`-0.25`, `-0.5`, and `-1.0`, respectively.

This structure expresses the intended ordering directly:

- Any passing answer outranks any failing answer.
- Among passing answers, writing quality supplies the entire ranking signal.
- Among failing answers, fixing a more severe substantive defect is rewarded.
- Extra detail has no inherent benefit after the answer fulfills the task.

Clarity is a multiplier because short sentences and approved words do not make
a fragmented or ambiguous answer good. The `0.6/0.4` simplicity-to-STE split is
an initial preference to calibrate, not a universal constant. The structural
choice—the correctness gate and clarity prerequisite—is more important than the
first set of coefficients.

This design supersedes the earlier prompt-relative reward experiment. That
design guaranteed ordering between correctness grades and increased the
available style signal, but it still relied on fine-grained score differences
that later GRPO normalization could amplify.

## Validation before another full run

The reward should be tested as a preference model before it is used as a
training objective. The minimum validation set should include deliberately
edited answer pairs:

| Edit                                                    | Expected result                             |
| ------------------------------------------------------- | ------------------------------------------- |
| Add accurate but unnecessary paragraphs                 | Same correctness; lower or equal simplicity |
| Remove a redundant summary                              | Same correctness; higher simplicity         |
| Remove a necessary caveat                               | Fail the gate                               |
| Replace jargon with equally precise familiar words      | Same correctness; higher simplicity         |
| Split coherent prose into disconnected fragments        | Lower clarity                               |
| Replace a technical term with an inaccurate common word | Fail the gate                               |
| Improve an applicable STE rule without changing meaning | Higher compliance                           |

Tests must cover the complete reward-to-advantage path, not only raw reward
ordering. Repeated judge calls and shuffled candidate order should measure
whether close decisions are stable. Coarse, anchored scores are preferable to
unsupported precision.

The final evaluation must also treat correctness as a constraint rather than
only another average:

```text
maximize writing quality
subject to substantive-failure rate <= the accepted threshold.
```

Reject checkpoints that exceed the tolerated correctness or clarity regression,
then select among eligible checkpoints using simplicity and STE compliance. A
gated reward guarantees ordering among sampled completions; it cannot guarantee
model-wide correctness preservation because every update changes shared model
parameters.

## Conclusions

Run 08 clarified that "correctness first" should not mean "reward every increase
in completeness." Correctness must protect the necessary meaning and task
requirements. Once that condition is met, the reward should give the optimizer
a direct reason to prefer clear, simple, controlled language.

## References

- [Run 08 review and reward-design discussion](https://chatgpt.com/share/6aa4fcaf-a1c8-83ec-a6e3-af344a93ffe4)
