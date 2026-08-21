# Looping and Experiment 06 analysis

## Summary

The SFT model produced severe loops during Experiment 05. Responses started normally, then repeated a phrase or list until they reached the 2,048-token limit.

Three changes reduced the problem:

1. Replacing greedy decoding with sampling.
2. Reducing the LoRA adapter scale from 0.5 to 0.25.
3. Adding a presence penalty of 1.5 in Experiment 06.

Experiment 06 completed all 100 prompts with no truncations or severe phrase loops. Mean output length fell by 5.2%, and mean generation time fell by 5.9% relative to the final Experiment 05 run.

The presence penalty did not improve accuracy. Judge scores were broadly unchanged, while several mechanical Simple English scores declined. The penalty is therefore a useful anti-looping guardrail, but 1.5 may be stronger than necessary.

## The problem

We observed two kinds of repetition:

- **Structural repetition:** too many headings, bullets, parallel sentences, or summaries. The answer remains readable but is longer than necessary.
- **Degenerate looping:** a word or phrase cycle takes over the response. The model usually continues until it reaches the token limit.

Experiment 05 exposed the second and more serious failure mode.

### Examples

In the first Experiment 05 attempt, `NET-06` repeated an expanding BGP phrase:

> ... from a single exit point from a single next hop from a single next hop from a single next hop ...

The answer reached 2,048 tokens. A six-word sequence appeared 86 times.

`API-03` repeatedly cycled through four Protocol Buffers bullets:

> The new field must not be a message field.
>
> The new field must not be a group field.
>
> The new field must not be a map field.
>
> The new field must not be a oneof field.

In the final Experiment 05 run, `DBS-10` repeated variants of:

> ... per second per query per second per second per second ...

Its most common six-word sequence appeared 578 times.

Sampling also produced a less common compound-token loop. `API-07` generated names such as `X-Request-Request-Request-Request-Request-ID` until truncation. Sampling reduced the problem but did not remove it.

## What we tried

The table shows the saved output from each attempt. The first three attempts are incomplete, so their counts are diagnostic rather than full benchmark results.

| Run                      | Decoder and adapter                     | Saved responses | Truncated | Severe loops |
| ------------------------ | --------------------------------------- | --------------: | --------: | -----------: |
| Experiment 05, attempt 1 | Greedy, adapter scale 0.5               |              58 |        10 |           10 |
| Experiment 05, attempt 2 | Sampling, adapter scale 0.5             |              72 |         5 |            5 |
| Experiment 05, attempt 3 | Sampling, adapter scale 0.25            |              75 |         2 |            1 |
| Experiment 05, final     | Sampling, revised adapter at scale 0.25 |      99 visible |         2 |            2 |
| Experiment 06            | Same setup plus presence penalty 1.5    |             100 |         0 |            0 |

A severe loop in this table means that one normalized six-word sequence appeared at least 15 times. This retrospective check is not part of the current scorer.

### Sampling

We changed greedy decoding to temperature 0.7, top-p 0.8, and top-k 20. This produced the largest improvement.

Greedy decoding always selects the most likely next token. Once it enters a repetitive pattern, that pattern can reinforce itself. Sampling gives the model a chance to select a different continuation.

### Lower adapter scale and revised training data

Reducing the adapter scale from 0.5 to 0.25 lowered the number of loops in the saved runs. The final Experiment 05 run also used a new adapter trained on revised answers.

These changes were not tested independently. We cannot separate the effects of adapter scale, training data, and random sampling from these runs alone.

### Output limit and validity checks

The 2,048-token limit bounds the cost of a loop. The scorer rejects truncated answers, which prevents them from affecting quality averages.

These measures contain the failure but do not prevent it. Excluding truncated answers can also make quality averages look better by removing the worst outputs.

### Presence penalty

Experiment 06 subtracts 1.5 from the logit of every token that has already appeared in the generated response. Prompt tokens are not penalized, and repeated tokens receive the penalty once rather than once per occurrence.

This makes repeated continuations less likely. It also discourages valid reuse of common words and technical terms.

## Experiment 06 results

Experiment 06 uses the same model revision, evaluation set, seed, sampling settings, adapter, and adapter scale as the final Experiment 05 run. The intended generation difference is the presence penalty.

Both experiments are single stochastic runs. Their comparison does not measure variation across seeds. The final Experiment 05 artifact is also missing one visible prediction, so paired analysis uses the 99 common prompts.

### Generation

| Measure              | Experiment 05 | Experiment 06 |                 Change |
| -------------------- | ------------: | ------------: | ---------------------: |
| Truncated responses  |             2 |             0 | Eliminated in this run |
| Mean output tokens   |         679.2 |         643.6 |                  -5.2% |
| Median output tokens |           613 |           567 |                  -7.5% |
| Mean generation time |       35.71 s |       33.62 s |                  -5.9% |
| Wall time            |    3,601.61 s |    3,392.07 s |                  -5.8% |

Experiment 06 was shorter on 52 of the 99 common prompts and longer on 47. Most of the total saving came from fixing the two Experiment 05 truncations:

- `DBS-10` fell from 2,048 to 1,250 tokens.
- `CLD-07` fell from 2,048 to 1,059 tokens.

The penalty improved the worst cases rather than making every answer shorter. Experiment 06 still produced 19 answers with at least 1,000 tokens.

### Answer quality

| Judge dimension       | Experiment 05 | Experiment 06 |   Change |
| --------------------- | ------------: | ------------: | -------: |
| Clarity and coherence |        90.63% |        92.68% | +2.05 pp |
| Semantic simplicity   |        79.21% |        83.25% | +4.04 pp |
| Task fulfillment      |        95.57% |        95.92% | +0.35 pp |
| Technical adequacy    |        71.88% |        71.43% | -0.45 pp |

The small gains in clarity and simplicity are not conclusive. In paired comparisons, the approximate 95% confidence interval for every judge dimension included zero. Technical adequacy was effectively unchanged: 24 answers improved, 46 tied, and 24 declined.

Removing a loop also did not fix factual errors. The Experiment 06 versions of `DBS-10` and `CLD-07` became complete and scorable, but both still contained incorrect operational advice.

### Simple English rules

| Rule metric             | Experiment 05 | Experiment 06 |      Change |
| ----------------------- | ------------: | ------------: | ----------: |
| Average sentence length |         15.37 |         16.15 | +0.79 words |
| Long-sentence fraction  |        15.76% |        17.83% |    +2.07 pp |
| Sentence mechanics      |        91.54% |        88.89% |    -2.65 pp |
| Verb forms and modals   |        89.11% |        86.77% |    -2.34 pp |
| Controlled vocabulary   |        54.03% |        50.30% |    -3.72 pp |
| Terminology consistency |        84.69% |        77.00% |    -7.69 pp |

Paired declines in controlled vocabulary, sentence mechanics, and verb forms were statistically clear. The terminology decline was large but less certain.

This result matches the penalty's design. Simple English favors common words and consistent terms. The presence penalty pushes the model away from words and terms it has already used.

### Examples of the trade-off

- `PLR-08` improved from a 591-token answer with headings, bullets, and a diagram to one clear 139-token paragraph. All judge dimensions reached 1.0.
- `CLD-04` moved in the opposite direction. It grew from 471 to 1,473 tokens and added unnecessary sections and a summary.
- `API-09` remained clear but omitted required OAuth credentials and introduced other technical errors. Its technical score fell from 0.5 to 0.25.

The penalty reduces repeated tokens. It does not reliably control total length, structure, or correctness.

## Detection gap

The current validity check detects one token repeated eight times in a row. It also rejects truncated answers. It does not detect the loops seen here:

- alternating phrases such as `per second per second`;
- repeated groups of bullets;
- compound tokens such as `Request-Request-Request`;
- loops that stop before the token limit.

One sampled `DST-03` response repeated a six-word sequence 127 times but ended at 1,771 tokens. It was not truncated and could pass the current repetition check.

## Key lessons

1. Sampling was the most effective first response to greedy-decoding loops.
2. A lower adapter scale helped, but its effect is not isolated.
3. Presence penalty 1.5 removed severe loops and truncations in Experiment 06.
4. The benefit is concentrated in rare, severe failures rather than all responses.
5. Loop prevention does not improve factual accuracy.
6. A strong presence penalty conflicts with controlled vocabulary and consistent terminology.
7. Truncation must be reported beside quality scores because invalid-output filtering hides the worst cases.

## Next steps

1. Test presence penalties 0.0, 0.5, 1.0, and 1.5 with the same adapter across at least three fixed seeds.
2. Add repeated n-gram or repeated-span metrics to every prediction. Detect phrases across punctuation, not only identical adjacent tokens.
3. Keep the token limit and invalid-output reporting as containment controls.
4. Review SFT answers for duplicated bullets, repeated headings, and redundant summaries.
5. Accept a new configuration only if it removes severe loops without a meaningful decline in technical adequacy or task fulfillment.

Experiment 06 is the best anti-looping result so far, but it does not establish 1.5 as the best penalty. The next experiment should find the smallest penalty that prevents loops while preserving simple, consistent language.

## Run artifacts

- `runs/05_qwen35_4b_sft-20260818-221718-723918`
- `runs/05_qwen35_4b_sft-20260819-213410-696833`
- `runs/05_qwen35_4b_sft-20260819-222101-677229`
- `runs/05_qwen35_4b_sft-20260821-131434-300571`
- `runs/06_qwen35_4b_sft_presence_penalty-20260821-135951-794560`

See [Experiment 05 analysis](03-experiment-05-analysis.md) for the broader SFT quality comparison.
