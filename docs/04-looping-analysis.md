# Looping and penalty ablation analysis

## Summary

The SFT model produced severe loops during Experiment 05. Responses started normally, then repeated phrases or lists until they reached the 2,048-token limit.

The mitigation sequence was:

1. Replace greedy decoding with sampling.
2. Reduce the LoRA adapter scale from 0.5 to 0.25.
3. Add presence penalty 1.5 in Experiment 06.
4. Replace it with presence penalty 0.5 plus repetition penalty 1.05 in Experiment 07.

Experiments 06 and 07 both completed 100 prompts with no truncations or severe phrase loops. Experiment 07 preserved the judged quality of Experiment 06 and slightly improved semantic simplicity. It also produced fewer answers over 1,000 tokens.

Experiment 07 is therefore good enough for the main goal: comparable judged simplicity with no degenerate looping. It does not eliminate structural repetition, restore every mechanical Simple English metric to Experiment 05 levels, or preserve generation throughput.

## The problem

We observed two different forms of repetition:

- **Structural repetition:** too many headings, bullets, parallel sections, or repeated summaries. The answer remains readable but is longer and more fragmented than necessary.
- **Degenerate looping:** a token or phrase cycle takes over the response and usually continues until the output limit.

Experiment 05 exposed the second and more serious failure.

### Examples

In the first Experiment 05 attempt, `NET-06` repeated an expanding BGP phrase:

> ... from a single exit point from a single next hop from a single next hop from a single next hop ...

The response reached 2,048 tokens. One six-word sequence appeared 86 times.

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

Its most common six-word sequence appeared 578 times. Sampling also produced compound-token loops such as `X-Request-Request-Request-Request-Request-ID` in `API-07`.

## Mitigation sequence

The first three Experiment 05 attempts are incomplete. Their counts describe the saved output, not full benchmark runs.

| Run                      | Decoder and penalties                        | Saved responses | Truncated | Severe loops |
| ------------------------ | -------------------------------------------- | --------------: | --------: | -----------: |
| Experiment 05, attempt 1 | Greedy, adapter scale 0.5                    |              58 |        10 |           10 |
| Experiment 05, attempt 2 | Sampling, adapter scale 0.5                  |              72 |         5 |            5 |
| Experiment 05, attempt 3 | Sampling, adapter scale 0.25                 |              75 |         2 |            1 |
| Experiment 05, final     | Sampling, revised adapter at scale 0.25      |      99 visible |         2 |            2 |
| Experiment 06            | Same setup, presence 1.5                     |             100 |         0 |            0 |
| Experiment 07            | Same setup, presence 0.5 and repetition 1.05 |             100 |         0 |            0 |

A severe loop means that one normalized six-word sequence appears at least 15 times. This is a retrospective check and is not part of the current scorer.

### Sampling and adapter scale

Changing from greedy decoding to temperature 0.7, top-p 0.8, and top-k 20 produced the largest improvement. Sampling gives the model a chance to leave a repetitive high-probability continuation.

Reducing the adapter scale from 0.5 to 0.25 reduced loops further in the saved runs. The final Experiment 05 run also used a newly trained adapter. These effects are not isolated because the stochastic path and training artifact changed at the same time.

### Presence penalty in Experiment 06

Experiment 06 subtracts 1.5 from the logit of every token already generated. Prompt tokens are excluded. This removed severe loops, but also discouraged valid reuse of common words and technical terms.

Mechanical Simple English scores declined relative to Experiment 05, especially controlled vocabulary, sentence mechanics, and terminology consistency. Judge scores remained broadly unchanged.

### Combined penalties in Experiment 07

Experiment 07 reduces presence penalty to 0.5 and adds repetition penalty 1.05. Repetition penalty is applied first and excludes prompt tokens. Presence penalty then applies a smaller fixed reduction.

This milder combination also removed severe loops. Its judged quality was comparable to Experiment 06, but it did not recover all mechanical Simple English losses and added generation overhead.

## Experiment comparison

Experiments 06 and 07 use the same model revision, adapter, adapter scale, evaluation set, seed, sampling settings, GPU type, and token limit. Their intended difference is the penalty configuration.

Both are single stochastic runs. Paired confidence intervals measure variation across prompts, not across seeds. The final Experiment 05 prediction file also contains only 99 rows even though its summary reports 100 generations.

### Generation

| Measure                       |    Experiment 05 | Experiment 06 | Experiment 07 |
| ----------------------------- | ---------------: | ------------: | ------------: |
| Truncated responses           |                2 |             0 |             0 |
| Severe loops                  |                2 |             0 |             0 |
| Mean output tokens            |            679.2 |         643.6 |         622.3 |
| Median output tokens          |              613 |           567 |         563.5 |
| Answers at least 1,000 tokens | 18 of 99 visible |            19 |            12 |
| Mean generation time          |          35.71 s |       33.62 s |       39.06 s |
| Output throughput             |      19.02 tok/s |   19.14 tok/s |   15.93 tok/s |

Experiment 07 was shorter than Experiment 06 on 50 prompts and longer on 50. Its paired mean reduction was 21 tokens, with a confidence interval that included zero. The output-length difference is therefore small, although Experiment 07 produced fewer answers over 1,000 tokens.

Experiment 07's mean generation time was 16.2% higher and throughput was 16.8% lower than Experiment 06. The recorded wall time conflicts with the summed generation timings, so end-to-end latency is uncertain. Per-response timings still show overhead from the additional processor.

### Judge-assessed quality

| Judge dimension       | Experiment 05 | Experiment 06 | Experiment 07 |
| --------------------- | ------------: | ------------: | ------------: |
| Clarity and coherence |        90.63% |        92.68% |        92.09% |
| Semantic simplicity   |        79.21% |        83.25% |        85.46% |
| Task fulfillment      |        95.57% |        95.92% |        95.71% |
| Technical adequacy    |        71.88% |        71.43% |        70.15% |

Paired Experiment 07 changes relative to Experiment 06 were small:

| Dimension             | Paired change | Approximate 95% CI |
| --------------------- | ------------: | -----------------: |
| Clarity and coherence |      -0.51 pp |  -3.26 to +2.24 pp |
| Semantic simplicity   |      +2.06 pp |  -1.58 to +5.70 pp |
| Task fulfillment      |      -0.26 pp |  -3.17 to +2.66 pp |
| Technical adequacy    |      -1.28 pp |  -5.74 to +3.18 pp |

Every interval includes zero. Experiment 07 is comparable to Experiment 06 on all four dimensions.

Relative to Experiment 05, Experiment 07 improves paired semantic simplicity by 6.18 points. Its clarity, task fulfillment, and technical adequacy changes are inconclusive. This meets the goal if semantic simplicity is the primary measure.

Loop prevention does not fix factual errors. Several concise Experiment 07 answers still contain incorrect technical claims, and technical adequacy did not improve.

### Mechanical Simple English

For sentence length and long-sentence fraction, lower is better. For other metrics, higher is better.

| Rule metric             | Experiment 05 | Experiment 06 | Experiment 07 |
| ----------------------- | ------------: | ------------: | ------------: |
| Average sentence length |         15.37 |         16.15 |         15.32 |
| Long-sentence fraction  |        15.76% |        17.83% |        15.35% |
| Sentence mechanics      |        91.54% |        88.89% |        88.68% |
| Verb forms and modals   |        89.11% |        86.77% |        88.09% |
| Controlled vocabulary   |        54.03% |        50.30% |        49.60% |
| Terminology consistency |        84.69% |        77.00% |        78.00% |

Experiment 07 is mechanically comparable to Experiment 06. No paired rule change between them is conclusive.

Compared with Experiment 05, sentence length and long-sentence fraction are unchanged, while semantic simplicity improves. Controlled vocabulary declines by 4.38 paired points and sentence mechanics by 2.94 points; both declines are statistically clear. Experiment 07 therefore does not meet a strict requirement that every Simple English metric match Experiment 05.

## Looping verdict

### Degenerate looping is addressed

Experiment 07 produced:

- no truncations;
- no six-word sequence repeated 15 times;
- a maximum six-word repetition count of six;
- no expanding phrase or repeated-list cycles like those in Experiment 05.

The repeated lines found in `API-07` and `CLD-09` are legitimate command examples, such as HTTP headers and Kubernetes RBAC declarations.

### Structural repetition remains

Seven Experiment 07 answers received semantic-simplicity scores of 0.5 because of excessive sections, repeated summaries, or duplicated comparison structures. Experiment 06 had nine such answers, including one scored 0.25.

The worst structural tail improved, but the problem remains in answers such as `NET-01`, `API-06`, `PLR-04`, and `ARC-03`. Token penalties are not a reliable way to control whole-answer organization.

## Detection gap

The current validity check detects one token repeated eight times in a row and rejects truncated answers. It misses:

- alternating phrases such as `per second per second`;
- repeated groups of bullets;
- compound tokens such as `Request-Request-Request`;
- loops that stop before the token limit.

One Experiment 05 `DST-03` response repeated a six-word sequence 127 times but ended at 1,771 tokens. A repeated n-gram or repeated-span metric must be added before judge scoring.

## Conclusions and next steps

1. Sampling was the most effective first fix for greedy-decoding loops.
2. Experiments 06 and 07 both removed degenerate loops on this 100-prompt run.
3. Experiment 07 preserved judged simplicity and quality with milder penalties.
4. Experiment 07 still trails Experiment 05 on controlled vocabulary and sentence mechanics.
5. Structural repetition and factual accuracy require separate interventions.
6. Experiment 07 is slower than Experiment 06 because of the additional logits processor.

Further penalty tuning is not necessary for the current goal. Experiment 07 is a reasonable stopping point if semantic simplicity and loop prevention are primary and the throughput cost is acceptable. If latency matters more, Experiment 06 remains the faster tested configuration.

The remaining work should focus on direct repetition detection and training examples that demonstrate concise whole-answer structure. Increasing token penalties is unlikely to solve headings, duplicated summaries, or factual errors.

## Run artifacts

- `runs/05_qwen35_4b_sft-20260818-221718-723918`
- `runs/05_qwen35_4b_sft-20260819-213410-696833`
- `runs/05_qwen35_4b_sft-20260819-222101-677229`
- `runs/05_qwen35_4b_sft-20260821-131434-300571`
- `runs/06_qwen35_4b_sft_presence_penalty-20260821-135951-794560`
- `runs/07_qwen35_4b_sft_combined_penalties-20260821-171407-507484`

See [Experiment 05 analysis](03-experiment-05-analysis.md) for the broader SFT quality comparison.
