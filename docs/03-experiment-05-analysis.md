# Experiment 05 analysis: Qwen3.5 4B base vs. SFT

## Summary

Experiment 05 produced simpler, shorter answers than the base model without a clear loss of correctness.

Compared with Run 03, the SFT configuration:

- improved paired semantic simplicity by 14.0 percentage points;
- improved paired technical adequacy by 4.6 points;
- reduced mean output length by 45.5%;
- reduced mean generation time by 42.3%;
- improved most mechanical Simple English metrics.

Clarity and task fulfillment fell by 1.6 and 1.9 paired points, but neither change was conclusive. The SFT configuration also performed better than the system prompt in Run 04, which improved sentence-level style but reduced answer quality.

The main weakness is factual reliability. Some Experiment 05 answers are clear and concise but contain important technical errors. The model also remains too fond of headings, lists, and repeated summaries.

This experiment does not isolate the effect of SFT. Run 05 also changes decoding, model provenance, and adapter scale. Its results describe the complete configuration: a scale-0.25 SFT adapter with stochastic sampling.

## Experiment setup

Runs 03, 04, and 05 use:

- `Qwen/Qwen3.5-4B` on an NVIDIA L4;
- `torch.bfloat16`, seed 42, and disabled thinking;
- the same 100 prompts and evaluation hash;
- a 2,048-token output limit.

The important differences are:

| Run | Condition | Decoding |
| --- | --- | --- |
| 03 | Base model, no system prompt | Greedy |
| 04 | Base model with Simple English system prompt | Greedy |
| 05 | Rank-16 LoRA adapter merged at scale 0.25 | Temperature 0.7, top-p 0.8, top-k 20 |

Run 05 uses pinned model revision `851bf6e...`. Runs 03 and 04 do not record a revision, so exact base-weight equality cannot be verified. Run 05 also uses 901 training examples and 99 held-out training-evaluation examples. No benchmark prompt exactly matches either SFT split.

These differences prevent a clean causal claim about SFT alone. A later experiment must use the same model revision and decoder for every adapter condition.

## Artifact limitations

The final Run 05 artifacts do not fully agree:

- `summary.json` reports 100 generations and 67,922 output tokens.
- `predictions.jsonl` contains 99 rows and is missing `ARC-10`.
- `rule_scores.json` contains `ARC-10` and reports 100 prompts.
- `judge_scores.json` covers the 99 visible predictions.

Generation totals below come from `summary.json`. Row-level comparisons use only prompts present in both runs.

`DBS-10` and `CLD-07` reached the token limit and were excluded from scoring. Five judge dimensions also failed across `DEV-01` and `DEV-02`. Paired judge comparisons therefore contain 93 or 94 answers, depending on the dimension.

Each answer received one deterministic pass from `deepseek-v4-pro`. Confidence intervals measure variation across prompts, not judge or decoding variance.

## Results

### Efficiency

| Measure | Run 03: base | Run 04: prompted | Run 05: SFT | SFT vs. base |
| --- | ---: | ---: | ---: | ---: |
| Truncated responses | 2 | 0 | 2 | No change |
| Mean input tokens | 37.7 | 541.7 | 37.6 on visible rows | No material change |
| Mean output tokens | 1,245.8 | 279.0 | 679.2 | -45.5% |
| Median output tokens | 1,273.5 | 271.0 | 613 on visible rows | -51.9% |
| Mean generation time | 61.91 s | 13.47 s | 35.71 s | -42.3% |
| Wall time | 6,221.42 s | 1,394.37 s | 3,601.61 s | -42.1% |

SFT learned substantial concision without Run 04's 504-token system-prompt overhead. It found a middle point between the long base answers and the aggressive brevity of the prompted answers.

The result is still uneven. Among the 99 visible Run 05 answers, 18 use at least 1,000 tokens and two truncate. See [Looping and Experiment 06 analysis](04-looping-analysis.md) for the repeated-output failures and later mitigations.

### Mechanical Simple English

For sentence length and long-sentence fraction, lower is better. For the other metrics, higher is better.

| Rule metric | Run 03: base | Run 04: prompted | Run 05: SFT | SFT vs. base |
| --- | ---: | ---: | ---: | ---: |
| Average sentence length | 17.93 | 11.15 | 15.37 | -2.56 words |
| Long-sentence fraction | 25.44% | 4.22% | 15.76% | -9.68 pp |
| Sentence mechanics | 88.05% | 99.79% | 91.54% | +3.49 pp |
| Verb forms and modals | 84.81% | 96.03% | 89.11% | +4.29 pp |
| Controlled vocabulary | 51.97% | 55.88% | 54.03% | +2.05 pp |
| Terminology consistency | 70.41% | 73.50% | 84.69% | +14.29 pp |
| Document limits | 99.88% | 89.45% | 100.00% | +0.12 pp |

SFT improved sentence length and most rule metrics, but less strongly than the system prompt. In return, it avoided Run 04's document-limit failure, where many short sentences formed long, fragmented paragraphs.

The terminology gain is concentrated: 75 of 96 paired answers tie, 18 improve, and three decline. Controlled vocabulary remains weak at 54.0%. The model learned sentence shape more reliably than word choice.

### Judge-assessed quality

| Judge dimension | Run 03: base | Run 04: prompted | Run 05: SFT | SFT vs. base |
| --- | ---: | ---: | ---: | ---: |
| Clarity and coherence | 92.09% | 85.25% | 90.63% | -1.47 pp |
| Semantic simplicity | 65.82% | 74.75% | 79.21% | +13.39 pp |
| Task fulfillment | 97.96% | 87.75% | 95.57% | -2.39 pp |
| Technical adequacy | 68.30% | 59.18% | 71.88% | +3.58 pp |

Aggregate denominators differ because of missing and invalid scores. Paired results are more reliable:

| Dimension | Paired change, SFT - base | Approximate 95% CI | Improved / tied / declined |
| --- | ---: | ---: | ---: |
| Clarity and coherence | -1.60 pp | -5.29 to +2.10 pp | 15 / 59 / 20 |
| Semantic simplicity | +13.98 pp | +8.98 to +18.97 pp | 48 / 36 / 9 |
| Task fulfillment | -1.86 pp | -4.23 to +0.51 pp | 5 / 79 / 10 |
| Technical adequacy | +4.57 pp | +0.02 to +9.12 pp | 34 / 37 / 22 |

The strongest result is semantic simplicity. Its confidence interval is well above zero, and more than five times as many answers improve as decline.

Technical adequacy also improves, but its interval only just excludes zero. The result is promising rather than decisive because the comparison changes decoding, uses one judge pass, and tests several dimensions.

Relative to Run 04, SFT recovers much of the quality lost to prompt-only style control. Paired technical adequacy improves by 12.5 points, task fulfillment by 8.3 points, and clarity by 5.5 points. SFT is therefore the better of the two tested approaches for balancing simple language with useful answers.

## Where SFT worked

The best responses became much shorter without losing the requested explanation.

- `API-04` defines API idempotency in 264 tokens instead of 979. All judge dimensions improve, and technical adequacy reaches 1.0.
- `ML-05` explains embeddings in 259 tokens instead of 940. Clarity and simplicity reach 1.0.
- `ARC-02` explains dependency inversion in 220 tokens instead of 957. Clarity, simplicity, and task fulfillment improve, although it incorrectly states that SOLID has four principles.

Run 05 also avoids Run 04's most serious behavior: exposing or debating the system instructions instead of answering. For example, `ARC-01` recovers from zero technical adequacy and task fulfillment in Run 04 to 1.0 on both.

These answers show the desired pattern: answer directly, keep enough explanation, and do not discuss the writing rules.

## Where SFT failed

### Factual accuracy

Twenty-two of 93 paired answers decline in technical adequacy. Several errors concern exact rules and failure conditions:

- `API-03` gives incorrect Protocol Buffers reservation and field-removal rules.
- `DST-07` treats false sharing as a distributed-lock problem and omits lease expiry, fencing, pauses, and partitions.
- `CLD-09` gives unsafe or incomplete Kubernetes operator lifecycle instructions.

These are not language failures. The model presents incorrect details clearly. More training data will help only if the examples preserve exact invariants, exceptions, and safe procedures.

### Whole-answer simplicity

SFT learned shorter sentences more reliably than shorter information design. It still overuses headings, bullets, and closing summaries.

`NET-01` uses 13 numbered sections and 1,048 tokens for an IPv4 and IPv6 comparison. `ARC-01` gives a strong answer, but six sections repeat feature toggles, measurement, and decision criteria. Both lose semantic-simplicity points for excessive structure.

A length target alone may remove useful detail. Training examples must show how to select, group, and stop.

## Key lessons

1. SFT is more effective than the current system prompt at balancing simple language and answer quality.
2. The strongest gain is semantic simplicity, followed by a smaller technical-adequacy signal.
3. The experiment does not isolate SFT because decoder settings and model provenance also changed.
4. Sentence-level style improved more than controlled vocabulary or whole-answer structure.
5. Clear writing can hide factual errors. Correctness must remain a separate evaluation dimension.
6. Truncation, repetition, and inconsistent artifacts must be reported beside average scores.
7. The current benchmark covers ten technical domains and does not establish performance on the general domains in the SFT data.

## Next steps

1. **Run a controlled ablation.** Pin one base revision and decoder. Compare adapter scales 0.0, 0.25, 0.5, and 1.0 across several fixed seeds.
2. **Target precise technical failures.** Add reviewed examples for protocol evolution, distributed leases and fencing, and safe upgrade or removal procedures.
3. **Teach global concision.** Use examples that remove redundant sections and summaries without dropping requested content.
4. **Strengthen evaluation.** Track factual errors, missing qualifications, unsafe instructions, redundancy, and mechanical rule failures separately. Use more than one judge pass for close correctness claims.
5. **Add regression gates.** Require a simplicity gain, no meaningful loss in technical adequacy or task fulfillment, fewer truncations, and consistent artifact IDs and totals.
6. **Expand the benchmark.** Add held-out prompts from the general domains represented in training.

Experiment 05 is a successful first SFT result, but it is not evidence that correctness is solved. The next experiment should preserve its simplicity gain while isolating the adapter effect and targeting the factual and structural failures that remain.

## Run artifacts

- `runs/03_qwen35_4b_base-20260812-212130-695745`
- `runs/04_qwen35_4b_simple_english-20260812-212154-599693`
- `runs/05_qwen35_4b_sft-20260821-131434-300571`
