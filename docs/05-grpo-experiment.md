# GRPO experiment: improving on the SFT adapter

## Introduction

The purpose of this experiment is to evaluate whether Group Relative Policy
Optimization (GRPO) can improve performance on top of the existing Qwen3.5-4B
SFT LoRA adapter. The target is not only better writing quality, but better
writing without reducing technical correctness or task fulfillment.

## Reward functions

We experimented with two reward functions using correctness (`C`), semantic
simplicity (`S`), clarity and coherence (`H`), and ASD-STE100-style compliance
(`A`).

### Multiplicative reward

```text
R = C(0.70 + 0.15S + 0.15A)
```

Multiplying by correctness prevents a useless but well-written answer from
receiving a high reward. However, correctness and style remain continuously
exchangeable: an answer with `C=0.75` and perfect style receives `0.75`, while
an answer with `C=1` and zero style receives `0.70`. The reward can also favor
longer answers if the judge interprets optional detail as greater correctness.

### Gated reward

Define writing quality as

```text
Q = H(0.6S + 0.4A).
```

Then use

```text
         Q       if C = 1
R = {
        -(1-C)   otherwise.
```

Correctness uses four severity grades: `1.0` for a pass, `0.75` for a localized
failure, `0.5` for a major failure, and `0.0` for an unusable answer. Any
passing answer therefore outranks any failing answer; writing quality ranks
only passing answers, while failure severity ranks the rest. Optional detail
has no inherent reward once the prompt is fulfilled.

The gated reward replaces the multiplicative reward and is the reward used for
the rest of this document. Its ordering is better aligned with the objective,
but it still depends on the correctness judge and cannot guarantee that shared
model updates preserve correctness on unseen prompts.

## Experiment results

Run 07 is the baseline: `07_qwen35_4b_sft_combined_penalties-20260821-171407-507484`.

Run 08 evaluates the gated GRPO adapter: `08_qwen35_4b_grpo-20260913-155919-398168`.

Both runs used the same 100 prompts, model revision, adapter scale (`0.25`),
seed, and recorded decoding configuration. Neither run had generation
failures or truncations.

| Metric                 | Run 07 SFT | Run 08 GRPO | GRPO - SFT |
| ---------------------- | ---------: | ----------: | ---------: |
| Judge macro-average    |     0.8585 |      0.8635 |    +0.0049 |
| Technical adequacy     |     0.7015 |      0.7071 |    +0.0055 |
| Task fulfillment       |     0.9571 |      0.9592 |    +0.0021 |
| Clarity and coherence  |     0.9209 |      0.9304 |    +0.0095 |
| Semantic simplicity    |     0.8546 |      0.8571 |    +0.0026 |
| Mean output tokens     |      622.3 |       635.7 |      +13.4 |
| Mean sentence length   |      15.32 |       16.30 |      +0.98 |
| Long-sentence fraction |      0.153 |       0.181 |     +0.027 |

### Key insights

- GRPO did not produce a reliable overall improvement. On the 95 prompts with
  complete scores for both models, its mean composite gain was only `0.0053`,
  with a paired bootstrap 95% interval of approximately
  `[-0.0158, 0.0257]`.
- GRPO won 42 paired prompts, lost 28, and tied 25. The movement was real but
  uneven rather than a broad improvement.
- Both models had 38 answers with technical adequacy at or below `0.5`. GRPO
  changed which prompts failed but did not reduce the weak technical tail.
- GRPO worked best when it became shorter. Shorter GRPO answers improved by
  `0.0426` on average; longer answers declined by `0.0270`. Extra detail often
  introduced unsupported claims.
- Results varied by domain. GRPO improved most in ML (`+0.050` composite) but
  declined in architecture (`-0.044`), developer tooling (`-0.027`), and
  networking (`-0.019`). Each domain has only ten prompts, so these are
  diagnostic signals rather than stable rankings.
- Rule-based style results were mixed. GRPO improved the narrow terminology
  consistency check, but produced longer sentences, more long sentences, and
  no paired improvement in judged semantic simplicity.

### Examples from the predictions

`DST-08`, distributed key-value store, is a clear GRPO success. GRPO reduced
the response from 608 to 274 tokens, used consistent hashing, and removed SFT's
incorrect definition of strong consistency. Technical adequacy rose from `0.5`
to `1.0`.

`ARC-01`, monolith versus microservices, shows the same positive mode. GRPO
reduced the answer from 790 to 375 tokens and focused on domain boundaries,
team ownership, measured operational friction, and independent scaling. It
removed SFT's misleading observability and team-size rules.

`ARC-04`, control plane versus data plane, is a severe regression. SFT gave a
correct 232-token answer. GRPO expanded to 508 tokens and claimed that the
control plane performs per-packet route lookup, firewall checks, and QoS
enforcement, which are data-plane operations.

`ARC-10`, architecture decision records, shows that verbosity did not guarantee
coverage. GRPO used essentially the same number of tokens as SFT but omitted
the explicitly requested considered-alternatives section.

`DBS-03`, transaction isolation, remained unreliable. Both models scored
`0.25` for technical adequacy, but GRPO added 410 tokens containing incorrect
database defaults, nonexistent implementation details, and unsupported
performance multipliers.

These examples support the reward-design concern: concise answers can remove
opportunities for error, while additional detail can make a fluent response
less reliable. They do not prove that the reward alone caused the outcome. The
saved evaluation does not contain the training rollouts, component rewards, or
trainer logs needed for that causal claim.

## Next steps

The Run 08 checkpoint should not replace Run 07 SFT as the default. The gated
reward has the right preference structure, but this run did not convert that
structure into a dependable model-wide gain.

Before another full GRPO run:

1. Calibrate the correctness judge on edited answer pairs. Optional detail must
   not increase correctness, while factual errors, unsafe guidance, and missing
   requested sections must fail the gate.
2. Validate the complete reward-to-advantage path, including GRPO group
   normalization, rather than checking raw reward ordering alone.
