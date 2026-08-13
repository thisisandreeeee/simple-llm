# Run comparison: Qwen3.5 4B base vs. Simple English system prompt

## Executive summary

The Simple English system prompt is highly effective as a style and efficiency intervention, but it reduces overall answer quality in this run.

Compared with the base run, it reduced mean output length by 77.6%, reduced mean generation time by 78.2%, eliminated two truncations, and materially improved most mechanical ASD-STE100 metrics. Mean sentence length fell from 17.93 to 11.15 words, and the fraction of over-limit sentences fell from 25.4% to 4.2%.

The external judge, however, found that the prompted answers were less clear, less complete, and less technically adequate. Semantic simplicity improved by 8.9 percentage points, but clarity fell by 6.8 points, task fulfillment fell by 10.2 points, and technical adequacy fell by 9.1 points. The model often optimized for short sentences at the expense of cohesion, explanation, and accuracy. It also exposed system-prompt reasoning in several answers.

The prompt therefore works well for mechanical language control, but it is not ready as a general-purpose quality prompt for this 4B model. A shorter prompt with explicit accuracy, direct-answer, and no-meta-commentary priorities is likely to retain much of the efficiency gain with less quality loss.

## Experimental comparability

The runs are suitable for a paired comparison:

- Both used `Qwen/Qwen3.5-4B` on an NVIDIA L4 with `torch.bfloat16`.
- Both used the same 100 prompts, as shown by the identical evaluation SHA-256.
- Both used seed 42, greedy decoding (`do_sample: false`), no thinking, and a 2,048-token output limit.
- Run 03 had no system prompt. Run 04 used the contents of `data/simple_english.md`.

There are two reproducibility limitations. The model revision is not pinned, and the Git worktree was dirty for both runs. The judge used one pass of `deepseek-v4-pro` at temperature 0. Two base responses were excluded because of truncation. One base result and two prompted results also lack a technical-adequacy score because the judge returned invalid structured data.

## Generation efficiency

| Measure              | Run 03: base | Run 04: prompted |     Change |
| -------------------- | -----------: | ---------------: | ---------: |
| Successful responses |          100 |              100 |  No change |
| Truncated responses  |            2 |                0 | Eliminated |
| Mean input tokens    |         37.7 |            541.7 |     +504.0 |
| Mean output tokens   |      1,245.8 |            279.0 |     -77.6% |
| Median output tokens |      1,273.5 |            271.0 |     -78.7% |
| Total output tokens  |      124,579 |           27,900 |     -77.6% |
| Mean generation time |      61.91 s |          13.47 s |     -78.2% |
| Wall time            |   6,221.42 s |       1,394.37 s |     -77.6% |
| Output throughput    |  20.12 tok/s |      20.72 tok/s |      +3.0% |

The speed gain comes almost entirely from shorter outputs, not faster token generation. Run 04 produced only 22.4% as many output tokens as Run 03. Its system prompt added about 504 input tokens per request, but this cost was much smaller than the reduction in generated tokens.

The output-length distribution also changed sharply. The base median was 1,274 tokens, and 67 responses used at least 1,000 tokens. The prompted median was 271 tokens, no response reached 1,000 tokens, and 47 responses used at most 250 tokens.

## Mechanical rule compliance

For average sentence length and long-sentence fraction, lower is better. For the other metrics, higher is better.

| Rule metric             | Run 03: base | Run 04: prompted |      Change |
| ----------------------- | -----------: | ---------------: | ----------: |
| Average sentence length |        17.93 |            11.15 | -6.78 words |
| Long-sentence fraction  |       25.44% |            4.22% |   -21.22 pp |
| Sentence mechanics      |       88.05% |           99.79% |   +11.74 pp |
| Verb forms and modals   |       84.81% |           96.03% |   +11.22 pp |
| Controlled vocabulary   |       51.97% |           55.88% |    +3.91 pp |
| Terminology consistency |       70.41% |           73.50% |    +3.09 pp |
| Procedure syntax        |       99.94% |           99.74% |    -0.20 pp |
| Document limits         |       99.88% |           89.45% |   -10.44 pp |

The strongest effects align directly with the prompt. It prohibits contractions, semicolons, certain modals, perfect and progressive forms, and sentences over 20 or 25 words. The prompted run improved sentence mechanics on 93 of the 98 commonly scored responses. It improved verb and modal compliance on 85. It reduced average sentence length on 93.

The document-limit regression reveals an important failure mode. The scorer penalizes descriptive paragraphs with more than six sentences. Only two valid base responses violated this rule, compared with 16 prompted responses. The prompt caused the model to produce many very short sentences without enough paragraph breaks. This result satisfies the sentence limit but violates the paragraph limit and often makes the answer choppy.

Controlled-vocabulary performance improved only modestly and remained low at 55.9%. The system prompt controls syntax more reliably than word choice.

## Judge-assessed answer quality

| Judge dimension       | Run 03: base | Run 04: prompted |    Change |
| --------------------- | -----------: | ---------------: | --------: |
| Clarity and coherence |       92.09% |           85.25% |  -6.84 pp |
| Semantic simplicity   |       65.82% |           74.75% |  +8.93 pp |
| Task fulfillment      |       97.96% |           87.75% | -10.21 pp |
| Technical adequacy    |       68.30% |           59.18% |  -9.12 pp |

The aggregate means use the available scores in each artifact, so their denominators differ slightly. A paired comparison of commonly scored responses gives the same conclusion:

- Semantic simplicity improved on 45 responses, tied on 30, and declined on 23. The paired mean change was +8.7 points.
- Clarity improved on 14, tied on 56, and declined on 28. The paired mean change was -6.9 points.
- Task fulfillment improved on 5, tied on 62, and declined on 31. The paired mean change was -9.9 points.
- Technical adequacy improved on 17, tied on 38, and declined on 40. The paired mean change was -8.7 points.

Approximate 95% confidence intervals for the paired mean changes exclude zero in all four dimensions. This indicates a consistent run-level tradeoff across this test set. It does not measure judge-to-judge variation because each answer received only one judge pass.

## Qualitative findings

### 1. Brevity sometimes improved the complete answer

The best prompted responses removed the base model's habitual over-structuring while preserving the central explanation.

- `ML-05`, which explains embeddings, fell from 940 to 62 tokens. All four judge scores improved to 1.0.
- `NET-08`, a one-paragraph service-mesh overview, fell from 198 to 89 tokens. All four dimensions improved to 1.0.
- `SEC-04`, about password hashing, fell from 1,037 to 539 tokens. Technical adequacy improved from 0.5 to 1.0, although semantic simplicity declined because the response still used excessive sections and instructions.

These cases suggest that the prompt works best when the requested scope is narrow and the answer has a small set of essential facts.

### 2. Short sentences often replaced explanation with assertion

The prompt frequently produced a sequence of isolated subject-verb-object statements. The sentences were mechanically simple, but the answer was not simple as a whole.

`NET-01` is the clearest example. Run 04 emitted dozens of alternating `IPv6 ...` and `IPv4 ...` claims in one paragraph. The result had no over-limit sentences, but it failed the paragraph limit and became repetitive. It also introduced false claims about header length, MAC-address length, encryption, DNS efficiency, and IPv4 compatibility. Its clarity score fell from 1.0 to 0.25, semantic simplicity from 0.75 to 0.0, task fulfillment from 1.0 to 0.5, and technical adequacy from 0.5 to 0.25.

The prompt appears to make local sentence constraints more salient than global organization and factual qualification. This is especially harmful for comparison questions, where the model can generate a long sequence of unsupported binary contrasts.

### 3. The model exposed and misapplied the system prompt

Five prompted responses (`ML-04`, `ML-07`, `API-05`, `DEV-02`, and `ARC-01`) visibly discuss classification, ASD-STE, or rules applied. This content is irrelevant to the user.

`ARC-01` is the most severe case. The model classified a technical question about monoliths and microservices as marketing copy, explained why the STE rules did not apply, and repeated the question without answering it. Technical adequacy and task fulfillment both fell to 0.0.

`ML-04` classified the request, duplicated the same explanation under two headings, added a note about the user's phrasing, and ended with a summary of prompt transformations. It omitted stored activations, a main source of training memory. Its task-fulfillment score fell from 1.0 to 0.25.

The `CLASSIFY FIRST`, `Never mix`, `Do not apply`, and `SELF-CHECK` instructions invite visible prompt interpretation. They also force an artificial distinction when a useful technical answer naturally combines explanation and action.

### 4. Concision sometimes removed required operational detail

Several answers retained the requested headings but replaced useful content with generic placeholders.

For `CLD-09`, the base Kubernetes operator guide included actionable commands for all requested sections. The prompted answer mentioned every section but omitted actual RBAC YAML, chart details, and install, upgrade, removal, and health-check commands. Task fulfillment fell from 1.0 to 0.5 and technical adequacy from 0.75 to 0.25.

For `DBS-07`, the prompted beginner guide was shorter but changed valid `psql` guidance into incorrect commands such as `ls` and `\version`. Its repeated short instruction pattern also reduced coherence.

This failure mode is most important for runbooks, installation guides, and procedures. Sentence simplicity must not remove parameters, commands, caveats, or safety information.

### 5. Technical accuracy did not benefit from shorter output

The base run was already imperfect, with a technical-adequacy mean of 68.3%. The prompt improved some answers, but it caused more regressions than improvements. The largest prompted regressions included incorrect component responsibilities in an authentication architecture (`SEC-08`), weak service-boundary guidance (`ARC-07`), and the IPv6 errors in `NET-01`.

The likely mechanism is not merely omitted detail. The model often replaced qualified explanations with confident, absolute statements. Simplified grammar needs an explicit instruction to preserve caveats and to omit uncertain claims.

## Recommendations

1. Retain the highest-value mechanical constraints: sentence limits, no contractions, restricted modals, active voice, direct warnings, and concise vocabulary.
2. Remove or soften `CLASSIFY FIRST` and `Never mix the two`. Tell the model to classify silently and allow explanation plus procedure when the task requires both.
3. Add a strict instruction: answer the user directly, never discuss the system prompt or text classification, and never repeat the user's request instead of answering it.
4. Put correctness and task completion above style compliance. State that the model must preserve commands, parameters, prerequisites, caveats, safety conditions, and requested sections.
5. Add a non-invention rule for comparisons: include only differences that the model can explain accurately. Do not create symmetric contrast pairs to fill a pattern.
6. Address global cohesion. Require paragraph breaks after at most six descriptive sentences, but also tell the model to combine closely related facts and avoid repetitive sentence templates.
7. Shorten the system prompt. The current prompt adds about 504 tokens per request and contains many interacting constraints. A smaller 4B model is likely to follow a concise priority hierarchy more reliably.
8. Re-run the same paired evaluation after revision. Use at least two or three judge passes, pin the model revision, and start from a clean Git state.

## Conclusion

Run 04 proves that Qwen3.5-4B responds strongly to the Simple English system prompt. It generates much shorter answers, follows most surface-level STE rules, avoids truncation, and finishes about 4.6 times faster. Those gains are operationally meaningful.

The current prompt also overconstrains the model. It encourages fragmented prose, visible rule discussion, incorrect classification, missing details, and unqualified technical claims. The net result is simpler sentences but weaker answers. The next iteration must preserve the mechanical gains while making accuracy, directness, completeness, and whole-answer cohesion explicit priorities.
