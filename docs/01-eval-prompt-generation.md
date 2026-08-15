# Eval and SFT Dataset Generation Guide

## Goal

Create datasets that improve and measure simple technical writing without trading away correctness, completeness, or safety.

## Common principles

Apply these principles to both datasets:

1. **Correctness is a gate, not a weighted preference.** A simple but materially incorrect answer always fails.
2. **Task completion comes before style.** Preserve every requested topic, command, parameter, prerequisite, caveat, and safety condition.
3. **Evaluate simplicity across the complete answer.** Sentence length and output tokens are diagnostics, not sufficient measures of simplicity.
4. **Use authoritative grounding.** Never invent a tool, product name, command, API, architecture, or technical fact.
5. **Match length to the task.** A definition can be short. A runbook must remain as long as necessary.
6. **Preserve literal technical content.** Do not alter code blocks, identifiers, CLI commands, file paths, quoted errors, or product names.
7. **Keep responses user-facing.** Exclude chain-of-thought, classification commentary, self-check reports, editing notes, and references to system instructions.
8. **Prevent leakage.** Keep near-duplicates, paraphrases, and prompts derived from the same source in one split. Never reuse eval prompts for training.
9. **Record provenance.** Keep the source, generation method, verification result, and reviewer decision for every example.

## Evaluation dataset

### Purpose

Generate realistic prompts that evaluate technical explanations and documentation. Favor explanations, at about two thirds, over documentation tasks, at about one third.

### Coverage

Use 10 prompts from each subject:

- Networking and internet
- Machine learning
- Data formats and APIs
- Databases and data systems
- Distributed systems
- Security and identity
- Cloud and infrastructure
- Programming languages and runtimes
- Developer tools
- Software architecture

Maintain these totals:

| Dimension               | Distribution                      |
| ----------------------- | --------------------------------- |
| Difficulty              | 30 easy, 40 medium, 30 hard       |
| Expected length         | 30 short, 40 medium, 30 long      |
| Technical terminology   | 30 minimal, 40 moderate, 30 heavy |
| Oversimplification risk | 30 low, 35 medium, 35 high        |

Do not make hard, long, terminology-heavy, and high-risk synonymous. Test varied combinations.

Oversample known failure modes:

- Comparisons that invite false symmetric claims
- Procedures where brevity can remove required steps
- Questions that require qualifications or exceptions
- Exact CLI, database, and configuration instructions
- Requests that combine explanation and procedure
- Prompts with first-person language
- Requests with explicit format or length constraints
- Prompts that a model can incorrectly classify as non-technical

### Prompt styles

Mix natural ways that people ask for help:

- Direct definitions and comparisons
- “Why” and “how” questions
- Misconception correction
- Troubleshooting from an observed symptom
- Practical decisions and tradeoffs
- Requests tailored to beginners, operators, or engineers
- Getting-started guides, architecture overviews, concept guides, migrations, runbooks, and troubleshooting documentation

Phrase prompts as genuine user requests, not labels from a test plan. Avoid repeatedly starting with “Explain” or “Write.” Give each prompt one clear primary purpose.

### Grounding

- For real tools, use stable facts from authoritative documentation.
- For user-provided or repository projects, use only supplied or discoverable facts.
- If product facts are unavailable, request a generic template or keep the request high-level.
- Documentation prompts must identify the intended artifact, audience, and scope when relevant.
- Avoid near-duplicates and superficial rewrites of the same question.
- High-risk prompts must require caveats, boundary conditions, tradeoffs, uncertainty, or correction of a tempting but incomplete answer.

### Scoring and model selection

Score these dimensions separately:

- Technical adequacy
- Task fulfillment
- Clarity and coherence
- Semantic simplicity
- Mechanical language compliance

Use a correctness-constrained evaluation:

1. Reject responses with material technical errors.
2. Reject responses that omit required parts of the task.
3. Compare simplicity only among the remaining responses.

Track output length, sentence length, paragraph limits, and banned forms as diagnostics. Do not combine them into a score that can allow a short, incorrect response to win.

Report at least:

- Material-error rate
- Technical-adequacy pass rate
- Task-fulfillment pass rate
- Semantic simplicity among correct and complete responses
- All-dimension pass rate
- Mean and median output tokens
- Truncation rate
- System-prompt commentary rate
- Results by subject and prompt type

Use paired comparisons on the same prompts. Keep generation settings fixed. Use multiple independent judge passes when practical, and send disagreements or high-risk answers for human review.

### Output

Write one JSON object per line:

```json
{ "id": "NET-01", "prompt": "What is the difference between IPv4 and IPv6?" }
```

Every row must contain exactly `id` and `prompt`.

## SFT dataset

### Purpose

Teach the instruction-tuned model to produce answers that are correct, complete, cohesive, direct, and proportionately concise. The SFT dataset must establish this output distribution without teaching minimum length as the goal.

### Initial size and split

Start with:

- 2,000–5,000 training examples
- About 1–3 million assistant-output tokens
- 85–90% training data
- 5–10% validation data
- 5–10% internal test data

Split by source, topic, and prompt family. Keep the existing 100-prompt evaluation set as an untouched external holdout.

### Prompt coverage

Keep about half of the SFT prompts aligned with the evaluation subjects and failure modes, but use different prompts and source material. Use the remaining capacity for the broader subject and topic catalog in `data/sft_topics.json`. This adds writing variety without weakening technical coverage. Include definitions, comparisons, architecture decisions, troubleshooting, security guidance, code and CLI instructions, installation guides, and runbooks.

Use this initial prompt-intent mix:

| Prompt intent             | Share |
| ------------------------ | ----: |
| Explanation               |   25% |
| Documentation              |   20% |
| Troubleshooting            |   15% |
| Procedure                  |   15% |
| Comparison                 |   15% |
| Misconception correction   |   10% |

### Target generation

For each prompt, create a factual specification before generating prose:

- Required facts and requested sections
- Necessary caveats and safety conditions
- Valid commands or code
- Known false or prohibited claims
- Appropriate answer type and approximate length

Generate several candidates from strong teacher models and the current model. Include complete, concise, and moderately detailed variants. Select or edit the shortest cohesive candidate that preserves the factual specification.

Do not automatically use a teacher response or a previous base-model response as the target. Both require verification.

### Target length

Use task-dependent guidance rather than a universal limit:

| Task type                           |      Typical target |
| ----------------------------------- | ------------------: |
| Definition or simple explanation    |       50–200 tokens |
| Moderate technical explanation      |      150–400 tokens |
| Comparison or architecture overview |      250–700 tokens |
| Troubleshooting procedure           |      300–900 tokens |
| Installation guide or runbook       | As long as required |

At least 20% of targets must exceed 500 tokens. This protects complex procedures from a systematic brevity bias.

### Target acceptance

Accept a target only if it:

- Contains no material factual error
- Fulfills every explicit request
- Preserves necessary detail and qualifications
- Contains valid commands and code
- Uses cohesive paragraphs and proportionate structure
- Avoids redundant headings, summaries, and examples
- Avoids repeated short sentence patterns
- Does not mention classification, writing rules, or system instructions

Use the strongest available verification method:

- Execute code and commands in a safe environment.
- Apply deterministic validators to structured output.
- Check factual claims against authoritative sources.
- Use two independent judges for subjective cases.
- Require human review for disagreements and high-risk material.

Store only the final user-facing response as the assistant target. Exclude teacher reasoning and review notes.

### Training format

Use the exact chat template and short system message planned for production. For example:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Why does training use more memory than inference?"
    },
    {
      "role": "assistant",
      "content": "Training stores activations for backpropagation, parameter gradients, and optimizer states. Inference does not need most of these tensors. Training also often uses larger batches."
    }
  ]
}
```

Apply training loss only to assistant tokens. Ensure that the complete prompt and response fit within the training context window without truncation.

Keep provenance and verification metadata in a separate manifest if the trainer requires rows that contain only `messages`.
