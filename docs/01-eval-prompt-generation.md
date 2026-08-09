# Eval Prompt Generation Guide

## Goal

Generate realistic prompts that evaluate technical explanations and documentation. Favor explanations (about two thirds) over documentation tasks (about one third).

## Coverage

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

## Prompt styles

Mix natural ways that people ask for help:

- Direct definitions and comparisons
- “Why” and “how” questions
- Misconception correction
- Troubleshooting from an observed symptom
- Practical decisions and tradeoffs
- Requests tailored to beginners, operators, or engineers
- Getting-started guides, architecture overviews, concept guides, migrations, runbooks, and troubleshooting documentation

Phrase prompts as genuine user requests, not labels from a test plan. Avoid repeatedly starting with “Explain” or “Write.” Give each prompt one clear primary purpose.

## Grounding rules

- Never invent a tool, product name, command, API, or architecture.
- For real tools, use stable facts from authoritative documentation.
- For user-provided or repository projects, use only supplied or discoverable facts.
- If product facts are unavailable, request a generic template or keep the request high-level; do not fabricate details.
- Documentation prompts should identify the intended artifact, audience, and scope when relevant.
- Avoid near-duplicates and superficial rewrites of the same question.

High oversimplification-risk prompts should require caveats, boundary conditions, tradeoffs, uncertainty, or correction of a tempting but incomplete answer.

## Output

Write one JSON object per line:

```json
{ "id": "NET-01", "prompt": "What is the difference between IPv4 and IPv6?" }
```

Every row must contain exactly `id` and `prompt`.
