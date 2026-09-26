# How jev-long works

The whole model, with every constant. Nothing here is configurable except where noted.

## 1. Chunk size

Jev's context window is 32,000 tokens. Each chunk gets:

```
chunk budget = (32,000 - question tokens) × 0.9
```

The 10% headroom covers Jev's own prompt and errors in the token estimate. A question that leaves less than 1,000 tokens for the input is rejected.

## 2. Counting tokens

```
tokens = ceil(UTF-8 bytes / 3)        // estimateTokens()
```

Jev's tokenizer is not public, so jev-long uses a rule that is simple, conservative, and exactly additive. Common tokenizers average about 4 bytes per token on English prose, so the rule over-counts English by about a third. Code, JSON, and CJK text come out close to 3 bytes per token. `maxInputTokens` (default 250,000) is checked with the same rule.

If Jev still rejects a chunk as too long (HTTP 413, or a 400 or 422 that mentions the context length), the budget shrinks to 75% of that chunk's estimate and the whole input is planned again. After 3 shrinks, the call fails.

## 3. Chunks

1. Split the input at line breaks and sentence ends (`.`, `!`, `?` followed by whitespace, or `。！？`). A sentence longer than half the overlap is split at spaces, and a single word longer than that is split anywhere, never inside a surrogate pair.
2. Pack those pieces greedily into chunks up to the budget. Each chunk after the first starts about 5% of the budget before the previous chunk ended.
3. Use the smallest chunk size that still needs the same number of chunks, so all chunks are about the same size and the last one is not a scrap.

Every character is in at least one chunk. `chunks[i]` gives `start` and `end`, and `input.slice(start, end)` is exactly what Jev saw.

**Weights.** A chunk's weight is the share of the input that only it covers:

```
weight_i = bytes(input from the end of chunk i-1 to the end of chunk i) / bytes(input)
```

Weights sum to 1. Overlap is counted once, in the chunk that covered it first.

## 4. Combining answers

Each chunk's answer is read as probabilities over the question's labels: one per option for `choice` (in your order), one per level for `score`, and `[1 - p, p]` for `noul`.

With `combine: "average"` (the default):

| | Combined answer |
| --- | --- |
| probabilities | `P = Σ weight_i · P_i` |
| `choice` | The option with the highest `P`. Exact ties go to the first option in your `criteria`. |
| `score` | `Σ weight_i · score_i`, which equals `Σ level · P[level]` |
| `noul` | `Σ weight_i · noul_i` |
| `confidence` | `Σ weight_i · confidence_i`, for `choice` and `score`. Left out if Jev left it out for any chunk. |

With `combine: "max"` (for `noul` and `score` only), the answer is the chunk with the highest `noul` or `score`, returned as Jev gave it, including its confidence. The earliest chunk wins ties.

An input that fits in one chunk returns Jev's own answer, with its probabilities rescaled to sum to exactly 1.

## 5. Agreement

Agreement is 1 minus the weighted average difference between every pair of chunks:

```
agreement = 1 - Σ_{i<j} w_i·w_j·d(i, j) / Σ_{i<j} w_i·w_j        (1 for a single chunk)
```

| Question | Difference `d(i, j)`, from 0 to 1 |
| --- | --- |
| `choice` | Total variation distance, `½ Σ \|P_i - P_j\|`: the share of probability that differs |
| `noul` | `\|noul_i - noul_j\|` |
| `score` | `\|score_i - score_j\| / (levels - 1)`, because levels are ordered |

Identical answers give 1. Two chunks that give opposite, certain answers give 0. Pairs are weighted by the product of their weights, so a small chunk that disagrees moves agreement only a little. Agreement does not depend on `combine`.

## 6. Requests

- Four at a time, each with a 60-second timeout.
- Retried up to 3 times, with exponential backoff (about 0.5 s, 1 s, and 2 s, with jitter): HTTP 408, 429, and 5xx, network errors, and timeouts. `Retry-After` is honored, up to 30 seconds.
- Any other failure, or a malformed answer, fails the call with `request_failed` and cancels the chunks still in flight. No answer is computed from part of the input.
- Error messages include the provider's own message, unless it quotes the input (any 20-character run in common). That way input text never ends up in an error that gets logged.

## Design notes

The first version (see [original-spec.md](original-spec.md)) had much more machinery. Each piece was removed because it did not improve the answer enough to justify another concept a developer has to trust.

- **C3 confidence calibration** raised Jev's confidence when chunks agreed. It used an effective chunk count, a saturation curve, an agreement gate, and four parameters (λ, agreement floor, exponent, cap). None of it was fitted to data, and there was no evidence that chunking lowers Jev's confidence in the first place. In practice it nudged confidence up by small amounts. jev-long now reports Jev's confidence (averaged) and agreement as two separate signals, and adjusts neither.
- **Averaged confidence cannot see disagreement.** Each chunk's confidence measures how sure that chunk is, not whether the chunks agree. That is exactly why `agreement` is always returned next to it.
- **Five aggregation methods** (weighted mean, mean, median, min, max) became two. Mean and median gave a short final chunk a full vote. Min and max were applied per probability and renormalized, which for `noul` produced 0.5 whenever chunks disagreed, the opposite of what "anywhere" should mean. `average` and a `max` that picks the strongest chunk cover the two kinds of question people actually ask.
- **Agreement was measured against the average.** With two chunks it could never go below 0.5. Pairwise agreement reaches 0 for opposite answers, compares scores on their own scale, and does not depend on how answers are combined.
- **A calibrated heuristic tokenizer** had per-script costs tuned against cl100k and o200k. Jev uses neither, so the precision was false. Bytes divided by 3 is honest and fits in one line.
- **Options that existed because the implementation supported them** are now fixed defaults or gone: overlap, context and protocol reserves, retries, concurrency, re-chunking policy, custom tokenizers, confidence parameters, event hooks, serialization options, a model-catalog lookup, and a second built-in transport. A bounded `maxInputTokens` replaced the unbounded "any length" promise.

## Validating the numbers

None of the constants are fitted to data yet. With labeled documents, the useful checks are:

- Whether the combined answer matches Jev's answer on the same document when it fits in one request.
- Whether high-agreement results are right more often than low-agreement ones.
- Jev's real token counts against `estimateTokens`, to tune the 3-bytes rule. `npm run smoke:live` prints this ratio.
