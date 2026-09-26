> **Superseded.** This is the original v0.1 specification, written when the project was called JevInfiniteCTX, and kept for reference. jev-long 0.2 deliberately simplified it: C3 confidence calibration and the extra aggregation methods were removed, agreement became pairwise, the input is bounded by `maxInputTokens` (250K by default), and most options became fixed defaults. See [how-it-works.md](how-it-works.md) for the current model and the reasons for each change.

# JevInfiniteCTX — Long-Context Decision Wrapper

**Status:** Draft specification 0.1  
**Date:** 2026-09-25  
**Working package name:** `JevInfiniteCTX`  
**Core confidence method:** **C3 — Cross-Chunk Confidence Calibration**

## 1. Summary

JevInfiniteCTX is a thin TypeScript library that lets applications use Jev's native decision primitives against input strings larger than Jev's context window without silently truncating data.

The caller supplies:

1. an arbitrary-length input string,
2. one Jev output type — `choice`, `score`, or `noul`,
3. the Jev instructions and criteria,
4. an aggregation strategy,
5. optional chunking and confidence-calibration settings.

JevInfiniteCTX then:

1. computes a safe state budget,
2. splits the input into overlapping token-aware chunks,
3. sends the same Jev question across every chunk,
4. converts each result into a common probability representation,
5. aggregates those probabilities,
6. measures cross-chunk agreement,
7. applies a bounded confidence correction called **C3**, and
8. returns one final answer plus transparent chunk-level metadata.

The package must never hide truncation, silently discard a failed chunk, or represent its adjusted confidence as Jev's original confidence.

---

## 2. Problem

Jev is optimized for fast, typed decisions rather than text generation. Its three native question types are:

- **Choice** — select one key from a caller-defined set; returns the selected key, per-option probabilities, and Jev confidence.
- **Score** — evaluate against an ordered rubric; returns a probability-weighted score, per-level probabilities, a legend, and Jev confidence.
- **Noul** — yes/no as a probability; returns a `0..1` probability that the answer is yes. Noul does **not** return a separate Jev confidence field.

On OpenRouter, the current Jev family is exposed with a 32K context window. Large documents therefore cannot be sent in a single request.

Naive chunking creates three problems:

- context can be lost at chunk boundaries,
- duplicate overlap can distort aggregation,
- confidence can become difficult to interpret as the number of chunks increases.

JevInfiniteCTX addresses those issues while retaining Jev's native semantics.

---

## 3. Goals

### 3.1 Required

- Accept an arbitrarily long text string.
- Never blindly truncate input.
- Support all three Jev question types: `choice`, `score`, and `noul`.
- Preserve caller-provided Choice keys exactly.
- Use token-aware chunking with configurable overlap.
- Default overlap to **5%** of the usable chunk size.
- Ensure interior chunks have context from both neighboring regions through normal sliding-window overlap.
- Support a small set of numeric aggregation methods.
- Correct for duplicated overlap when weighting chunks.
- Return a single final decision.
- Return raw chunk decisions for inspectability.
- Return both unadjusted and C3-adjusted confidence signals.
- Keep adjusted confidence bounded.
- Work independently of transport so Jev can be called through OpenRouter or another compatible Jev provider.

### 3.2 Non-goals for V1

- Summarization or generative text output.
- Semantic/RAG chunk selection.
- Reordering document content.
- Recursive LLM summarization.
- Hidden thresholding that turns probabilities into business policy.
- Claiming C3 is statistically calibrated before it has been tested against labeled datasets.
- Multi-document retrieval.

---

## 4. Native Jev Contract

JevInfiniteCTX should preserve the meaning of Jev's native output types rather than inventing a fourth decision type.

### Choice

Caller supplies an option map. The map keys can be arbitrary strings such as:

```ts
{
  tree: "The text primarily describes a tree.",
  rock: "The text primarily describes a rock.",
  other: "Neither tree nor rock is the best description."
}
```

Jev returns one of those keys. It is not generating the label as free text; the returned value is a selection from the caller-defined option set.

### Score

Caller supplies an ordered rubric of 2–10 levels. Jev returns a probability-weighted score over level indices, so the native score range is:

```text
0 .. (numberOfLevels - 1)
```

It is **not inherently a 0–1 score**. If an application needs 0–1, JevInfiniteCTX may expose an optional normalized score in metadata, but must preserve the native score.

### Noul

Jev returns:

```text
0.0 = no
1.0 = yes
```

The value is the probability of yes. Noul has no separate native confidence field.

---

## 5. Proposed Public API

V1 should expose one primary function.

```ts
const result = await JevInfiniteCTX.decide({
  input: veryLargeString,

  question: {
    type: "choice", // "choice" | "score" | "noul"
    instructions: "What is the primary subject of this material?",
    criteria: {
      tree: "Primarily about trees or forests.",
      rock: "Primarily about rocks or geology.",
      other: "Neither is the primary subject."
    }
  },

  aggregation: "weighted_mean",

  chunking: {
    overlap: 0.05,
    maxStateTokens: "auto"
  },

  confidence: {
    method: "c3"
  },

  provider: {
    transport: "openrouter",
    model: "typesafe/jev-1.13"
  }
});
```

### 5.1 Minimal required input

```ts
type JevInfiniteCTXRequest = {
  input: string;
  question: JevQuestion;
};
```

Everything else has defaults.

### 5.2 Question types

```ts
type NoulQuestion = {
  type: "noul";
  instructions: string | object | unknown[];
  criteria?: {
    true: unknown;
    false: unknown;
  };
};

type ChoiceQuestion = {
  type: "choice";
  instructions: string | object | unknown[];
  criteria: Record<string, unknown>;
};

type ScoreQuestion = {
  type: "score";
  instructions: string | object | unknown[];
  criteria: unknown[];
};
```

V1 may initially narrow `unknown` to Jev-supported JSON-safe values in the concrete TypeScript implementation.

---

## 6. Chunking

### 6.1 Context budget

The library must not simply split state at 32,000 tokens.

The Jev request also includes model/request overhead, instructions, and criteria. JevInfiniteCTX should calculate a safe state budget:

```text
usableStateTokens = contextWindow
                  - estimatedQuestionTokens
                  - protocolReserve
                  - safetyReserve
```

Recommended V1 defaults:

```text
contextWindow:      provider/model metadata, currently 32,000 for Jev on OpenRouter
safetyReserve:      8% of remaining context
protocolReserve:    implementation-defined small fixed reserve
```

If reliable provider token metadata is unavailable, default to a conservative maximum state size of approximately **28K estimated tokens** and shrink automatically on a provider context-limit error.

### 6.2 Token-aware splitting

Chunk boundaries must be selected by token count, not JavaScript string length.

Preferred boundary order near the target split point:

1. paragraph boundary,
2. sentence boundary,
3. whitespace boundary,
4. hard token boundary as a last resort.

No source text may be omitted.

### 6.3 Overlap

Default:

```ts
overlap: 0.05
```

Overlap size:

```text
overlapTokens = floor(usableStateTokens * overlap)
```

The implementation uses a sliding window. Therefore an interior chunk naturally overlaps the prior chunk on its left and the following chunk on its right.

Example:

```text
Chunk 1: [----------------------]
Chunk 2:                    [----------------------]
Chunk 3:                                       [----------------------]
                    ^^^^^                     ^^^^^
                   shared                    shared
```

### 6.4 Unique-content weighting

Overlap must not cause duplicated text to receive extra voting power.

Each chunk receives a weight based on the amount of **new source content** it contributes:

```text
uniqueTokens_1 = chunkTokens_1
uniqueTokens_i = chunkTokens_i - overlapWithPrevious_i

weight_i = uniqueTokens_i / sum(uniqueTokens)
```

This weighting is used by the default `weighted_mean` aggregator and by C3.

---

## 7. Execution

For `N` chunks, JevInfiniteCTX sends the same Jev question against each chunk as state.

Requests may run concurrently with bounded concurrency.

Recommended default:

```text
maxConcurrency = 4
```

The library must preserve original chunk order in the returned metadata regardless of completion order.

### Error behavior

- Retry provider `429`, transient `5xx`, and overload responses with exponential backoff.
- If a request fails because the input exceeds the context limit, reduce the state budget and re-chunk the entire document.
- If any chunk permanently fails, fail the overall operation by default.
- Never silently aggregate a partial document.

A future `allowPartial` mode can be considered, but it is out of scope for V1.

---

## 8. Common Probability Representation

The cleanest implementation is to convert all three Jev outputs into a probability vector internally.

### Choice

```text
P_i = Jev's returned probability map
```

Example:

```text
{ tree: 0.80, rock: 0.15, other: 0.05 }
```

### Score

```text
P_i = Jev's returned probability map across ordered level indices
```

Example:

```text
{ 0: 0.05, 1: 0.20, 2: 0.70, 3: 0.05 }
```

### Noul

Represent the returned yes probability `p` internally as:

```text
P_i = [1 - p, p]
```

Example:

```text
noul = 0.82
P_i = [0.18, 0.82]
```

This gives the aggregation and agreement layers one common mathematical representation.

---

## 9. Aggregation

### 9.1 Supported methods

V1:

```ts
type AggregationMethod =
  | "weighted_mean"
  | "mean"
  | "median"
  | "min"
  | "max";
```

Default:

```text
weighted_mean
```

### 9.2 Vector reduction

For Choice and Score, the selected aggregation function is applied independently to each probability dimension.

After aggregation, the resulting vector is normalized so all dimensions sum to 1.

For `weighted_mean`:

```text
P_agg[j] = Σ weight_i * P_i[j]
```

For `mean`, `median`, `min`, and `max`, apply that reducer over the chunk values for each dimension and then renormalize.

### 9.3 Final Choice

```text
choice = argmax(P_agg)
```

Return the aggregated probability map alongside it.

### 9.4 Final Score

After obtaining the aggregated level distribution:

```text
score = Σ levelIndex_j * P_agg[j]
```

The legend remains the caller-provided ordered rubric.

Optional normalized score:

```text
normalizedScore = score / (numberOfLevels - 1)
```

This is metadata only; it must not replace Jev's native score semantics.

### 9.5 Final Noul

```text
noul = P_agg[yes]
```

---

## 10. Cross-Chunk Agreement

Confidence correction should not depend on chunk count alone.

Ten chunks that agree strongly should be treated differently from two chunks that sharply disagree.

JevInfiniteCTX computes agreement using **total variation distance** over the common probability vectors.

For chunk `i`:

```text
TV_i = 0.5 * Σ_j |P_i[j] - P_agg[j]|
```

`TV_i` is bounded from 0 to 1.

Weighted disagreement:

```text
disagreement = Σ weight_i * TV_i
```

Agreement:

```text
A = 1 - disagreement
```

Therefore:

```text
0 <= A <= 1
```

Interpretation:

```text
A ≈ 1.0   chunks strongly agree
A ≈ 0.5   material disagreement
A ≈ 0.0   extreme disagreement
```

No hard meaning should be assigned to these bands until observed against real data.

---

## 11. C3 — Cross-Chunk Confidence Calibration

### 11.1 Purpose

Chunking can alter the confidence characteristics of the underlying Jev calls. C3 is a bounded correction layer intended to compensate for fragmentation **only when chunk results agree**.

C3 must never overwrite or mislabel Jev's original confidence.

### 11.2 Base confidence

For **Choice** and **Score**:

```text
C_base = Σ weight_i * jevConfidence_i
```

For **Noul**, Jev does not provide confidence. JevInfiniteCTX derives a certainty signal from the aggregated yes probability:

```text
C_base = |2 * noul - 1|
```

So:

```text
noul = 0.50 -> C_base = 0.00
noul = 0.75 -> C_base = 0.50
noul = 1.00 -> C_base = 1.00
```

This field must be labeled `derived`, not `jev`.

### 11.3 Effective chunk count

Overlap reduces how much new content each additional chunk contributes.

For a fixed overlap ratio `r`:

```text
N_eff ≈ 1 + (N - 1) * (1 - r)
```

The actual implementation should derive this from unique-token weights rather than relying only on the approximation.

### 11.4 Saturation term

C3 uses a saturating exponential rather than an unbounded logarithm:

```text
S = 1 - exp(-λ * (N_eff - 1))
```

Properties:

- `N = 1` -> `S = 0` -> no correction.
- More chunks increase the possible correction.
- The increase rapidly tapers off.
- `S` never exceeds 1.

Recommended provisional default:

```text
λ = 0.25
```

### 11.5 Agreement gate

Low agreement should not receive a fragmentation boost.

Define:

```text
G = clamp((A - A_floor) / (1 - A_floor), 0, 1)^γ
```

Provisional defaults:

```text
A_floor = 0.50
gamma   = 2.0
```

This means:

- agreement at or below `0.50` receives no boost,
- moderate agreement receives a small boost,
- near-perfect agreement receives most of the available boost.

### 11.6 Adjusted confidence

Let:

```text
C_cap = 0.98
```

Then:

```text
C_adjusted = C_base + (C_cap - C_base) * S * G
```

Properties:

- one chunk -> adjusted confidence equals base confidence,
- adjustment is monotonic with chunk count only when agreement supports it,
- disagreement suppresses the correction,
- adjusted confidence asymptotically approaches an agreement-limited ceiling,
- adjusted confidence cannot exceed `C_cap`.

### 11.7 Important interpretation

C3 is initially a **heuristic normalization signal**, not a statistically proven probability of correctness.

The names `confidence.adjusted` and `c3` are acceptable. The library must not call it `calibratedProbability` until the parameters have been fit and validated on labeled data.

The provisional constants should later be learned from an evaluation corpus by minimizing calibration error rather than defended as universal constants.

---

## 12. Result Shape

The package should return one final answer plus transparent metadata.

### Choice example

```ts
{
  type: "choice",
  choice: "tree",
  probabilities: {
    tree: 0.78,
    rock: 0.16,
    other: 0.06
  },

  confidence: {
    base: 0.81,
    adjusted: 0.86,
    source: "jev",
    agreement: 0.93,
    method: "c3-v1"
  },

  aggregation: {
    method: "weighted_mean"
  },

  chunks: {
    count: 3,
    effectiveCount: 2.9,
    overlap: 0.05,
    results: [/* raw per-chunk Jev results */]
  },

  usage: {
    inputTokens: 74219,
    outputTokens: 102,
    costUsd: 0.0031,
    elapsedMs: 612
  }
}
```

### Noul example

```ts
{
  type: "noul",
  noul: 0.84,

  confidence: {
    base: 0.68,
    adjusted: 0.76,
    source: "derived",
    agreement: 0.91,
    method: "c3-v1"
  },

  chunks: {
    count: 4,
    effectiveCount: 3.85,
    overlap: 0.05,
    results: [/* raw results */]
  }
}
```

The raw chunk results should be optional in serialization for applications that do not want the payload size, but they should be available to the caller at runtime by default.

---

## 13. Provider Abstraction

The chunking, aggregation, and C3 logic must not depend on OpenRouter.

```ts
interface JevTransport {
  decide(request: NativeJevRequest): Promise<NativeJevResponse>;
  contextWindow(model: string): Promise<number> | number;
}
```

Initial adapters:

```text
OpenRouterJevTransport
DirectJevTransport
```

The OpenRouter adapter should use the Decisions API rather than the chat-completions API.

Provider-specific model names, authentication, retries, and usage parsing belong in the adapter layer.

---

## 14. Defaults

```ts
const defaults = {
  aggregation: "weighted_mean",

  chunking: {
    overlap: 0.05,
    maxStateTokens: "auto",
    contextSafetyReserve: 0.08,
    preferNaturalBoundaries: true
  },

  execution: {
    maxConcurrency: 4,
    retries: 3
  },

  confidence: {
    method: "c3",
    cap: 0.98,
    lambda: 0.25,
    agreementFloor: 0.50,
    agreementExponent: 2.0
  }
};
```

All confidence parameters are provisional until benchmarked.

---

## 15. Edge Cases

### Input fits in one chunk

Do not chunk unnecessarily.

Return the native Jev decision with:

```text
chunkCount = 1
agreement = 1
adjustedConfidence = baseConfidence
```

### Empty input

Reject before calling Jev.

### Choice with unknown cases

JevInfiniteCTX must not invent an `other` option. Documentation should recommend that callers include `other` / `none` when the domain is not exhaustive.

### Conflicting chunks

Do not hide the conflict. Low agreement should suppress C3 adjustment and remain visible in metadata.

### Extremely large input

Process incrementally without holding duplicate tokenized copies of the entire document where practical. The result must be the same logical operation regardless of input size.

### Provider context error

Automatically reduce chunk size and retry the full operation. Never drop the overflow.

### Provider partial failure

Fail closed by default. Do not return a seemingly complete aggregate from incomplete source coverage.

---

## 16. Observability

Expose enough data to evaluate whether the wrapper is actually working.

Recommended event payload per call:

```text
model
provider
inputTokensEstimated
inputTokensActual
chunkCount
effectiveChunkCount
overlap
aggregationMethod
agreement
baseConfidence
adjustedConfidence
finalAnswer
perChunkProbabilities
cost
latency
retryCount
```

Do not log source text by default.

---

## 17. Validation Plan

C3 should be treated as provisional until evaluated.

Create a labeled benchmark with documents at multiple sizes:

```text
< 1 chunk
2 chunks
3–5 chunks
6–10 chunks
10+ chunks
```

For every source document:

1. obtain a trusted label / expected score,
2. run the full document when it fits,
3. create controlled longer variants that force chunking,
4. compare raw aggregate accuracy,
5. compare C3-adjusted confidence against actual correctness,
6. measure calibration error by chunk count and agreement,
7. fit `lambda`, `agreementFloor`, `agreementExponent`, and `cap` from data.

Useful metrics:

```text
accuracy / F1 for Choice
MAE or rank error for Score
Brier score for Noul
ECE (Expected Calibration Error)
Brier score for confidence where meaningful
```

The key test is not whether C3 makes confidence numerically larger. It is whether similarly scored results are correct at approximately the expected rate across different chunk counts.

---

## 18. Acceptance Criteria

V1 is complete when:

- [ ] A 100K+ token string can be processed without truncation.
- [ ] Input fitting in one chunk produces no confidence adjustment.
- [ ] Default overlap is 5% and no source text is lost.
- [ ] Overlap is not double-counted in `weighted_mean`.
- [ ] Choice works with arbitrary caller-defined string keys.
- [ ] Score returns native weighted level index plus probabilities.
- [ ] Noul returns aggregated yes probability.
- [ ] Choice and Score preserve Jev's raw chunk confidence values.
- [ ] Noul clearly labels package-derived certainty as derived, not Jev confidence.
- [ ] Agreement is calculated from all chunk probability distributions.
- [ ] C3 adjustment is bounded and equals zero for one chunk.
- [ ] Low agreement suppresses the adjustment.
- [ ] A permanently failed chunk fails the whole call by default.
- [ ] Raw chunk results are inspectable.
- [ ] Provider implementation is replaceable through `JevTransport`.
- [ ] No business threshold is imposed by the library.

---

## 19. Future Extensions

Not required for V1:

- multiple Jev questions in one JevInfiniteCTX call,
- aggregation strategies specific to semantics such as `any`, `all`, `majority`, and percentile,
- hierarchical reduction for millions of tokens,
- adaptive overlap based on semantic boundaries,
- benchmark-trained C3 parameters by domain,
- pluggable calibration curves / isotonic regression,
- streaming source ingestion,
- persisted chunk result cache,
- cross-provider Jev fallback.

---

## 20. Design Principle

JevInfiniteCTX should feel like Jev with a larger effective input surface, not like a new reasoning system.

The library's job is to preserve all source data, preserve Jev's three native decision semantics, aggregate the numeric evidence transparently, and expose exactly how much confidence adjustment it applied and why.

