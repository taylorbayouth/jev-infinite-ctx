# JevInfiniteCTX

JevInfiniteCTX lets you ask a TypeSafe Jev `choice`, `score`, or `noul` question about an input of any length. On OpenRouter, Jev has a 32K-token context window. A long document does not fit in one request, and truncating it silently changes the answer. The library splits the input into token-budgeted chunks that overlap, asks the same question about every chunk, and converts each answer into a probability vector. It then combines the vectors with weights that stop overlapping text from counting twice, measures how well the chunks agree, and applies a bounded confidence correction called C3. The result is one final answer with Jev's native meaning, plus every per-chunk result so you can check it. The library never drops text and never combines a partial document. It never labels its own adjusted confidence as Jev's.

TypeScript, ESM only, Node 20 or newer, zero runtime dependencies. The design is specified in [docs/specification.md](docs/specification.md).

- [Install](#install)
- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [Result shape](#result-shape)
- [Options reference](#options-reference)
- [Transports](#transports)
- [Tokenizers](#tokenizers)
- [Errors and fail-closed semantics](#errors-and-fail-closed-semantics)
- [Observability](#observability)
- [Serialization](#serialization)
- [Lower-level API](#lower-level-api)
- [Caveats](#caveats)
- [Development](#development)

## Install

The package is not on the npm registry yet. To use it, build it from source:

```bash
git clone https://github.com/taylorbayouth/jev-infinite-ctx.git
cd jev-infinite-ctx
npm install            # also builds dist/ (the "prepare" script runs `npm run build`)
```

Then install it into your project from the checkout, or from a tarball you make with `npm pack` (which also builds first):

```bash
npm install ../jev-infinite-ctx
```

The built-in transports read their API keys from the environment:

```bash
export OPENROUTER_API_KEY=sk-or-...   # default transport ("openrouter")
export TYPESAFE_API_KEY=...           # only for transport "direct"
```

## Quick start

Each example is a complete ES module. You can run it with any TypeScript runner, for example `npx tsx choice.ts ./document.txt`. The only required fields are `input` and `question`. By default, requests go through OpenRouter to `typesafe/jev-1.13`.

### Choice

```ts
import { readFile } from "node:fs/promises";
import { decide } from "jev-infinite-ctx";

const input = await readFile(process.argv[2] ?? "document.txt", "utf8");

const result = await decide({
  input,
  question: {
    type: "choice",
    instructions: "What is the primary subject of this material?",
    criteria: {
      tree: "Primarily about trees or forests.",
      rock: "Primarily about rocks or geology.",
      other: "Neither is the primary subject.",
    },
  },
});

// result.choice is typed "tree" | "rock" | "other"
console.log(result.choice);                  // "tree"
console.log(result.probabilities);           // { tree: 0.7714, rock: 0.1686, other: 0.06 }
console.log(result.confidence.base);         // 0.8066, weighted mean of Jev's chunk confidences
console.log(result.confidence.adjusted);     // 0.8448, after C3
console.log(result.confidence.agreement);    // 0.949, cross-chunk agreement
console.log(`${result.chunks.count} chunks, ${result.usage.requests} requests, $${result.usage.costUsd ?? "n/a"}`);
```

### Score

```ts
import { readFile } from "node:fs/promises";
import { decide } from "jev-infinite-ctx";

const input = await readFile(process.argv[2] ?? "contract.txt", "utf8");

const result = await decide({
  input,
  question: {
    type: "score",
    instructions: "How much contractual risk does this agreement carry for the buyer?",
    criteria: ["Minimal risk", "Some risk", "Significant risk", "Severe risk"], // lowest level first
  },
  aggregation: "weighted_mean",
  chunking: { overlap: 0.1 },
  execution: { maxConcurrency: 8 },
  provider: { transport: "openrouter", model: "typesafe/jev-1.13" },
});

console.log(result.score);            // 1.9378, native scale 0..(levels - 1)
console.log(result.normalizedScore);  // 0.6459, score / (levels - 1), metadata only
console.log(result.legend[String(Math.round(result.score))]); // "Significant risk"
console.log(result.probabilities);    // { "0": 0.0368, "1": 0.1412, "2": 0.6693, "3": 0.1527 }
```

### Noul

```ts
import { readFile } from "node:fs/promises";
import { decide } from "jev-infinite-ctx";

const input = await readFile(process.argv[2] ?? "contract.txt", "utf8");

const result = await decide({
  input,
  question: {
    type: "noul",
    instructions: "Does this agreement allow either party to terminate for convenience?",
    // Optional for noul. When present, both keys are required.
    criteria: {
      true: "A party may end the agreement without cause.",
      false: "Termination requires cause, or is not addressed.",
    },
  },
  signal: AbortSignal.timeout(120_000),
  onEvent: (event) => {
    if (event.type === "chunk.retry") {
      console.error(`chunk ${event.chunkIndex} retrying in ${Math.round(event.delayMs)} ms (${event.errorKind})`);
    }
    if (event.type === "rechunk") {
      console.error(`context limit hit, budget ${event.previousBudget} -> ${event.newBudget}`);
    }
  },
});

console.log(result.noul);               // 0.8525, aggregated probability of yes
console.log(result.confidence.base);    // 0.7051 = |2 * noul - 1|
console.log(result.confidence.source);  // "derived" (Jev returns no confidence for noul)
```

`decide` is also available as `JevInfiniteCTX.decide` on the default export (`import JevInfiniteCTX from "jev-infinite-ctx"`).

### Question rules

The library checks every question before it makes any request. A violation throws `JevValidationError`.

| Type | `criteria` |
| --- | --- |
| `choice` | Object with at least 2 keys. The keys come back exactly as you wrote them. |
| `score` | Array of 2 to 10 levels, lowest first. |
| `noul` | Optional `{ true, false }`. When present, both keys are required. |

`instructions` and every criterion must be a non-empty string, or a JSON object or array. The question must be JSON-serializable. Unknown fields anywhere in the request are rejected, so a mistyped option name never silently falls back to its default. The library does not invent an `other` option for `choice`. If the options might not cover every case, include an `other` or `none` option yourself.

## How it works

1. Compute a safe per-chunk state budget from the model's context window.
2. Plan chunks that fit the budget and overlap. Chunks end at natural boundaries where possible.
3. Send the same question with every chunk as the `state`. Requests run concurrently, with a limit.
4. Convert each answer into a probability vector over a fixed set of labels.
5. Combine the vectors into one.
6. Measure how far each chunk is from the combined vector (agreement).
7. Compute base confidence and apply C3.
8. Return the final answer plus every chunk result, in document order.

### State budget

Every chunk is sent together with the question, so the question's size comes out of every chunk's budget:

```text
questionTokens = tokenizer.count(JSON.stringify({ type, instructions, criteria }))
remaining      = contextWindow - questionTokens - protocolReserve
safetyReserve  = ceil(max(0, remaining) * contextSafetyReserve)
usable         = floor(remaining - safetyReserve)
```

- `contextWindow` comes from `transport.contextWindow(model)`. See [Transports](#transports).
- If the transport returns `undefined`, throws, or returns a value that is not a positive number, the window is treated as unknown. The same formula then runs with 32,000, and the result is capped at 28,000 (`source: "fallback"`).
- A numeric `chunking.maxStateTokens` below `usable` replaces it (`source: "caller"`). It can only lower the budget, never raise it.
- If `usable` is below 256 tokens, the call throws `JevValidationError` before any request, because the question is too large for the window.

Example: a 32,000-token window with a 67-token question gives `remaining = 30,909`, `safetyReserve = 2,473`, and `usable = 28,436`.

### Token-aware chunking with overlap

All chunk boundaries are chosen by token count, never by string length.

- If the whole input fits in the budget, the plan is a single chunk and the input is not split.
- Otherwise the planner uses a greedy sliding window. Each chunk extends as far as the budget allows. With `preferNaturalBoundaries` (default `true`), the chunk end moves back to the best boundary within the last 20% of the chunk's characters. A paragraph break (a blank line or U+2029) is best. A sentence end is next: `.`, `!`, `?`, `…` or `。！？`, optionally followed by closing quotes or brackets, then whitespace. The CJK forms `。！？` need no whitespace after them, since CJK text has none. Any whitespace is last. The planner takes the candidate closest to the maximum end within the best class it finds. If there is no candidate, it cuts at the token limit (`boundary: "hard"`). The final chunk has `boundary: "end"`.
- Each chunk after the first starts early enough that the shared region `[start_i, end_{i-1})` holds about `floor(budget * overlap)` tokens. The default overlap of 0.05 with a 28,436-token budget gives 1,421 tokens. When natural boundaries are preferred, the start moves back up to 32 characters so it falls at the start of a word.
- Offsets are UTF-16 indices into the input. A chunk is exactly `input.slice(start, end)`, and a boundary never splits a surrogate pair.
- Guarantees: the first chunk starts at 0 and the last ends at `input.length`. Chunks together cover every character. Starts and ends strictly increase, and every chunk fits the budget. The planner only counts tokens in slices about one chunk long, so the total counting work is O(n log chunk), not O(n²).

Every interior chunk overlaps the chunk before it and the chunk after it, so text near each boundary appears in both neighboring chunks.

### Unique-content weights

Overlapping text must not get extra voting power:

```text
overlapTokens_i = tokens(input.slice(start_i, end_{i-1}))   (0 for the first chunk)
uniqueTokens_i  = max(1, tokens_i - overlapTokens_i)
weight_i        = uniqueTokens_i / Σ uniqueTokens
```

The weights sum to 1. A short final chunk gets a proportionally small weight. For example, a plan with chunks of 28,384, 28,259, and 10,786 tokens at 5% overlap gets weights of 0.439, 0.416, and 0.145. Agreement and base confidence always use these weights. Among the aggregation methods, only `weighted_mean` uses them.

### Common probability representation

Each chunk's answer becomes a `Distribution` over a fixed, ordered set of labels:

| Type | Labels | Vector |
| --- | --- | --- |
| `choice` | `Object.keys(criteria)` | Jev's `probabilities` map |
| `score` | `"0"` to `String(levels - 1)` | Jev's `probabilities` map over level indices |
| `noul` | `["no", "yes"]` | `[1 - p, p]`, where `p` is Jev's `noul` |

Answers are validated strictly. The answer's `type` must match the question's type. A `choice` must be one of the criteria keys. `choice` and `score` answers must include `probabilities`. A key that is not a label is an error, and a missing label counts as 0. Each value must be finite, at least 0, and at most 1.1 (float noise down to -1e-9 is clamped to 0). The sum must be within 1 ± 0.1, and the vector is then renormalized to sum to exactly 1. A native `score` must lie in `[0, levels - 1]` and a native `noul` in `[0, 1]`, each with 1e-6 slack. Any violation is a `JevResponseError`, and the call fails.

Labels follow JavaScript's own-key order. Integer-like keys such as `"1"` and `"2"` come first, in ascending order, followed by the other keys in insertion order. This order only matters for breaking ties in `choice`.

### Aggregation

The chosen reducer runs separately on each probability dimension. The result is then renormalized to sum to 1.

| `aggregation` | Reducer per dimension `j` | Uses weights |
| --- | --- | --- |
| `weighted_mean` (default) | `Σ weight_i * P_i[j]` | yes |
| `mean` | arithmetic mean | no |
| `median` | median (mean of the two middle values for an even count) | no |
| `min` | minimum | no |
| `max` | maximum | no |

If the reduced vector sums to 0 (for example, `min` over chunks that put all their mass on different labels), the aggregate becomes uniform and `aggregation.degenerate` is `true`. The fallback is never silent.

`min` and `max` are numeric reducers, not logical "all" or "any". For noul, `min` over `[0.1, 0.9]` and `[0.9, 0.1]` gives `[0.1, 0.1]`, which renormalizes to 0.5. `mean`, `median`, `min`, and `max` give a short final chunk the same vote as a full chunk.

The final answer for more than one chunk:

```text
choice          = argmax_j P_agg[j]        (ties within 1e-12 go to the earliest label)
score           = Σ_j j * P_agg[j]
normalizedScore = score / (levels - 1)     (metadata only; score keeps Jev's scale)
legend          = { "0": criteria[0], "1": criteria[1], ... }
noul            = P_agg["yes"]
```

If the input fits in one chunk, the result carries Jev's native `choice`, `score`, or `noul` unchanged. `probabilities` is that chunk's normalized distribution, agreement is 1, and adjusted confidence equals base confidence.

### Agreement via total variation

```text
TV_i         = 0.5 * Σ_j |P_i[j] - P_agg[j]|      in [0, 1]
disagreement = Σ_i weight_i * TV_i
A            = 1 - disagreement                    in [0, 1]
```

A single chunk has `A = 1`. Each chunk's `TV_i` is reported as `chunks.results[i].totalVariation`, so you can find the chunks that disagree. `A` is measured against the aggregate, so its practical range depends on how many chunks there are. Two equally weighted chunks that fully disagree give `A = 0.5`. `A` only approaches 0 when many chunks each back a different label. No fixed meaning is assigned to particular values of `A`.

### C3: Cross-Chunk Confidence Calibration

C3 can raise confidence when many chunks agree and never lowers it. The base value is always reported next to the adjusted one, and its `source` says where it came from.

**Base confidence**

```text
choice, score:  C_base = Σ_i weight_i * jevConfidence_i            source "jev"
                if any chunk omitted confidence:
                C_base = 1 - H(P_agg) / ln(K)                       source "derived"
noul:           C_base = |2 * noul - 1|                             source "derived"
```

`H` is the natural-log entropy of the aggregate and `K` is the number of labels.

**Correction**

```text
N_eff      = clamp(Σ_i uniqueTokens_i / max_i tokens_i, 1, N)
S          = 1 - exp(-λ * (N_eff - 1))
G          = clamp((A - A_floor) / (1 - A_floor), 0, 1) ^ γ
adjustment = max(0, cap - C_base) * S * G
C_adjusted = C_base + adjustment
```

The defaults are `λ = 0.25`, `A_floor = 0.5`, `γ = 2`, and `cap = 0.98`. With these defaults:

| `N_eff` | 1 | 2 | 3 | 5 | 10 |
| --- | --- | --- | --- | --- | --- |
| `S` | 0 | 0.221 | 0.393 | 0.632 | 0.895 |

| `A` | ≤ 0.5 | 0.6 | 0.75 | 0.9 | 0.95 | 1 |
| --- | --- | --- | --- | --- | --- | --- |
| `G` | 0 | 0.04 | 0.25 | 0.64 | 0.81 | 1 |

**Interpretation choices.** Where the spec leaves room, the implementation makes these choices:

- **`max(0, cap - C_base)`**. The spec writes `(cap - C_base) * S * G`, which turns negative when Jev reports a confidence above the cap and would pull Jev's own number down. C3 only adds. If `C_base >= cap`, the adjustment is 0. As a result, `C_adjusted >= C_base` always. When `C_base < cap`, `C_adjusted <= cap`. The adjusted value is above the cap only when Jev's base already was, and in that case it is unchanged.
- **One chunk means no adjustment.** `S = 0` at `N_eff = 1`, and a single-chunk result is also short-circuited, so the adjustment is exactly 0.
- **`N_eff` comes from real unique-token counts**, not the approximation `1 + (N - 1)(1 - r)`. It is the new content of all chunks, measured in units of the largest chunk, so each chunk counts in proportion to the new content it carries. For equal chunks it equals the approximation. A short final chunk adds only its share of a full chunk: without overlap, a 10-token tail after a 1,000-token chunk adds 0.01.
- **Noul certainty is derived and labelled that way.** Jev has no confidence field for noul. `|2 * noul - 1|` measures how far the aggregate is from 0.5 and carries `source: "derived"`. It is not independent evidence.
- **A missing chunk confidence switches to a derived base.** If some choice or score chunks omit `confidence`, averaging only the chunks that reported it would over-weight them. Instead the base comes from the entropy of the aggregate, marked `"derived"`.
- **`confidence.method: "none"`** reports `adjusted = base`, `adjustment = 0`, and `method: "none"`. The components are still computed and reported.

**Worked example** (the three-chunk choice above): `N_eff = (28384 + 26838 + 9365) / 28384 = 2.275`, `A = 0.949`, `C_base = 0.8066`.

```text
S          = 1 - exp(-0.25 * 1.275)          = 0.2730
G          = ((0.949 - 0.5) / 0.5) ^ 2       = 0.8063
adjustment = (0.98 - 0.8066) * 0.2730 * 0.8063 = 0.0382
C_adjusted = 0.8066 + 0.0382                 = 0.8448
```

With the same chunks at `A = 0.6`, `G` drops to 0.04 and the adjustment to 0.0019. At `A <= 0.5` the adjustment is 0.

## Result shape

`decide` resolves to a `ChoiceResult`, `ScoreResult`, or `NoulResult`, according to `question.type`. For `choice`, the result type carries your literal keys through (`ChoiceResult<"tree" | "rock" | "other">`). The values below are rounded and illustrative.

### ChoiceResult

```ts
{
  type: "choice",
  choice: "tree",                                  // argmax of probabilities (native Jev value when there is 1 chunk)
  probabilities: { tree: 0.7714, rock: 0.1686, other: 0.06 }, // aggregated, sums to 1

  confidence: {
    base: 0.8066,                                  // C_base: weighted mean of Jev's chunk confidences
    adjusted: 0.8448,                              // C_adjusted after C3; >= base
    adjustment: 0.0382,                            // adjusted - base
    source: "jev",                                 // "jev" | "derived"
    agreement: 0.949,                              // A = 1 - Σ w_i * TV_i
    method: "c3-v1",                               // "c3-v1" | "none"
    components: {
      effectiveChunkCount: 2.275,                  // N_eff
      saturation: 0.2730,                          // S
      gate: 0.8063,                                // G
      cap: 0.98, lambda: 0.25, agreementFloor: 0.5, agreementExponent: 2,
    },
  },

  aggregation: { method: "weighted_mean", degenerate: false },

  chunks: {
    count: 3,
    effectiveCount: 2.275,
    overlap: 0.05,                                 // ratio
    overlapTokens: 1421,                           // floor(stateTokenBudget * overlap)
    stateTokenBudget: 28436,                       // budget of the final (successful) pass
    results: [                                     // in document order; results[1] shown, 0 and 2 have the same shape
      /* results[0] */
      {
        index: 1,
        start: 91672, end: 187751,                 // UTF-16 offsets: input.slice(start, end)
        estimatedTokens: 28259,                    // package tokenizer estimate
        uniqueTokens: 26838,                       // estimatedTokens minus overlap with chunk 0
        weight: 0.4155,
        boundary: "paragraph",                     // "paragraph" | "sentence" | "whitespace" | "hard" | "end"
        answer: {                                  // Jev's raw answer, unmodified
          type: "choice", choice: "tree",
          probabilities: { tree: 0.71, rock: 0.23, other: 0.06 },
          confidence: 0.77,
        },
        probabilities: { tree: 0.71, rock: 0.23, other: 0.06 }, // normalized common representation
        jevConfidence: 0.77,                       // undefined for noul or when Jev omitted it
        totalVariation: 0.0614,                    // TV between this chunk and the aggregate
        usage: { inputTokens: 21031, outputTokens: 35, costUsd: 0.00077 },
        attempts: 1,                               // 1 = no retries
        elapsedMs: 588,
        model: "typesafe/jev-1.13-20260917",       // exact build that answered
        responseId: "gen-dec-1790265859-EaXKST7hul1Wcqots1ZK",
      },
      /* results[2] */
    ],
  },

  usage: {
    inputTokens: 50143,          // provider-reported, summed over every response received, including discarded re-chunk passes
    outputTokens: 105,
    costUsd: 0.00184,            // undefined when no response reported cost
    elapsedMs: 1240,             // wall clock for the whole call
    inputTokensEstimated: 64587, // package estimate of the full input
    requests: 3,
    retries: 0,
    rechunks: 0,
  },

  model: "typesafe/jev-1.13-20260917", // model reported by the first chunk response
  provider: "openrouter",              // transport.name
}
```

### ScoreResult

The fields shared with every result (`confidence`, `aggregation`, `chunks`, `usage`, `model`, `provider`) look as they do above.

```ts
{
  type: "score",
  score: 1.9378,               // Σ j * P_agg[j], on Jev's native scale 0..(levels - 1)
  normalizedScore: 0.6459,     // score / (levels - 1); metadata only
  probabilities: { "0": 0.0368, "1": 0.1412, "2": 0.6693, "3": 0.1527 },
  legend: {                    // your rubric, keyed by level index
    "0": "Minimal risk", "1": "Some risk", "2": "Significant risk", "3": "Severe risk",
  },
  confidence: {
    base: 0.7024, adjusted: 0.7476, adjustment: 0.0452,
    source: "jev", agreement: 0.9409, method: "c3-v1",
    components: { effectiveChunkCount: 1.9397, saturation: 0.2094, gate: 0.7776, /* parameters */ },
  },
  chunks: { count: 2, effectiveCount: 1.9397, /* ... */ },
  // aggregation, usage, model, provider
}
```

### NoulResult

```ts
{
  type: "noul",
  noul: 0.8525,                // aggregated probability of yes (P_agg["yes"])
  confidence: {
    base: 0.7051,              // |2 * 0.8525 - 1|, package-derived certainty
    adjusted: 0.8195,
    adjustment: 0.1144,
    source: "derived",         // always "derived" for noul
    agreement: 0.953,
    method: "c3-v1",
    components: { effectiveChunkCount: 3.8285, saturation: 0.5069, gate: 0.8207, /* parameters */ },
  },
  chunks: {
    count: 4,
    effectiveCount: 3.8285,
    results: [
      {
        index: 0,
        answer: { type: "noul", noul: 0.91 },       // Jev's raw answer
        probabilities: { no: 0.09, yes: 0.91 },     // [1 - p, p]
        // no jevConfidence: Jev has none for noul
        totalVariation: 0.0575,
        /* offsets, tokens, weight, usage, attempts, ... */
      },
      /* chunks 1 to 3 */
    ],
    /* overlap, overlapTokens, stateTokenBudget */
  },
  // aggregation, usage, model, provider
}
```

## Options reference

Defaults are exported as `DEFAULTS` (from `src/defaults.ts`). An option you leave out takes its default. An option set to an invalid value throws `JevValidationError`, and the message names the field path (for example `chunking.overlap`).

| Option | Default | Allowed | Notes |
| --- | --- | --- | --- |
| `input` | required | string with a non-whitespace character | Empty input is rejected before any request. |
| `question` | required | see [Question rules](#question-rules) | |
| `aggregation` | `"weighted_mean"` | `weighted_mean`, `mean`, `median`, `min`, `max` | |
| `chunking.overlap` | `0.05` | `[0, 0.5]` | Share of the state budget repeated between neighboring chunks. |
| `chunking.maxStateTokens` | `"auto"` | `"auto"` or integer >= 256 | A number caps the budget: `min(number, auto budget)`. |
| `chunking.contextSafetyReserve` | `0.08` | `[0, 0.5]` | Fraction of the remaining window held back. |
| `chunking.protocolReserve` | `1024` | integer >= 0 | Tokens reserved for Jev's request scaffolding. |
| `chunking.preferNaturalBoundaries` | `true` | boolean | `false` means hard token cuts only. |
| `chunking.maxChunks` | `undefined` | integer >= 1 | Guard on the number of chunks. A first plan that exceeds it throws `JevValidationError` before any request. A re-chunk pass that would exceed it throws `JevContextBudgetError`, and the requests of earlier passes have already been made and billed. |
| `execution.maxConcurrency` | `4` | integer >= 1 | Maximum number of requests in flight at once. |
| `execution.retries` | `3` | integer >= 0 | Retries per chunk for retryable failures. |
| `execution.retryBaseDelayMs` | `500` | >= 0 | Backoff base. |
| `execution.retryMaxDelayMs` | `8000` | >= 0 | Maximum computed backoff. |
| `execution.maxRechunks` | `4` | integer >= 0 | Shrink-and-rechunk passes after context-limit errors. |
| `execution.rechunkShrinkFactor` | `0.75` | `(0, 1)` | Budget multiplier for each re-chunk, applied to `min(budget, failing chunk tokens)`. |
| `confidence.method` | `"c3"` | `"c3"`, `"none"` | `"none"` reports `adjusted = base`. |
| `confidence.cap` | `0.98` | `(0, 1]` | `C_cap` |
| `confidence.lambda` | `0.25` | > 0 | `λ`, saturation rate |
| `confidence.agreementFloor` | `0.5` | `[0, 1)` | `A_floor` |
| `confidence.agreementExponent` | `2` | > 0 | `γ` |
| `provider.transport` | `"openrouter"` | `"openrouter"`, `"direct"`, or a `JevTransport` | |
| `provider.model` | transport's `defaultModel` | non-empty string | `typesafe/jev-1.13` (OpenRouter), `jev-latest` (direct). |
| `provider.apiKey` | from the environment | string | `OPENROUTER_API_KEY` or `TYPESAFE_API_KEY`. Ignored when `transport` is an object. |
| `tokenizer` | `defaultTokenizer` | `Tokenizer` | See [Tokenizers](#tokenizers). |
| `signal` | none | `AbortSignal` | Cancels the whole call, including requests in flight. |
| `onEvent` | none | `(event) => void` | See [Observability](#observability). |

All confidence parameters are provisional until they are fitted on labelled data (spec 17).

## Transports

Transports are what make chunking, aggregation, and C3 independent of the provider. A transport sends one chunk's request and returns Jev's answers.

### OpenRouter Decisions API (default)

`transport: "openrouter"` creates an `OpenRouterJevTransport`. It uses the Decisions API, not chat completions: `POST https://openrouter.ai/api/alpha/decisions` with `{ model, state, questions }`. To set more options, construct the transport yourself and pass it in:

```ts
import { readFile } from "node:fs/promises";
import { decide, OpenRouterJevTransport } from "jev-infinite-ctx";

const transport = new OpenRouterJevTransport({
  apiKey: process.env.OPENROUTER_API_KEY,
  appName: "contract-review",           // X-OpenRouter-Title and X-Title
  appUrl: "https://example.com",        // HTTP-Referer
  sessionId: "batch-2026-09-25",        // body.session_id
  timeoutMs: 30_000,
});

const result = await decide({
  input: await readFile("contract.txt", "utf8"),
  question: { type: "noul", instructions: "Is there an auto-renewal clause?" },
  provider: { transport, model: "typesafe/jev-1.13" },
});
console.log(result.noul);
```

| Option | Default |
| --- | --- |
| `apiKey` | `process.env.OPENROUTER_API_KEY`. The constructor throws `JevValidationError` if neither is set. |
| `baseUrl` | `"https://openrouter.ai"` |
| `defaultModel` | `"typesafe/jev-1.13"` |
| `fetch` | global `fetch` |
| `timeoutMs` | `60_000` per request |
| `headers` | none. Extra headers cannot replace `Authorization` or `Content-Type`. |
| `appName`, `appUrl`, `sessionId`, `user` | none |
| `contextWindow` | none. An explicit window skips the lookup. |
| `resolveContextWindow` | `true` |
| `metadataTimeoutMs` | `5_000` |

How the context window is resolved: an explicit `contextWindow` is used first. Next, the transport looks the model up in `GET {baseUrl}/api/v1/models?output_modalities=decisions` and uses the entry's `context_length`. A model matches by `id`, by `canonical_slug`, or as an alias without the leading `~`. Successful lookups are cached for an hour and failed ones for a minute. If the lookup fails, any model id containing `jev` gets a static 32,000. Otherwise the window is `undefined`, which selects the conservative fallback budget. The lookup never throws.

### TypeSafe direct

`transport: "direct"` creates a `DirectJevTransport`. It uses the same wire format, sent to `POST https://api.typesafe.ai/v1/systemone` with `Authorization: Bearer $TYPESAFE_API_KEY`. The default model is `jev-latest` and the context window is 32,000 (the documented state limit). It accepts `apiKey`, `baseUrl`, `path`, `defaultModel`, `fetch`, `timeoutMs`, `headers`, and `contextWindow`. TypeSafe does not report cost, so `usage.costUsd` is `undefined`.

### How the built-in transports map errors

| Response | `JevProviderError.kind` | Retried |
| --- | --- | --- |
| 429 | `rate_limit` | yes |
| 503, 529, or another 5xx whose body mentions "overload" | `overloaded` | yes |
| other 5xx | `server` | yes |
| 408, request timeout | `timeout` | yes |
| `fetch` rejected | `network` | yes |
| 413, or 400/422 with context-length wording | `context_limit` | no; the whole document is re-chunked |
| other 400/422 | `bad_request` | no |
| 401/403, 402, 404 | `auth`, `payment`, `not_found` | no |
| malformed body | `invalid_response` | no |
| anything else | `unknown` | no |

A `Retry-After` header (in seconds or as an HTTP date) is used as the delay. A 200 response with a top-level `error` object is classified by its `code` in the same way. Without a `code`, it is classified by its wording alone: context-length wording gives `context_limit`, "overload" gives `overloaded`, and anything else gives `unknown`. The kind is chosen from the status and the first 1,000,000 characters of the body, including any part of the state the provider echoed. So an echoed document that mentions the context window can make a 400 or 422 look like a context-limit error. The re-chunk that follows is unnecessary, but the operation still fails closed (with `JevContextBudgetError` if the wording recurs in every pass). In the same way, an echoed document that mentions "overload" makes a code-less 200 `error` object `overloaded`, so it is retried up to `execution.retries` times, and makes a 5xx `overloaded` rather than `server` (both are retried). "overload" is never read for a 4xx, so an echo cannot make a client error retryable.

When the request carries a non-empty state, error messages never quote the provider's error text, so they never contain the state. (Without a state, a message may quote up to 300 characters of it.) An `invalid_response` message quotes a response key only when the request sent it (a question key, a criteria key, or a score level) and describes any other string the provider returned by its length, and the JSON parser's error, which quotes the body, is not kept as `cause`. The provider's response body is kept on `body`, truncated to 2,000 characters. If any run of 8 characters in that truncated body, or in the 128 characters after the cut, also occurs in the state, the whole body is replaced by `"[withheld: provider error body overlaps the request state]"`. Nothing is partially redacted. Both texts are compared with each whitespace run collapsed to one space, and the body is checked as sent and with up to three levels of JSON string escapes decoded (`\uXXXX` in either case, `\n`, `\"`, and the rest), so an echo nested in a JSON string, as in OpenRouter's `metadata.raw`, is found too. A state shorter than 8 characters withholds the body when the body contains it. A shorter fragment of a longer state is not detected.

### Writing a custom JevTransport

```ts
interface JevTransport {
  readonly name: string;          // used in result.provider and events
  readonly defaultModel: string;  // used when provider.model is not set
  decide(request: NativeJevRequest): Promise<NativeJevResponse>;
  contextWindow(model: string): Promise<number | undefined> | number | undefined;
}
```

The contract:

- `decide` receives `{ model, state, questions, signal }`. `state` is the chunk text. `questions` has exactly one key, `"decision"`, and the response's `answers` must use the same key.
- Return the normalized response. `model` is required. `usage` is optional and camelCase: `{ inputTokens, outputTokens, costUsd }`. `answers.decision` is Jev's raw answer (`{ type: "noul", noul }`, `{ type: "choice", choice, probabilities, confidence }`, or `{ type: "score", score, probabilities, legend, confidence }`).
- To report failures, throw `JevProviderError`. The kinds `rate_limit`, `server`, `overloaded`, `timeout`, and `network` are retried with backoff. `context_limit` triggers a re-chunk. Any other error, including a plain `Error`, is permanent, and the call fails.
- Pass `request.signal` to your HTTP client. The library aborts it when the caller cancels, when a sibling chunk fails, or when a pass is abandoned for re-chunking.
- `contextWindow` returns the model's total window in tokens. Return `undefined` when you do not know it, so the conservative fallback applies.
- Keep `state` out of error messages and logs.

This example sends requests through an internal gateway:

```ts
import { readFile } from "node:fs/promises";
import {
  decide,
  JevProviderError,
  type JevTransport,
  type NativeJevRequest,
  type NativeJevResponse,
} from "jev-infinite-ctx";

class GatewayTransport implements JevTransport {
  readonly name = "gateway";
  readonly defaultModel = "jev-latest";

  constructor(
    private readonly baseUrl: string,
    private readonly token: string,
  ) {}

  async decide(request: NativeJevRequest): Promise<NativeJevResponse> {
    let response: Response;
    try {
      response = await fetch(`${this.baseUrl}/decisions`, {
        method: "POST",
        headers: { authorization: `Bearer ${this.token}`, "content-type": "application/json" },
        body: JSON.stringify({ model: request.model, state: request.state, questions: request.questions }),
        signal: request.signal,
      });
    } catch (cause) {
      if (request.signal?.aborted) throw cause; // cancellation, not a network failure
      throw new JevProviderError("gateway unreachable", { kind: "network", cause });
    }
    if (!response.ok) {
      const kind =
        response.status === 413 ? "context_limit"
        : response.status === 429 ? "rate_limit"
        : response.status >= 500 ? "server"
        : "bad_request";
      throw new JevProviderError(`gateway returned HTTP ${response.status}`, { kind, status: response.status });
    }
    const body = (await response.json()) as {
      model: string;
      answers: NativeJevResponse["answers"];
      usage?: { input_tokens?: number; output_tokens?: number };
    };
    return {
      model: body.model,
      answers: body.answers,
      usage: { inputTokens: body.usage?.input_tokens, outputTokens: body.usage?.output_tokens },
    };
  }

  contextWindow(): number {
    return 32_000;
  }
}

const result = await decide({
  input: await readFile("contract.txt", "utf8"),
  question: { type: "noul", instructions: "Does the contract contain a non-compete clause?" },
  provider: { transport: new GatewayTransport("https://jev-gateway.internal", process.env.GATEWAY_TOKEN ?? "") },
});
console.log(result.noul);
```

A transport is also the simplest test double. The one below needs no network access:

```ts
import { decide, type JevTransport } from "jev-infinite-ctx";

const fake: JevTransport = {
  name: "fake",
  defaultModel: "fake-jev",
  contextWindow: () => 4_000,
  async decide(request) {
    const noul = request.state.includes("terminate for convenience") ? 0.9 : 0.1;
    return { model: "fake-jev", answers: { decision: { type: "noul", noul } } };
  },
};

const result = await decide({
  input: "Lorem ipsum. ".repeat(5_000) + "Either party may terminate for convenience.",
  question: { type: "noul", instructions: "Can either party terminate for convenience?" },
  provider: { transport: fake },
});
console.log(result.chunks.count, result.noul, result.confidence.agreement);
```

## Tokenizers

Chunk budgets and boundaries depend on a `Tokenizer`:

```ts
interface Tokenizer {
  readonly name: string;
  count(text: string): number; // deterministic, integer >= 0, 0 for "", roughly monotone as text grows
}
```

**The default is conservative.** Jev's tokenizer is not public, so exact counts are impossible. `defaultTokenizer`, a `HeuristicTokenizer` named `"heuristic-v1"`, makes one O(n) pass. It follows how BPE tokenizers pre-split text (letter runs, camelCase humps, digit groups of three, whitespace, punctuation, CJK, emoji) and charges each run a fixed cost. The costs are calibrated against `cl100k_base` and `o200k_base` so that estimates err high. On English prose it counts about 1.25 to 1.35 times what those tokenizers report (about 3.4 to 4 characters per estimated token on plain prose, about 3 on Markdown-heavy text). On JSON it counts about 1.1 to 1.35 times. Code, logs, CSV, numeric tables, HTML, Markdown, CJK (NFC), and emoji also estimate at or above both. The two failure modes have different costs. Overestimating means slightly smaller chunks and a few more requests. Underestimating means chunks that exceed Jev's real window, and each of those causes a context-limit error and a complete re-chunk pass. Known low estimates: accented Latin and some other scripts can fall below `cl100k_base`. Text dense in combining marks (pointed Hebrew, Arabic with harakat, decomposed Vietnamese) can also fall below `o200k_base`, by about 10 to 30%, and zalgo text by about 55%. Decomposed (NFD) Hangul runs about 35% below both, and halfwidth Katakana up to about 10%. Lines of numbers indented by two or three spaces run up to about 20% below both. Random strings run below both: letter runs such as DNA or protein sequences by about 55 to 60%, random Unicode code points by about 45 to 50%, base64 by about 30%, random printable ASCII by about 10%, and hex by about 5%. The safety reserve and the shrink-and-rechunk loop absorb these misses.

**Plugging in js-tiktoken.** `createTokenizer` wraps any counting function. Each result is validated: a non-finite or negative count throws `JevValidationError`, and fractional counts are rounded up.

```ts
import { readFile } from "node:fs/promises";
import { getEncoding } from "js-tiktoken";
import { createTokenizer, decide } from "jev-infinite-ctx";

const o200k = getEncoding("o200k_base");
// Passing [] and [] makes "<|endoftext|>"-style text in the input count as plain text.
// By default, js-tiktoken throws on it.
const tokenizer = createTokenizer((text) => o200k.encode(text, [], []).length, "o200k_base");

const result = await decide({
  input: await readFile("document.txt", "utf8"),
  question: { type: "noul", instructions: "Does the document mention a data breach?" },
  tokenizer,
  chunking: { contextSafetyReserve: 0.15 }, // optional: more headroom, since o200k only approximates Jev
});
console.log(result.noul, result.chunks.count);
```

Trade-offs: a real BPE tokenizer puts more text in each chunk, so there are fewer requests. It is still only an approximation of Jev's tokenizer, and it is much slower. The planner calls `count` many times on chunk-sized slices. In one measurement, planning 400K characters took about 0.1 s with the heuristic and about 3 s with js-tiktoken `o200k_base`.

## Errors and fail-closed semantics

Every error extends `JevInfiniteCTXError`.

| Error | When | Fields |
| --- | --- | --- |
| `JevValidationError` | Invalid request or options, empty input, a question too large for the window, a missing API key, or a first plan that exceeds `chunking.maxChunks`. Thrown before any request. Also thrown when a tokenizer returns an invalid count or throws (its error is the `cause`); on a re-chunk pass this can happen after requests were made. | |
| `JevProviderError` | Thrown by transports. You see it as the `cause` of the errors below. | `kind`, `status`, `retryAfterMs`, `retryable`, `body` |
| `JevResponseError` | Jev's answer does not match the question: wrong type, unknown label, invalid probabilities, or no `decision` answer. | |
| `JevChunkFailedError` | A chunk failed permanently: a non-retryable error, or retries used up. | `chunkIndex`, `attempts`, `cause` |
| `JevContextBudgetError` | Context-limit errors continued after `maxRechunks` re-chunks, the shrunk budget would fall below 256 tokens, or a re-chunk pass would exceed `chunking.maxChunks`. | `lastBudget`, `rechunks`, `cause` |
| `JevAbortError` | The caller's `signal` fired. Its `name` is `"AbortError"`. | `cause` (the signal's reason) |

How each kind of failure is handled:

- **Transient failures** (`rate_limit`, `server`, `overloaded`, `timeout`, `network`) are retried up to `execution.retries` times per chunk. The delay is `min(retryMaxDelayMs, retryBaseDelayMs * 2^(attempt - 1)) * (0.5 + 0.5 * random())`. A server `Retry-After` value overrides it, capped at `max(retryMaxDelayMs, 60 s)`.
- **Context limit** from any chunk: the current pass is cancelled and the new budget is `floor(min(budget, failingChunkTokens) * rechunkShrinkFactor)`. The entire input is then re-planned and every chunk runs again. With the defaults, a 28,436-token budget shrinks to about 21,300, then about 16,000, and so on. Using the failing chunk's size keeps a short input from resending the same oversized request. The overflow is never dropped.
- **Any other failure** cancels the requests in flight (through their `signal`) and rejects with a single error: the first failure, never a partial aggregate. A permanent chunk failure, including a `JevResponseError`, arrives as `JevChunkFailedError`, with the original error as `cause`.
- **Usage from discarded passes** is still counted. `usage.requests` counts every request sent. `usage.inputTokens`, `usage.outputTokens`, and `usage.costUsd` sum the usage reported in every response received, including responses from discarded re-chunk passes. A request that fails, times out, or is aborted in flight (siblings cancelled when a pass is discarded or fails) returns no usage. It counts in `usage.requests` but not in tokens or cost, so the provider may bill slightly more than `usage.costUsd` reports.

```ts
import { readFile } from "node:fs/promises";
import {
  decide,
  JevAbortError,
  JevChunkFailedError,
  JevContextBudgetError,
  JevProviderError,
  JevValidationError,
} from "jev-infinite-ctx";

try {
  const result = await decide({
    input: await readFile("document.txt", "utf8"),
    question: { type: "noul", instructions: "Is this document a contract?" },
  });
  console.log(result.noul);
} catch (error) {
  if (error instanceof JevChunkFailedError) {
    const cause = error.cause instanceof JevProviderError ? `${error.cause.kind} (${error.cause.status})` : error.cause;
    console.error(`chunk ${error.chunkIndex} failed after ${error.attempts} attempt(s):`, cause);
  } else if (error instanceof JevContextBudgetError) {
    console.error(`still over the context limit at ${error.lastBudget} tokens after ${error.rechunks} re-chunks`);
  } else if (error instanceof JevValidationError || error instanceof JevAbortError) {
    console.error(error.message);
  } else {
    throw error;
  }
}
```

## Observability

`onEvent` receives structured events. Events never contain source text. An exception thrown by the hook, or a rejected promise it returns, is swallowed.

| Event `type` | When | Fields |
| --- | --- | --- |
| `chunk.retry` | Before each backoff sleep | `chunkIndex`, `attempt` (the attempt that failed, 1-based), `delayMs`, `errorKind`, `status?` |
| `rechunk` | A context-limit error shrinks the budget | `previousBudget`, `newBudget`, `chunkIndex`, `status?` |
| `decision.completed` | Once, on success | `model`, `provider`, `questionType`, `inputTokensEstimated`, `inputTokensActual`, `chunkCount`, `effectiveChunkCount`, `overlap`, `stateTokenBudget`, `aggregationMethod`, `agreement`, `baseConfidence`, `adjustedConfidence`, `confidenceSource`, `finalAnswer`, `perChunkProbabilities`, `costUsd`, `latencyMs`, `retryCount`, `rechunkCount` |
| `decision.failed` | Once, on any failure after the request passes validation | `model`, `provider`, `errorName`, `errorKind?`, `chunkIndex?`, `latencyMs`, `retryCount`, `rechunkCount` |

`inputTokensActual` is the sum of the input tokens the provider reported. `finalAnswer` is the choice key, the score, or the noul probability. Put `decision.completed` events next to your ground-truth labels to evaluate C3 (spec 17).

```ts
import type { JevInfiniteCTXEvent } from "jev-infinite-ctx";

function logEvent(event: JevInfiniteCTXEvent): void {
  if (event.type === "decision.completed") {
    console.log(JSON.stringify({
      model: event.model,
      chunks: event.chunkCount,
      agreement: event.agreement,
      base: event.baseConfidence,
      adjusted: event.adjustedConfidence,
      answer: event.finalAnswer,
      costUsd: event.costUsd,
      latencyMs: event.latencyMs,
    }));
  }
}
// decide({ ..., onEvent: logEvent })
```

## Serialization

The runtime result always includes the raw per-chunk results. `serializeResult` returns a JSON-safe deep copy that shares no references with the result. You can leave out the chunk results or only the raw answers:

```ts
import { decide, serializeResult } from "jev-infinite-ctx";

const result = await decide({
  input: "The quarterly report shows revenue grew 12% year over year.",
  question: { type: "noul", instructions: "Does the text report revenue growth?" },
});

const full = serializeResult(result);                                     // everything
const compact = serializeResult(result, { includeChunkResults: false });  // drops chunks.results, keeps count etc.
const lean = serializeResult(result, { includeRawAnswers: false });       // keeps chunk results, drops each raw `answer`

await fetch("https://example.com/api/decisions", {
  method: "POST",
  headers: { "content-type": "application/json" },
  body: JSON.stringify(compact),
});
```

Both options default to `true`. The output survives `JSON.parse(JSON.stringify(x))` unchanged, and caller keys such as `"__proto__"` in probability maps are kept as ordinary keys.

## Lower-level API

The building blocks are exported too: `computeStateBudget`, `planChunks`, `aggregate`, `computeAgreement`, `totalVariation`, `effectiveChunkCount`, `baseConfidence`, `applyC3`, and `resolveTransport`. For example, you can see how many requests a document needs without calling Jev:

```ts
import { readFile } from "node:fs/promises";
import { computeStateBudget, DEFAULTS, defaultTokenizer, planChunks } from "jev-infinite-ctx";

const input = await readFile("document.txt", "utf8");
const budget = computeStateBudget({
  question: { type: "noul", instructions: "Is this document a contract?" },
  contextWindow: 32_000,
  tokenizer: defaultTokenizer,
  chunking: DEFAULTS.chunking,
});
const plan = planChunks(input, {
  stateTokenBudget: budget.usableStateTokens,
  overlap: DEFAULTS.chunking.overlap,
  tokenizer: defaultTokenizer,
  preferNaturalBoundaries: true,
});
console.log(budget.usableStateTokens, plan.chunks.length, plan.effectiveCount);
```

## Caveats

- **C3 is a heuristic, not a calibrated probability.** Its constants (`λ`, `A_floor`, `γ`, `cap`) are provisional. A `confidence.adjusted` of 0.9 does not mean a 90% chance of being correct. Until the parameters are fitted on labelled data (spec 17), read it as a signal that is normalized for fragmentation. The library deliberately does not call it a calibrated probability.
- **High base confidence can coexist with strong disagreement.** For choice and score, `C_base` averages each chunk's own confidence. Three chunks that each confidently pick a different option produce a high base. Read `confidence.agreement` (and the `totalVariation` of each chunk) alongside it. C3 does not boost low agreement, but it does not lower the base either.
- **Budgets are estimates.** Jev's tokenizer is private, so every budget is computed with a proxy tokenizer. The default errs high, and context-limit errors are handled by re-chunking, but no count is exact.
- **Noul confidence is derived.** It is `|2 * noul - 1|`, a restatement of the aggregate, labelled `source: "derived"`.
- **Chunks see only their own text.** A fact that needs context from far apart in the document may not be detectable from any single chunk. Overlap only helps near boundaries.
- **Cost grows with length.** Each chunk is one request that includes the question. Overlap and the conservative default tokenizer add some extra requests. Use `chunking.maxChunks` as a guard.
- **No business thresholds.** The library returns probabilities and confidence signals. Deciding what counts as "yes" or "confident enough" is up to you.

## Development

```bash
npm install
npm run typecheck    # tsc --noEmit over src and test
npm test             # vitest run (unit and fast-check property tests)
npm run test:watch   # vitest in watch mode
npm run build        # compile src/ to dist/ (tsconfig.build.json)
npm run check        # typecheck + test + build
```

Source layout:

| File | Responsibility |
| --- | --- |
| `src/index.ts` | Public entry point |
| `src/decide.ts` | Orchestrator: validation, budget, chunk passes, retries, re-chunking, result assembly |
| `src/validation.ts` | Request validation and default resolution |
| `src/tokenizer.ts` | `HeuristicTokenizer`, `createTokenizer` |
| `src/chunking.ts` | `computeStateBudget`, `planChunks` |
| `src/probability.ts` | Common probability representation and answer validation |
| `src/aggregation.ts` | The five aggregation methods |
| `src/agreement.ts` | Total variation and weighted agreement |
| `src/c3.ts` | Effective chunk count, base confidence, C3 |
| `src/concurrency.ts`, `src/retry.ts` | Concurrency limit and fail-fast cancellation; exponential backoff |
| `src/serialize.ts` | `serializeResult` |
| `src/transports/` | OpenRouter and TypeSafe direct transports, shared HTTP and error mapping |
| `src/types.ts`, `src/errors.ts`, `src/defaults.ts` | Public types, error classes, defaults |

License: MIT.
