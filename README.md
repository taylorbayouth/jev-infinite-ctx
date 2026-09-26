# jev-long

Ask [Jev](https://openrouter.ai/typesafe/jev-1.13) one question about an input of up to 250K tokens, and get one answer back.

Jev is TypeSafe's decision model. It answers typed questions (`choice`, `score`, `noul`) with probabilities instead of prose, and its context window is about 32K tokens. jev-long is for the times your input is bigger than that: a long contract, a support history, a transcript, a log file.

## The idea

```
  your input (up to 250K tokens)
               │
  overlapping chunks that each fit Jev
               │
  the same question, asked about every chunk
               │
  answers averaged, weighted by the new text each chunk covers
               │
  one answer, plus how much the chunks agreed
```

## Install

```bash
npm install github:taylorbayouth/jev-infinite-ctx   # not on npm yet
```

jev-long calls Jev through OpenRouter's Decisions API. Set `OPENROUTER_API_KEY`, or pass `apiKey`.

## Quick start

```ts
import { decide } from "jev-long";

const result = await decide({
  input: document, // any string up to about 250K tokens
  question: {
    type: "choice",
    instructions: "What kind of document is this?",
    criteria: {
      contract: "A legal agreement between parties",
      report: "A report, analysis, or research paper",
      correspondence: "Emails, letters, or chat logs",
      other: "Anything else",
    },
  },
});

result.choice;        // "contract"
result.probabilities; // { contract: 0.91, report: 0.05, correspondence: 0.02, other: 0.02 }
result.confidence;    // 0.88
result.agreement;     // 0.96
```

Questions are exactly Jev's: `choice` picks one of your options, `score` places the input on your ordered rubric, and `noul` gives the probability of yes.

## What you get back

| Field | What it is |
| --- | --- |
| `choice`, `score`, or `noul` | The answer, in Jev's format for the question type. |
| `probabilities` | Combined probability of each option or score level (`choice` and `score`). |
| `confidence` | Jev's own confidence, averaged across chunks (`choice` and `score`, as in Jev). |
| `agreement` | How consistently the chunks answered, from 0 to 1. 1 means every chunk said the same thing. |
| `chunks` | Where each chunk sits in the input, its weight, and Jev's answer for it. |

Read `confidence` and `agreement` together. High confidence with low agreement means each chunk was sure, but not of the same answer. `result.chunks` shows you which ones.

## "Does it appear anywhere?"

Averaging suits questions about the input as a whole: what kind of document is this, how frustrated is this customer overall. It is wrong for questions about whether something appears anywhere. If only one chunk in ten contains an automatic renewal clause, the average says "probably not".

For those questions, use `combine: "max"`. The answer comes from the chunk with the highest yes-probability or score.

```ts
const renewal = await decide({
  input: contract,
  question: { type: "noul", instructions: "Does this text contain an automatic renewal clause?" },
  combine: "max",
});

renewal.noul; // 0.97, from the chunk that contains the clause
```

`max` works for `noul` and `score` questions ("How severe is the worst problem described?").

## How it works

1. **Split.** The input is cut at sentence and line breaks into chunks that fit Jev's window, all about the same size. Each chunk repeats about 5% of the one before it, so no sentence loses its context at a boundary.
2. **Ask.** Jev answers the same question about every chunk, four requests at a time. Rate limits, server errors, and timeouts are retried.
3. **Combine.** Each chunk's probabilities count in proportion to the text only it covers, so the repeated overlap never counts twice. The weighted average is the answer.
4. **Measure agreement.** Agreement is 1 minus the average difference between every pair of chunk answers.

Formulas and design notes: [docs/how-it-works.md](docs/how-it-works.md).

## Guarantees

- **Nothing is silently dropped.** Every character of the input is in a chunk. An input over the limit is rejected before any request is sent.
- **No partial answers.** If any chunk still fails after retries, the whole call fails. You never get an answer computed from part of the input.
- **Overlap never counts twice.**
- **Jev's meaning is kept.** Choice keys come back exactly as you wrote them, and scores stay on your rubric's scale. An input that fits in one chunk returns Jev's own answer.
- **Deterministic.** The same chunk answers always combine into the same result.

## Tradeoffs

- **Token counts are estimates.** Jev's tokenizer is not public, so jev-long counts 3 bytes of UTF-8 as one token. That is deliberately high for English, so 250K tokens is roughly 750KB of text. If Jev still rejects a chunk as too long, jev-long makes the chunks smaller and starts over.
- **Each chunk sees only its own text.** Questions whose answer depends on connecting distant parts ("does the conclusion contradict the introduction?") are beyond what any single chunk can see.
- **Cost grows with length.** A 250K-token input takes about ten Jev requests, which costs about a cent.

## Options

```ts
await decide({
  input,
  question,
  combine: "average",       // or "max", for noul and score "appears anywhere" questions
  maxInputTokens: 250_000,  // raise or lower the limit
  model: "typesafe/jev-1.13",
  apiKey: process.env.OPENROUTER_API_KEY,
  signal,                   // an AbortSignal, to cancel
});
```

To call another Decisions endpoint (such as TypeSafe's own API) or to swap the HTTP call for your own function, see [docs/api.md](docs/api.md).

## Errors

`decide()` throws a `DecideError` with a `code`:

| `code` | When |
| --- | --- |
| `invalid_request` | The options or the question are invalid. Nothing was sent. |
| `input_too_large` | The input is over `maxInputTokens`. Nothing was sent. |
| `request_failed` | A chunk failed after retries. `error.chunk` and `error.status` say which and why. |
| `aborted` | Your `signal` fired. |

## Development

```bash
npm install
npm run check        # typecheck, test, build
npm run smoke:live   # four real decisions against OpenRouter (needs OPENROUTER_API_KEY, about a cent)
```

MIT License.
