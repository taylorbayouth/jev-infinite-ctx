# API reference

```ts
import { decide, estimateTokens, DecideError } from "jev-long";
```

## `decide(options)`

Returns a promise of `ChoiceResult`, `ScoreResult`, or `NoulResult`, depending on the question type. With TypeScript, `result.choice` is typed with your option keys.

| Option | Default | |
| --- | --- | --- |
| `input` | required | The text. Must contain something other than whitespace, and be at most `maxInputTokens`. |
| `question` | required | A Jev question (below). |
| `combine` | `"average"` | `"average"`, or `"max"` for `noul` and `score` questions about whether something appears anywhere. TypeScript rejects `"max"` for `choice`. |
| `maxInputTokens` | `250000` | Largest input accepted, in estimated tokens. A positive integer. |
| `model` | `"typesafe/jev-1.13"` | Model id sent with each request. |
| `apiKey` | `process.env.OPENROUTER_API_KEY` | Bearer key for the default transport. The environment key is used only when `url` is left at its default, so it is never sent anywhere else. |
| `url` | `"https://openrouter.ai/api/alpha/decisions"` | Endpoint for the default transport: `http` or `https`, with no username or password. Pass `apiKey` with it. |
| `transport` | HTTP POST to `url` | Your own request function (below). `apiKey` and `url` are then unused. |
| `signal` | none | An `AbortSignal` to cancel the call. |

Unknown options are rejected, so a typo never silently falls back to a default.

### Questions

```ts
{ type: "choice", instructions, criteria: { billing: "...", technical: "...", other: "..." } } // at least 2 options
{ type: "score",  instructions, criteria: ["lowest level", "...", "highest level"] }             // 2 to 10 levels
{ type: "noul",   instructions, criteria?: { true: "...", false: "..." } }
```

`instructions` and each criterion can be a non-empty string, or JSON such as `{ what, not_for, examples }`, exactly as Jev accepts them.

### Results

```ts
interface ChoiceResult<K extends string> {
  type: "choice";
  choice: K;
  probabilities: Record<K, number>;
  confidence?: number;       // Jev's, averaged across chunks; absent only if Jev omitted it
  agreement: number;         // 0 to 1
  chunks: Array<{ start: number; end: number; weight: number; choice: K; probabilities: Record<K, number>; confidence?: number }>;
  model: string;             // the exact build that answered, e.g. "typesafe/jev-1.13-20260917"
  usage: { requests: number; inputTokens: number; costUsd?: number };
}

interface ScoreResult {
  type: "score";
  score: number;                          // 0 to levels - 1
  probabilities: Record<string, number>;  // keyed "0", "1", ...
  confidence?: number;
  agreement: number;
  chunks: Array<{ start: number; end: number; weight: number; score: number; probabilities: Record<string, number>; confidence?: number }>;
  model: string;
  usage: { requests: number; inputTokens: number; costUsd?: number };
}

interface NoulResult {
  type: "noul";
  noul: number;              // probability of yes
  agreement: number;
  chunks: Array<{ start: number; end: number; weight: number; noul: number }>;
  model: string;
  usage: { requests: number; inputTokens: number; costUsd?: number };
}
```

- `chunks` are in input order. `input.slice(start, end)` is exactly the text Jev saw, and the weights sum to 1.
- `usage.requests` counts every request, including retries and replanned passes. `inputTokens` and `costUsd` are what the provider reported.
- Results are plain data, so `JSON.stringify(result)` works. Drop `chunks` if you don't want to store them.

## `estimateTokens(text)`

`ceil(UTF-8 bytes / 3)`. This is the unit `maxInputTokens` uses. Call it to check an input's size before deciding.

## `DecideError`

The only error `decide()` throws.

| Property | |
| --- | --- |
| `code` | `"invalid_request"`, `"input_too_large"`, `"request_failed"`, or `"aborted"` |
| `message` | What went wrong. The built-in transport leaves out the provider's message if it shares 20 or more consecutive characters with the input. |
| `chunk` | For `request_failed`: the 0-based index of the chunk that failed. |
| `status` | For `request_failed`: the provider's HTTP status, if there was one. A rejection for length is always reported as 413. |
| `cause` | The underlying error. |

## Calling TypeSafe directly

TypeSafe's own API uses the same request and response format:

```ts
await decide({
  input,
  question,
  url: "https://api.typesafe.ai/v1/systemone",
  model: "jev-latest",
  apiKey: process.env.TYPESAFE_API_KEY,
});
```

This endpoint comes from third-party integration notes and has not been verified against TypeSafe's documentation yet.

## Custom transports

A transport sends one Decisions API request and returns the response body:

```ts
import type { Transport } from "jev-long";

const transport: Transport = async (request, signal) => {
  // request is { model, state, questions: { decision: question } }
  const response = await fetch("https://gateway.example.com/decisions", {
    method: "POST",
    headers: { "content-type": "application/json", authorization: `Bearer ${key}` },
    body: JSON.stringify(request),
    signal,
  });
  if (!response.ok) throw Object.assign(new Error(`HTTP ${response.status}`), { status: response.status });
  return response.json(); // { answers: { decision: <Jev answer> }, model?, usage? }
};
```

To control what happens on failure, throw an error with a numeric `status`:

- `408`, `429`, and `5xx` are retried. Set `retryAfterMs` to control the wait.
- `413` means the chunk was too long, so the input is planned again with smaller chunks.
- Anything else fails the call.

A `TypeError` counts as a network error and is retried. Honor `signal`: when it fires, `decide()` stops waiting either way, and a late response is dropped.

In tests, a transport can simply return canned answers:

```ts
const fake: Transport = async () => ({ answers: { decision: { type: "noul", noul: 0.9 } } });
```
