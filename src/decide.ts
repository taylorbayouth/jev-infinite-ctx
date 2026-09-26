import { type ChunkAnswer, combine, isRecord, readAnswer, validateQuestion } from "./answers.js";
import { estimateTokens, planChunks, type Span } from "./chunks.js";
import { DecideError } from "./errors.js";
import { OPENROUTER_URL, httpTransport } from "./http.js";
import type { DecideOptions, DecisionsRequest, Question, ResultFor, Transport, Usage } from "./types.js";

const DEFAULT_MODEL = "typesafe/jev-1.13";
const DEFAULT_MAX_INPUT_TOKENS = 250_000;
/** Jev's context window, in tokens. */
const CONTEXT_WINDOW = 32_000;
/** Share of the window left free for Jev's own prompt and for estimation error. */
const HEADROOM = 0.1;
/** Chunks smaller than this are not worth asking about; a question that leaves less is rejected. */
const MIN_CHUNK_TOKENS = 1_000;
const CONCURRENCY = 4;
const RETRIES = 3;
const TIMEOUT_MS = 60_000;
/** Times Jev may reject a chunk as too long before the call fails. Each time, chunks shrink by a quarter. */
const MAX_REPLANS = 3;

const OPTIONS = new Set(["input", "question", "combine", "maxInputTokens", "model", "apiKey", "url", "transport", "signal"]);

/**
 * Asks Jev one question about `input` (up to `maxInputTokens`) by asking it
 * about overlapping chunks and combining their answers. Throws `DecideError`.
 */
export async function decide<Q extends Question>(options: DecideOptions<Q>): Promise<ResultFor<Q>> {
  const run = prepare(options);
  let maxTokens = chunkTokens(run.question);
  for (let replans = 0; ; replans++) {
    const spans = planChunks(run.input, maxTokens);
    try {
      const answers = await askAll(run, spans);
      const result = {
        ...combine(run.question, spans, answers, run.combine),
        model: run.answeredBy ?? run.model,
        usage: run.usage,
      };
      return result as unknown as ResultFor<Q>;
    } catch (error) {
      if (!(error instanceof TooLong)) throw error;
      // Our estimate undercounted this input: shrink every chunk and plan the whole input again.
      const span = spans[error.chunk]!;
      maxTokens = Math.floor(Math.min(maxTokens, estimateTokens(run.input.slice(span.start, span.end))) * 0.75);
      if (replans === MAX_REPLANS || maxTokens < MIN_CHUNK_TOKENS) {
        throw failed(error.chunk, spans.length, error.cause, "Jev rejected it as too long, even after shrinking chunks");
      }
    }
  }
}

interface Run {
  input: string;
  question: Question;
  combine: "average" | "max";
  model: string;
  transport: Transport;
  signal: AbortSignal | undefined;
  usage: Usage;
  /** The exact model build that answered, as reported by the first response. */
  answeredBy: string | undefined;
}

function prepare(options: DecideOptions): Run {
  if (!isRecord(options)) throw invalid("decide() expects an options object.");
  for (const key of Object.keys(options)) if (!OPTIONS.has(key)) throw invalid(`Unknown option "${key}".`);
  const { input, question, combine = "average", maxInputTokens = DEFAULT_MAX_INPUT_TOKENS, model = DEFAULT_MODEL, signal } = options;
  if (typeof input !== "string" || input.trim() === "") throw invalid("input must be a non-empty string.");
  validateQuestion(question);
  if (combine !== "average" && combine !== "max") throw invalid('combine must be "average" or "max".');
  if (combine === "max" && question.type === "choice") {
    throw invalid('combine: "max" is for noul and score questions; choice answers are always averaged.');
  }
  if (!Number.isSafeInteger(maxInputTokens) || maxInputTokens <= 0) throw invalid("maxInputTokens must be a positive integer.");
  if (typeof model !== "string" || model.trim() === "") throw invalid("model must be a non-empty string.");
  if (signal !== undefined && !(signal instanceof AbortSignal)) throw invalid("signal must be an AbortSignal.");
  const transport = options.transport ?? defaultTransport(options.url, options.apiKey);
  if (typeof transport !== "function") throw invalid("transport must be a function.");
  chunkTokens(question); // Rejects an oversized question before any request.

  const tokens = estimateTokens(input);
  if (tokens > maxInputTokens) {
    throw new DecideError(
      "input_too_large",
      `The input is about ${tokens.toLocaleString("en-US")} tokens, over the limit of ` +
        `${maxInputTokens.toLocaleString("en-US")}. Split it, or raise maxInputTokens.`,
    );
  }
  if (signal?.aborted) throw aborted(signal);
  return { input, question, combine, model, transport, signal, usage: { requests: 0, inputTokens: 0 }, answeredBy: undefined };
}

function defaultTransport(url: unknown, apiKey: unknown): Transport {
  const key = apiKey ?? (typeof process === "undefined" ? undefined : process.env["OPENROUTER_API_KEY"]);
  if (typeof key !== "string" || key.trim() === "") throw invalid("No API key: set OPENROUTER_API_KEY, or pass apiKey.");
  const endpoint = url ?? OPENROUTER_URL;
  if (typeof endpoint !== "string" || !URL.canParse(endpoint)) throw invalid("url must be an absolute URL.");
  return httpTransport(endpoint, key);
}

/** The largest chunk Jev can take alongside this question. */
function chunkTokens(question: Question): number {
  const tokens = Math.floor((CONTEXT_WINDOW - estimateTokens(JSON.stringify(question))) * (1 - HEADROOM));
  if (tokens < MIN_CHUNK_TOKENS) {
    throw invalid("The question is too long: it leaves too little of Jev's context window for the input.");
  }
  return tokens;
}

/** Asks about every chunk, a few at a time. The first failure cancels the rest and fails the call. */
async function askAll(run: Run, spans: readonly Span[]): Promise<ChunkAnswer[]> {
  const pass = new AbortController();
  const signal = run.signal ? AbortSignal.any([run.signal, pass.signal]) : pass.signal;
  const answers: ChunkAnswer[] = [];
  let next = 0;
  const worker = async (): Promise<void> => {
    while (next < spans.length && !signal.aborted) {
      const index = next++;
      try {
        answers[index] = await ask(run, spans, index, signal);
      } catch (error) {
        pass.abort();
        throw error;
      }
    }
  };
  await Promise.all(Array.from({ length: Math.min(CONCURRENCY, spans.length) }, worker));
  // A caller abort between requests stops the workers without an error of their own.
  if (run.signal?.aborted) throw aborted(run.signal);
  return answers;
}

/** Asks about one chunk, retrying rate limits, server errors, network errors, and timeouts. */
async function ask(run: Run, spans: readonly Span[], index: number, pass: AbortSignal): Promise<ChunkAnswer> {
  const span = spans[index]!;
  const request: DecisionsRequest = {
    model: run.model,
    state: run.input.slice(span.start, span.end),
    questions: { decision: run.question },
  };
  for (let attempt = 1; ; attempt++) {
    const signal = AbortSignal.any([pass, AbortSignal.timeout(TIMEOUT_MS)]);
    let response: unknown;
    try {
      run.usage.requests++;
      response = await run.transport(request, signal);
    } catch (error) {
      if (pass.aborted) throw stopped(run, error);
      const timedOut = signal.aborted;
      const status = statusOf(error);
      if (status === 413) throw new TooLong(index, error);
      const retryable = timedOut || error instanceof TypeError || status === 408 || status === 429 || (status ?? 0) >= 500;
      if (!retryable || attempt > RETRIES) {
        throw failed(index, spans.length, error, timedOut ? `no response after ${TIMEOUT_MS / 1000}s` : undefined);
      }
      try {
        await sleep(backoff(attempt, error), pass);
      } catch {
        throw stopped(run, error);
      }
      continue;
    }
    record(run, response);
    try {
      return readAnswer(run.question, response);
    } catch (error) {
      throw failed(index, spans.length, error);
    }
  }
}

/** Jev said a chunk was too long; decide() shrinks the chunks and tries again. */
class TooLong extends Error {
  constructor(
    readonly chunk: number,
    override readonly cause: unknown,
  ) {
    super("Chunk too long for Jev.");
  }
}

function record(run: Run, response: unknown): void {
  if (!isRecord(response)) return;
  if (run.answeredBy === undefined && typeof response.model === "string" && response.model !== "") {
    run.answeredBy = response.model;
  }
  const usage = isRecord(response.usage) ? response.usage : {};
  if (typeof usage.input_tokens === "number" && usage.input_tokens >= 0) run.usage.inputTokens += usage.input_tokens;
  if (typeof usage.cost === "number" && usage.cost >= 0) run.usage.costUsd = (run.usage.costUsd ?? 0) + usage.cost;
}

function backoff(attempt: number, error: unknown): number {
  const after = isRecord(error) ? error.retryAfterMs : undefined;
  if (typeof after === "number" && after >= 0) return Math.min(after, 30_000);
  return 500 * 2 ** (attempt - 1) * (0.5 + Math.random() / 2);
}

function sleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) return reject(signal.reason);
    const onAbort = (): void => {
      clearTimeout(timer);
      reject(signal.reason);
    };
    const timer = setTimeout(() => {
      signal.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    signal.addEventListener("abort", onAbort, { once: true });
  });
}

function statusOf(error: unknown): number | undefined {
  const status = isRecord(error) ? error.status : undefined;
  return typeof status === "number" ? status : undefined;
}

/** Why a request stopped once its pass was cancelled: the caller aborted, or a sibling failed (that error wins). */
function stopped(run: Run, error: unknown): unknown {
  return run.signal?.aborted ? aborted(run.signal) : error;
}

function failed(index: number, count: number, error: unknown, reason?: string): DecideError {
  const detail = reason ?? (error instanceof Error ? error.message : String(error));
  return new DecideError("request_failed", `Chunk ${index + 1} of ${count} failed: ${detail}`, {
    chunk: index,
    status: statusOf(error),
    cause: error,
  });
}

function aborted(signal: AbortSignal): DecideError {
  return new DecideError("aborted", "The operation was aborted.", { cause: signal.reason });
}

function invalid(message: string): DecideError {
  return new DecideError("invalid_request", message);
}
