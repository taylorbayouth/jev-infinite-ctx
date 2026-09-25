/**
 * The orchestrator (spec 5–12, 15, 16). It validates a request, budgets and
 * plans chunks, asks Jev the same question about every chunk, and reduces the
 * answers to one decision with transparent chunk metadata and C3 confidence.
 *
 * Fail-closed rules (spec 7, 15):
 * - a chunk that fails permanently fails the whole call (JevChunkFailedError);
 * - a provider context-limit error shrinks the budget and re-plans and re-runs
 *   the ENTIRE document, never just the overflowing chunk;
 * - nothing is ever aggregated from a partial document.
 *
 * Source text never leaves this module except as request state: events and
 * the errors created here carry offsets, counts, labels, and error kinds only
 * (spec 16).
 */

import { aggregate } from "./aggregation.js";
import { computeAgreement } from "./agreement.js";
import { applyC3, baseConfidence, effectiveChunkCount } from "./c3.js";
import { computeStateBudget, planChunks } from "./chunking.js";
import { mapWithConcurrency } from "./concurrency.js";
import { MIN_STATE_TOKENS, NOUL_LABELS, QUESTION_KEY } from "./defaults.js";
import {
  JevAbortError,
  JevChunkFailedError,
  JevContextBudgetError,
  JevInfiniteCTXError,
  JevProviderError,
  JevResponseError,
  JevValidationError,
} from "./errors.js";
import { clamp01, distributionToRecord, extractConfidence, toDistribution } from "./probability.js";
import { withRetry } from "./retry.js";
import { defaultTokenizer } from "./tokenizer.js";
import { DirectJevTransport } from "./transports/direct.js";
import { OpenRouterJevTransport } from "./transports/openrouter.js";
import { resolveOptions, validateRequest } from "./validation.js";
import type {
  ChunkPlan,
  ChunkResult,
  ChunkSpan,
  ChunkUsage,
  ConfidenceInfo,
  DecisionCompletedEvent,
  DecisionFailedEvent,
  Distribution,
  JevInfiniteCTXEvent,
  JevInfiniteCTXRequest,
  JevInfiniteCTXResult,
  JevQuestion,
  JevTransport,
  NativeJevAnswer,
  NativeJevResponse,
  ProviderOptions,
  ResolvedOptions,
  ResultFor,
  Tokenizer,
} from "./types.js";

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Turns `provider.transport` into a transport instance: "openrouter" (the
 * default) and "direct" construct the built-in transports with
 * `provider.apiKey` (falling back to their environment variable); an object is
 * used as is. Throws JevValidationError for a missing API key or an
 * unrecognized transport.
 */
export function resolveTransport(provider: ProviderOptions | undefined): JevTransport {
  const transport: unknown = provider?.transport ?? "openrouter";
  if (transport === "openrouter") return new OpenRouterJevTransport({ apiKey: provider?.apiKey });
  if (transport === "direct") return new DirectJevTransport({ apiKey: provider?.apiKey });
  if (
    typeof transport === "object" &&
    transport !== null &&
    typeof (transport as Partial<JevTransport>).decide === "function" &&
    typeof (transport as Partial<JevTransport>).contextWindow === "function"
  ) {
    return transport as JevTransport;
  }
  throw new JevValidationError(
    'provider.transport must be "openrouter", "direct", or an object implementing JevTransport.',
  );
}

/**
 * Answers one Jev question about an input of any length (spec 5).
 *
 * Rejects with:
 * - JevValidationError for an invalid request, before any provider call and
 *   before any event (the request, including `onEvent`, is not trusted until
 *   it validates), or when the question leaves no room for state, or when the
 *   plan exceeds `chunking.maxChunks`;
 * - JevChunkFailedError when a chunk fails permanently (non-retryable error,
 *   retries exhausted, or an answer that does not fit the question);
 * - JevContextBudgetError when context-limit errors persist after
 *   `execution.maxRechunks` re-chunk passes, or the budget would fall below
 *   MIN_STATE_TOKENS, or a re-chunk pass would exceed `chunking.maxChunks`;
 * - JevAbortError when `signal` fires.
 * Every rejection after validation emits "decision.failed"; success emits
 * "decision.completed".
 */
export async function decide<Q extends JevQuestion>(
  request: JevInfiniteCTXRequest<Q>,
): Promise<ResultFor<Q>> {
  const startedAt = performance.now();
  const req = validateRequest(request);
  const options = resolveOptions(req);
  const emit = createEmitter(req.onEvent);
  const tally: Tally = {
    requests: 0,
    retries: 0,
    rechunks: 0,
    inputTokens: 0,
    outputTokens: 0,
    costUsd: undefined,
  };

  let transport: JevTransport | undefined;
  try {
    throwIfAborted(req.signal);
    transport = resolveTransport(req.provider);
    const model = req.provider?.model ?? transport.defaultModel;
    const run: Run = {
      input: req.input,
      question: req.question,
      questions: { [QUESTION_KEY]: req.question },
      options,
      tokenizer: req.tokenizer ?? defaultTokenizer,
      transport,
      model,
      signal: req.signal,
      emit,
      tally,
    };

    const contextWindow = await lookupContextWindow(transport, model, req.signal);
    const budget = computeStateBudget({
      question: run.question,
      contextWindow,
      tokenizer: run.tokenizer,
      chunking: options.chunking,
    });
    const { plan, outcomes } = await runPasses(run, budget.usableStateTokens);
    const result = buildResult(run, plan, outcomes, elapsedSince(startedAt));
    emit(completedEvent(result));
    // ResultFor<Q> is a conditional type TypeScript cannot narrow; buildResult
    // returns the variant matching question.type, which is Q's type.
    return result as ResultFor<Q>;
  } catch (error) {
    emit(
      failedEvent(error, {
        provider: transport?.name ?? requestedProviderName(req.provider),
        model: req.provider?.model ?? transport?.defaultModel ?? "",
        latencyMs: elapsedSince(startedAt),
        tally,
      }),
    );
    throw error;
  }
}

// ---------------------------------------------------------------------------
// Run state
// ---------------------------------------------------------------------------

type Emit = (event: JevInfiniteCTXEvent) => void;

/** Everything a decision shares across passes and chunks. */
interface Run {
  readonly input: string;
  readonly question: JevQuestion;
  /** `{ [QUESTION_KEY]: question }`, one object reused by every request. */
  readonly questions: Record<string, JevQuestion>;
  readonly options: ResolvedOptions;
  readonly tokenizer: Tokenizer;
  readonly transport: JevTransport;
  readonly model: string;
  readonly signal: AbortSignal | undefined;
  readonly emit: Emit;
  readonly tally: Tally;
}

/** Counters over every request of the call, including discarded passes (types.ts UsageInfo). */
interface Tally {
  requests: number;
  retries: number;
  rechunks: number;
  inputTokens: number;
  outputTokens: number;
  /** Undefined until some response reports a cost. */
  costUsd: number | undefined;
}

/** One chunk's successful, validated answer. */
interface ChunkOutcome {
  answer: NativeJevAnswer;
  distribution: Distribution;
  confidence: number | undefined;
  usage: ChunkUsage | undefined;
  model: string;
  responseId: string | undefined;
  attempts: number;
  elapsedMs: number;
}

interface ContextLimitFailure {
  chunkIndex: number;
  error: JevProviderError;
}

// ---------------------------------------------------------------------------
// Context window and passes
// ---------------------------------------------------------------------------

/**
 * The transport's context window, or undefined when it is unknown, invalid,
 * or the lookup fails: metadata is advisory, and computeStateBudget then
 * uses its conservative fallback (spec 6.1). Only a caller abort rejects.
 */
function lookupContextWindow(
  transport: JevTransport,
  model: string,
  signal: AbortSignal | undefined,
): Promise<number | undefined> {
  // Never rejects: a throwing (even synchronously throwing) lookup means "unknown".
  const lookup = Promise.resolve()
    .then(() => transport.contextWindow(model))
    .then(
      (value: unknown) =>
        typeof value === "number" && Number.isFinite(value) && value > 0 ? value : undefined,
      () => undefined,
    );
  if (signal === undefined) return lookup;
  // A slow catalog lookup (bounded only by the transport's own timeout) must
  // not delay a caller abort.
  return new Promise((resolve, reject) => {
    const onAbort = (): void => reject(abortError(signal));
    if (signal.aborted) {
      onAbort();
      return;
    }
    signal.addEventListener("abort", onAbort, { once: true });
    void lookup.then((value) => {
      signal.removeEventListener("abort", onAbort);
      resolve(value);
    });
  });
}

/**
 * Plans the whole input and runs every chunk. On a context-limit error from
 * any chunk the pass is abandoned, the budget shrinks, and the ENTIRE input is
 * re-planned and re-run (spec 7, 15: never drop the overflow).
 */
async function runPasses(
  run: Run,
  initialBudget: number,
): Promise<{ plan: ChunkPlan; outcomes: ChunkOutcome[] }> {
  const { chunking } = run.options;
  let budget = initialBudget;
  let lastContextError: JevProviderError | undefined;
  for (;;) {
    const plan = planChunks(run.input, {
      stateTokenBudget: budget,
      overlap: chunking.overlap,
      tokenizer: run.tokenizer,
      preferNaturalBoundaries: chunking.preferNaturalBoundaries,
    });
    if (chunking.maxChunks !== undefined && plan.chunks.length > chunking.maxChunks) {
      throw tooManyChunks(plan, chunking.maxChunks, run.tally.rechunks, lastContextError);
    }

    let outcomes: ChunkOutcome[];
    try {
      outcomes = await runPass(run, plan);
    } catch (error) {
      const failure = contextLimitFailure(error);
      if (failure === undefined) throw error;
      budget = shrinkBudget(run, plan, failure);
      lastContextError = failure.error;
      continue;
    }
    return { plan, outcomes };
  }
}

/**
 * The budget for the next pass after `failure`, or JevContextBudgetError
 * when no further pass is allowed.
 *
 * The shrink factor applies to the failed chunk's estimated size, which is
 * at most the pass budget. For a multi-chunk plan the two are nearly equal;
 * for a short input (or a short final chunk) scaling the unused budget would
 * re-send the identical oversized request, wasting a pass on a known failure.
 */
function shrinkBudget(run: Run, plan: ChunkPlan, failure: ContextLimitFailure): number {
  const { maxRechunks, rechunkShrinkFactor } = run.options.execution;
  const budget = plan.stateTokenBudget;
  const failedTokens = plan.chunks[failure.chunkIndex]?.tokens ?? budget;
  const newBudget = Math.floor(Math.min(budget, failedTokens) * rechunkShrinkFactor);
  const rechunks = run.tally.rechunks;
  if (rechunks >= maxRechunks || newBudget < MIN_STATE_TOKENS) {
    const reason =
      rechunks >= maxRechunks
        ? `after ${rechunks} re-chunk pass(es) (execution.maxRechunks is ${maxRechunks})`
        : `and the state budget cannot shrink below ${MIN_STATE_TOKENS} tokens`;
    throw new JevContextBudgetError(
      `The provider rejected chunk ${failure.chunkIndex} as exceeding its context limit at a ` +
        `${budget}-token state budget ${reason}.`,
      { lastBudget: budget, rechunks, cause: failure.error },
    );
  }
  run.tally.rechunks += 1;
  run.emit({
    type: "rechunk",
    previousBudget: budget,
    newBudget,
    chunkIndex: failure.chunkIndex,
    ...withStatus(failure.error.status),
  });
  return newBudget;
}

/**
 * On the first pass this is a configuration problem caught before any
 * provider call. On a re-chunk pass the provider has already been called, and
 * the real cause is that the context limit cannot be met within maxChunks.
 */
function tooManyChunks(
  plan: ChunkPlan,
  maxChunks: number,
  rechunks: number,
  lastContextError: JevProviderError | undefined,
): JevInfiniteCTXError {
  const message =
    `The input needs ${plan.chunks.length} chunks at a ${plan.stateTokenBudget}-token state budget, ` +
    `more than chunking.maxChunks (${maxChunks}).`;
  if (lastContextError === undefined) return new JevValidationError(message);
  return new JevContextBudgetError(`${message} The budget was reduced after provider context-limit errors.`, {
    lastBudget: plan.stateTokenBudget,
    rechunks,
    cause: lastContextError,
  });
}

function contextLimitFailure(error: unknown): ContextLimitFailure | undefined {
  if (
    error instanceof JevChunkFailedError &&
    error.cause instanceof JevProviderError &&
    error.cause.kind === "context_limit"
  ) {
    return { chunkIndex: error.chunkIndex, error: error.cause };
  }
  return undefined;
}

/** Runs every chunk of a plan with bounded concurrency; results keep chunk order (spec 7). */
function runPass(run: Run, plan: ChunkPlan): Promise<ChunkOutcome[]> {
  return mapWithConcurrency(
    plan.chunks,
    run.options.execution.maxConcurrency,
    (chunk, _index, signal) => runChunk(run, chunk, plan.chunks.length, signal),
    { signal: run.signal },
  );
}

/**
 * Asks Jev about one chunk, retrying transient failures (spec 7). `signal`
 * is the pass's signal: it fires on a caller abort or a sibling's failure and
 * cancels both the backoff sleep and the in-flight request.
 */
async function runChunk(
  run: Run,
  chunk: ChunkSpan,
  chunkCount: number,
  signal: AbortSignal,
): Promise<ChunkOutcome> {
  const { execution } = run.options;
  const startedAt = performance.now();
  // The state is sliced per chunk and dropped after the call, so the input is
  // never held in duplicate (spec 15, extremely large input).
  const state = run.input.slice(chunk.start, chunk.end);
  let attempts = 0;
  try {
    const { value } = await withRetry(
      async (attempt) => {
        attempts = attempt;
        run.tally.requests += 1;
        if (attempt > 1) run.tally.retries += 1;
        const response = await run.transport.decide({
          model: run.model,
          state,
          questions: run.questions,
          signal,
        });
        return interpretResponse(run, response);
      },
      {
        retries: execution.retries,
        baseDelayMs: execution.retryBaseDelayMs,
        maxDelayMs: execution.retryMaxDelayMs,
        signal,
        onRetry: ({ attempt, delayMs, error }) =>
          run.emit({
            type: "chunk.retry",
            chunkIndex: chunk.index,
            attempt,
            delayMs,
            errorKind: error.kind,
            ...withStatus(error.status),
          }),
      },
    );
    return { ...value, attempts, elapsedMs: elapsedSince(startedAt) };
  } catch (error) {
    // Once the pass is cancelled (caller abort or a sibling's failure) this
    // chunk's error is a consequence, not the cause; mapWithConcurrency
    // reports the first failure.
    if (signal.aborted) throw error;
    throw new JevChunkFailedError(describeChunkFailure(chunk.index, chunkCount, attempts, error), {
      chunkIndex: chunk.index,
      attempts,
      cause: error,
    });
  }
}

/**
 * Validates a transport response against the question. Usage is recorded
 * first, so a response that is paid for but unusable still counts. Custom
 * transports are not type-checked at runtime, hence the shape guards.
 */
function interpretResponse(
  run: Run,
  response: NativeJevResponse,
): Omit<ChunkOutcome, "attempts" | "elapsedMs"> {
  const raw: unknown = response;
  if (!isRecord(raw)) {
    throw new JevResponseError("Jev response must be an object.");
  }
  const usage = recordUsage(run.tally, raw["usage"]);
  const answers = raw["answers"];
  if (!isRecord(answers) || !Object.hasOwn(answers, QUESTION_KEY)) {
    throw new JevResponseError(`Jev response has no answer for question "${QUESTION_KEY}".`);
  }
  const answer = answers[QUESTION_KEY] as NativeJevAnswer;
  const distribution = toDistribution(run.question, answer);
  const confidence = extractConfidence(answer);
  const model = raw["model"];
  const id = raw["id"];
  return {
    answer,
    distribution,
    confidence,
    usage,
    model: typeof model === "string" && model !== "" ? model : run.model,
    responseId: typeof id === "string" ? id : undefined,
  };
}

/** Adds a response's usage to the tally and returns its valid fields (non-negative finite numbers). */
function recordUsage(tally: Tally, value: unknown): ChunkUsage | undefined {
  if (!isRecord(value)) return undefined;
  const inputTokens = usageNumber(value["inputTokens"]);
  const outputTokens = usageNumber(value["outputTokens"]);
  const costUsd = usageNumber(value["costUsd"]);
  if (inputTokens !== undefined) tally.inputTokens += inputTokens;
  if (outputTokens !== undefined) tally.outputTokens += outputTokens;
  if (costUsd !== undefined) tally.costUsd = (tally.costUsd ?? 0) + costUsd;
  if (inputTokens === undefined && outputTokens === undefined && costUsd === undefined) return undefined;
  return {
    ...(inputTokens === undefined ? {} : { inputTokens }),
    ...(outputTokens === undefined ? {} : { outputTokens }),
    ...(costUsd === undefined ? {} : { costUsd }),
  };
}

function usageNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 ? value : undefined;
}

/**
 * Names the chunk and the failure kind. Only messages of this package's own
 * errors are quoted: they are built never to contain state, while a custom
 * transport's message might. The original error stays on `cause`.
 */
function describeChunkFailure(index: number, count: number, attempts: number, error: unknown): string {
  const head = `Chunk ${index} of ${count} failed after ${attempts} attempt${attempts === 1 ? "" : "s"}`;
  if (error instanceof JevInfiniteCTXError) return `${head}: ${error.message}`;
  return `${head}: ${error instanceof Error ? error.name : typeof error}`;
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

/**
 * Reduces chunk outcomes to the final result (spec 9–12). With one chunk the
 * answer is Jev's native one (spec 15): nothing is recomputed, agreement is 1,
 * and C3 makes no adjustment.
 */
function buildResult(
  run: Run,
  plan: ChunkPlan,
  outcomes: ChunkOutcome[],
  elapsedMs: number,
): JevInfiniteCTXResult {
  const { question, options, tally } = run;
  const weights = plan.chunks.map((chunk) => chunk.weight);
  const distributions = outcomes.map((outcome) => outcome.distribution);
  const native = outcomes.length === 1 ? outcomes[0] : undefined;

  const { distribution: combined, degenerate } =
    native === undefined
      ? aggregate(distributions, weights, options.aggregation)
      : { distribution: native.distribution, degenerate: false };
  const agreement = computeAgreement(distributions, combined, weights);
  const base = baseConfidence({
    type: question.type,
    aggregate: combined,
    chunkConfidences: outcomes.map((outcome) => outcome.confidence),
    weights,
  });
  const effectiveCount = effectiveChunkCount(plan.chunks);
  const c3 = applyC3({
    base: base.base,
    agreement: agreement.agreement,
    effectiveCount,
    chunkCount: plan.chunks.length,
    options: options.confidence,
  });

  const confidence: ConfidenceInfo = {
    base: base.base,
    adjusted: c3.adjusted,
    adjustment: c3.adjustment,
    source: base.source,
    agreement: agreement.agreement,
    method: c3.method,
    components: c3.components,
  };
  const results = plan.chunks.map((chunk, i) =>
    toChunkResult(chunk, outcomeAt(outcomes, i), agreement.perChunk[i] ?? 0),
  );
  const common = {
    confidence,
    aggregation: { method: options.aggregation, degenerate },
    chunks: {
      count: plan.chunks.length,
      effectiveCount,
      overlap: plan.overlapRatio,
      overlapTokens: plan.overlapTokens,
      stateTokenBudget: plan.stateTokenBudget,
      results,
    },
    usage: {
      inputTokens: tally.inputTokens,
      outputTokens: tally.outputTokens,
      costUsd: tally.costUsd,
      elapsedMs,
      inputTokensEstimated: plan.totalTokens,
      requests: tally.requests,
      retries: tally.retries,
      rechunks: tally.rechunks,
    },
    model: outcomeAt(outcomes, 0).model,
    provider: run.transport.name,
  };

  const answer = native?.answer;
  switch (question.type) {
    case "choice": {
      // spec 9.3: argmax of the aggregate.
      const choice = answer?.type === "choice" ? answer.choice : argmaxLabel(combined);
      return { type: "choice", choice, probabilities: distributionToRecord(combined), ...common };
    }
    case "score": {
      const maxLevel = question.criteria.length - 1;
      // spec 9.4: Σ j · P_agg[j]. Clamped because toDistribution accepts a
      // native score up to 1e-6 outside the level range.
      const raw = answer?.type === "score" ? answer.score : expectedLevel(combined);
      const score = Math.min(maxLevel, Math.max(0, raw));
      return {
        type: "score",
        score,
        normalizedScore: score / maxLevel,
        probabilities: distributionToRecord(combined),
        // The legend remains the caller's rubric, keyed by level index.
        legend: Object.fromEntries(question.criteria.map((criterion, level) => [String(level), criterion])),
        ...common,
      };
    }
    case "noul": {
      // spec 9.5: P_agg[yes].
      const noul = answer?.type === "noul" ? clamp01(answer.noul) : probabilityOf(combined, NOUL_LABELS[1]);
      return { type: "noul", noul, ...common };
    }
  }
}

function toChunkResult(chunk: ChunkSpan, outcome: ChunkOutcome, totalVariation: number): ChunkResult {
  return {
    index: chunk.index,
    start: chunk.start,
    end: chunk.end,
    estimatedTokens: chunk.tokens,
    uniqueTokens: chunk.uniqueTokens,
    weight: chunk.weight,
    boundary: chunk.boundary,
    answer: outcome.answer,
    probabilities: distributionToRecord(outcome.distribution),
    ...(outcome.confidence === undefined ? {} : { jevConfidence: outcome.confidence }),
    totalVariation,
    ...(outcome.usage === undefined ? {} : { usage: outcome.usage }),
    attempts: outcome.attempts,
    elapsedMs: outcome.elapsedMs,
    model: outcome.model,
    ...(outcome.responseId === undefined ? {} : { responseId: outcome.responseId }),
  };
}

/** mapWithConcurrency returns one outcome per chunk; a gap is an internal bug. */
function outcomeAt(outcomes: readonly ChunkOutcome[], index: number): ChunkOutcome {
  const outcome = outcomes[index];
  if (outcome === undefined) {
    throw new JevInfiniteCTXError(`Internal error: no outcome for chunk ${index}.`);
  }
  return outcome;
}

/** The most probable label; ties go to the earliest label in criteria order. */
function argmaxLabel(distribution: Distribution): string {
  let best = 0;
  distribution.probs.forEach((p, j) => {
    if (p > (distribution.probs[best] ?? -Infinity)) best = j;
  });
  const label = distribution.labels[best];
  if (label === undefined) throw new JevInfiniteCTXError("Internal error: empty distribution.");
  return label;
}

/** Σ j · P[j] over level indices. */
function expectedLevel(distribution: Distribution): number {
  return distribution.probs.reduce((sum, p, level) => sum + level * p, 0);
}

function probabilityOf(distribution: Distribution, label: string): number {
  const p = distribution.probs[distribution.labels.indexOf(label)];
  if (p === undefined) throw new JevInfiniteCTXError(`Internal error: no "${label}" probability.`);
  return p;
}

// ---------------------------------------------------------------------------
// Events (spec 16). Payloads hold numbers, labels, and error kinds only.
// ---------------------------------------------------------------------------

/** Wraps `onEvent` so a throwing (or rejecting) hook can never affect the decision. */
function createEmitter(onEvent: JevInfiniteCTXRequest["onEvent"]): Emit {
  if (onEvent === undefined) return () => undefined;
  return (event) => {
    try {
      const returned: unknown = onEvent(event);
      // An async hook's rejection is swallowed too, instead of going unhandled.
      if (returned instanceof Promise) returned.catch(() => undefined);
    } catch {
      // Swallowed by contract (types.ts onEvent).
    }
  };
}

function completedEvent(result: JevInfiniteCTXResult): DecisionCompletedEvent {
  return {
    type: "decision.completed",
    model: result.model,
    provider: result.provider,
    questionType: result.type,
    inputTokensEstimated: result.usage.inputTokensEstimated,
    inputTokensActual: result.usage.inputTokens,
    chunkCount: result.chunks.count,
    effectiveChunkCount: result.chunks.effectiveCount,
    overlap: result.chunks.overlap,
    stateTokenBudget: result.chunks.stateTokenBudget,
    aggregationMethod: result.aggregation.method,
    agreement: result.confidence.agreement,
    baseConfidence: result.confidence.base,
    adjustedConfidence: result.confidence.adjusted,
    confidenceSource: result.confidence.source,
    finalAnswer: finalAnswerOf(result),
    // Copies, so a hook cannot mutate the result. Spread keeps "__proto__" keys.
    perChunkProbabilities: result.chunks.results.map((chunk) => ({ ...chunk.probabilities })),
    costUsd: result.usage.costUsd,
    latencyMs: result.usage.elapsedMs,
    retryCount: result.usage.retries,
    rechunkCount: result.usage.rechunks,
  };
}

function finalAnswerOf(result: JevInfiniteCTXResult): string | number {
  switch (result.type) {
    case "choice":
      return result.choice;
    case "score":
      return result.score;
    case "noul":
      return result.noul;
  }
}

function failedEvent(
  error: unknown,
  context: { provider: string; model: string; latencyMs: number; tally: Tally },
): DecisionFailedEvent {
  const providerError = providerErrorOf(error);
  return {
    type: "decision.failed",
    model: context.model,
    provider: context.provider,
    errorName: error instanceof Error ? error.name : typeof error,
    ...(providerError === undefined ? {} : { errorKind: providerError.kind }),
    ...(error instanceof JevChunkFailedError ? { chunkIndex: error.chunkIndex } : {}),
    latencyMs: context.latencyMs,
    retryCount: context.tally.retries,
    rechunkCount: context.tally.rechunks,
  };
}

/** The provider error behind a failure: the error itself or its direct cause. */
function providerErrorOf(error: unknown): JevProviderError | undefined {
  if (error instanceof JevProviderError) return error;
  if (error instanceof JevInfiniteCTXError && error.cause instanceof JevProviderError) return error.cause;
  return undefined;
}

/** Provider label for a failure that happened before the transport was built. */
function requestedProviderName(provider: ProviderOptions | undefined): string {
  const transport = provider?.transport ?? "openrouter";
  return typeof transport === "string" ? transport : transport.name;
}

function withStatus(status: number | undefined): { status?: number } {
  return status === undefined ? {} : { status };
}

// ---------------------------------------------------------------------------
// Small utilities
// ---------------------------------------------------------------------------

function elapsedSince(startedAt: number): number {
  return Math.round(performance.now() - startedAt);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function abortError(signal: AbortSignal): JevAbortError {
  return new JevAbortError("The operation was aborted.", { cause: signal.reason });
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted) throw abortError(signal);
}
