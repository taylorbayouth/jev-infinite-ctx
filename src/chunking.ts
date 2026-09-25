/**
 * State budgeting and token-aware chunk planning (spec 6).
 *
 * `planChunks` never omits source text: chunks are a sliding window whose
 * union is the whole input, each chunk respects the token budget, chunk
 * boundaries prefer paragraph, sentence and whitespace breaks, and
 * neighbouring chunks overlap by about floor(budget × overlap) tokens.
 */

import {
  FALLBACK_CONTEXT_WINDOW,
  FALLBACK_MAX_STATE_TOKENS,
  MAX_OVERLAP,
  MIN_STATE_TOKENS,
} from "./defaults.js";
import { JevInfiniteCTXError, JevValidationError } from "./errors.js";
import { normalizeTokenCount } from "./tokenizer.js";
import type {
  ChunkBoundary,
  ChunkPlan,
  ChunkSpan,
  JevQuestion,
  ResolvedOptions,
  StateBudget,
  Tokenizer,
} from "./types.js";

/** Natural boundaries are searched for in this trailing fraction of a chunk's characters. */
const NATURAL_BOUNDARY_WINDOW = 0.2;
/** How far (in UTF-16 units) a chunk start may move back to reach a word start. */
const WORD_SNAP_MAX_CHARS = 32;
/** Initial characters-per-token estimate; later chunks use the observed ratio. */
const INITIAL_CHARS_PER_TOKEN = 4;
/** Smallest forward step when galloping towards a chunk's end. */
const MIN_GALLOP_STEP = 16;

// ---------------------------------------------------------------------------
// State budget (spec 6.1)
// ---------------------------------------------------------------------------

/**
 * Computes the per-chunk state budget:
 * usable = window − questionTokens − protocolReserve − safetyReserve, where
 * safetyReserve = ceil(contextSafetyReserve × remaining). An unknown window
 * falls back to FALLBACK_CONTEXT_WINDOW and is capped at
 * FALLBACK_MAX_STATE_TOKENS; a numeric `maxStateTokens` caps the result.
 * Throws JevValidationError when fewer than MIN_STATE_TOKENS remain.
 */
export function computeStateBudget(args: {
  question: JevQuestion;
  contextWindow: number | undefined;
  tokenizer: Tokenizer;
  chunking: ResolvedOptions["chunking"];
}): StateBudget {
  const { question, contextWindow, tokenizer, chunking } = args;
  assertTokenizer(tokenizer, "tokenizer");

  let serialized: string;
  try {
    serialized = JSON.stringify({
      type: question.type,
      instructions: question.instructions,
      criteria: question.criteria,
    });
  } catch (error) {
    throw new JevValidationError("question must be JSON-serializable", { cause: error });
  }
  const questionTokens = normalizeTokenCount(tokenizer.count(serialized), tokenizer.name);

  const windowKnown =
    typeof contextWindow === "number" && Number.isFinite(contextWindow) && contextWindow > 0;
  const window = windowKnown ? contextWindow : FALLBACK_CONTEXT_WINDOW;
  const remaining = window - questionTokens - chunking.protocolReserve;
  const safetyReserve = Math.ceil(Math.max(0, remaining) * chunking.contextSafetyReserve);

  let usable = Math.floor(remaining - safetyReserve);
  let source: StateBudget["source"] = windowKnown ? "metadata" : "fallback";
  if (!windowKnown) usable = Math.min(usable, FALLBACK_MAX_STATE_TOKENS);
  if (typeof chunking.maxStateTokens === "number" && chunking.maxStateTokens < usable) {
    usable = chunking.maxStateTokens;
    source = "caller";
  }

  if (usable < MIN_STATE_TOKENS) {
    if (source === "caller") {
      throw new JevValidationError(
        `chunking.maxStateTokens (${usable}) is below the minimum state budget of ${MIN_STATE_TOKENS} tokens`,
      );
    }
    throw new JevValidationError(
      `The question is too large for the context window: it uses about ${questionTokens} tokens, ` +
        `leaving ${Math.max(0, usable)} state tokens in a ${window}-token window after a ` +
        `${chunking.protocolReserve}-token protocol reserve and a ${safetyReserve}-token safety reserve ` +
        `(minimum ${MIN_STATE_TOKENS}). Shorten the instructions or criteria.`,
    );
  }

  return {
    contextWindow: windowKnown ? window : undefined,
    questionTokens,
    protocolReserve: chunking.protocolReserve,
    safetyReserve,
    usableStateTokens: usable,
    source,
  };
}

// ---------------------------------------------------------------------------
// Chunk planning (spec 6.2–6.4, 11.3)
// ---------------------------------------------------------------------------

/**
 * Splits `input` into chunks of at most `stateTokenBudget` tokens.
 *
 * Offsets are UTF-16 indices and never split a surrogate pair. When the whole
 * input fits, the plan is a single chunk. Otherwise each chunk extends as far
 * as the budget allows, is pulled back to the best natural boundary in its
 * last ~20% of characters when `preferNaturalBoundaries` is set, and the next
 * chunk starts early enough to share about floor(budget × overlap) tokens
 * with it. Weights and the effective chunk count come from unique tokens, so
 * overlap never gains extra voting power (spec 6.4).
 *
 * `stateTokenBudget` may be any positive integer (the orchestrator never goes
 * below MIN_STATE_TOKENS); a JevValidationError is thrown if the tokenizer
 * reports a single character as larger than the budget.
 */
export function planChunks(
  input: string,
  args: {
    stateTokenBudget: number;
    /** Ratio in [0, MAX_OVERLAP]. */
    overlap: number;
    tokenizer: Tokenizer;
    preferNaturalBoundaries: boolean;
  },
): ChunkPlan {
  const { stateTokenBudget: budget, overlap, tokenizer, preferNaturalBoundaries } = args;
  if (typeof input !== "string") {
    throw new JevValidationError("planChunks: input must be a string");
  }
  if (!Number.isSafeInteger(budget) || budget < 1) {
    throw new JevValidationError("planChunks: stateTokenBudget must be a positive integer");
  }
  if (typeof overlap !== "number" || !Number.isFinite(overlap) || overlap < 0 || overlap > MAX_OVERLAP) {
    throw new JevValidationError(`planChunks: overlap must be a finite number in [0, ${MAX_OVERLAP}]`);
  }
  if (typeof preferNaturalBoundaries !== "boolean") {
    throw new JevValidationError("planChunks: preferNaturalBoundaries must be a boolean");
  }
  assertTokenizer(tokenizer, "planChunks: tokenizer");

  const count = (text: string): number => normalizeTokenCount(tokenizer.count(text), tokenizer.name);
  const totalTokens = count(input);
  // Always < budget because overlap <= 0.5, which leaves room for new content.
  const overlapTarget = Math.floor(budget * overlap);

  const spans: Span[] =
    totalTokens <= budget
      ? [{ start: 0, end: input.length, tokens: totalTokens, overlapTokens: 0, boundary: "end" }]
      : new SlidingWindowPlanner(input, budget, overlapTarget, preferNaturalBoundaries, count, tokenizer.name).plan();

  const chunks: ChunkSpan[] = spans.map((span, index) => ({
    index,
    start: span.start,
    end: span.end,
    tokens: span.tokens,
    overlapTokens: span.overlapTokens,
    uniqueTokens: Math.max(1, span.tokens - span.overlapTokens),
    weight: 0,
    boundary: span.boundary,
  }));
  const uniqueSum = chunks.reduce((sum, chunk) => sum + chunk.uniqueTokens, 0);
  let effectiveCount = 1;
  for (const chunk of chunks) {
    chunk.weight = chunk.uniqueTokens / uniqueSum;
    // N_eff = 1 + Σ_{i≥1} unique_i / tokens_i (spec 11.3).
    if (chunk.index > 0) effectiveCount += chunk.uniqueTokens / Math.max(1, chunk.tokens);
  }

  return {
    chunks,
    stateTokenBudget: budget,
    overlapTokens: overlapTarget,
    overlapRatio: overlap,
    totalTokens,
    effectiveCount,
  };
}

interface Span {
  start: number;
  end: number;
  tokens: number;
  /** Tokens of [start, previous end). */
  overlapTokens: number;
  boundary: ChunkBoundary;
}

/** Where the next chunk starts, plus an end already known to fit from there. */
interface NextStart {
  start: number;
  overlapTokens: number;
  /** A code-point boundary strictly after the previous chunk's end. */
  fitEnd: number;
  fitTokens: number;
}

/**
 * Greedy sliding-window planner. Token counts are only ever taken over
 * slices about one chunk long, and each search needs O(log chunk) probes,
 * so the total counting work is O(n · log chunk) characters, never O(n²).
 */
class SlidingWindowPlanner {
  /** Observed characters per token, used to aim the first probe of each search. */
  private charsPerToken = INITIAL_CHARS_PER_TOKEN;

  constructor(
    private readonly input: string,
    private readonly budget: number,
    private readonly overlapTarget: number,
    private readonly preferNatural: boolean,
    private readonly count: (text: string) => number,
    private readonly tokenizerName: string,
  ) {}

  plan(): Span[] {
    const { input, budget } = this;
    const spans: Span[] = [];
    let start = 0;
    let overlapTokens = 0;
    let previousEnd = 0;
    // For the first chunk the only end known to fit is the empty slice.
    let fitEnd = 0;
    let fitTokens = 0;

    for (;;) {
      const fit = this.maxFittingEnd(start, fitEnd, fitTokens);
      let { end, tokens } = fit;
      let boundary: ChunkBoundary = end === input.length ? "end" : "hard";

      if (boundary === "hard" && this.preferNatural) {
        const natural = findNaturalBoundary(input, start, end, previousEnd);
        if (natural !== undefined) {
          // Re-verify: a real tokenizer is only approximately monotone.
          const naturalTokens = natural.end === end ? tokens : this.tokensBetween(start, natural.end);
          if (naturalTokens <= budget) {
            end = natural.end;
            tokens = naturalTokens;
            boundary = natural.kind;
          }
        }
      }

      if (tokens > budget || end <= previousEnd || start > previousEnd) {
        throw new JevInfiniteCTXError("Internal error: chunk plan violated its coverage or budget invariants");
      }
      spans.push({ start, end, tokens, overlapTokens, boundary });
      if (end === input.length) return spans;

      const next = this.nextStart(start, end);
      previousEnd = end;
      ({ start, overlapTokens, fitEnd, fitTokens } = next);
    }
  }

  private tokensBetween(from: number, to: number): number {
    return this.count(this.input.slice(from, to));
  }

  /**
   * The largest code-point boundary `end` with tokens(start, end) <= budget,
   * given `fitEnd` (>= start) already known to fit. Gallops forward from an
   * estimate until a probe overflows, then narrows the bracket, so the
   * remainder of a long input is never counted wholesale.
   */
  private maxFittingEnd(start: number, fitEnd: number, fitTokens: number): { end: number; tokens: number } {
    const { input, budget } = this;
    const length = input.length;
    let lo = fitEnd;
    let loTokens = fitTokens;
    let probe = Math.max(nextBoundary(input, lo), start + Math.ceil(budget * this.charsPerToken));
    let step = 0;
    let hi: number;
    let hiTokens: number;
    for (;;) {
      probe = alignUp(input, Math.min(probe, length));
      const tokens = this.tokensBetween(start, probe);
      if (tokens > budget) {
        hi = probe;
        hiTokens = tokens;
        break;
      }
      lo = probe;
      loTokens = tokens;
      if (probe === length) return this.observe(start, lo, loTokens);
      if (step === 0) step = Math.max(MIN_GALLOP_STEP, Math.floor((probe - start) / 8));
      probe += step;
      step *= 2;
    }

    const found = narrow(
      input,
      { pos: lo, value: loTokens },
      { pos: hi, value: hiTokens },
      budget + 0.5,
      this.probeUnit(),
      (end) => this.tokensBetween(start, end),
      (tokens) => tokens <= budget,
    );
    if (found.pos === start) throw this.characterTooLarge(start);
    return this.observe(start, found.pos, found.value);
  }

  /** Typical distance in characters between token-count steps. */
  private probeUnit(): number {
    return Math.max(1, Math.round(this.charsPerToken));
  }

  private observe(start: number, end: number, tokens: number): { end: number; tokens: number } {
    this.charsPerToken = (end - start) / Math.max(1, tokens);
    return { end, tokens };
  }

  /**
   * Chooses where the chunk after [start, end) begins (spec 6.3). The shared
   * region [next, end) holds about `overlapTarget` tokens and, when natural
   * boundaries are preferred, begins at a word start. Candidates fall back
   * from snapped to hard to no overlap, taking the first from which the next
   * chunk can extend at least one code point past `end`; that guarantees
   * strictly increasing ends and therefore termination for every input.
   */
  private nextStart(start: number, end: number): NextStart {
    const { input, budget } = this;
    const fitEnd = nextBoundary(input, end);
    const candidates: Array<{ start: number; overlapTokens: number | undefined }> = [];

    if (this.overlapTarget > 0) {
      const hard = this.overlapStart(start, end);
      if (hard !== undefined) {
        if (this.preferNatural) {
          // Snapping may at most double the shared region and never reaches this chunk's start.
          const limit = Math.max(start + 1, hard.start - Math.min(WORD_SNAP_MAX_CHARS, end - hard.start));
          const snapped = snapToWordStart(input, hard.start, limit);
          if (snapped !== hard.start) candidates.push({ start: snapped, overlapTokens: undefined });
        }
        candidates.push({ start: hard.start, overlapTokens: hard.tokens });
      }
    }
    candidates.push({ start: end, overlapTokens: 0 });

    for (const candidate of candidates) {
      const fitTokens = this.tokensBetween(candidate.start, fitEnd);
      if (fitTokens <= budget) {
        return {
          start: candidate.start,
          overlapTokens: candidate.overlapTokens ?? this.tokensBetween(candidate.start, end),
          fitEnd,
          fitTokens,
        };
      }
    }
    throw this.characterTooLarge(end);
  }

  /**
   * The largest code-point boundary s in (start, end) with
   * tokens(s, end) >= overlapTarget, searched leftwards from `end`. When even
   * the whole chunk after its first code point is below the target, returns
   * that position (maximal overlap). Undefined for a single-code-point chunk.
   */
  private overlapStart(start: number, end: number): { start: number; tokens: number } | undefined {
    const { input, overlapTarget: target } = this;
    const minStart = nextBoundary(input, start);
    if (minStart >= end) return undefined;

    let hi = end;
    let hiTokens = 0;
    let step = Math.max(1, Math.ceil(target * this.charsPerToken));
    let lo: number;
    let loTokens: number;
    for (;;) {
      const probe = Math.max(minStart, alignDown(input, Math.max(0, end - step)));
      const tokens = this.tokensBetween(probe, end);
      if (tokens >= target) {
        lo = probe;
        loTokens = tokens;
        break;
      }
      if (probe === minStart) return { start: minStart, tokens };
      hi = probe;
      hiTokens = tokens;
      step *= 2;
    }

    const found = narrow(
      input,
      { pos: lo, value: loTokens },
      { pos: hi, value: hiTokens },
      target - 0.5,
      this.probeUnit(),
      (from) => this.tokensBetween(from, end),
      (tokens) => tokens >= target,
    );
    return { start: found.pos, tokens: found.value };
  }

  private characterTooLarge(offset: number): JevValidationError {
    return new JevValidationError(
      `stateTokenBudget of ${this.budget} tokens cannot hold the single character at offset ${offset} ` +
        `according to tokenizer "${this.tokenizerName}"`,
    );
  }
}

// ---------------------------------------------------------------------------
// Bracket search
// ---------------------------------------------------------------------------

interface Probe {
  pos: number;
  value: number;
}

/**
 * Narrows a bracket of code-point boundaries lo.pos < hi.pos, where
 * accept(lo.value) holds and accept(hi.value) does not, down to adjacent
 * boundaries, and returns the accepted end.
 *
 * Token counts are roughly linear in characters, so each probe interpolates
 * towards `target` (midway between the last accepted and first rejected
 * count). The answer usually lies within a token of one endpoint while the
 * other is far away, which makes plain interpolation creep in from one side;
 * so each consecutive move of the same endpoint pushes the next guess
 * further towards the other one, by unit·(2^streak − 1) characters. Three
 * probes in a row that fail to halve the bracket force a bisection, which
 * bounds the probes at O(log width). Every returned value was measured, so
 * the result is verified even if the tokenizer is not perfectly monotone.
 */
function narrow(
  input: string,
  lo: Probe,
  hi: Probe,
  target: number,
  unit: number,
  evaluate: (pos: number) => number,
  accept: (value: number) => boolean,
): Probe {
  let streak = 0;
  let loMovedLast = true;
  let stalls = 0;
  for (;;) {
    const first = nextBoundary(input, lo.pos);
    if (first >= hi.pos) return lo;
    const width = hi.pos - lo.pos;
    let guess: number;
    if (stalls >= 3 || hi.value === lo.value) {
      guess = lo.pos + Math.floor(width / 2);
    } else {
      const bias = unit * (2 ** streak - 1);
      const estimate = lo.pos + ((target - lo.value) / (hi.value - lo.value)) * width;
      guess = Math.floor(loMovedLast ? estimate + bias : estimate - bias);
    }
    guess = Math.min(Math.max(guess, first), hi.pos - 1);
    // `first` is a boundary, so stepping back from a split pair stays >= first.
    if (!isBoundary(input, guess)) guess -= 1;

    const value = evaluate(guess);
    const accepted = accept(value);
    if (accepted) lo = { pos: guess, value };
    else hi = { pos: guess, value };
    streak = accepted === loMovedLast ? streak + 1 : 1;
    loMovedLast = accepted;
    stalls = (hi.pos - lo.pos) * 2 > width ? stalls + 1 : 0;
  }
}

// ---------------------------------------------------------------------------
// UTF-16 boundaries
// ---------------------------------------------------------------------------

function isHighSurrogate(code: number): boolean {
  return code >= 0xd800 && code <= 0xdbff;
}

function isLowSurrogate(code: number): boolean {
  return code >= 0xdc00 && code <= 0xdfff;
}

/** True unless `pos` falls between the halves of a surrogate pair. */
function isBoundary(input: string, pos: number): boolean {
  return (
    pos <= 0 ||
    pos >= input.length ||
    !(isHighSurrogate(input.charCodeAt(pos - 1)) && isLowSurrogate(input.charCodeAt(pos)))
  );
}

function alignUp(input: string, pos: number): number {
  return isBoundary(input, pos) ? pos : pos + 1;
}

function alignDown(input: string, pos: number): number {
  return isBoundary(input, pos) ? pos : pos - 1;
}

/** The first code-point boundary after `pos` (which must be < input.length). */
function nextBoundary(input: string, pos: number): number {
  return alignUp(input, pos + 1);
}

// ---------------------------------------------------------------------------
// Natural boundaries (spec 6.2)
// ---------------------------------------------------------------------------

const LF = 0x0a;
const PARAGRAPH_SEPARATOR = 0x2029;

/** Whitespace that may separate words. Excludes no-break spaces and U+FEFF. */
function isBreakingSpace(code: number): boolean {
  return (
    code === 0x20 ||
    (code >= 0x09 && code <= 0x0d) ||
    code === 0x85 ||
    code === 0x1680 ||
    (code >= 0x2000 && code <= 0x2006) ||
    (code >= 0x2008 && code <= 0x200a) ||
    code === 0x2028 ||
    code === 0x2029 ||
    code === 0x205f ||
    code === 0x3000
  );
}

/** Whitespace allowed on a "blank" line; includes CR so CRLF text works. */
function isBlankLineSpace(code: number): boolean {
  return code === 0x20 || code === 0x09 || code === 0x0d || code === 0x0b || code === 0x0c;
}

/** Sentence terminators that are followed by whitespace: . ! ? … and their CJK forms. */
function isTerminator(code: number): boolean {
  return (
    code === 0x2e ||
    code === 0x21 ||
    code === 0x3f ||
    code === 0x2026 ||
    isFullwidthTerminator(code)
  );
}

/** 。！？ end a sentence even without whitespace, since CJK text has none. */
function isFullwidthTerminator(code: number): boolean {
  return code === 0x3002 || code === 0xff01 || code === 0xff1f;
}

/** Closing quotes, brackets and emphasis that may follow a terminator. */
function isCloser(code: number): boolean {
  switch (code) {
    case 0x22: // "
    case 0x27: // '
    case 0x29: // )
    case 0x2a: // *
    case 0x5d: // ]
    case 0x5f: // _
    case 0x7d: // }
    case 0xbb: // »
    case 0x2019: // ’
    case 0x201d: // ”
    case 0x203a: // ›
    case 0x3009: // 〉
    case 0x300b: // 》
    case 0x300d: // 」
    case 0x300f: // 』
    case 0x3011: // 】
    case 0x3015: // 〕
    case 0xff09: // ）
    case 0xff3d: // ］
    case 0xff5d: // ｝
      return true;
    default:
      return false;
  }
}

/** Whether the text before `pos` ends with a terminator plus optional closers. */
function endsWithTerminator(input: string, pos: number, terminator: (code: number) => boolean): boolean {
  let k = pos - 1;
  while (k >= 0 && isCloser(input.charCodeAt(k))) k--;
  return k >= 0 && terminator(input.charCodeAt(k));
}

/**
 * The best natural end in the last NATURAL_BOUNDARY_WINDOW of [start, maxEnd]
 * and strictly after the previous chunk's end: the candidate closest to
 * `maxEnd` of the highest-priority class present (paragraph, then sentence,
 * then whitespace). Every candidate follows a BMP character, so it never
 * splits a surrogate pair.
 */
function findNaturalBoundary(
  input: string,
  start: number,
  maxEnd: number,
  previousEnd: number,
): { end: number; kind: "paragraph" | "sentence" | "whitespace" } | undefined {
  const lowest = Math.max(
    maxEnd - Math.floor((maxEnd - start) * NATURAL_BOUNDARY_WINDOW),
    previousEnd + 1,
    start + 1,
  );
  let end = lastParagraphEnd(input, lowest, maxEnd);
  if (end >= 0) return { end, kind: "paragraph" };
  end = lastSentenceEnd(input, lowest, maxEnd);
  if (end >= 0) return { end, kind: "sentence" };
  end = lastWhitespaceEnd(input, lowest, maxEnd);
  if (end >= 0) return { end, kind: "whitespace" };
  return undefined;
}

/** Largest e in [lowest, highest] just after a blank line (/\n[ \t\r]*\n/) or U+2029, else -1. */
function lastParagraphEnd(input: string, lowest: number, highest: number): number {
  for (let e = highest; e >= lowest; e--) {
    const code = input.charCodeAt(e - 1);
    if (code === PARAGRAPH_SEPARATOR) return e;
    if (code !== LF) continue;
    let k = e - 2;
    while (k >= 0 && isBlankLineSpace(input.charCodeAt(k))) k--;
    if (k >= 0 && input.charCodeAt(k) === LF) return e;
  }
  return -1;
}

/**
 * Largest e in [lowest, highest] that ends a sentence, else -1. Preferred is
 * the end of the whitespace after a terminator and optional closers; also
 * accepted is the position right after the closers, when whitespace follows
 * (reachable only at `highest`, when that whitespace does not fit) or when
 * the terminator is a CJK one. Each run is examined once, so the scan is
 * linear in the window.
 */
function lastSentenceEnd(input: string, lowest: number, highest: number): number {
  let e = highest;
  while (e >= lowest) {
    if (isBreakingSpace(input.charCodeAt(e - 1))) {
      let runStart = e - 1;
      while (runStart > 0 && isBreakingSpace(input.charCodeAt(runStart - 1))) runStart--;
      if (endsWithTerminator(input, runStart, isTerminator)) return e;
      e = runStart;
      continue;
    }
    const next = input.charCodeAt(e);
    // Never between closers, so closing quotes and brackets stay with their sentence.
    if (!isCloser(next)) {
      const terminator = isBreakingSpace(next) ? isTerminator : isFullwidthTerminator;
      if (endsWithTerminator(input, e, terminator)) return e;
    }
    e--;
  }
  return -1;
}

/** Largest e in [lowest, highest] just after a breaking whitespace character, else -1. */
function lastWhitespaceEnd(input: string, lowest: number, highest: number): number {
  for (let e = highest; e >= lowest; e--) {
    if (isBreakingSpace(input.charCodeAt(e - 1))) return e;
  }
  return -1;
}

/**
 * Moves a chunk start back (earlier) to the nearest word start (just after
 * breaking whitespace) at or above `limit`, so chunks do not begin mid-word.
 * Returns `pos` unchanged when it already is one or none is in reach.
 */
function snapToWordStart(input: string, pos: number, limit: number): number {
  for (let p = pos; p >= limit; p--) {
    if (isBreakingSpace(input.charCodeAt(p - 1))) return p;
  }
  return pos;
}

function assertTokenizer(tokenizer: Tokenizer, label: string): void {
  if (
    tokenizer === null ||
    typeof tokenizer !== "object" ||
    typeof tokenizer.count !== "function" ||
    typeof tokenizer.name !== "string"
  ) {
    throw new JevValidationError(`${label} must be an object with a count function and a string name`);
  }
}
