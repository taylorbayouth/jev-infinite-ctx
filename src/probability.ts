/**
 * Common probability representation (spec 8).
 *
 * Every Jev answer type is converted into a `Distribution` over a fixed,
 * ordered label set so aggregation (spec 9), agreement (spec 10), and C3
 * (spec 11) share one mathematical representation.
 *
 * Validation here is strict: an answer that does not match its question
 * fails loudly with `JevResponseError` instead of being coerced into a
 * plausible-looking distribution. Error messages only ever mention labels,
 * types, and numbers from the answer, never chunk text.
 */

import { NOUL_LABELS, PROBABILITY_SUM_TOLERANCE } from "./defaults.js";
import { JevInfiniteCTXError, JevResponseError } from "./errors.js";
import type { Distribution, JevQuestion, NativeJevAnswer } from "./types.js";

/** Tiny negative probabilities (float noise from the provider) are clamped to 0. */
const NEGATIVE_PROBABILITY_TOLERANCE = 1e-9;

/** Slack allowed on scalar outputs (noul, score, confidence) before rejecting them. */
const SCALAR_RANGE_TOLERANCE = 1e-6;

/**
 * Absorbs float rounding in the sum check: for a sum of 1.1, `sum - 1`
 * evaluates to 0.10000000000000009, just outside a 0.1 tolerance.
 */
const SUM_FLOAT_SLACK = 1e-12;

/** Longest label/type echoed back in an error message. */
const MAX_ECHO_LENGTH = 64;

/**
 * Ordered labels for a question (spec 8):
 * - choice: `Object.keys(criteria)` (the caller's key order, as JS enumerates it)
 * - score: "0" .. String(levels - 1)
 * - noul: ["no", "yes"]
 */
export function labelsFor(question: JevQuestion): string[] {
  switch (question.type) {
    case "choice":
      return Object.keys(question.criteria);
    case "score":
      return Array.from({ length: question.criteria.length }, (_, level) => String(level));
    case "noul":
      return [...NOUL_LABELS];
    default:
      return unknownQuestionType(question);
  }
}

/**
 * Converts one native Jev answer into the common representation.
 * Throws `JevResponseError` when the answer does not match the question.
 */
export function toDistribution(question: JevQuestion, answer: NativeJevAnswer): Distribution {
  assertAnswerObject(answer);
  if (answer.type !== question.type) {
    throw new JevResponseError(
      `Jev returned answer type ${describeLabel(answer.type)} for a "${question.type}" question.`,
    );
  }

  const labels = labelsFor(question);
  switch (answer.type) {
    case "choice": {
      const choice: unknown = answer.choice;
      if (typeof choice !== "string" || !labels.includes(choice)) {
        throw new JevResponseError(
          `Jev choice ${describeLabel(choice)} is not one of the question's criteria keys.`,
        );
      }
      return { labels, probs: parseProbabilityMap(labels, answer.probabilities, "choice") };
    }
    case "score": {
      // The native score is reported as-is for a single chunk (spec 15), so it
      // must be a valid level index even though the distribution ignores it.
      checkScalar(answer.score, 0, labels.length - 1, "score");
      return { labels, probs: parseProbabilityMap(labels, answer.probabilities, "score") };
    }
    case "noul": {
      const p = checkScalar(answer.noul, 0, 1, "noul");
      return { labels, probs: [1 - p, p] }; // spec 8: P = [1 - p, p]
    }
    default:
      return unknownAnswerType(answer);
  }
}

/**
 * Keyed view of a distribution: `{ [label]: probability }`. Built with
 * `Object.fromEntries` so keys such as "__proto__" become own properties.
 */
export function distributionToRecord(d: Distribution): Record<string, number> {
  if (d.labels.length !== d.probs.length) {
    throw new JevInfiniteCTXError(
      `Distribution has ${d.labels.length} labels but ${d.probs.length} probabilities.`,
    );
  }
  return Object.fromEntries(d.probs.map((p, j) => [d.labels[j], p]));
}

/**
 * Jev's native confidence for choice/score, clamped into [0, 1]; `undefined`
 * when omitted (the schema marks it optional) and always for noul, which has
 * no native confidence (spec 4, 11.2).
 */
export function extractConfidence(answer: NativeJevAnswer): number | undefined {
  assertAnswerObject(answer);
  switch (answer.type) {
    case "noul":
      return undefined;
    case "choice":
    case "score": {
      // null is treated like an omitted field; either way C3 falls back to a
      // derived base confidence rather than failing the whole decision.
      const confidence: unknown = answer.confidence;
      if (confidence === undefined || confidence === null) return undefined;
      return checkScalar(confidence, 0, 1, "confidence");
    }
    default:
      return unknownAnswerType(answer);
  }
}

// ---------------------------------------------------------------------------
// Shared numeric helpers for aggregation, agreement, and C3 (not re-exported
// from the package entry point).
// ---------------------------------------------------------------------------

/** True when two label arrays have the same labels in the same order. */
export function sameLabels(a: readonly string[], b: readonly string[]): boolean {
  return a.length === b.length && a.every((label, j) => label === b[j]);
}

/**
 * Validates chunk weights and rescales them to sum to 1. Chunk weights
 * already sum to 1 (spec 6.4); renormalizing again is a defensive step so
 * callers passing raw unique-token counts get the same result.
 */
export function normalizeWeights(weights: readonly number[], expectedLength: number): number[] {
  if (weights.length !== expectedLength) {
    throw new JevInfiniteCTXError(`Expected ${expectedLength} weights, got ${weights.length}.`);
  }
  weights.forEach((w, i) => {
    if (!Number.isFinite(w) || w < 0) {
      throw new JevInfiniteCTXError(`Weight ${i} must be a finite number >= 0, got ${w}.`);
    }
  });
  const sum = weights.reduce((acc, w) => acc + w, 0);
  if (!(sum > 0) || !Number.isFinite(sum)) {
    throw new JevInfiniteCTXError(`Weights must have a positive finite sum, got ${sum}.`);
  }
  return weights.map((w) => w / sum);
}

/** Σ values[i] · weights[i] over two arrays of equal length. */
export function weightedSum(values: readonly number[], weights: readonly number[]): number {
  if (values.length !== weights.length) {
    throw new JevInfiniteCTXError(`Expected ${weights.length} values, got ${values.length}.`);
  }
  // Non-null assertion is safe: the lengths were checked above.
  return values.reduce((acc, v, i) => acc + v * weights[i]!, 0);
}

export function clamp01(x: number): number {
  return Math.min(1, Math.max(0, x));
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

/** Custom transports are not type-checked at runtime, so guard the answer's shape. */
function assertAnswerObject(answer: unknown): void {
  if (typeof answer !== "object" || answer === null || Array.isArray(answer)) {
    throw new JevResponseError(`Jev answer must be an object, got ${describeKind(answer)}.`);
  }
}

/**
 * Validates a probability map against the question's labels and returns the
 * probabilities in label order, renormalized to sum to 1 (spec 8). Missing
 * labels count as 0; unknown keys are an error because they mean Jev
 * answered a different question than the one asked.
 */
function parseProbabilityMap(labels: readonly string[], map: unknown, type: string): number[] {
  if (map === undefined || map === null) {
    throw new JevResponseError(`Jev ${type} answer is missing probabilities.`);
  }
  if (typeof map !== "object" || Array.isArray(map)) {
    throw new JevResponseError(
      `Jev ${type} probabilities must be an object keyed by label, got ${describeKind(map)}.`,
    );
  }

  // Map (not a plain object) so labels like "__proto__" index safely.
  const indexOf = new Map(labels.map((label, j) => [label, j]));
  const probs = new Array<number>(labels.length).fill(0);
  for (const [key, value] of Object.entries(map)) {
    const j = indexOf.get(key);
    if (j === undefined) {
      throw new JevResponseError(
        `Jev ${type} probabilities contain unknown label ${describeLabel(key)}.`,
      );
    }
    if (typeof value !== "number" || !Number.isFinite(value)) {
      throw new JevResponseError(
        `Jev ${type} probability for ${describeLabel(key)} must be a finite number, got ${describeKind(value)}.`,
      );
    }
    if (value < -NEGATIVE_PROBABILITY_TOLERANCE || value > 1 + PROBABILITY_SUM_TOLERANCE) {
      throw new JevResponseError(
        `Jev ${type} probability for ${describeLabel(key)} is out of range: ${value}.`,
      );
    }
    probs[j] = Math.max(0, value);
  }

  const sum = probs.reduce((acc, p) => acc + p, 0);
  if (sum <= 0) {
    throw new JevResponseError(`Jev ${type} probabilities sum to 0.`);
  }
  if (Math.abs(sum - 1) > PROBABILITY_SUM_TOLERANCE + SUM_FLOAT_SLACK) {
    throw new JevResponseError(
      `Jev ${type} probabilities sum to ${sum}, outside 1 ± ${PROBABILITY_SUM_TOLERANCE}.`,
    );
  }
  return probs.map((p) => p / sum);
}

/**
 * Checks that `value` is a finite number within [min, max] up to a small
 * tolerance, and clamps it into [min, max].
 */
function checkScalar(value: unknown, min: number, max: number, field: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new JevResponseError(`Jev ${field} must be a finite number, got ${describeKind(value)}.`);
  }
  if (value < min - SCALAR_RANGE_TOLERANCE || value > max + SCALAR_RANGE_TOLERANCE) {
    throw new JevResponseError(`Jev ${field} ${value} is outside [${min}, ${max}].`);
  }
  return Math.min(max, Math.max(min, value));
}

function unknownQuestionType(question: never): never {
  const type: unknown = (question as { type?: unknown }).type;
  throw new JevInfiniteCTXError(`Unknown question type ${describeLabel(type)}.`);
}

function unknownAnswerType(answer: never): never {
  const type: unknown = (answer as { type?: unknown }).type;
  throw new JevResponseError(`Unknown Jev answer type ${describeLabel(type)}.`);
}

/**
 * Quoted, truncated rendering of a label or answer type for error messages.
 * Labels come from the caller's criteria or Jev's option selection, never
 * from chunk text.
 */
function describeLabel(value: unknown): string {
  if (typeof value !== "string") return describeKind(value);
  const shown = value.length > MAX_ECHO_LENGTH ? `${value.slice(0, MAX_ECHO_LENGTH)}…` : value;
  return JSON.stringify(shown);
}

/** Numbers and booleans as-is; anything else by kind only (string content is never echoed). */
function describeKind(value: unknown): string {
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  if (value === null) return "null";
  if (Array.isArray(value)) return "an array";
  if (typeof value === "string") return "a string";
  return typeof value;
}
