/**
 * C3: Cross-Chunk Confidence Calibration (spec 11).
 *
 * C3 is a bounded, heuristic correction (spec 11.7): it may raise the base
 * confidence toward `cap` when many chunks agree, and never lowers it. It is
 * not a calibrated probability and must not be reported as Jev's own
 * confidence; the base is kept alongside it.
 */

import { C3_METHOD, NOUL_LABELS } from "./defaults.js";
import { JevInfiniteCTXError, JevValidationError } from "./errors.js";
import { clamp01, normalizeWeights, weightedSum } from "./probability.js";
import type {
  C3Components,
  ChunkSpan,
  ConfidenceInfo,
  ConfidenceSource,
  Distribution,
  JevQuestionType,
  ResolvedOptions,
} from "./types.js";

export interface BaseConfidenceResult {
  base: number;
  source: ConfidenceSource;
}

export interface C3Result {
  adjusted: number;
  /** adjusted − base; always >= 0. */
  adjustment: number;
  method: ConfidenceInfo["method"];
  components: C3Components;
}

/**
 * N_eff = Σ_i uniqueTokens_i / max_i tokens_i, clamped to [1, N] (spec 11.3):
 * the new content across all chunks, measured in units of the largest
 * chunk. It is derived from the actual unique-token counts, the same ones
 * that set the weights, so a chunk counts in proportion to the new content
 * it carries: a short final chunk adds only its share of a full chunk. For
 * N chunks of equal size with overlap ratio r it equals the approximation
 * 1 + (N − 1)(1 − r).
 */
export function effectiveChunkCount(
  chunks: ReadonlyArray<Pick<ChunkSpan, "tokens" | "uniqueTokens">>,
): number {
  if (chunks.length === 0) {
    throw new JevInfiniteCTXError("effectiveChunkCount() requires at least one chunk.");
  }
  let uniqueSum = 0;
  let largest = 1;
  chunks.forEach(({ tokens, uniqueTokens }, index) => {
    if (!Number.isFinite(tokens) || tokens < 0 || !Number.isFinite(uniqueTokens) || uniqueTokens < 0) {
      throw new JevInfiniteCTXError(
        `Chunk ${index} token counts must be finite and >= 0 (tokens ${tokens}, uniqueTokens ${uniqueTokens}).`,
      );
    }
    uniqueSum += uniqueTokens;
    largest = Math.max(largest, tokens);
  });
  // No chunk adds more than one chunk's worth of new content, so N_eff <= N.
  return Math.min(chunks.length, Math.max(1, uniqueSum / largest));
}

/**
 * C_base (spec 11.2).
 * - noul: |2 · P_agg[yes] − 1|, labeled "derived" (Jev has no noul confidence).
 * - choice/score: Σ w_i · jevConfidence_i, labeled "jev". If any chunk omitted
 *   its confidence, a weighted mean over the rest would silently over-weight
 *   the chunks that reported one, so the base is instead derived from the
 *   aggregate: 1 − H(P_agg) / ln K (natural-log entropy, 0 · ln 0 = 0).
 */
export function baseConfidence(args: {
  type: JevQuestionType;
  aggregate: Distribution;
  chunkConfidences: ReadonlyArray<number | undefined>;
  weights: readonly number[];
}): BaseConfidenceResult {
  const { type, aggregate, chunkConfidences } = args;
  if (chunkConfidences.length === 0) {
    throw new JevInfiniteCTXError("baseConfidence() requires at least one chunk.");
  }
  const weights = normalizeWeights(args.weights, chunkConfidences.length);
  aggregate.probs.forEach((p, j) => {
    if (!Number.isFinite(p) || p < 0) {
      throw new JevInfiniteCTXError(`Aggregate probability ${j} must be a finite number >= 0, got ${p}.`);
    }
  });

  if (type === "noul") {
    const yes = aggregate.labels.indexOf(NOUL_LABELS[1]);
    const p = aggregate.probs[yes];
    if (yes === -1 || p === undefined) {
      throw new JevInfiniteCTXError(`Noul aggregate has no "${NOUL_LABELS[1]}" probability.`);
    }
    return { base: clamp01(Math.abs(2 * p - 1)), source: "derived" };
  }

  const reported = chunkConfidences.filter((c): c is number => c !== undefined);
  reported.forEach((c) => {
    if (!Number.isFinite(c)) {
      throw new JevInfiniteCTXError(`Chunk confidence must be a finite number, got ${c}.`);
    }
  });
  if (reported.length === chunkConfidences.length) {
    return { base: clamp01(weightedSum(reported, weights)), source: "jev" };
  }
  return { base: clamp01(1 - normalizedEntropy(aggregate.probs)), source: "derived" };
}

/**
 * C_adjusted = C_base + max(0, C_cap − C_base) · S · G (spec 11.4–11.6), with
 *   S = 1 − exp(−λ · (N_eff − 1))                       (N_eff < 1 treated as 1)
 *   G = clamp((A − A_floor) / (1 − A_floor), 0, 1)^γ
 * Guarantees: one chunk ⇒ no adjustment; base <= adjusted <= max(base, cap),
 * so C3 never lowers Jev's confidence and never raises anything above cap;
 * A <= A_floor ⇒ no adjustment; non-decreasing in N_eff and in A.
 * With method "none" the components are still computed and reported.
 */
export function applyC3(args: {
  base: number;
  agreement: number;
  effectiveCount: number;
  chunkCount: number;
  options: ResolvedOptions["confidence"];
}): C3Result {
  const { base, agreement, effectiveCount, chunkCount, options } = args;
  validateConfidenceOptions(options);
  requireUnitInterval(base, "base");
  requireUnitInterval(agreement, "agreement");
  if (Number.isNaN(effectiveCount)) {
    throw new JevInfiniteCTXError("C3 effectiveCount must be a number, got NaN.");
  }
  if (!Number.isInteger(chunkCount) || chunkCount < 1) {
    throw new JevInfiniteCTXError(`C3 chunkCount must be an integer >= 1, got ${chunkCount}.`);
  }

  const { cap, lambda, agreementFloor, agreementExponent } = options;
  const nEff = Math.max(1, effectiveCount);
  // −expm1(−x) is 1 − e^(−x) without cancellation for small x; it is exactly 0 at N_eff = 1.
  const saturation = -Math.expm1(-lambda * (nEff - 1));
  const gate = clamp01((agreement - agreementFloor) / (1 - agreementFloor)) ** agreementExponent;
  const components: C3Components = {
    effectiveChunkCount: nEff,
    saturation,
    gate,
    cap,
    lambda,
    agreementFloor,
    agreementExponent,
  };

  if (options.method === "none") {
    return { adjusted: base, adjustment: 0, method: "none", components };
  }
  // spec 15: a single chunk is Jev's native answer; its confidence is never adjusted.
  if (chunkCount === 1) {
    return { adjusted: base, adjustment: 0, method: C3_METHOD, components };
  }

  // The outer min guards against a 1-ulp overshoot of cap from float rounding.
  const adjusted = Math.min(
    Math.max(base, cap),
    base + Math.max(0, cap - base) * saturation * gate,
  );
  return { adjusted, adjustment: adjusted - base, method: C3_METHOD, components };
}

/** H(p) / ln K in [0, 1]; 0 when there is a single outcome. */
function normalizedEntropy(probs: readonly number[]): number {
  if (probs.length <= 1) return 0;
  const entropy = probs.reduce((h, p) => (p > 0 ? h - p * Math.log(p) : h), 0);
  return entropy / Math.log(probs.length);
}

function requireUnitInterval(value: number, field: string): void {
  if (!(value >= 0 && value <= 1)) {
    throw new JevInfiniteCTXError(`C3 ${field} must be a number in [0, 1], got ${value}.`);
  }
}

/**
 * `applyC3` is public, so it re-checks the ranges `resolveOptions` enforces;
 * out-of-range parameters would otherwise yield NaN or unbounded output.
 */
function validateConfidenceOptions(options: ResolvedOptions["confidence"]): void {
  const { method, cap, lambda, agreementFloor, agreementExponent } = options;
  if (method !== "c3" && method !== "none") {
    throw new JevValidationError(`confidence.method must be "c3" or "none", got ${String(method)}.`);
  }
  if (!(cap > 0 && cap <= 1)) {
    throw new JevValidationError(`confidence.cap must be in (0, 1], got ${cap}.`);
  }
  if (!(lambda > 0 && Number.isFinite(lambda))) {
    throw new JevValidationError(`confidence.lambda must be a finite number > 0, got ${lambda}.`);
  }
  if (!(agreementFloor >= 0 && agreementFloor < 1)) {
    throw new JevValidationError(`confidence.agreementFloor must be in [0, 1), got ${agreementFloor}.`);
  }
  if (!(agreementExponent > 0 && Number.isFinite(agreementExponent))) {
    throw new JevValidationError(
      `confidence.agreementExponent must be a finite number > 0, got ${agreementExponent}.`,
    );
  }
}
