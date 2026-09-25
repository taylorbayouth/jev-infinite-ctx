/**
 * Cross-chunk agreement (spec 10).
 *
 * Agreement is measured with total variation distance between each chunk's
 * distribution and the aggregate, weighted by unique-content weights so
 * overlap does not double-count (spec 6.4). It gates C3 (spec 11.5): chunks
 * that disagree receive no fragmentation boost.
 */

import { JevInfiniteCTXError } from "./errors.js";
import { clamp01, normalizeWeights, sameLabels, weightedSum } from "./probability.js";
import type { Distribution } from "./types.js";

export interface AgreementResult {
  /** A = 1 − disagreement, in [0, 1]. */
  agreement: number;
  /** Σ w_i · TV_i, in [0, 1]. */
  disagreement: number;
  /** TV_i per chunk, in original chunk order. */
  perChunk: number[];
}

/**
 * Total variation distance 0.5 · Σ_j |p_j − q_j| (spec 10). Bounded to
 * [0, 1] for probability vectors; the clamp only absorbs float noise.
 */
export function totalVariation(p: readonly number[], q: readonly number[]): number {
  if (p.length !== q.length) {
    throw new JevInfiniteCTXError(
      `totalVariation() needs vectors of equal length, got ${p.length} and ${q.length}.`,
    );
  }
  // Non-null assertion is safe: the lengths were checked above.
  const distance = 0.5 * p.reduce((acc, pj, j) => acc + Math.abs(pj - q[j]!), 0);
  if (!Number.isFinite(distance)) {
    throw new JevInfiniteCTXError("totalVariation() received a non-finite probability.");
  }
  return clamp01(distance);
}

/**
 * Weighted agreement of chunk distributions with their aggregate.
 * `weights` are the chunk weights; they are renormalized defensively.
 */
export function computeAgreement(
  distributions: Distribution[],
  aggregate: Distribution,
  weights: number[],
): AgreementResult {
  if (distributions.length === 0) {
    throw new JevInfiniteCTXError("computeAgreement() requires at least one distribution.");
  }
  const w = normalizeWeights(weights, distributions.length);
  distributions.forEach((d, i) => {
    if (!sameLabels(d.labels, aggregate.labels)) {
      throw new JevInfiniteCTXError(`Distribution ${i} has different labels than the aggregate.`);
    }
  });

  // spec 15: a single chunk agrees with itself by definition.
  if (distributions.length === 1) {
    return { agreement: 1, disagreement: 0, perChunk: [0] };
  }

  const perChunk = distributions.map((d) => totalVariation(d.probs, aggregate.probs));
  // A convex combination of values in [0, 1]; the clamp only absorbs float noise.
  const disagreement = clamp01(weightedSum(perChunk, w));
  return { agreement: 1 - disagreement, disagreement, perChunk };
}
