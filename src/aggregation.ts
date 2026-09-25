/**
 * Aggregation of per-chunk distributions (spec 9).
 *
 * The selected reducer is applied independently to each probability
 * dimension, then the result is renormalized to sum to 1 (spec 9.2). Only
 * `weighted_mean` uses the unique-content chunk weights (spec 6.4); the
 * other methods treat every chunk equally by design.
 */

import { JevInfiniteCTXError } from "./errors.js";
import { normalizeWeights, sameLabels, weightedSum } from "./probability.js";
import type { AggregationMethod, Distribution } from "./types.js";

export interface AggregateResult {
  distribution: Distribution;
  /**
   * True when the reduced vector summed to zero (for example `min` over
   * chunks whose mass is on disjoint labels) and the aggregate fell back to
   * uniform. Surfaced so the fallback is never silent.
   */
  degenerate: boolean;
}

/**
 * Reduces chunk distributions to one distribution. Pure: inputs are never
 * mutated and the returned labels array is a fresh copy.
 */
export function aggregate(
  distributions: Distribution[],
  weights: number[],
  method: AggregationMethod,
): AggregateResult {
  const { labels, columns } = toColumns(distributions);
  const w = normalizeWeights(weights, distributions.length);
  const reduce = reducerFor(method, w);

  const reduced = columns.map(reduce);
  const sum = reduced.reduce((acc, p) => acc + p, 0);
  if (!(sum > 0) || !Number.isFinite(sum)) {
    return {
      distribution: { labels: [...labels], probs: labels.map(() => 1 / labels.length) },
      degenerate: true,
    };
  }
  return {
    distribution: { labels: [...labels], probs: reduced.map((p) => p / sum) },
    degenerate: false,
  };
}

/** Per-dimension reducer: maps one column (a label's value in every chunk) to a number. */
function reducerFor(
  method: AggregationMethod,
  weights: readonly number[],
): (column: readonly number[]) => number {
  switch (method) {
    case "weighted_mean":
      return (column) => weightedSum(column, weights); // spec 9.2: Σ w_i · P_i[j]
    case "mean":
      return mean;
    case "median":
      return median;
    case "min":
      return (column) => column.reduce((a, b) => Math.min(a, b));
    case "max":
      return (column) => column.reduce((a, b) => Math.max(a, b));
    default: {
      const unknown: never = method;
      throw new JevInfiniteCTXError(`Unknown aggregation method ${JSON.stringify(unknown)}.`);
    }
  }
}

function mean(values: readonly number[]): number {
  return values.reduce((acc, v) => acc + v, 0) / values.length;
}

/** Middle value; for an even count, the mean of the two middle values. */
function median(values: readonly number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return mean(sorted.length % 2 === 1 ? sorted.slice(mid, mid + 1) : sorted.slice(mid - 1, mid + 1));
}

/**
 * Validates that every distribution shares the first one's labels and holds
 * finite, non-negative probabilities, and returns them column-wise:
 * `columns[j][i]` is chunk i's probability for label j. Mismatched vectors
 * are a programming error, never something to aggregate around.
 */
function toColumns(distributions: readonly Distribution[]): {
  labels: readonly string[];
  columns: number[][];
} {
  const first = distributions[0];
  if (first === undefined) {
    throw new JevInfiniteCTXError("aggregate() requires at least one distribution.");
  }
  const labels = first.labels;
  if (labels.length === 0) {
    throw new JevInfiniteCTXError("aggregate() requires at least one label.");
  }

  const columns = labels.map((): number[] => []);
  distributions.forEach((d, i) => {
    if (!sameLabels(d.labels, labels)) {
      throw new JevInfiniteCTXError(`Distribution ${i} has different labels than distribution 0.`);
    }
    if (d.probs.length !== labels.length) {
      throw new JevInfiniteCTXError(
        `Distribution ${i} has ${d.probs.length} probabilities for ${labels.length} labels.`,
      );
    }
    columns.forEach((column, j) => {
      const p = d.probs[j]; // undefined only for a sparse array
      if (p === undefined || !Number.isFinite(p) || p < 0) {
        throw new JevInfiniteCTXError(
          `Distribution ${i} probability ${j} must be a finite number >= 0, got ${p}.`,
        );
      }
      column.push(p);
    });
  });
  return { labels, columns };
}
