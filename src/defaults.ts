/**
 * Default configuration (spec 14). All confidence parameters are provisional
 * until benchmarked (spec 11.7, 17).
 */

export const DEFAULTS = {
  aggregation: "weighted_mean",

  chunking: {
    overlap: 0.05,
    maxStateTokens: "auto",
    contextSafetyReserve: 0.08,
    protocolReserve: 1024,
    preferNaturalBoundaries: true,
    maxChunks: undefined,
  },

  execution: {
    maxConcurrency: 4,
    retries: 3,
    retryBaseDelayMs: 500,
    retryMaxDelayMs: 8000,
    maxRechunks: 4,
    rechunkShrinkFactor: 0.75,
  },

  confidence: {
    method: "c3",
    cap: 0.98,
    lambda: 0.25,
    agreementFloor: 0.5,
    agreementExponent: 2.0,
  },
} as const;

/** Context window assumed when the transport cannot report one (spec 6.1). */
export const FALLBACK_CONTEXT_WINDOW = 32_000;

/** Hard ceiling on state size when the context window is unknown (spec 6.1: ~28K). */
export const FALLBACK_MAX_STATE_TOKENS = 28_000;

/** Below this state budget the request is rejected as unworkable. */
export const MIN_STATE_TOKENS = 256;

/** Question key used for the single question sent with every chunk. */
export const QUESTION_KEY = "decision";

/** Label pair used for noul in the common probability representation. */
export const NOUL_LABELS = ["no", "yes"] as const;

/** Allowed range for the overlap ratio. */
export const MAX_OVERLAP = 0.5;

/** Score rubric level bounds (spec 4). */
export const MIN_SCORE_LEVELS = 2;
export const MAX_SCORE_LEVELS = 10;

/** Tolerance when checking that a returned probability map sums to 1. */
export const PROBABILITY_SUM_TOLERANCE = 0.1;

export const C3_METHOD = "c3-v1";
