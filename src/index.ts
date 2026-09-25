/**
 * JevInfiniteCTX: Jev's native choice, score, and noul decisions over inputs
 * of any length (spec 1). `decide` is the primary API; the building blocks
 * (chunking, aggregation, agreement, C3) are exported for inspection,
 * evaluation (spec 17), and custom pipelines.
 */

import { decide } from "./decide.js";
import { serializeResult } from "./serialize.js";

export type * from "./types.js";
export * from "./errors.js";
export { DEFAULTS } from "./defaults.js";

export { decide, resolveTransport } from "./decide.js";
export { serializeResult, type SerializeOptions } from "./serialize.js";

export {
  OpenRouterJevTransport,
  OPENROUTER_DEFAULT_MODEL,
  type OpenRouterJevTransportOptions,
} from "./transports/openrouter.js";
export {
  DirectJevTransport,
  DIRECT_DEFAULT_MODEL,
  type DirectJevTransportOptions,
} from "./transports/direct.js";

export { HeuristicTokenizer, defaultTokenizer, createTokenizer } from "./tokenizer.js";
export { planChunks, computeStateBudget } from "./chunking.js";
export { aggregate, type AggregateResult } from "./aggregation.js";
export { computeAgreement, totalVariation, type AgreementResult } from "./agreement.js";
export {
  effectiveChunkCount,
  baseConfidence,
  applyC3,
  type BaseConfidenceResult,
  type C3Result,
} from "./c3.js";

/** Namespace-style entry point: `JevInfiniteCTX.decide({ input, question })` (spec 5). */
export const JevInfiniteCTX = { decide, serializeResult } as const;
export default JevInfiniteCTX;
