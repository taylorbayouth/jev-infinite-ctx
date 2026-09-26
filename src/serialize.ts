/**
 * JSON serialization of results (spec 12).
 *
 * Raw chunk results are always available on the runtime result object; this
 * helper lets applications drop them (or just the raw Jev answers) when the
 * payload size matters.
 */

import { JevInfiniteCTXError } from "./errors.js";
import type { ChunkResult, JevInfiniteCTXResult, JsonObject } from "./types.js";

export interface SerializeOptions {
  /** Include `chunks.results`. Default true. The other `chunks` fields are always kept. */
  includeChunkResults?: boolean | undefined;
  /** Include each chunk result's raw Jev `answer`. Default true. Irrelevant when chunk results are excluded. */
  includeRawAnswers?: boolean | undefined;
}

/**
 * Returns a JSON-safe deep copy of `result` that shares no references with
 * it and survives `JSON.parse(JSON.stringify(x))` unchanged.
 *
 * The copy is produced by a JSON round trip on purpose: the output is then
 * by construction exactly what JSON can represent (undefined fields dropped,
 * `-0` → 0, `toJSON` honored on caller-provided legend criteria), and
 * `JSON.parse` defines every key as an own data property, so caller keys
 * such as "__proto__" in probability maps survive instead of rewriting the
 * prototype.
 */
export function serializeResult<R extends JevInfiniteCTXResult>(
  result: R,
  options: SerializeOptions = {},
): JsonObject {
  const includeChunkResults = options.includeChunkResults ?? true;
  const includeRawAnswers = options.includeRawAnswers ?? true;

  // Spread overrides keep each key in its original position.
  const chunks = includeChunkResults
    ? {
        ...result.chunks,
        results: includeRawAnswers ? result.chunks.results : result.chunks.results.map(withoutAnswer),
      }
    : withoutResults(result.chunks);

  return toJsonObject({ ...result, chunks });
}

function withoutAnswer(chunk: ChunkResult): Omit<ChunkResult, "answer"> {
  const { answer: _omitted, ...rest } = chunk;
  return rest;
}

function withoutResults<C extends { results: unknown }>(chunks: C): Omit<C, "results"> {
  const { results: _omitted, ...rest } = chunks;
  return rest;
}

function toJsonObject(value: object): JsonObject {
  let parsed: unknown;
  try {
    parsed = JSON.parse(JSON.stringify(value));
  } catch (error) {
    // Only possible for a hand-built result (cycle, BigInt, throwing toJSON).
    throw new JevInfiniteCTXError("Result is not JSON-serializable.", { cause: error });
  }
  if (!isJsonObject(parsed)) {
    // Only reachable when a top-level toJSON replaces the result with a non-object.
    throw new JevInfiniteCTXError("Result did not serialize to a JSON object.");
  }
  return parsed;
}

/** JSON.parse output is JSON by construction; only the top-level shape needs checking. */
function isJsonObject(value: unknown): value is JsonObject {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
