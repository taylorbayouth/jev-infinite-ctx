/**
 * OpenRouter transport (spec 13). Uses OpenRouter's Decisions API
 * (`POST /api/alpha/decisions`), not chat completions, and resolves the
 * model's context window from OpenRouter's public model catalog.
 */

import { JevValidationError } from "../errors.js";
import type { JevTransport, NativeJevRequest, NativeJevResponse } from "../types.js";
import {
  encodeDecisionBody,
  fetchText,
  isRecord,
  mergeHeaders,
  normalizeBaseUrl,
  optionalString,
  own,
  postDecision,
  resolveApiKey,
  resolveContextWindowOption,
  resolveFetch,
  resolveTimeoutMs,
  type HttpDecisionConfig,
} from "./http.js";

export const OPENROUTER_DEFAULT_MODEL = "typesafe/jev-1.13";

/**
 * Every field also accepts an explicit `undefined`, read as omitted, so
 * options built from possibly-unset values (`process.env.X`) compile under
 * `exactOptionalPropertyTypes`.
 */
export interface OpenRouterJevTransportOptions {
  /** Default: process.env.OPENROUTER_API_KEY. */
  apiKey?: string | undefined;
  /** Default "https://openrouter.ai". A trailing slash is tolerated. */
  baseUrl?: string | undefined;
  /** Default "typesafe/jev-1.13". */
  defaultModel?: string | undefined;
  /** Default: the global fetch, looked up at request time. */
  fetch?: typeof fetch | undefined;
  /** Per decision request, including reading the body. Default 60_000. */
  timeoutMs?: number | undefined;
  /** Extra request headers. They cannot replace Authorization or Content-Type. */
  headers?: Record<string, string> | undefined;
  /** Sent as X-OpenRouter-Title and X-Title (app attribution). */
  appName?: string | undefined;
  /** Sent as HTTP-Referer (app attribution). */
  appUrl?: string | undefined;
  /** Sent as body.session_id (observability grouping). */
  sessionId?: string | undefined;
  /** Sent as body.user (per-end-user attribution). */
  user?: string | undefined;
  /** Explicit context window in tokens; skips the metadata lookup. */
  contextWindow?: number | undefined;
  /** Default true: look the model up in GET {baseUrl}/api/v1/models?output_modalities=decisions. */
  resolveContextWindow?: boolean | undefined;
  /** Deadline for the model catalog request. Default 5_000. */
  metadataTimeoutMs?: number | undefined;
}

const LABEL = "OpenRouter";
const DEFAULT_BASE_URL = "https://openrouter.ai";
const DECISIONS_PATH = "/api/alpha/decisions";
const MODELS_PATH = "/api/v1/models?output_modalities=decisions";
const DEFAULT_TIMEOUT_MS = 60_000;
const DEFAULT_METADATA_TIMEOUT_MS = 5_000;

/** Spec 2: the Jev family is exposed on OpenRouter with a 32K context window. */
const JEV_STATIC_CONTEXT_WINDOW = 32_000;

/**
 * The catalog changes rarely, so a successful lookup is reused for an hour.
 * A failed lookup is remembered briefly so a burst of calls does not each wait
 * out a dead endpoint, then retried.
 */
const MODEL_LIST_TTL_MS = 60 * 60_000;
const MODEL_LIST_FAILURE_TTL_MS = 60_000;

interface ModelMetadata {
  id: string;
  canonicalSlug: string | undefined;
  contextLength: number | undefined;
}

interface ModelListCacheEntry {
  promise: Promise<readonly ModelMetadata[]>;
  /** Infinity while the request is in flight, so concurrent callers share it. */
  expiresAt: number;
}

/**
 * Module-level catalog cache, shared across transport instances (the
 * orchestrator creates one per call). Keyed by fetch implementation, then by
 * catalog URL, so instances with different injected fetches never see each
 * other's data.
 */
const modelListCache = new WeakMap<typeof fetch, Map<string, ModelListCacheEntry>>();

export class OpenRouterJevTransport implements JevTransport {
  readonly name = "openrouter";
  readonly defaultModel: string;

  readonly #decision: HttpDecisionConfig;
  readonly #modelsUrl: string;
  readonly #modelsHeaders: Record<string, string>;
  readonly #sessionId: string | undefined;
  readonly #user: string | undefined;
  readonly #contextWindow: number | undefined;
  readonly #resolveContextWindow: boolean;
  readonly #metadataTimeoutMs: number;

  /** Throws JevValidationError on bad options or a missing API key, before any request. */
  constructor(options: OpenRouterJevTransportOptions = {}) {
    if (!isRecord(options)) throw new JevValidationError(`${LABEL}: options must be an object`);

    const apiKey = resolveApiKey(options.apiKey, "OPENROUTER_API_KEY", LABEL);
    const baseUrl = normalizeBaseUrl(options.baseUrl, DEFAULT_BASE_URL, LABEL);
    const fetchFn = resolveFetch(options.fetch, LABEL);

    const defaultModel = options.defaultModel ?? OPENROUTER_DEFAULT_MODEL;
    if (typeof defaultModel !== "string" || defaultModel.trim() === "") {
      throw new JevValidationError(`${LABEL}: defaultModel must be a non-empty string`);
    }
    this.defaultModel = defaultModel;

    const appName = optionalString(options.appName, "appName", LABEL);
    const appUrl = optionalString(options.appUrl, "appUrl", LABEL);
    this.#sessionId = optionalString(options.sessionId, "sessionId", LABEL);
    this.#user = optionalString(options.user, "user", LABEL);

    const authorization = `Bearer ${apiKey}`;
    this.#decision = {
      url: `${baseUrl}${DECISIONS_PATH}`,
      headers: mergeHeaders(options.headers, {
        "HTTP-Referer": appUrl,
        "X-OpenRouter-Title": appName,
        "X-Title": appName,
        Authorization: authorization,
        "Content-Type": "application/json",
        Accept: "application/json",
      }, LABEL),
      fetch: fetchFn,
      timeoutMs: resolveTimeoutMs(options.timeoutMs, DEFAULT_TIMEOUT_MS, "timeoutMs", LABEL),
      providerName: LABEL,
    };

    // The catalog is public; the key is sent anyway (harmless, and some
    // proxies in front of OpenRouter require it).
    this.#modelsUrl = `${baseUrl}${MODELS_PATH}`;
    this.#modelsHeaders = mergeHeaders(options.headers, {
      Authorization: authorization,
      Accept: "application/json",
    }, LABEL);

    this.#contextWindow = resolveContextWindowOption(options.contextWindow, LABEL);
    const resolve = options.resolveContextWindow ?? true;
    if (typeof resolve !== "boolean") {
      throw new JevValidationError(`${LABEL}: resolveContextWindow must be a boolean`);
    }
    this.#resolveContextWindow = resolve;
    this.#metadataTimeoutMs = resolveTimeoutMs(
      options.metadataTimeoutMs,
      DEFAULT_METADATA_TIMEOUT_MS,
      "metadataTimeoutMs",
      LABEL,
    );
  }

  decide(request: NativeJevRequest): Promise<NativeJevResponse> {
    // encodeDecisionBody throws synchronously on a malformed request; surface it as a rejection.
    return Promise.resolve().then(() => {
      const body = {
        ...encodeDecisionBody(request),
        ...(this.#sessionId === undefined ? {} : { session_id: this.#sessionId }),
        ...(this.#user === undefined ? {} : { user: this.#user }),
      };
      return postDecision(this.#decision, body, request.signal);
    });
  }

  /**
   * Explicit option, else the catalog entry's `context_length`, else a static
   * fallback (32K for Jev models, undefined otherwise so the orchestrator
   * uses its conservative budget, spec 6.1). Never rejects.
   */
  async contextWindow(model: string): Promise<number | undefined> {
    if (this.#contextWindow !== undefined) return this.#contextWindow;
    if (this.#resolveContextWindow && typeof model === "string") {
      try {
        const models = await loadModelList(
          this.#decision.fetch,
          this.#modelsUrl,
          this.#modelsHeaders,
          this.#metadataTimeoutMs,
        );
        const found = findModel(models, model)?.contextLength;
        if (found !== undefined) return found;
      } catch {
        // Metadata is advisory: any failure falls through to the static fallback.
      }
    }
    return staticContextWindow(model);
  }
}

function staticContextWindow(model: unknown): number | undefined {
  return typeof model === "string" && /jev/i.test(model) ? JEV_STATIC_CONTEXT_WINDOW : undefined;
}

function loadModelList(
  fetchFn: typeof fetch,
  url: string,
  headers: Record<string, string>,
  timeoutMs: number,
): Promise<readonly ModelMetadata[]> {
  let byUrl = modelListCache.get(fetchFn);
  if (byUrl === undefined) {
    byUrl = new Map();
    modelListCache.set(fetchFn, byUrl);
  }
  const cached = byUrl.get(url);
  if (cached !== undefined && cached.expiresAt > Date.now()) return cached.promise;

  const entry: ModelListCacheEntry = {
    promise: fetchModelList(fetchFn, url, headers, timeoutMs),
    expiresAt: Number.POSITIVE_INFINITY,
  };
  byUrl.set(url, entry);
  entry.promise.then(
    () => {
      entry.expiresAt = Date.now() + MODEL_LIST_TTL_MS;
    },
    () => {
      entry.expiresAt = Date.now() + MODEL_LIST_FAILURE_TTL_MS;
    },
  );
  return entry.promise;
}

async function fetchModelList(
  fetchFn: typeof fetch,
  url: string,
  headers: Record<string, string>,
  timeoutMs: number,
): Promise<readonly ModelMetadata[]> {
  const { response, text } = await fetchText(
    fetchFn,
    url,
    { method: "GET", headers },
    { timeoutMs, providerName: `${LABEL} model catalog` },
  );
  if (!response.ok) throw new Error(`model catalog returned HTTP ${response.status}`);
  const json: unknown = JSON.parse(text);
  const data = isRecord(json) ? own(json, "data") : undefined;
  if (!Array.isArray(data)) throw new Error("model catalog has no data array");

  const models: ModelMetadata[] = [];
  for (const entry of data) {
    if (!isRecord(entry)) continue;
    const id = own(entry, "id");
    if (typeof id !== "string" || id === "") continue;
    const slug = own(entry, "canonical_slug");
    const contextLength = own(entry, "context_length");
    models.push({
      id,
      canonicalSlug: typeof slug === "string" && slug !== "" ? slug : undefined,
      contextLength:
        typeof contextLength === "number" && Number.isFinite(contextLength) && contextLength >= 1
          ? Math.floor(contextLength)
          : undefined,
    });
  }
  return models;
}

/**
 * Exact id, then exact canonical (build) slug, then an alias match ignoring
 * the leading "~" OpenRouter uses for floating aliases such as
 * "~typesafe/jev-latest".
 */
function findModel(models: readonly ModelMetadata[], model: string): ModelMetadata | undefined {
  const bare = stripAliasPrefix(model);
  return (
    models.find((m) => m.id === model) ??
    models.find((m) => m.canonicalSlug === model) ??
    models.find((m) => stripAliasPrefix(m.id) === bare)
  );
}

function stripAliasPrefix(id: string): string {
  return id.startsWith("~") ? id.slice(1) : id;
}
