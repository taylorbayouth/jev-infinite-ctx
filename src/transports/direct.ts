/**
 * Direct TypeSafe transport (spec 13): the same Decisions wire format
 * (`{ model, state, questions }` → `{ model, answers, usage }`) sent straight
 * to TypeSafe's API instead of through OpenRouter.
 */

import { JevValidationError } from "../errors.js";
import type { JevTransport, NativeJevRequest, NativeJevResponse } from "../types.js";
import {
  encodeDecisionBody,
  isRecord,
  mergeHeaders,
  normalizeBaseUrl,
  postDecision,
  resolveApiKey,
  resolveContextWindowOption,
  resolveFetch,
  resolveTimeoutMs,
  type HttpDecisionConfig,
} from "./http.js";

export const DIRECT_DEFAULT_MODEL = "jev-latest";

export interface DirectJevTransportOptions {
  /** Default: process.env.TYPESAFE_API_KEY. */
  apiKey?: string;
  /** Default "https://api.typesafe.ai". A trailing slash is tolerated. */
  baseUrl?: string;
  /** Default "/v1/systemone". A missing leading slash is added. */
  path?: string;
  /** Default "jev-latest". */
  defaultModel?: string;
  /** Default: the global fetch, looked up at request time. */
  fetch?: typeof fetch;
  /** Per decision request, including reading the body. Default 60_000. */
  timeoutMs?: number;
  /** Extra request headers. They cannot replace Authorization or Content-Type. */
  headers?: Record<string, string>;
  /** Context window in tokens. Default 32_000 (TypeSafe documents a 32K state limit). */
  contextWindow?: number;
}

const LABEL = "TypeSafe";
const DEFAULT_BASE_URL = "https://api.typesafe.ai";
const DEFAULT_PATH = "/v1/systemone";
const DEFAULT_TIMEOUT_MS = 60_000;
const DEFAULT_CONTEXT_WINDOW = 32_000;

export class DirectJevTransport implements JevTransport {
  readonly name = "direct";
  readonly defaultModel: string;

  readonly #decision: HttpDecisionConfig;
  readonly #contextWindow: number;

  /** Throws JevValidationError on bad options or a missing API key, before any request. */
  constructor(options: DirectJevTransportOptions = {}) {
    if (!isRecord(options)) throw new JevValidationError(`${LABEL}: options must be an object`);

    const apiKey = resolveApiKey(options.apiKey, "TYPESAFE_API_KEY", LABEL);
    const baseUrl = normalizeBaseUrl(options.baseUrl, DEFAULT_BASE_URL, LABEL);
    const path = options.path ?? DEFAULT_PATH;
    if (typeof path !== "string" || /[?#]/.test(path)) {
      throw new JevValidationError(`${LABEL}: path must be a string without a query string or fragment`);
    }

    const defaultModel = options.defaultModel ?? DIRECT_DEFAULT_MODEL;
    if (typeof defaultModel !== "string" || defaultModel.trim() === "") {
      throw new JevValidationError(`${LABEL}: defaultModel must be a non-empty string`);
    }
    this.defaultModel = defaultModel;

    this.#decision = {
      url: `${baseUrl}${path.startsWith("/") ? path : `/${path}`}`,
      headers: mergeHeaders(options.headers, {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
        Accept: "application/json",
      }, LABEL),
      fetch: resolveFetch(options.fetch, LABEL),
      timeoutMs: resolveTimeoutMs(options.timeoutMs, DEFAULT_TIMEOUT_MS, "timeoutMs", LABEL),
      providerName: LABEL,
    };
    this.#contextWindow = resolveContextWindowOption(options.contextWindow, LABEL) ?? DEFAULT_CONTEXT_WINDOW;
  }

  decide(request: NativeJevRequest): Promise<NativeJevResponse> {
    // encodeDecisionBody throws synchronously on a malformed request; surface it as a rejection.
    return Promise.resolve().then(() => postDecision(this.#decision, encodeDecisionBody(request), request.signal));
  }

  /** TypeSafe publishes no model catalog; the window is the configured (documented) limit. */
  contextWindow(_model: string): number {
    return this.#contextWindow;
  }
}
