/** The default transport: POST the Decisions request to an endpoint (OpenRouter by default). */

import type { DecisionsResponse, Transport } from "./types.js";

export const OPENROUTER_URL = "https://openrouter.ai/api/alpha/decisions";

/** A 400 or 422 whose message says this means the chunk was too long for Jev. */
const TOO_LONG = /context (length|window)|maximum context|too many tokens|token limit|too long|too large/i;

/** An HTTP failure. `status` 413 means "too long", whatever status the provider used. */
export class HttpError extends Error {
  override name = "HttpError";
  constructor(
    message: string,
    readonly status: number | undefined,
    readonly retryAfterMs?: number,
  ) {
    super(message);
  }
}

export function httpTransport(url: string, apiKey: string): Transport {
  const host = new URL(url).host;
  return async (request, signal) => {
    // Network failures reject with a TypeError, which decide() retries.
    const response = await fetch(url, {
      method: "POST",
      headers: { authorization: `Bearer ${apiKey}`, "content-type": "application/json" },
      body: JSON.stringify(request),
      signal,
    });
    const body = await response.text();
    if (!response.ok) {
      const tooLong = (response.status === 400 || response.status === 422) && TOO_LONG.test(body);
      const detail = providerMessage(body, request.state);
      throw new HttpError(
        `${host} returned HTTP ${response.status}${detail ? `: ${detail}` : ""}`,
        tooLong ? 413 : response.status,
        retryAfter(response.headers.get("retry-after")),
      );
    }
    try {
      return JSON.parse(body) as DecisionsResponse;
    } catch {
      throw new HttpError(`${host} returned a response that is not JSON`, undefined);
    }
  };
}

/**
 * The provider's own error message, for diagnostics. Withheld if it quotes
 * the chunk (any 20-character run in common), so input text never reaches an
 * error message that might be logged.
 */
function providerMessage(body: string, state: string): string | undefined {
  let message: unknown;
  try {
    const parsed = JSON.parse(body) as { error?: { message?: unknown }; message?: unknown } | null;
    message = parsed?.error?.message ?? parsed?.message;
  } catch {
    return undefined;
  }
  if (typeof message !== "string") return undefined;
  const text = message.trim().slice(0, 200);
  for (let i = 0; i === 0 || i + 20 <= text.length; i += 10) {
    if (state.includes(text.slice(i, i + 20))) return undefined;
  }
  return text || undefined;
}

/** Retry-After in milliseconds, from seconds or an HTTP date. */
function retryAfter(value: string | null): number | undefined {
  if (value === null || value.trim() === "") return undefined;
  const seconds = Number(value);
  if (Number.isFinite(seconds)) return Math.max(0, seconds * 1000);
  const date = Date.parse(value);
  return Number.isNaN(date) ? undefined : Math.max(0, date - Date.now());
}
