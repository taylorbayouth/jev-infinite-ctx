/** The default transport: POST the Decisions request to an endpoint (OpenRouter by default). */

import type { DecisionsResponse, Transport } from "./types.js";

export const OPENROUTER_URL = "https://openrouter.ai/api/alpha/decisions";

/** Phrases a provider uses when the input exceeds the model's context. */
const TOO_LONG = /context[ _-]?(length|window|limit)|maximum context|too many (input )?tokens|(input|prompt|state) (is )?too (long|large)/i;

/** An HTTP failure. `status` 413 means "too long", whatever status the provider used for it. */
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
      const { message, raw } = providerError(body);
      const tooLong = (response.status === 400 || response.status === 422) && TOO_LONG.test(`${message}\n${raw}`);
      const detail = safeDetail(message, request.state);
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

/** The provider's error message, and the upstream error OpenRouter nests in `metadata.raw`. */
function providerError(body: string): { message: string; raw: string } {
  try {
    const parsed = JSON.parse(body) as { error?: { message?: unknown; metadata?: { raw?: unknown } }; message?: unknown } | null;
    const message = parsed?.error?.message ?? parsed?.message;
    const raw = parsed?.error?.metadata?.raw;
    return { message: typeof message === "string" ? message : "", raw: typeof raw === "string" ? raw : "" };
  } catch {
    return { message: "", raw: "" };
  }
}

/**
 * The provider's message, trimmed, for diagnostics. Withheld if it shares any
 * 20 consecutive characters with the chunk, so a provider that quotes the
 * input does not put it in an error message that might be logged.
 */
function safeDetail(message: string, state: string): string | undefined {
  const text = message.trim().slice(0, 200);
  for (let i = 0; i === 0 || i + 20 <= text.length; i++) {
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
