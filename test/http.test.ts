import { afterEach, describe, expect, it, vi } from "vitest";
import { HttpError, OPENROUTER_URL, httpTransport } from "../src/http.js";
import type { DecisionsRequest } from "../src/index.js";

const STATE = "Granite forms when magma cools slowly deep underground.";
const REQUEST: DecisionsRequest = {
  model: "typesafe/jev-1.13",
  state: STATE,
  questions: { decision: { type: "noul", instructions: "Is this about geology?" } },
};

function respond(status: number, body: unknown, headers: Record<string, string> = {}) {
  const text = typeof body === "string" ? body : JSON.stringify(body);
  return vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(text, { status, headers }));
}

const call = () => httpTransport(OPENROUTER_URL, "sk-test")(REQUEST, new AbortController().signal);

afterEach(() => vi.restoreAllMocks());

describe("httpTransport", () => {
  it("POSTs the request with a bearer key and returns the parsed response", async () => {
    const body = { model: "typesafe/jev-1.13-20260917", answers: { decision: { type: "noul", noul: 0.9 } } };
    const fetch = respond(200, body);
    await expect(call()).resolves.toEqual(body);
    const [url, init] = fetch.mock.calls[0]!;
    expect(url).toBe(OPENROUTER_URL);
    expect(init).toMatchObject({ method: "POST", headers: { authorization: "Bearer sk-test", "content-type": "application/json" } });
    expect(JSON.parse(init!.body as string)).toEqual(REQUEST);
  });

  it("reports the status and the provider's message", async () => {
    respond(401, { error: { code: 401, message: "No auth credentials found" } });
    await expect(call()).rejects.toMatchObject({ status: 401, message: "openrouter.ai returned HTTP 401: No auth credentials found" });
  });

  it("reads Retry-After in seconds or as a date", async () => {
    respond(429, "slow down", { "retry-after": "2" });
    await expect(call()).rejects.toMatchObject({ status: 429, retryAfterMs: 2000 });
    respond(503, "busy", { "retry-after": new Date(Date.now() + 60_000).toUTCString() });
    const error = (await call().catch((e: unknown) => e)) as HttpError;
    expect(error.retryAfterMs).toBeGreaterThan(55_000);
  });

  it("reports a 400 that says the input is too long as 413", async () => {
    respond(400, { error: { message: "This endpoint's maximum context length is 32000 tokens." } });
    await expect(call()).rejects.toMatchObject({ status: 413 });
    respond(400, { error: { message: "criteria must have at least two options" } });
    await expect(call()).rejects.toMatchObject({ status: 400 });
  });

  it("never quotes the input in an error message", async () => {
    respond(400, { error: { message: `Invalid state: ${STATE}` } });
    const error = (await call().catch((e: unknown) => e)) as HttpError;
    expect(error.message).toBe("openrouter.ai returned HTTP 400");
  });

  it("rejects a success response that is not JSON", async () => {
    respond(200, "<html>oops</html>");
    await expect(call()).rejects.toMatchObject({ name: "HttpError", status: undefined });
  });

  it("lets network failures through as TypeError, which decide() retries", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new TypeError("fetch failed"));
    await expect(call()).rejects.toBeInstanceOf(TypeError);
  });
});
