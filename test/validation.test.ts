import { describe, expect, it } from "vitest";
import { DEFAULTS, MAX_SCORE_LEVELS, MIN_STATE_TOKENS } from "../src/defaults.js";
import { JevValidationError } from "../src/errors.js";
import type { JevInfiniteCTXRequest, JevTransport } from "../src/types.js";
import {
  resolveOptions,
  validateQuestion,
  validateRequest,
  validateRequestAndResolve,
} from "../src/validation.js";

// Tests deliberately feed malformed shapes, so they work with loosely typed objects.
type Loose = Record<string, unknown>;

function baseRequest(): Loose {
  return {
    input: "The quick brown fox.",
    question: {
      type: "choice",
      instructions: "What is this about?",
      criteria: { animal: "About an animal.", other: "Anything else." },
    },
  };
}

function withQuestion(question: unknown): Loose {
  return { ...baseRequest(), question };
}

/** Runs `run`, asserts it threw a JevValidationError, and returns the message. */
function validationMessage(run: () => unknown): string {
  let caught: unknown;
  try {
    run();
  } catch (error) {
    caught = error;
  }
  expect(caught).toBeInstanceOf(JevValidationError);
  return (caught as JevValidationError).message;
}

function expectInvalid(run: () => unknown, ...fragments: string[]): void {
  const message = validationMessage(run);
  for (const fragment of fragments) {
    expect(message).toContain(fragment);
  }
}

class FakeTransport implements JevTransport {
  readonly name = "fake";
  readonly defaultModel = "fake-model";
  async decide(): Promise<never> {
    throw new Error("not used");
  }
  contextWindow(): number {
    return 32_000;
  }
}

describe("validateRequest: top level", () => {
  it("accepts a minimal request and returns the same object", () => {
    const request = baseRequest();
    expect(validateRequest(request)).toBe(request);
  });

  it("accepts a fully specified request", () => {
    const request: JevInfiniteCTXRequest = {
      input: "text",
      question: { type: "noul", instructions: "Is it?", criteria: { true: "yes", false: "no" } },
      aggregation: "median",
      chunking: {
        overlap: 0.1,
        maxStateTokens: 4000,
        contextSafetyReserve: 0.1,
        protocolReserve: 512,
        preferNaturalBoundaries: false,
        maxChunks: 20,
      },
      execution: {
        maxConcurrency: 2,
        retries: 1,
        retryBaseDelayMs: 100,
        retryMaxDelayMs: 1000,
        maxRechunks: 2,
        rechunkShrinkFactor: 0.5,
      },
      confidence: { method: "none", cap: 0.9, lambda: 0.5, agreementFloor: 0.2, agreementExponent: 1 },
      provider: { transport: new FakeTransport(), model: "m", apiKey: "k" },
      tokenizer: { name: "t", count: (text: string) => text.length },
      signal: new AbortController().signal,
      onEvent: () => {},
    };
    expect(validateRequest(request)).toBe(request);
  });

  it.each([null, undefined, "text", 42, []])("rejects a non-object request (%s)", (request) => {
    expectInvalid(() => validateRequest(request), "request must be an object");
  });

  it("rejects unknown top-level fields, naming them", () => {
    expectInvalid(() => validateRequest({ ...baseRequest(), inputs: "x" }), "inputs", "not a recognized field");
    expectInvalid(() => validateRequest({ ...baseRequest(), "odd key": 1 }), '["odd key"]');
  });

  it("validates option sections too", () => {
    expectInvalid(() => validateRequest({ ...baseRequest(), chunking: { overlap: 0.9 } }), "chunking.overlap");
  });
});

describe("validateRequest: input (spec 15)", () => {
  it.each([undefined, null, 42, ["text"], { text: "x" }])("rejects a non-string input (%s)", (input) => {
    expectInvalid(() => validateRequest({ ...baseRequest(), input }), "input must be a string");
  });

  it.each(["", " ", "\n\t\r  ", "  　﻿"])(
    "rejects empty or whitespace-only input %j before any provider call",
    (input) => {
      expectInvalid(() => validateRequest({ ...baseRequest(), input }), "input", "non-whitespace");
    },
  );

  it("accepts input with any non-whitespace character", () => {
    expect(() => validateRequest({ ...baseRequest(), input: "   x   " })).not.toThrow();
    expect(() => validateRequest({ ...baseRequest(), input: "\n\n😀" })).not.toThrow();
  });

  it("never echoes the input in an error", () => {
    const secret = "TOP-SECRET-DOCUMENT-TEXT";
    const message = validationMessage(() =>
      validateRequest({ ...baseRequest(), input: secret, chunking: { overlap: 2 } }),
    );
    expect(message).not.toContain(secret);
  });
});

describe("validateQuestion: shape", () => {
  it.each([null, undefined, "choice", 1, []])("rejects a non-object question (%s)", (question) => {
    expectInvalid(() => validateQuestion(question), "question must be an object");
  });

  it.each([undefined, "multi", "Choice", 3])("rejects question.type %s", (type) => {
    expectInvalid(
      () => validateQuestion({ type, instructions: "x", criteria: { a: "a", b: "b" } }),
      "question.type",
      '"choice", "score", "noul"',
    );
  });

  it("rejects unknown question fields (fails loudly instead of dropping them)", () => {
    expectInvalid(
      () => validateQuestion({ type: "choice", instructions: "x", options: { a: "a", b: "b" } }),
      "question.options",
      "not a recognized field",
    );
    expectInvalid(
      () => validateQuestion({ type: "score", instructions: "x", criteria: ["a", "b"], legend: {} }),
      "question.legend",
    );
  });

  it("reads own enumerable fields only, matching what JSON.stringify sends", () => {
    const inherited = Object.create({ type: "noul" }) as Loose;
    inherited["instructions"] = "Is it?";
    expectInvalid(() => validateQuestion(inherited), "question.type");
  });

  it("returns the same question object", () => {
    const question = { type: "noul", instructions: "Is it?" };
    expect(validateQuestion(question)).toBe(question);
  });
});

describe("validateQuestion: instructions", () => {
  it.each([undefined, "", "   \n", 42, null, true])("rejects instructions %j", (instructions) => {
    expectInvalid(() => validateQuestion({ type: "noul", instructions }), "question.instructions");
  });

  it.each([["a string", "Decide."], ["an object", { goal: "Decide.", audience: "ops" }], ["an array", ["Step 1", { step: 2 }]]])(
    "accepts instructions as %s",
    (_label, instructions) => {
      expect(() => validateQuestion({ type: "noul", instructions })).not.toThrow();
    },
  );

  it("rejects cyclic instructions (the question must be JSON-serializable)", () => {
    const instructions: Loose = { goal: "x" };
    instructions["self"] = instructions;
    expectInvalid(
      () => validateQuestion({ type: "noul", instructions }),
      "question.instructions",
      "JSON-serializable",
    );
  });

  it("rejects instructions that JSON.stringify cannot handle (BigInt, throwing toJSON)", () => {
    expectInvalid(
      () => validateQuestion({ type: "noul", instructions: { n: 1n } }),
      "question.instructions",
      "JSON-serializable",
    );
    const throwing = {
      toJSON() {
        throw new Error("nope");
      },
    };
    expectInvalid(() => validateQuestion({ type: "noul", instructions: throwing }), "question.instructions");
  });

  it("rejects instructions that serialize to nothing", () => {
    expectInvalid(
      () => validateQuestion({ type: "noul", instructions: { toJSON: () => undefined } }),
      "question.instructions",
      "serializes to nothing",
    );
  });

  it("detects a cycle back to the question itself", () => {
    const question: Loose = { type: "noul" };
    question["instructions"] = { question };
    expectInvalid(() => validateQuestion(question), "question.instructions", "JSON-serializable");
  });
});

describe("validateQuestion: choice criteria", () => {
  const choice = (criteria: unknown): Loose => ({ type: "choice", instructions: "Pick.", criteria });

  it("accepts string, object, and array criteria values", () => {
    expect(() =>
      validateQuestion(choice({ a: "A", b: { what: "B", not_for: ["x"] }, c: ["C", "see also"] })),
    ).not.toThrow();
  });

  it.each([
    ["missing", undefined],
    ["null", null],
    ["an array", ["a", "b"]],
    ["a string", "a,b"],
    ["a Map", new Map([["a", "A"], ["b", "B"]])],
    ["a class instance", new (class Options { a = "A"; b = "B"; })()],
  ])("rejects criteria that is %s", (_label, criteria) => {
    expectInvalid(() => validateQuestion(choice(criteria)), "question.criteria", "plain object");
  });

  it.each([{}, { only: "one" }])("rejects fewer than 2 options (%j)", (criteria) => {
    expectInvalid(() => validateQuestion(choice(criteria)), "question.criteria", "at least 2 options");
  });

  it.each([null, 3, true, undefined])("rejects an option value of %s, naming the option", (value) => {
    expectInvalid(
      () => validateQuestion(choice({ good: "fine", bad: value })),
      "question.criteria.bad",
      "must be a string, an object, or an array",
    );
  });

  // README "Question rules": every criterion must be a non-empty string (or JSON object/array).
  it.each(["", "  \n"])("rejects a blank option description %j, naming the option", (value) => {
    expectInvalid(
      () => validateQuestion(choice({ a: "A", b: value })),
      "question.criteria.b",
      "non-empty string",
    );
    expectInvalid(() => validateQuestion(choice({ a: value, b: "B" })), "question.criteria.a");
  });

  it("still accepts empty object and array option descriptions", () => {
    expect(() => validateQuestion(choice({ a: {}, b: [] }))).not.toThrow();
  });

  it("names odd option keys with bracket notation", () => {
    expectInvalid(() => validateQuestion(choice({ ok: "x", "my key": null })), 'question.criteria["my key"]');
  });

  it("rejects a cyclic option value, naming the option", () => {
    const cyclic: Loose = {};
    cyclic["self"] = cyclic;
    expectInvalid(() => validateQuestion(choice({ a: "A", b: cyclic })), "question.criteria.b", "JSON-serializable");
  });

  it('accepts arbitrary keys including "__proto__" and null-prototype maps', () => {
    const parsed = JSON.parse('{"__proto__": "proto option", "constructor": "ctor option"}') as Loose;
    expect(Object.keys(parsed)).toEqual(["__proto__", "constructor"]);
    expect(() => validateQuestion(choice(parsed))).not.toThrow();

    const bare = Object.assign(Object.create(null) as Loose, { "": "empty key", x: "x" });
    expect(() => validateQuestion(choice(bare))).not.toThrow();
  });

  it('checks the value of an own "__proto__" option', () => {
    const parsed = JSON.parse('{"__proto__": null, "b": "B"}') as Loose;
    expectInvalid(() => validateQuestion(choice(parsed)), 'question.criteria.__proto__');
  });
});

describe("validateQuestion: score criteria", () => {
  const score = (criteria: unknown): Loose => ({ type: "score", instructions: "Rate.", criteria });
  const levels = (n: number): string[] => Array.from({ length: n }, (_, i) => `level ${i}`);

  it("accepts 2 to 10 levels of strings, objects, or arrays", () => {
    expect(() => validateQuestion(score(levels(2)))).not.toThrow();
    expect(() => validateQuestion(score(levels(MAX_SCORE_LEVELS)))).not.toThrow();
    expect(() => validateQuestion(score(["low", { what: "mid" }, ["high"]]))).not.toThrow();
  });

  it.each([undefined, null, { 0: "a", 1: "b" }, "a,b"])("rejects non-array criteria %j", (criteria) => {
    expectInvalid(() => validateQuestion(score(criteria)), "question.criteria", "array of rubric levels");
  });

  it.each([0, 1, 11])("rejects %i levels", (n) => {
    expectInvalid(() => validateQuestion(score(levels(n))), "question.criteria", "between 2 and 10 levels", `got ${n}`);
  });

  it.each([null, 5, false])("rejects a level of %s, naming its index", (value) => {
    expectInvalid(() => validateQuestion(score(["low", value, "high"])), "question.criteria[1]");
  });

  it.each(["", " ", "\t\n"])("rejects a blank level %j, naming its index", (value) => {
    expectInvalid(() => validateQuestion(score(["low", value])), "question.criteria[1]", "non-empty string");
  });

  it("rejects holes in a sparse rubric", () => {
    const sparse: unknown[] = ["low"];
    sparse[2] = "high";
    expectInvalid(() => validateQuestion(score(sparse)), "question.criteria[1]");
  });
});

describe("validateQuestion: noul criteria", () => {
  const noul = (criteria: unknown): Loose => ({ type: "noul", instructions: "Is it?", criteria });

  it("accepts omitted criteria, or exactly true and false", () => {
    expect(() => validateQuestion({ type: "noul", instructions: "Is it?" })).not.toThrow();
    expect(() => validateQuestion(noul(undefined))).not.toThrow();
    expect(() => validateQuestion(noul({ false: "no", true: { what: "yes" } }))).not.toThrow();
  });

  it.each([null, "yes/no", ["yes", "no"]])("rejects criteria %j", (criteria) => {
    expectInvalid(() => validateQuestion(noul(criteria)), "question.criteria", '"true" and "false"');
  });

  it.each([{ true: "yes" }, { false: "no" }, { true: "y", false: "n", maybe: "m" }, { yes: "y", no: "n" }, {}])(
    "rejects criteria without exactly the keys true and false (%j)",
    (criteria) => {
      expectInvalid(() => validateQuestion(noul(criteria)), "question.criteria", 'exactly the keys "true" and "false"');
    },
  );

  it("names the invalid side", () => {
    expectInvalid(() => validateQuestion(noul({ true: null, false: "no" })), "question.criteria.true");
    expectInvalid(() => validateQuestion(noul({ true: "yes", false: 0 })), "question.criteria.false");
  });

  it.each(["", "   "])("rejects a blank description %j, naming the side", (value) => {
    expectInvalid(() => validateQuestion(noul({ true: value, false: "no" })), "question.criteria.true", "non-empty string");
    expectInvalid(() => validateQuestion(noul({ true: "yes", false: value })), "question.criteria.false", "non-empty string");
  });
});

describe("validateRequest: tokenizer, onEvent, signal", () => {
  it("accepts a tokenizer class instance whose count lives on the prototype", () => {
    class Words {
      readonly name = "words";
      count(text: string): number {
        return text.split(/\s+/).length;
      }
    }
    expect(() => validateRequest({ ...baseRequest(), tokenizer: new Words() })).not.toThrow();
  });

  it.each<[string, unknown, string]>([
    ["not an object", "tiktoken", "tokenizer must be an object"],
    ["missing count", { name: "t" }, "tokenizer.count must be a function"],
    ["missing name", { count: (): number => 1 }, "tokenizer.name must be a string"],
    ["non-string name", { name: 1, count: (): number => 1 }, "tokenizer.name must be a string"],
  ])("rejects a tokenizer that is %s", (_label, tokenizer, fragment) => {
    expectInvalid(() => validateRequest({ ...baseRequest(), tokenizer }), fragment);
  });

  it.each([null, "log", {}])("rejects onEvent %j", (onEvent) => {
    expectInvalid(() => validateRequest({ ...baseRequest(), onEvent }), "onEvent must be a function");
  });

  it("accepts a real AbortSignal and a duck-typed one", () => {
    expect(() => validateRequest({ ...baseRequest(), signal: AbortSignal.abort() })).not.toThrow();
    const duck = { aborted: false, addEventListener: () => {}, removeEventListener: () => {} };
    expect(() => validateRequest({ ...baseRequest(), signal: duck })).not.toThrow();
  });

  it.each([
    ["null", null],
    ["a boolean", true],
    ["an empty object", {}],
    ["non-boolean aborted", { aborted: "no", addEventListener: () => {}, removeEventListener: () => {} }],
    ["missing addEventListener", { aborted: false, removeEventListener: () => {} }],
    ["missing removeEventListener", { aborted: false, addEventListener: () => {} }],
  ])("rejects a signal that is %s", (_label, signal) => {
    expectInvalid(() => validateRequest({ ...baseRequest(), signal }), "signal");
  });
});

describe("validateRequest: provider", () => {
  const withProvider = (provider: unknown): Loose => ({ ...baseRequest(), provider });

  it("accepts built-in names, a transport class instance, and a plain transport object", () => {
    expect(() => validateRequest(withProvider({ transport: "openrouter" }))).not.toThrow();
    expect(() => validateRequest(withProvider({ transport: "direct", apiKey: "k" }))).not.toThrow();
    expect(() => validateRequest(withProvider({ transport: new FakeTransport() }))).not.toThrow();
    const plain = { name: "p", defaultModel: "m", decide: async () => ({}), contextWindow: () => undefined };
    expect(() => validateRequest(withProvider({ transport: plain, model: "other" }))).not.toThrow();
  });

  it.each([null, "openrouter", []])("rejects provider %j", (provider) => {
    expectInvalid(() => validateRequest(withProvider(provider)), "provider must be an object");
  });

  it("rejects unknown provider fields", () => {
    expectInvalid(() => validateRequest(withProvider({ key: "k" })), "provider.key");
  });

  it.each(["anthropic", "OpenRouter", ""])("rejects transport name %j", (transport) => {
    expectInvalid(() => validateRequest(withProvider({ transport })), "provider.transport", '"openrouter", "direct"');
  });

  it.each([null, 5, () => {}])("rejects transport value %s", (transport) => {
    expectInvalid(() => validateRequest(withProvider({ transport })), "provider.transport");
  });

  const transportMissing = (field: string): Loose => {
    const t: Loose = { name: "p", defaultModel: "m", decide: async () => ({}), contextWindow: () => 1 };
    delete t[field];
    return t;
  };

  it.each([
    ["decide", "provider.transport.decide must be a function"],
    ["contextWindow", "provider.transport.contextWindow must be a function"],
    ["name", "provider.transport.name must be a string"],
    ["defaultModel", "provider.transport.defaultModel must be a string"],
  ])("rejects a transport object without %s", (field, fragment) => {
    expectInvalid(() => validateRequest(withProvider({ transport: transportMissing(field) })), fragment);
  });

  it("requires provider.model when the custom transport's defaultModel is blank", () => {
    const t = { name: "p", defaultModel: " ", decide: async () => ({}), contextWindow: () => 1 };
    expectInvalid(() => validateRequest(withProvider({ transport: t })), "provider.model is required");
    expect(() => validateRequest(withProvider({ transport: t, model: "m" }))).not.toThrow();
  });

  it.each(["", "   ", 42, null])("rejects provider.model %j", (model) => {
    expectInvalid(() => validateRequest(withProvider({ model })), "provider.model must be a non-empty string");
  });

  it("rejects a non-string apiKey without echoing it", () => {
    const message = validationMessage(() => validateRequest(withProvider({ apiKey: 123456789 })));
    expect(message).toContain("provider.apiKey must be a string");
    expect(message).not.toContain("123456789");
  });
});

describe("resolveOptions", () => {
  it("applies every default (spec 14)", () => {
    expect(resolveOptions(baseRequest() as unknown as JevInfiniteCTXRequest)).toEqual({
      aggregation: DEFAULTS.aggregation,
      chunking: { ...DEFAULTS.chunking },
      execution: { ...DEFAULTS.execution },
      confidence: { ...DEFAULTS.confidence },
    });
  });

  it("returns fresh objects that do not alias DEFAULTS or the request", () => {
    const chunking = { overlap: 0.2 };
    const resolved = resolveOptions({ ...baseRequest(), chunking } as unknown as JevInfiniteCTXRequest);
    expect(resolved.chunking).not.toBe(chunking);
    expect(resolved.chunking).not.toBe(DEFAULTS.chunking);
    expect(resolved.chunking.overlap).toBe(0.2);
    expect(resolved.chunking.protocolReserve).toBe(DEFAULTS.chunking.protocolReserve);
  });

  it("treats explicitly undefined options as omitted", () => {
    const resolved = resolveOptions({
      ...baseRequest(),
      aggregation: undefined,
      chunking: { overlap: undefined, maxChunks: undefined },
      execution: undefined,
    } as unknown as JevInfiniteCTXRequest);
    expect(resolved.aggregation).toBe("weighted_mean");
    expect(resolved.chunking.overlap).toBe(0.05);
    expect(resolved.chunking.maxChunks).toBeUndefined();
    expect(resolved.execution.retries).toBe(3);
  });

  it.each(["chunking", "execution", "confidence"])("rejects a non-object %s section", (name) => {
    expectInvalid(() => resolveOptions({ ...baseRequest(), [name]: 5 } as unknown as JevInfiniteCTXRequest), `${name} must be an object`);
    expectInvalid(() => resolveOptions({ ...baseRequest(), [name]: [] } as unknown as JevInfiniteCTXRequest), `${name} must be an object`);
    expectInvalid(() => resolveOptions({ ...baseRequest(), [name]: null } as unknown as JevInfiniteCTXRequest), `${name} must be an object`);
  });

  it.each([
    ["chunking", "overlapp"],
    ["execution", "retry"],
    ["confidence", "floor"],
  ])("rejects unknown field %s.%s (typos never fall back to defaults silently)", (name, key) => {
    expectInvalid(
      () => resolveOptions({ ...baseRequest(), [name]: { [key]: 1 } } as unknown as JevInfiniteCTXRequest),
      `${name}.${key}`,
      "not a recognized field",
    );
  });

  // [path, invalid values, valid values, expected message fragment]
  const rules: Array<[string, unknown[], unknown[], string]> = [
    ["aggregation", ["avg", "Mean", 1, null], ["weighted_mean", "mean", "median", "min", "max"], 'one of "weighted_mean"'],
    ["chunking.overlap", [-0.01, 0.51, Number.NaN, Infinity, "0.1", null], [0, 0.05, 0.5], "in [0, 0.5]"],
    ["chunking.maxStateTokens", [MIN_STATE_TOKENS - 1, 1000.5, 0, "big", Infinity, null], ["auto", MIN_STATE_TOKENS, 100_000], `"auto" or an integer >= ${MIN_STATE_TOKENS}`],
    ["chunking.contextSafetyReserve", [-0.1, 0.51, Number.NaN], [0, 0.08, 0.5], "in [0, 0.5]"],
    ["chunking.protocolReserve", [-1, 1.5, Infinity, "1024"], [0, 1024], "an integer >= 0"],
    ["chunking.preferNaturalBoundaries", ["true", 1, null], [true, false], "a boolean"],
    ["chunking.maxChunks", [0, -1, 1.5, Infinity, null], [1, 50], "an integer >= 1"],
    ["execution.maxConcurrency", [0, 1.5, Infinity, -2], [1, 16], "an integer >= 1"],
    ["execution.retries", [-1, 0.5, Number.NaN], [0, 3], "an integer >= 0"],
    ["execution.retryBaseDelayMs", [-1, Number.NaN, Infinity], [0, 250.5], "a finite number >= 0"],
    ["execution.retryMaxDelayMs", [-1, Number.NaN, Infinity], [0, 8000], "a finite number >= 0"],
    ["execution.maxRechunks", [-1, 2.5], [0, 4], "an integer >= 0"],
    ["execution.rechunkShrinkFactor", [0, 1, -0.5, 1.5, Number.NaN], [0.5, 0.99], "in (0, 1)"],
    ["confidence.method", ["C3", "platt", null], ["c3", "none"], 'one of "c3", "none"'],
    ["confidence.cap", [0, 1.01, -1, Number.NaN], [1, 0.5, 0.98], "in (0, 1]"],
    ["confidence.lambda", [0, -1, Infinity], [0.001, 10], "a finite number > 0"],
    ["confidence.agreementFloor", [1, -0.1, Number.NaN], [0, 0.99], "in [0, 1)"],
    ["confidence.agreementExponent", [0, -2, Infinity], [0.5, 3], "a finite number > 0"],
  ];

  const requestWith = (path: string, value: unknown): JevInfiniteCTXRequest => {
    const [head, leaf] = path.split(".") as [string, string | undefined];
    const option = leaf === undefined ? value : Object.fromEntries([[leaf, value]]);
    return { ...baseRequest(), [head]: option } as unknown as JevInfiniteCTXRequest;
  };
  const read = (resolved: unknown, path: string): unknown =>
    path.split(".").reduce<unknown>((obj, key) => (obj as Loose)[key], resolved);

  it.each(rules)("%s: rejects invalid values and names the field", (path, invalid, _valid, fragment) => {
    for (const value of invalid) {
      expectInvalid(() => resolveOptions(requestWith(path, value)), path, fragment);
      expectInvalid(() => validateRequest(requestWith(path, value)), path);
    }
  });

  it.each(rules)("%s: accepts and keeps valid values", (path, _invalid, valid) => {
    for (const value of valid) {
      expect(read(resolveOptions(requestWith(path, value)), path)).toBe(value);
    }
  });

  it("describes the offending value in the message", () => {
    expect(validationMessage(() => resolveOptions(requestWith("chunking.overlap", 0.7)))).toBe(
      "chunking.overlap must be a finite number in [0, 0.5], got 0.7.",
    );
    expect(validationMessage(() => resolveOptions(requestWith("execution.retries", "3")))).toBe(
      'execution.retries must be an integer >= 0, got "3".',
    );
  });
});

describe("validateRequestAndResolve", () => {
  it("returns the same request object and its resolved options", () => {
    const request = { ...baseRequest(), chunking: { overlap: 0.2 }, execution: { retries: 1 } };
    const { request: validated, options } = validateRequestAndResolve(request);
    expect(validated).toBe(request);
    expect(options).toEqual(resolveOptions(request as unknown as JevInfiniteCTXRequest));
    expect(options.chunking.overlap).toBe(0.2);
    expect(options.execution.retries).toBe(1);
  });

  it("rejects what validateRequest rejects", () => {
    expectInvalid(() => validateRequestAndResolve({ ...baseRequest(), input: " " }), "input");
    expectInvalid(() => validateRequestAndResolve({ ...baseRequest(), chunking: { overlap: 0.9 } }), "chunking.overlap");
  });

  // Regression: validateRequest used to resolve the options and discard them, so
  // decide() resolved them a second time and read every option getter twice.
  it("reads each option section and option value exactly once", () => {
    let sectionReads = 0;
    let overlapReads = 0;
    const chunking = {};
    Object.defineProperty(chunking, "overlap", {
      enumerable: true,
      get() {
        overlapReads++;
        return 0.1;
      },
    });
    const request = baseRequest();
    Object.defineProperty(request, "chunking", {
      enumerable: true,
      get() {
        sectionReads++;
        return chunking;
      },
    });

    const { options } = validateRequestAndResolve(request);

    expect(options.chunking.overlap).toBe(0.1);
    expect(sectionReads).toBe(1);
    expect(overlapReads).toBe(1);
  });
});
