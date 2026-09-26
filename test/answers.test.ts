import { describe, expect, it } from "vitest";
import { type ChunkAnswer, combine, readAnswer, validateQuestion } from "../src/answers.js";
import type { Span } from "../src/chunks.js";
import { DecideError, type Question } from "../src/index.js";
import { choice, noul, score } from "./helpers.js";

const CHOICE: Question = { type: "choice", instructions: "Topic?", criteria: { rock: "Rocks", tree: "Trees", other: "Other" } };
const SCORE: Question = { type: "score", instructions: "Severity?", criteria: ["none", "low", "medium", "high", "critical"] };
const NOUL: Question = { type: "noul", instructions: "Mentions forests?" };

const read = (question: Question, answer: unknown) => readAnswer(question, { answers: { decision: answer } });
const spans = (...weights: number[]): Span[] => weights.map((weight, i) => ({ start: i * 10, end: i * 10 + 12, weight }));

describe("validateQuestion", () => {
  it("accepts each native question type", () => {
    for (const question of [CHOICE, SCORE, NOUL, { ...NOUL, criteria: { true: "yes", false: "no" } }]) {
      expect(() => validateQuestion(question)).not.toThrow();
    }
  });

  it.each([
    ["not an object", "choice?"],
    ["an unknown field", { ...CHOICE, critera: {} }],
    ["an unknown type", { ...CHOICE, type: "rank" }],
    ["empty instructions", { ...CHOICE, instructions: "  " }],
    ["one option", { ...CHOICE, criteria: { only: "One" } }],
    ["an empty option", { ...CHOICE, criteria: { a: "A", b: "" } }],
    ["one level", { ...SCORE, criteria: ["only"] }],
    ["eleven levels", { ...SCORE, criteria: Array.from({ length: 11 }, (_, i) => `level ${i}`) }],
    ["half of noul's criteria", { ...NOUL, criteria: { true: "yes" } }],
  ])("rejects %s", (_, question) => {
    expect(() => validateQuestion(question)).toThrow(expect.objectContaining({ code: "invalid_request" }));
  });

  it("rejects a rubric with holes in it", () => {
    const levels = ["none", "minor", "major"];
    delete levels[1];
    expect(() => validateQuestion({ ...SCORE, criteria: levels })).toThrow(expect.objectContaining({ code: "invalid_request" }));
  });

  it("rejects a question that is not JSON", () => {
    const criteria: Record<string, unknown> = { a: "A" };
    criteria.b = { loop: criteria };
    expect(() => validateQuestion({ ...CHOICE, criteria })).toThrow(DecideError);
  });
});

describe("readAnswer", () => {
  it("reads a choice, renormalizing Jev's rounded probabilities", () => {
    const answer = read(CHOICE, choice({ rock: 0.6, tree: 0.33, other: 0.06 }, 0.7));
    expect(answer.value).toBe("rock");
    expect(answer.probs.reduce((a, b) => a + b)).toBeCloseTo(1, 12);
    expect(answer.probs[0]).toBeCloseTo(0.6 / 0.99, 12);
    expect(answer.confidence).toBe(0.7);
  });

  it("treats options Jev left out as 0", () => {
    expect(read(CHOICE, choice({ tree: 1 })).probs).toEqual([0, 1, 0]);
  });

  it("reads scores and nouls", () => {
    expect(read(SCORE, score([0, 0, 0.5, 0.5, 0])).value).toBeCloseTo(2.5, 12);
    expect(read(NOUL, noul(0.8))).toEqual({ value: 0.8, probs: [0.19999999999999996, 0.8], confidence: undefined });
  });

  it("keeps caller keys exactly, including awkward ones", () => {
    const question = JSON.parse('{"type":"choice","instructions":"?","criteria":{"__proto__":"A","constructor":"B","":"C"}}');
    const answer = read(question, JSON.parse('{"type":"choice","choice":"__proto__","probabilities":{"__proto__":0.7,"constructor":0.2,"":0.1}}'));
    expect(answer.value).toBe("__proto__");
    [0.7, 0.2, 0.1].forEach((p, k) => expect(answer.probs[k]).toBeCloseTo(p, 12));
  });

  it.each([
    ["a different type", CHOICE, noul(0.5)],
    ["a choice that is not an option", CHOICE, { ...choice({ rock: 1 }), choice: "lava" }],
    ["missing probabilities", CHOICE, { type: "choice", choice: "rock" }],
    ["a probability for an unknown option", CHOICE, choice({ rock: 0.5, lava: 0.5 })],
    ["a negative probability", CHOICE, choice({ rock: 1.2, tree: -0.2 })],
    ["probabilities far from 1", CHOICE, choice({ rock: 0.3, tree: 0.2 })],
    ["confidence above 1", CHOICE, choice({ rock: 1 }, 1.5)],
    ["a score outside the rubric", SCORE, { ...score([1, 0, 0, 0, 0]), score: 7 }],
    ["noul outside 0 to 1", NOUL, noul(1.3)],
    ["no answer at all", NOUL, undefined],
  ])("rejects %s", (_, question, answer) => {
    expect(() => read(question, answer)).toThrow(/malformed answer/);
  });
});

describe("combine", () => {
  const answers = (question: Question, ...raw: unknown[]): ChunkAnswer[] => raw.map((answer) => read(question, answer));

  it("averages choice probabilities by chunk weight, and averages confidence the same way", () => {
    const chunks = answers(CHOICE, choice({ rock: 1 }, 0.9), choice({ tree: 1 }, 0.5));
    const result = combine(CHOICE, spans(0.75, 0.25), chunks, "average");
    expect(result).toMatchObject({ type: "choice", choice: "rock", probabilities: { rock: 0.75, tree: 0.25, other: 0 } });
    expect((result as { confidence: number }).confidence).toBeCloseTo(0.8, 12);
    expect(result.chunks.map((c) => [c.weight, (c as { choice: string }).choice])).toEqual([[0.75, "rock"], [0.25, "tree"]]);
  });

  it("breaks exact ties toward the first option", () => {
    const chunks = answers(CHOICE, choice({ tree: 1 }), choice({ rock: 1 }));
    expect(combine(CHOICE, spans(0.5, 0.5), chunks, "average")).toMatchObject({ choice: "rock" });
  });

  it("averages scores and nouls", () => {
    const scores = answers(SCORE, score([0, 1, 0, 0, 0]), score([0, 0, 0, 1, 0]));
    expect(combine(SCORE, spans(0.25, 0.75), scores, "average")).toMatchObject({ score: 2.5, probabilities: { "1": 0.25, "3": 0.75 } });
    const nouls = answers(NOUL, noul(0.2), noul(0.6));
    expect((combine(NOUL, spans(0.5, 0.5), nouls, "average") as { noul: number }).noul).toBeCloseTo(0.4, 12);
  });

  it('with "max", answers from the chunk with the highest score or noul', () => {
    const nouls = answers(NOUL, noul(0.1), noul(0.95), noul(0.95));
    expect(combine(NOUL, spans(0.4, 0.3, 0.3), nouls, "max")).toMatchObject({ noul: 0.95 });
    const scores = answers(SCORE, score([0, 1, 0, 0, 0], 0.6), score([0, 0, 0, 0, 1], 0.8));
    expect(combine(SCORE, spans(0.9, 0.1), scores, "max")).toMatchObject({ score: 4, confidence: 0.8 });
  });

  it("returns a single chunk exactly as Jev answered it", () => {
    const [only] = answers(CHOICE, { type: "choice", choice: "tree", probabilities: { rock: 0.5, tree: 0.5 }, confidence: 0.4 });
    expect(combine(CHOICE, spans(1), [only!], "average")).toMatchObject({ choice: "tree", confidence: 0.4, agreement: 1 });
  });

  it("leaves confidence out when any chunk lacks it", () => {
    const chunks = answers(CHOICE, choice({ rock: 1 }), { type: "choice", choice: "rock", probabilities: { rock: 1 } });
    expect(combine(CHOICE, spans(0.5, 0.5), chunks, "average")).not.toHaveProperty("confidence");
  });

  describe("agreement", () => {
    const agreement = (question: Question, weights: number[], ...raw: unknown[]) =>
      combine(question, spans(...weights), answers(question, ...raw), question.type === "choice" ? "average" : "max").agreement;

    it("is 1 when every chunk gives the same answer and 0 when two give opposite ones", () => {
      expect(agreement(CHOICE, [0.3, 0.3, 0.4], choice({ rock: 0.8, tree: 0.2 }), choice({ rock: 0.8, tree: 0.2 }), choice({ rock: 0.8, tree: 0.2 }))).toBe(1);
      expect(agreement(CHOICE, [0.5, 0.5], choice({ rock: 1 }), choice({ tree: 1 }))).toBe(0);
    });

    it("compares noul probabilities directly and scores on their scale", () => {
      expect(agreement(NOUL, [0.5, 0.5], noul(0.2), noul(0.8))).toBeCloseTo(0.4, 12);
      expect(agreement(SCORE, [0.5, 0.5], score([0, 1, 0, 0, 0]), score([0, 0, 0, 1, 0]))).toBeCloseTo(0.5, 12);
    });

    it("barely moves when only a tiny chunk disagrees", () => {
      expect(agreement(CHOICE, [0.49, 0.49, 0.02], choice({ rock: 1 }), choice({ rock: 1 }), choice({ tree: 1 }))).toBeCloseTo(1 - 0.0196 / 0.2597, 12);
    });
  });
});
