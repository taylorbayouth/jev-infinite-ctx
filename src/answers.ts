/**
 * Questions and answers: validating a question, reading Jev's answer for one
 * chunk, and combining the chunk answers into one result.
 */

import type { Span } from "./chunks.js";
import { DecideError } from "./errors.js";
import type { Question } from "./types.js";

/** Jev's probabilities are rounded, so a sum within this of 1 is renormalized. */
const SUM_TOLERANCE = 0.1;

/** One chunk's answer: the native value plus probabilities over the question's labels. */
export interface ChunkAnswer {
  /** The choice key, the score, or the noul probability. */
  value: string | number;
  /** Probability per label (see `labelsOf`); sums to 1. */
  probs: number[];
  confidence: number | undefined;
}

/** Choice keys in the caller's order, score levels "0".."n-1", or noul's no/yes. */
export function labelsOf(question: Question): string[] {
  switch (question.type) {
    case "choice":
      return Object.keys(question.criteria);
    case "score":
      return question.criteria.map((_, level) => String(level));
    case "noul":
      return ["no", "yes"];
  }
}

const QUESTION_FIELDS = new Set(["type", "instructions", "criteria"]);

export function validateQuestion(question: unknown): asserts question is Question {
  if (!isRecord(question)) throw invalid("question must be an object.");
  for (const key of Object.keys(question)) {
    if (!QUESTION_FIELDS.has(key)) throw invalid(`Unknown question field "${key}".`);
  }
  const { type, instructions, criteria } = question;
  if (type !== "choice" && type !== "score" && type !== "noul") {
    throw invalid('question.type must be "choice", "score", or "noul".');
  }
  if (!isCriterion(instructions)) throw invalid("question.instructions must be a non-empty string, or JSON.");
  if (type === "choice") {
    if (!isRecord(criteria) || Object.keys(criteria).length < 2) {
      throw invalid("A choice question needs at least 2 options in criteria.");
    }
    for (const [key, value] of Object.entries(criteria)) {
      if (!isCriterion(value)) throw invalid(`question.criteria["${key}"] must be a non-empty string, or JSON.`);
    }
  } else if (type === "score") {
    if (!Array.isArray(criteria) || criteria.length < 2 || criteria.length > 10) {
      throw invalid("A score question needs 2 to 10 levels in criteria.");
    }
    criteria.forEach((level, i) => {
      if (!isCriterion(level)) throw invalid(`question.criteria[${i}] must be a non-empty string, or JSON.`);
    });
  } else if (
    criteria !== undefined &&
    !(isRecord(criteria) && Object.keys(criteria).length === 2 && isCriterion(criteria.true) && isCriterion(criteria.false))
  ) {
    throw invalid("noul criteria, when given, must have exactly `true` and `false`.");
  }
  try {
    JSON.stringify(question);
  } catch {
    throw invalid("question must be JSON-serializable.");
  }
}

/** Reads and checks Jev's answer for one chunk. Throws on anything malformed. */
export function readAnswer(question: Question, response: unknown): ChunkAnswer {
  const answers = isRecord(response) ? response.answers : undefined;
  const answer = isRecord(answers) ? answers.decision : undefined;
  if (!isRecord(answer) || answer.type !== question.type) throw malformed(`expected a ${question.type} answer`);
  if (question.type === "noul") {
    const p = unitInterval(answer.noul, "noul");
    return { value: p, probs: [1 - p, p], confidence: undefined };
  }
  const labels = labelsOf(question);
  const probs = distribution(answer.probabilities, labels);
  const confidence = answer.confidence == null ? undefined : unitInterval(answer.confidence, "confidence");
  if (question.type === "choice") {
    if (typeof answer.choice !== "string" || !labels.includes(answer.choice)) {
      throw malformed("the choice is not one of the options");
    }
    return { value: answer.choice, probs, confidence };
  }
  const top = labels.length - 1;
  if (typeof answer.score !== "number" || !(answer.score >= -1e-6 && answer.score <= top + 1e-6)) {
    throw malformed("the score is outside the rubric");
  }
  return { value: Math.min(top, Math.max(0, answer.score)), probs, confidence };
}

/**
 * Combines chunk answers. "average" weights each chunk by the share of the
 * input only it covers; "max" takes the chunk with the highest score or noul.
 * A single chunk is returned as Jev answered it.
 */
export function combine(question: Question, spans: readonly Span[], answers: readonly ChunkAnswer[], how: "average" | "max") {
  const labels = labelsOf(question);
  const weights = spans.map((span) => span.weight);
  const answer =
    answers.length === 1 ? answers[0]! : how === "max" ? strongest(answers) : average(question, labels, answers, weights);
  return {
    type: question.type,
    ...fields(question, labels, answer),
    agreement: agreement(question, labels.length, answers, weights),
    chunks: spans.map((span, i) => ({ start: span.start, end: span.end, weight: span.weight, ...fields(question, labels, answers[i]!) })),
  };
}

function average(question: Question, labels: readonly string[], answers: readonly ChunkAnswer[], weights: readonly number[]): ChunkAnswer {
  const weighted = (value: (answer: ChunkAnswer) => number): number =>
    answers.reduce((sum, answer, i) => sum + weights[i]! * value(answer), 0);
  const probs = labels.map((_, k) => weighted((answer) => answer.probs[k]!));
  const confidence = answers.every((answer) => answer.confidence !== undefined)
    ? weighted((answer) => answer.confidence!)
    : undefined;
  if (question.type === "choice") {
    // The most probable option; the earliest in the caller's order wins a tie.
    const best = probs.reduce((top, p, k) => (p > probs[top]! ? k : top), 0);
    return { value: labels[best]!, probs, confidence };
  }
  const top = question.type === "score" ? labels.length - 1 : 1;
  return { value: Math.min(top, weighted((answer) => answer.value as number)), probs, confidence };
}

/** The chunk with the highest score or yes-probability; the earliest wins a tie. */
function strongest(answers: readonly ChunkAnswer[]): ChunkAnswer {
  return answers.reduce((best, answer) => ((answer.value as number) > (best.value as number) ? answer : best));
}

/** 1 minus the weighted average difference between every pair of chunk answers; 1 for one chunk. */
function agreement(question: Question, levels: number, answers: readonly ChunkAnswer[], weights: readonly number[]): number {
  let difference = 0;
  let pairs = 0;
  for (let i = 0; i < answers.length; i++) {
    for (let j = i + 1; j < answers.length; j++) {
      const weight = weights[i]! * weights[j]!;
      difference += weight * distance(question, levels, answers[i]!, answers[j]!);
      pairs += weight;
    }
  }
  return pairs > 0 ? Math.min(1, Math.max(0, 1 - difference / pairs)) : 1;
}

/** How different two chunk answers are, from 0 (same) to 1. */
function distance(question: Question, levels: number, a: ChunkAnswer, b: ChunkAnswer): number {
  // Scores are ordered, so compare the scores: 1 apart on a 0-4 rubric is 0.25.
  if (question.type === "score") return Math.abs((a.value as number) - (b.value as number)) / (levels - 1);
  // Otherwise the share of probability that differs (total variation); for noul, |p_a - p_b|.
  return a.probs.reduce((sum, p, k) => sum + Math.abs(p - b.probs[k]!), 0) / 2;
}

/** The public answer fields for a chunk or the combined result. */
function fields(question: Question, labels: readonly string[], answer: ChunkAnswer) {
  if (question.type === "noul") return { noul: answer.value as number };
  const probabilities = Object.fromEntries(labels.map((label, k) => [label, answer.probs[k]!]));
  const confidence = answer.confidence === undefined ? {} : { confidence: answer.confidence };
  return question.type === "choice"
    ? { choice: answer.value as string, probabilities, ...confidence }
    : { score: answer.value as number, probabilities, ...confidence };
}

function distribution(value: unknown, labels: readonly string[]): number[] {
  if (!isRecord(value)) throw malformed("the probabilities are missing");
  if (Object.keys(value).some((key) => !labels.includes(key))) throw malformed("a probability is for an unknown option");
  const probs = labels.map((label) => (Object.hasOwn(value, label) ? value[label] : 0));
  if (!probs.every((p): p is number => typeof p === "number" && p >= 0 && p <= 1 + 1e-6)) {
    throw malformed("a probability is not between 0 and 1");
  }
  const sum = probs.reduce((a, b) => a + b, 0);
  if (!(Math.abs(sum - 1) <= SUM_TOLERANCE)) throw malformed("the probabilities do not sum to 1");
  return probs.map((p) => p / sum);
}

function unitInterval(value: unknown, name: string): number {
  if (typeof value !== "number" || !(value >= -1e-6 && value <= 1 + 1e-6)) throw malformed(`${name} is not between 0 and 1`);
  return Math.min(1, Math.max(0, value));
}

function isCriterion(value: unknown): boolean {
  return typeof value === "string" ? value.trim() !== "" : typeof value === "object" && value !== null;
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function invalid(message: string): DecideError {
  return new DecideError("invalid_request", message);
}

function malformed(problem: string): Error {
  return new Error(`Jev returned a malformed answer: ${problem}.`);
}
