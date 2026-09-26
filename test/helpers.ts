import path from "node:path";
import { fileURLToPath } from "node:url";
import ts from "typescript";
import type { DecisionsRequest, Question, Transport } from "../src/index.js";

/** A fake Jev: `answer` decides each chunk's answer; every request is recorded. */
export function fakeJev(answer: (state: string, question: Question, call: number) => unknown) {
  const requests: DecisionsRequest[] = [];
  let inFlight = 0;
  let peak = 0;
  const transport: Transport = async (request) => {
    const call = requests.push(request) - 1;
    inFlight++;
    peak = Math.max(peak, inFlight);
    try {
      await new Promise((resolve) => setTimeout(resolve, 1));
      const decision = await answer(request.state, request.questions.decision, call);
      return {
        model: "typesafe/jev-1.13-test",
        answers: { decision },
        usage: { input_tokens: 100, cost: 0.001 },
      };
    } finally {
      inFlight--;
    }
  };
  return { transport, requests, peak: () => peak };
}

export const choice = (probabilities: Record<string, number>, confidence = 0.9) => ({
  type: "choice",
  choice: Object.entries(probabilities).reduce((a, b) => (b[1] > a[1] ? b : a))[0],
  probabilities,
  confidence,
});

export const score = (probabilities: number[], confidence = 0.9) => ({
  type: "score",
  score: probabilities.reduce((sum, p, level) => sum + p * level, 0),
  probabilities: Object.fromEntries(probabilities.map((p, level) => [String(level), p])),
  confidence,
});

export const noul = (p: number) => ({ type: "noul", noul: p });

/** An error the way an HTTP transport reports it. */
export const httpError = (status: number, retryAfterMs?: number) =>
  Object.assign(new Error(`HTTP ${status}`), { status, retryAfterMs });

/** Deterministic English-like prose: `sentences` sentences, a paragraph every eight. */
export function prose(sentences: number, topic = "rock"): string {
  const words = {
    rock: ["granite", "basalt", "magma", "sediment", "crystal", "quartz", "erosion", "strata"],
    tree: ["oak", "maple", "canopy", "seedling", "conifer", "bark", "forest", "sapling"],
  }[topic]!;
  let out = "";
  for (let i = 0; i < sentences; i++) {
    const w = (k: number) => words[(i * 7 + k * 3) % words.length];
    out += `The ${w(0)} near the ${w(1)} shows how ${w(2)} and ${w(3)} change over time (${i}).`;
    out += i % 8 === 7 ? "\n\n" : " ";
  }
  return out;
}

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

/** Typechecks in-memory snippets (which import "../src/index.js") and returns each one's errors. */
export function typeErrors(snippets: Readonly<Record<string, string>>): Record<string, string[]> {
  const config = ts.readConfigFile(path.join(root, "tsconfig.json"), (p) => ts.sys.readFile(p));
  const options = { ...ts.parseJsonConfigFileContent(config.config, ts.sys, root).options, noEmit: true };
  const files = new Map(
    Object.entries(snippets).map(([name, text]) => [path.join(root, "test", `__typecheck_${name}__.ts`), text]),
  );
  const host = ts.createCompilerHost(options);
  const getSourceFile = host.getSourceFile.bind(host);
  host.getSourceFile = (fileName, version, onError, create) => {
    const text = files.get(path.resolve(fileName));
    return text === undefined ? getSourceFile(fileName, version, onError, create) : ts.createSourceFile(fileName, text, version, true);
  };
  const fileExists = host.fileExists.bind(host);
  host.fileExists = (fileName) => files.has(path.resolve(fileName)) || fileExists(fileName);
  const program = ts.createProgram([...files.keys()], options, host);
  return Object.fromEntries(
    Object.keys(snippets).map((name) => {
      const sourceFile = program.getSourceFile(path.join(root, "test", `__typecheck_${name}__.ts`))!;
      const errors = ts.getPreEmitDiagnostics(program, sourceFile).map((d) => ts.flattenDiagnosticMessageText(d.messageText, "\n"));
      return [name, errors];
    }),
  );
}
