import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { describe, expect, it } from "vitest";
import * as api from "../src/index.js";
import { choice, fakeJev, typeErrors } from "./helpers.js";

const root = path.resolve(import.meta.dirname, "..");

describe("package", () => {
  it("exports three things", () => {
    expect(Object.keys(api).sort()).toEqual(["DecideError", "decide", "estimateTokens"]);
  });

  it("builds to dist/ and works when imported like a consumer would", async () => {
    const outDir = mkdtempSync(path.join(tmpdir(), "jev-long-dist-"));
    try {
      const tsc = path.join(root, "node_modules", "typescript", "bin", "tsc");
      execFileSync(process.execPath, [tsc, "-p", path.join(root, "tsconfig.build.json"), "--outDir", outDir]);
      const built = (await import(pathToFileURL(path.join(outDir, "index.js")).href)) as typeof api;
      const result = await built.decide({
        input: "Granite.",
        question: { type: "choice", instructions: "Topic?", criteria: { rock: "Rocks", tree: "Trees" } },
        transport: fakeJev(() => choice({ rock: 0.9, tree: 0.1 })).transport,
      });
      expect(result.choice).toBe("rock");
    } finally {
      rmSync(outDir, { recursive: true, force: true });
    }
  });

  it("publishes dist/, the README, and docs/ from package.json", () => {
    const pkg = JSON.parse(readFileSync(path.join(root, "package.json"), "utf8"));
    expect(pkg.exports["."]).toEqual({ types: "./dist/index.d.ts", import: "./dist/index.js" });
    expect(pkg.files).toEqual(["dist", "docs", "README.md"]);
  });
});

describe("types", () => {
  it("infers choice keys, narrows results by question type, and accepts `as const` questions", () => {
    const errors = typeErrors({
      inference: `
        import { decide, type ScoreQuestion } from "../src/index.js";
        const topic = await decide({
          input: "x",
          question: { type: "choice", instructions: "Topic?", criteria: { rock: "Rocks", tree: "Trees" } },
        });
        const key: "rock" | "tree" = topic.choice;
        const p: number = topic.probabilities.rock;

        const RUBRIC = ["low", "high"] as const;
        const severity: ScoreQuestion = { type: "score", instructions: "Severity?", criteria: RUBRIC };
        const scored = await decide({ input: "x", question: severity });
        const level: number = scored.score;

        const flagged = await decide({ input: "x", question: { type: "noul", instructions: "Flagged?" } as const, combine: "max" });
        const yes: number = flagged.noul;
        // @ts-expect-error noul answers are probabilities; Jev gives them no separate confidence
        flagged.confidence;
        export { key, p, level, yes };
      `,
    });
    expect(errors.inference).toEqual([]);
  });
});
