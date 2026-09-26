#!/usr/bin/env node
/**
 * Live smoke test against Jev through OpenRouter's Decisions API.
 *
 *   npm run smoke:live
 *
 * Environment:
 *   OPENROUTER_API_KEY   required
 *   JEV_MODEL            optional model id (default: the transport's default, typesafe/jev-1.13)
 *   OPENROUTER_BASE_URL  optional base URL (default https://openrouter.ai), for example a local mock
 *
 * Runs four decisions totalling roughly 200K input tokens (about $0.01 at
 * $0.042 per million input tokens): a short single-chunk choice, then a
 * choice, a score, and a noul over a ~250K-character document that forces
 * chunking. Prints answers, confidence, per-chunk answers, and Jev's token
 * counts next to the package's estimates. Never prints the document.
 */
import { decide, OpenRouterJevTransport } from "../dist/index.js";

if (!process.env.OPENROUTER_API_KEY) {
  console.error("OPENROUTER_API_KEY is not set. Export it (or add it to the environment) and rerun.");
  process.exit(1);
}

const transport = new OpenRouterJevTransport({
  appName: "jev-infinite-ctx smoke test",
  ...(process.env.OPENROUTER_BASE_URL ? { baseUrl: process.env.OPENROUTER_BASE_URL } : {}),
});
const provider = { transport, ...(process.env.JEV_MODEL ? { model: process.env.JEV_MODEL } : {}) };

const GEOLOGY = [
  "Granite forms when magma cools slowly deep underground, which lets large crystals grow.",
  "Basalt forms when lava cools quickly at the surface, so its crystals stay small.",
  "Sandstone is a sedimentary rock made of cemented sand grains, usually quartz.",
  "Marble and slate are metamorphic rocks that form under heat and pressure.",
  "Subduction zones recycle old crust, and mid-ocean ridges create new crust.",
  "Wind, water, and ice slowly erode mountains into sediment that settles in basins.",
];
const FORESTRY = [
  "Oak and maple forests shed their leaves each autumn to save water through winter.",
  "Conifers such as pine and spruce keep their needles all year.",
  "A mature canopy shades the forest floor and decides which seedlings survive.",
  "Foresters thin crowded stands so the remaining trees grow taller and healthier.",
];

/** `count` sentences cycled from `sentences`, in paragraphs of six. */
function section(sentences, count, offset) {
  const paragraphs = [];
  for (let i = 0; i < count; i += 6) {
    const paragraph = [];
    for (let j = i; j < Math.min(count, i + 6); j++) paragraph.push(sentences[(j + offset) % sentences.length]);
    paragraphs.push(paragraph.join(" "));
  }
  return paragraphs.join("\n\n");
}

const DOCUMENT = [section(GEOLOGY, 1150, 0), section(FORESTRY, 700, 0), section(GEOLOGY, 1150, 3)].join("\n\n");

const SUBJECT = {
  type: "choice",
  instructions: "What is the primary subject of this text?",
  criteria: {
    tree: "Trees or forests",
    rock: "Rocks or geology",
    other: "Neither trees nor rocks",
  },
};

const CASES = [
  {
    label: "choice, single chunk (native Jev answer, no C3 adjustment)",
    request: { input: "Basalt forms when lava cools quickly at the surface, so its crystals stay small.", question: SUBJECT },
    check: (r) => r.chunks.count === 1 && r.confidence.adjustment === 0 && r.confidence.agreement === 1,
  },
  {
    label: "choice, long document",
    request: { input: DOCUMENT, question: SUBJECT },
    check: (r) => r.chunks.count > 1,
  },
  {
    label: "score, long document",
    request: {
      input: DOCUMENT,
      question: {
        type: "score",
        instructions: "How technical is the language of this text?",
        criteria: [
          "Casual language with no technical terms",
          "Some technical terms, explained or easy to follow",
          "Dense technical or scientific language",
        ],
      },
    },
    check: (r) => r.chunks.count > 1 && r.score >= 0 && r.score <= 2,
  },
  {
    label: "noul, long document",
    request: { input: DOCUMENT, question: { type: "noul", instructions: "Does this text discuss trees or forests?" } },
    check: (r) => r.chunks.count > 1 && r.noul >= 0 && r.noul <= 1,
  },
];

function fmt(value, digits = 3) {
  return typeof value === "number" ? value.toFixed(digits) : String(value);
}

function answerOf(answer) {
  if (answer.type === "choice") return answer.choice;
  if (answer.type === "score") return fmt(answer.score, 2);
  return fmt(answer.noul, 2);
}

/** Every character covered, in order, by the chunk spans. */
function coversInput(result, input) {
  const chunks = result.chunks.results;
  return (
    chunks[0].start === 0 &&
    chunks.at(-1).end === input.length &&
    chunks.every((chunk, i) => i === 0 || (chunk.start <= chunks[i - 1].end && chunk.end > chunks[i - 1].end))
  );
}

function report(label, result) {
  const lines = [`\n## ${label}`];
  if (result.type === "choice") {
    const probabilities = Object.entries(result.probabilities).map(([k, p]) => `${k} ${fmt(p)}`).join(", ");
    lines.push(`answer: ${result.choice}   (${probabilities})`);
  } else if (result.type === "score") {
    lines.push(`answer: score ${fmt(result.score)} of ${Object.keys(result.legend).length - 1} (normalized ${fmt(result.normalizedScore)})`);
  } else {
    lines.push(`answer: noul ${fmt(result.noul)}`);
  }
  const c = result.confidence;
  lines.push(`confidence: base ${fmt(c.base)} (${c.source}) -> adjusted ${fmt(c.adjusted)}, agreement ${fmt(c.agreement)}`);
  lines.push(
    `chunks: ${result.chunks.count} (effective ${fmt(result.chunks.effectiveCount, 2)}, budget ${result.chunks.stateTokenBudget} tokens); ` +
      `per chunk: ${result.chunks.results.map((chunk) => answerOf(chunk.answer)).join(", ")}`,
  );
  const counted = result.chunks.results.reduce((sum, chunk) => sum + (chunk.usage?.inputTokens ?? 0), 0);
  const estimated = result.chunks.results.reduce((sum, chunk) => sum + chunk.estimatedTokens, 0);
  if (result.chunks.count > 1 && counted > 0) {
    lines.push(
      `tokens: Jev counted ${counted} input tokens (state plus its scaffolding) vs ${estimated} estimated state tokens, ` +
        `ratio ${fmt(counted / estimated, 2)} (below 1 means the estimate is conservative)`,
    );
  }
  const u = result.usage;
  lines.push(
    `usage: ${u.requests} requests, ${u.retries} retries, ${u.rechunks} rechunks, ` +
      `cost ${u.costUsd === undefined ? "n/a" : `$${u.costUsd.toFixed(6)}`}, ${u.elapsedMs} ms, model ${result.model}`,
  );
  console.log(lines.join("\n"));
}

let totalCost = 0;
for (const { label, request, check } of CASES) {
  try {
    const result = await decide({ provider, ...request });
    report(label, result);
    totalCost += result.usage.costUsd ?? 0;
    if (!check(result) || !coversInput(result, request.input)) {
      console.error(`FAILED ${label}: result did not satisfy the smoke checks.`);
      process.exit(1);
    }
  } catch (error) {
    // Library errors never contain source text, so they are safe to print.
    const cause = error?.cause;
    const detail = cause?.kind ? ` [${cause.kind}${cause.status ? ` ${cause.status}` : ""}]` : "";
    console.error(`\nFAILED ${label}: ${error?.name ?? "Error"}${detail}: ${error?.message ?? String(error)}`);
    process.exit(1);
  }
}
console.log(`\nAll ${CASES.length} decisions completed. Total reported cost: $${totalCost.toFixed(6)}.`);
