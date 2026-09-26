#!/usr/bin/env node
/**
 * Live smoke test against Jev on OpenRouter: npm run smoke:live
 *
 * Needs OPENROUTER_API_KEY. Optional: JEV_MODEL (model id), JEV_URL (another
 * Decisions endpoint, for example a local mock). Four decisions, roughly 200K
 * input tokens in total (about a cent). Never prints the document.
 */
import { decide, estimateTokens } from "../dist/index.js";

const extra = {
  ...(process.env.JEV_MODEL ? { model: process.env.JEV_MODEL } : {}),
  ...(process.env.JEV_URL ? { url: process.env.JEV_URL } : {}),
};

const repeat = (sentences, count, offset = 0) =>
  Array.from({ length: count }, (_, i) => sentences[(i + offset) % sentences.length] + (i % 6 === 5 ? "\n\n" : " ")).join("");
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
// Mostly geology, with a forestry section in the middle: about 250K characters.
const DOCUMENT = repeat(GEOLOGY, 1150) + repeat(FORESTRY, 700) + repeat(GEOLOGY, 1150, 3);

const TOPIC = {
  type: "choice",
  instructions: "What is the primary subject of this text?",
  criteria: { tree: "Trees or forests", rock: "Rocks or geology", other: "Neither" },
};

const CASES = [
  ["choice, one chunk (Jev's own answer)", { input: GEOLOGY[1], question: TOPIC }],
  ["choice, long document", { input: DOCUMENT, question: TOPIC }],
  [
    "score, long document",
    {
      input: DOCUMENT,
      question: {
        type: "score",
        instructions: "How technical is the language of this text?",
        criteria: ["Casual, no technical terms", "Some technical terms", "Dense technical language"],
      },
    },
  ],
  [
    'noul, long document, combine: "max" (anywhere in the text?)',
    { input: DOCUMENT, question: { type: "noul", instructions: "Does this text discuss trees or forests?" }, combine: "max" },
  ],
];

const fmt = (n) => (typeof n === "number" ? n.toFixed(3) : "n/a");
const answerOf = (a) => ("choice" in a ? a.choice : "score" in a ? fmt(a.score) : fmt(a.noul));

let cost = 0;
for (const [label, options] of CASES) {
  try {
    const result = await decide({ ...options, ...extra });
    const estimated = result.chunks.reduce((sum, c) => sum + estimateTokens(options.input.slice(c.start, c.end)), 0);
    console.log(`\n## ${label}`);
    console.log(`answer: ${answerOf(result)}${result.probabilities ? `   ${JSON.stringify(result.probabilities, (_, v) => (typeof v === "number" ? +v.toFixed(3) : v))}` : ""}`);
    console.log(`confidence: ${fmt(result.confidence)}   agreement: ${fmt(result.agreement)}`);
    console.log(`chunks: ${result.chunks.length}, per chunk: ${result.chunks.map(answerOf).join(", ")}`);
    console.log(
      `tokens: Jev counted ${result.usage.inputTokens} (with its prompt) vs ${estimated} estimated for the chunks, ` +
        `ratio ${(result.usage.inputTokens / estimated).toFixed(2)}; ${result.usage.requests} requests, ` +
        `cost $${(result.usage.costUsd ?? 0).toFixed(6)}, model ${result.model}`,
    );
    cost += result.usage.costUsd ?? 0;
  } catch (error) {
    console.error(`\nFAILED ${label}: ${error.code ?? error.name}: ${error.message}`);
    process.exit(1);
  }
}
console.log(`\nAll ${CASES.length} decisions completed. Total cost: $${cost.toFixed(6)}.`);
