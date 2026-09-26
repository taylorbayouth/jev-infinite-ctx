import { describe, expect, it } from "vitest";
import fc from "fast-check";
import { JevValidationError } from "../src/errors.js";
import {
  HeuristicTokenizer,
  createTokenizer,
  defaultTokenizer,
  normalizeTokenCount,
} from "../src/tokenizer.js";

const count = (text: string): number => defaultTokenizer.count(text);

const AUSTEN =
  "It is a truth universally acknowledged, that a single man in possession of a good fortune, " +
  "must be in want of a wife. However little known the feelings or views of such a man may be on " +
  "his first entering a neighbourhood, this truth is so well fixed in the minds of the surrounding " +
  "families, that he is considered the rightful property of some one or other of their daughters.";

/** Any code point, lone surrogate halves (which pair up when adjacent), and structural text. */
const unit = fc.oneof(
  { weight: 4, arbitrary: fc.string({ unit: "binary", minLength: 1, maxLength: 1 }) },
  { weight: 2, arbitrary: fc.integer({ min: 0xd800, max: 0xdfff }).map((c) => String.fromCharCode(c)) },
  { weight: 3, arbitrary: fc.string({ unit: "grapheme", minLength: 1, maxLength: 1 }) },
  {
    weight: 4,
    arbitrary: fc.constantFrom(
      " ", "  ", "\n", "\n\n", "\r\n", "\t", ".", "!\" ", "=>", "a", "B", "word", "Word", "camelCase",
      "123", "é", "e\u0301", "\u0301", "中", "ー", "한", "😀", "👨\u200d👩\u200d👧", "\u00a0", "\u200d", "\ufe0f",
      "д", "न", "\u094d", "\u0947", "\u05b0", "\u1100", "\u1161", "ㅋ", "ｶ", "\uff9e", " 1", "\n  4",
    ),
  },
);
const anyText = fc.string({ unit, maxLength: 60 });

describe("HeuristicTokenizer", () => {
  it("exposes a stable name and a frozen shared default instance", () => {
    expect(new HeuristicTokenizer().name).toBe("heuristic-v1");
    expect(defaultTokenizer).toBeInstanceOf(HeuristicTokenizer);
    expect(defaultTokenizer.name).toBe("heuristic-v1");
    expect(Object.isFrozen(defaultTokenizer)).toBe(true);
  });

  it("counts the empty string as zero", () => {
    expect(count("")).toBe(0);
  });

  it("works when the count method is detached from its instance", () => {
    const { count: detached } = new HeuristicTokenizer();
    expect(detached("hello world")).toBe(2);
  });

  describe("per-rule costs", () => {
    it.each([
      ["a", 1],
      ["hello", 1],
      ["because", 2], // ceil(7 / 5)
      ["internationalization", 4], // ceil(20 / 5)
      ["Hello", 1],
      ["HTTP", 1], // upper→upper is not a hump
      ["getElementById", 5], // get | Element | By | Id → 1 + 2 + 1 + 1
      ["HTTPServer", 2], // one hump of 10 units
      ["iPhone", 2], // i | Phone
    ])("Latin run %j → %i", (text, expected) => {
      expect(count(text)).toBe(expected);
    });

    it.each([
      ["é", 1], // 5 units
      ["café", 2], // 3 + 5 units
      ["e\u0301", 2], // e + combining acute: 1 + 5 units
      ["\u0301", 1], // stray combining mark
      ["Mädchen", 3], // 6 + 5 units
    ])("accented Latin %j → %i", (text, expected) => {
      expect(count(text)).toBe(expected);
    });

    it.each([
      ["1", 1],
      ["123", 1],
      ["1234", 2],
      ["1234567", 3],
    ])("ASCII digits %j → %i", (text, expected) => {
      expect(count(text)).toBe(expected);
    });

    it.each([
      [" ", 0], // a lone space merges into the next word
      ["hello world", 2],
      ["  ", 1],
      ["\t", 1],
      ["\n", 1],
      ["\n\n", 1],
      ["\r\n", 1],
      ["\n\n\n\n\n", 2],
      ["\u00a0", 1], // a lone no-break space is not free
      ["        ", 2],
    ])("whitespace %j → %i", (text, expected) => {
      expect(count(text)).toBe(expected);
    });

    // cl100k and o200k never merge a space into the digit group after it:
    // " 1 2" is [" ", "1", " ", "2"], so that space costs a token (F03).
    it.each([
      [" 1", 2],
      ["1 2 3", 5],
      ["a 1", 3],
      ["  1", 3], // ceil(2 / 4) + 1 + 1
      ["\t 1", 3],
      ["\n1", 2], // a run ending in a line break already costs its token
      ["1 ", 1], // nothing follows the space
      ["1 a", 2], // a space before a letter is still free
      ["1\u00a02", 3], // a no-break space costs its usual 1, with no extra token
    ])("whitespace before a digit %j → %i", (text, expected) => {
      expect(count(text)).toBe(expected);
    });

    it.each([
      [".", 1],
      ["=>", 1],
      ["();", 2],
      ["\u0000", 1], // control character
      ["—", 1], // General Punctuation
      ["“”", 2],
      ["…", 1],
      ["©", 1], // Latin-1 symbol
      ["。", 1], // CJK punctuation
      ["，", 1], // fullwidth form
      ["→", 2], // arrows are rarer
      ["\u200d", 2], // zero-width joiner glues emoji sequences
    ])("symbol %j → %i", (text, expected) => {
      expect(count(text)).toBe(expected);
    });

    it.each([
      ["中", 2], // ceil(1.25)
      ["日本語", 4], // ceil(3.75)
      ["こんにちは", 7], // ceil(6.25)
      ["コーヒー", 5], // "ー" counts as Katakana: ceil(4 × 1.25)
      ["한국어", 4],
      ["中。文", 1 + 3], // CJK letters are summed across runs: 1 + ceil(2.5)
    ])("CJK %j → %i", (text, expected) => {
      expect(count(text)).toBe(expected);
    });

    it.each([
      ["привет", 3],
      ["Ελλάδα", 3],
      // 4 letters (2) plus a virama and a vowel sign (1 each); o200k 4, cl100k 6.
      ["नमस्ते", 4],
      ["д\u0301\u0302\u0303", 4], // a mark costs 1 in a run, as it does when stray
    ])("other scripts %j → %i", (text, expected) => {
      expect(count(text)).toBe(expected);
    });

    it.each([
      ["ｶ", 2], // halfwidth Katakana
      ["ﾃﾞ", 4], // halfwidth Katakana and its voiced sound mark
      ["ㅋ", 2], // compatibility Jamo
      ["한".normalize("NFD"), 6], // three conjoining Jamo
      ["한", 2], // the composed syllable stays ceil(1.25)
    ])("Jamo and halfwidth forms %j → %i", (text, expected) => {
      expect(count(text)).toBe(expected);
    });

    it.each([
      ["😀", 3],
      ["🇺🇸", 6], // two regional indicators
      ["❤\ufe0f", 3], // dingbat (2) + variation selector (1)
      ["👨\u200d👩\u200d👧", 13], // three emoji and two joiners
      ["𠀀", 3], // astral Han is costed as astral
    ])("astral and emoji %j → %i", (text, expected) => {
      expect(count(text)).toBe(expected);
    });

    it.each([
      ["\ud800", 1],
      ["\udc00", 1],
      ["\udc00\ud800", 2], // reversed halves are not a pair
      ["a\ud83d", 2],
      ["\ud800a", 2], // a lone high surrogate does not swallow its neighbour
      ["\ud800😀", 4],
    ])("lone surrogates %j → %i without throwing", (text, expected) => {
      expect(count(text)).toBe(expected);
    });
  });

  describe("calibration against modern BPE tokenizers", () => {
    // Reference counts measured once with js-tiktoken (cl100k_base, o200k_base);
    // the heuristic must not under-estimate either, nor overshoot absurdly.
    const samples: Array<{ name: string; text: string; cl100k: number; o200k: number }> = [
      {
        name: "English prose",
        text:
          "Long documents rarely fit in a single request, so the library splits them into overlapping chunks. " +
          "Each chunk is sent with the same question, and the answers are combined into one decision. " +
          "Because neighbouring chunks share a small amount of text, a sentence that straddles a boundary is still read in full. " +
          "The weighting step makes sure that shared text does not receive extra voting power, and the confidence adjustment " +
          "only applies when the chunks broadly agree with one another. Nothing is silently truncated: if a request fails, " +
          "the whole operation fails rather than returning a partial answer.",
        cl100k: 111,
        o200k: 111,
      },
      {
        name: "TypeScript",
        text:
          "export function mapWithConcurrency<T, R>(items: readonly T[], limit: number, fn: (item: T) => Promise<R>): Promise<R[]> {\n" +
          "  const results: R[] = new Array(items.length);\n" +
          "  let next = 0;\n" +
          "  async function worker(): Promise<void> {\n" +
          "    while (next < items.length) {\n" +
          "      const index = next++;\n" +
          "      results[index] = await fn(items[index]!);\n" +
          "    }\n" +
          "  }\n" +
          "  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, worker));\n" +
          "  return results;\n" +
          "}\n",
        cl100k: 116,
        o200k: 115,
      },
      {
        name: "JSON",
        text: JSON.stringify(
          {
            id: "chunk_0042",
            model: "typesafe/jev-1.13",
            answers: {
              decision: {
                type: "choice",
                choice: "tree",
                probabilities: { tree: 0.81, rock: 0.14, other: 0.05 },
                confidence: 0.87,
              },
            },
            usage: { input_tokens: 27412, output_tokens: 18, cost: 0.000412 },
            tags: ["forest", "botany", "long-form"],
            createdAt: "2026-09-25T12:34:56.000Z",
          },
          null,
          2,
        ),
        cl100k: 175,
        o200k: 175,
      },
      {
        name: "Chinese",
        text: "长文档很少能放进单个请求中，因此该库会将其拆分为相互重叠的片段。每个片段都会使用相同的问题发送，然后将答案合并为一个决策。由于相邻片段共享少量文本，跨越边界的句子仍然会被完整读取。",
        cl100k: 100,
        o200k: 74,
      },
      {
        name: "Japanese",
        text: "長い文書は一つのリクエストに収まることがほとんどないため、ライブラリはそれを重なり合うチャンクに分割します。各チャンクには同じ質問が送信され、回答は一つの決定にまとめられます。",
        cl100k: 91,
        o200k: 68,
      },
      {
        name: "Korean",
        text: "긴 문서는 하나의 요청에 거의 들어가지 않으므로 라이브러리는 문서를 서로 겹치는 청크로 나눕니다. 각 청크에는 동일한 질문이 전송되고 답변은 하나의 결정으로 결합됩니다.",
        cl100k: 89,
        o200k: 53,
      },
      {
        name: "emoji chat",
        text: "Shipped it 🚀🎉 thanks everyone 👍👍 — the dashboard is live ✅ and the family photo is up 👨\u200d👩\u200d👧 🇺🇸 ❤\ufe0f see you tomorrow 😀🔥",
        cl100k: 56,
        o200k: 41,
      },
      {
        name: "Markdown",
        text:
          "## Results\n\n| Chunk | Tokens | Weight |\n|---|---:|---:|\n| 0 | 27000 | 0.34 |\n| 1 | 26975 | 0.33 |\n" +
          "| 2 | 26998 | 0.33 |\n\n- **Agreement:** 0.93\n- **Base confidence:** 0.81 (`jev`)\n- **Adjusted:** 0.86 (`c3-v1`)\n",
        cl100k: 95,
        o200k: 93,
      },
      {
        name: "English prose (Austen)",
        text: AUSTEN,
        cl100k: 76,
        o200k: 76,
      },
      // Numeric text: every space before a digit is its own token (F03).
      { name: "a lone space before a digit", text: " 1", cl100k: 2, o200k: 2 },
      {
        name: "space-separated integers",
        text: Array.from({ length: 100 }, (_, i) => String((i * 37) % 1000)).join(" "),
        cl100k: 199,
        o200k: 199,
      },
      {
        name: "comma-separated list",
        text: Array.from({ length: 100 }, (_, i) => String((i * 7) % 100)).join(", "),
        cl100k: 298,
        o200k: 298,
      },
      {
        name: "space-separated decimals",
        text: Array.from(
          { length: 50 },
          (_, i) => `${(i * 13) % 100}.${String((i * 29) % 100).padStart(2, "0")}`,
        ).join(" "),
        cl100k: 199,
        o200k: 199,
      },
      {
        name: "space-aligned numeric table",
        text: Array.from({ length: 10 }, (_, r) =>
          Array.from({ length: 5 }, (_, c) => String((r * 7919 + c * 104729) % 100000).padStart(6)).join(" "),
        ).join("\n"),
        cl100k: 200,
        o200k: 200,
      },
      {
        name: "log lines with timestamps and counts",
        text: Array.from(
          { length: 30 },
          (_, i) =>
            `2026-09-25 12:${String(i % 60).padStart(2, "0")}:07 INFO worker ${i % 7} processed ${i * 131} items in ${(i * 17) % 900} ms`,
        ).join("\n"),
        cl100k: 771,
        o200k: 771,
      },
    ];

    it.each(samples)("does not under-estimate $name", ({ text, cl100k, o200k }) => {
      const reference = Math.max(cl100k, o200k);
      const estimate = count(text);
      expect(estimate).toBeGreaterThanOrEqual(reference);
      expect(estimate).toBeLessThanOrEqual(reference * 1.75);
    });

    it.each([samples[0]!.text, AUSTEN])("estimates English prose at roughly 3.2-4 characters per token", (prose) => {
      const charsPerToken = prose.length / count(prose);
      expect(charsPerToken).toBeGreaterThanOrEqual(3.2);
      expect(charsPerToken).toBeLessThanOrEqual(4.2);
    });

    // Scripts whose letters cl100k splits finely stay at or above o200k.
    it.each([
      ["Russian", "Привет, мир! Это простое предложение на русском языке для проверки количества токенов.", 36, 19],
      ["Greek", "Γεια σου κόσμε! Αυτή είναι μια απλή πρόταση στα ελληνικά για τον έλεγχο των διακριτικών.", 76, 27],
      ["Hindi", "नमस्ते दुनिया। भारत एक विशाल देश है जिसमें अनेक भाषाएँ बोली जाती हैं और यहाँ की संस्कृति बहुत पुरानी है।", 105, 28],
      ["Thai", "สวัสดีชาวโลก ภาษาไทยเป็นภาษาที่มีวรรณยุกต์และสระที่ซับซ้อน", 61, 27],
      ["Arabic", "مرحبا بالعالم هذه جملة عربية بسيطة لاختبار عدد الرموز في النص", 44, 17],
    ])("estimates %s at or above o200k", (_name, text, cl100k, o200k) => {
      const estimate = count(text);
      expect(estimate).toBeGreaterThanOrEqual(o200k);
      expect(estimate).toBeLessThanOrEqual(cl100k * 1.75);
    });

    // The documented known low estimates (src/tokenizer.ts) stay within their
    // documented margins; this pins the documentation to the code.
    it.each([
      [
        "a DNA sequence (random letter run)",
        ">seq1\nGATTACACCGTAGCTAGGCTTACGATCGGATCCATGCAAGTCGTACGATGCTAGCTTAGCA\n" +
          "TTGACCGATGCATCGATCGGCTAGCTAACGTTAGCATGCGATCGTAGCTAGGCATCGATGC\n",
        70,
        0.4,
      ],
      ["NFD Hangul", "안녕하세요 세계".normalize("NFD"), 49, 0.6],
      ["halfwidth Katakana", "ｺﾝﾆﾁﾊ ｾｶｲ ﾃﾞｽ ｶﾞｯｺｳ ﾆ ｲｷﾏｽ", 47, 0.85],
      ["pointed Hebrew (vs o200k)", "בְּרֵאשִׁית בָּרָא אֱלֹהִים אֵת הַשָּׁמַיִם וְאֵת הָאָרֶץ", 48, 0.75],
      ["NFD Vietnamese (vs o200k)", "Tiếng Việt là ngôn ngữ chính thức của Việt Nam".normalize("NFD"), 33, 0.7],
      [
        "numbers indented by two spaces",
        Array.from({ length: 200 }, (_, i) => "  " + String(i * 37)).join("\n"),
        971,
        0.75,
      ],
    ])("keeps the known low estimate for %s within its documented margin", (_name, text, reference, ratio) => {
      expect(count(text)).toBeGreaterThanOrEqual(Math.floor(reference * ratio));
    });
  });

  describe("properties", () => {
    it("returns a non-negative integer, deterministically, for any string", () => {
      fc.assert(
        fc.property(anyText, (text) => {
          const n = count(text);
          expect(Number.isSafeInteger(n)).toBe(true);
          expect(n).toBeGreaterThanOrEqual(0);
          expect(count(text)).toBe(n);
          expect(new HeuristicTokenizer().count(text)).toBe(n);
        }),
        { numRuns: 500 },
      );
    });

    it("is monotone for combining marks after an other-script letter", () => {
      const marks = "\u0301\u0302\u0303";
      expect(count(marks)).toBe(3); // stray marks cost 1 each
      expect(count("д" + marks)).toBeGreaterThanOrEqual(count(marks));
      expect(count("न" + "\u094d\u0947")).toBeGreaterThanOrEqual(count("\u094d\u0947"));
    });

    it("is monotone: a substring never counts more than the string containing it", () => {
      fc.assert(
        fc.property(anyText, anyText, anyText, (before, middle, after) => {
          expect(count(before + middle + after)).toBeGreaterThanOrEqual(count(middle));
        }),
        { numRuns: 1000 },
      );
    });

    it("is nearly subadditive: joining two strings adds at most one token", () => {
      fc.assert(
        fc.property(anyText, anyText, (left, right) => {
          expect(count(left + right)).toBeLessThanOrEqual(count(left) + count(right) + 1);
        }),
        { numRuns: 1000 },
      );
    });

    it("charges a lone surrogate exactly one token whatever surrounds it", () => {
      const high = "\ud800";
      const low = "\udc00";
      fc.assert(
        fc.property(anyText, (text) => {
          const first = text.charCodeAt(0);
          const last = text.charCodeAt(text.length - 1);
          if (!(first >= 0xdc00 && first <= 0xdfff)) expect(count(high + text)).toBe(1 + count(text));
          if (!(last >= 0xd800 && last <= 0xdbff)) expect(count(text + low)).toBe(count(text) + 1);
        }),
        { numRuns: 500 },
      );
    });

    it("never charges more than two tokens per UTF-16 code unit", () => {
      fc.assert(
        fc.property(anyText, (text) => {
          expect(count(text)).toBeLessThanOrEqual(2 * text.length);
        }),
        { numRuns: 500 },
      );
    });
  });

  it("counts at >= 20 MB/s on English text", () => {
    const paragraph =
      "The quick brown fox jumps over the lazy dog, while the committee reviews every proposal. " +
      "Implementation details matter: overlapping chunks preserve context across boundaries.\n\n";
    const text = paragraph.repeat(Math.ceil(2_000_000 / paragraph.length));
    count(text); // warm up
    let best = Number.POSITIVE_INFINITY;
    for (let run = 0; run < 5; run++) {
      const started = performance.now();
      count(text);
      best = Math.min(best, performance.now() - started);
    }
    const megabytesPerSecond = text.length / 1e6 / (best / 1000);
    expect(megabytesPerSecond).toBeGreaterThanOrEqual(20);
  });
});

describe("createTokenizer", () => {
  it("wraps a counting function with a default name", () => {
    const tokenizer = createTokenizer((text) => text.length);
    expect(tokenizer.name).toBe("custom");
    expect(tokenizer.count("abc")).toBe(3);
    expect(tokenizer.count("")).toBe(0);
    expect(Object.isFrozen(tokenizer)).toBe(true);
  });

  it("uses a caller-provided name", () => {
    expect(createTokenizer(() => 1, "tiktoken-o200k").name).toBe("tiktoken-o200k");
  });

  it("rounds fractional counts up", () => {
    const tokenizer = createTokenizer((text) => text.length / 4);
    expect(tokenizer.count("abcde")).toBe(2);
    expect(tokenizer.count("abcd")).toBe(1);
  });

  it.each([
    ["NaN", Number.NaN],
    ["Infinity", Number.POSITIVE_INFINITY],
    ["negative", -1],
    ["string", "3"],
    ["undefined", undefined],
    ["null", null],
  ])("rejects a %s count with JevValidationError", (_label, value) => {
    const tokenizer = createTokenizer(() => value as number, "broken");
    expect(() => tokenizer.count("secret source text")).toThrow(JevValidationError);
    expect(() => tokenizer.count("secret source text")).toThrow(/Tokenizer "broken" returned/);
  });

  it("never includes the counted text in its error message", () => {
    const tokenizer = createTokenizer(() => Number.NaN);
    try {
      tokenizer.count("TOP SECRET PAYLOAD");
      expect.unreachable();
    } catch (error) {
      expect((error as Error).message).not.toContain("TOP SECRET");
    }
  });

  it("propagates errors thrown by the counting function unchanged", () => {
    const failure = new Error("boom");
    const tokenizer = createTokenizer(() => {
      throw failure;
    });
    expect(() => tokenizer.count("x")).toThrow(failure);
  });

  it("validates its arguments", () => {
    expect(() => createTokenizer("nope" as unknown as (text: string) => number)).toThrow(JevValidationError);
    expect(() => createTokenizer((text) => text.length, "")).toThrow(JevValidationError);
    expect(() => createTokenizer((text) => text.length, 42 as unknown as string)).toThrow(JevValidationError);
  });
});

describe("normalizeTokenCount", () => {
  it("accepts finite non-negative numbers and rounds them up", () => {
    expect(normalizeTokenCount(0, "t")).toBe(0);
    expect(normalizeTokenCount(7, "t")).toBe(7);
    expect(normalizeTokenCount(7.01, "t")).toBe(8);
  });

  it("rejects anything else, naming the tokenizer", () => {
    expect(() => normalizeTokenCount(-0.5, "mine")).toThrow(/Tokenizer "mine" returned -0.5/);
    expect(() => normalizeTokenCount({}, "mine")).toThrow(/returned object/);
  });
});
