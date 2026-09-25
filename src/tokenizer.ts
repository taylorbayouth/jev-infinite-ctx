/**
 * Token counting for budgeting and chunk boundaries (spec 6.1, 6.2).
 *
 * Jev's tokenizer is not public, so the default is a conservative heuristic:
 * it is meant to over- rather than under-estimate modern BPE tokenizers
 * (calibrated against cl100k_base and o200k_base) while staying close enough
 * that chunks are not needlessly small. Callers with a real tokenizer can plug
 * it in through `createTokenizer`.
 */

import { JevValidationError } from "./errors.js";
import type { Tokenizer } from "./types.js";

// ---------------------------------------------------------------------------
// Character classes
// ---------------------------------------------------------------------------

const UNCLASSIFIED = 0;
/** ASCII A-Z. */
const UPPER = 1;
/** ASCII a-z. */
const LOWER = 2;
/** Non-ASCII Latin-script letter (é, ř, ệ, ß, ...). */
const LATIN_EXT = 3;
/** Combining mark; extends the letter run it follows. */
const MARK = 4;
/** Han, Hiragana, Katakana, Hangul letter. */
const CJK = 5;
/** Letter of any other script (Cyrillic, Greek, Arabic, Devanagari, Thai, ...). */
const OTHER_LETTER = 6;
/** ASCII 0-9. Other decimal digits are multi-byte and costed as symbols. */
const DIGIT = 7;
const SPACE = 8;
/** ASCII punctuation. */
const PUNCT = 9;
/** Non-ASCII symbol that modern vocabularies usually hold as a single token. */
const SYMBOL = 10;
/** Non-ASCII symbol that often costs two tokens. */
const RARE_SYMBOL = 11;
const HIGH_SURROGATE = 12;
const LOW_SURROGATE = 13;

// ---------------------------------------------------------------------------
// Costs, calibrated against cl100k_base and o200k_base. English prose (~1.35x,
// 3.1-3.5 chars/token), code, JSON, CSV, logs, HTML, Markdown, CJK and emoji
// all estimate at or above both. Accented Latin and other scripts estimate at
// or above o200k but can fall below cl100k, whose vocabulary is English-heavy;
// random base64 or hex strings run up to ~30% below both. The orchestrator's
// safety reserve and shrink-and-rechunk loop absorb such misses (spec 6.1, 7).
// ---------------------------------------------------------------------------

/** Letter units per token in a Latin run. Common English words are one token up to ~5 letters. */
const LATIN_UNITS_PER_TOKEN = 5;
/**
 * Units charged per non-ASCII Latin letter or combining mark: roughly one
 * token each, because English-heavy vocabularies usually split words there.
 */
const NON_ASCII_LATIN_UNITS = 5;
/**
 * Tokens per Han, Kana or Hangul letter. cl100k spends ~1.1-1.15 per letter
 * on everyday prose (o200k ~0.7). Accumulated across the whole text and
 * rounded up once, so short runs are not over-charged.
 */
const CJK_TOKENS_PER_LETTER = 1.25;
/** Other-script letters per token (Cyrillic, Greek, Arabic, Indic, Thai, ...). */
const OTHER_LETTERS_PER_TOKEN = 2;
/** BPE vocabularies group ASCII digits in threes. */
const DIGITS_PER_TOKEN = 3;
const SPACES_PER_TOKEN = 4;
/** Runs such as `");` or `=>` are usually one token per one or two characters. */
const PUNCT_PER_TOKEN = 2;
/** Emoji and other astral code points are four UTF-8 bytes: up to three tokens. */
const ASTRAL_TOKENS = 3;

const CLASS_TABLE = new Uint8Array(0x10000);

for (let code = 0; code < 0x80; code++) {
  let cls: number;
  if (code >= 0x41 && code <= 0x5a) cls = UPPER;
  else if (code >= 0x61 && code <= 0x7a) cls = LOWER;
  else if (code >= 0x30 && code <= 0x39) cls = DIGIT;
  else if (code === 0x20 || (code >= 0x09 && code <= 0x0d)) cls = SPACE;
  else if (code > 0x20 && code < 0x7f) cls = PUNCT;
  else cls = SYMBOL; // other C0 controls and DEL
  CLASS_TABLE[code] = cls;
}

const WHITESPACE_RE = /^\s$/u;
const MARK_RE = /^\p{M}$/u;
const LETTER_RE = /^\p{L}$/u;
// Script_Extensions so that shared marks such as "ー" and "々" count as CJK.
const CJK_RE = /^[\p{scx=Han}\p{scx=Hiragana}\p{scx=Katakana}\p{scx=Hangul}]$/u;
const LATIN_RE = /^\p{sc=Latin}$/u;

/** Classifies a non-ASCII BMP code unit. Runs once per distinct code unit per process. */
function classifyNonAscii(code: number): number {
  if (code >= 0xd800 && code <= 0xdbff) return HIGH_SURROGATE;
  if (code >= 0xdc00 && code <= 0xdfff) return LOW_SURROGATE;
  const ch = String.fromCharCode(code);
  if (WHITESPACE_RE.test(ch)) return SPACE;
  if (MARK_RE.test(ch)) return MARK;
  if (LETTER_RE.test(ch)) {
    if (CJK_RE.test(ch)) return CJK;
    return LATIN_RE.test(ch) ? LATIN_EXT : OTHER_LETTER;
  }
  // Latin-1 symbols, General Punctuation (dashes, curly quotes, bullets,
  // ellipsis), CJK punctuation and fullwidth forms are single tokens. The
  // zero-width joiner glues emoji sequences and is not.
  if (
    code !== 0x200d &&
    (code < 0x0300 ||
      (code >= 0x2000 && code <= 0x206f) ||
      (code >= 0x3000 && code <= 0x303f) ||
      (code >= 0xff00 && code <= 0xffef))
  ) {
    return SYMBOL;
  }
  return RARE_SYMBOL;
}

function classOf(code: number): number {
  const cls = CLASS_TABLE[code] ?? UNCLASSIFIED;
  if (cls !== UNCLASSIFIED) return cls;
  const learned = classifyNonAscii(code);
  CLASS_TABLE[code] = learned;
  return learned;
}

// ---------------------------------------------------------------------------
// Heuristic tokenizer
// ---------------------------------------------------------------------------

/**
 * Conservative, dependency-free token estimator.
 *
 * One O(n) pass over UTF-16 code units, mirroring BPE pre-tokenization:
 * - Latin letter runs are split into humps at lower→upper case changes (as
 *   o200k does for camelCase) and each hump costs ceil(units / 5), where an
 *   ASCII letter is 1 unit and a non-ASCII Latin letter or combining mark is 5;
 * - Han, Hiragana, Katakana and Hangul letters cost 1.25 each, summed over
 *   the whole text and rounded up once;
 * - other-script letter runs cost ceil(length / 2);
 * - ASCII digit runs cost ceil(length / 3);
 * - a lone U+0020 is free (it merges into the next word); any other
 *   whitespace run costs ceil(length / 4), so "\n\n" is 1;
 * - ASCII punctuation runs cost ceil(length / 2);
 * - other symbols cost 1 or 2, astral code points (emoji) 3, lone surrogates 1.
 *
 * The count is 0 for "", deterministic, and monotone: a string never counts
 * more than any string that contains it. Lone surrogates never throw.
 */
export class HeuristicTokenizer implements Tokenizer {
  readonly name = "heuristic-v1";

  count(text: string): number {
    const length = text.length;
    let tokens = 0;
    let cjkLetters = 0;
    let i = 0;
    while (i < length) {
      const code = text.charCodeAt(i);
      const cls = classOf(code);
      let j = i + 1;
      switch (cls) {
        case UPPER:
        case LOWER:
        case LATIN_EXT: {
          let units = cls === LATIN_EXT ? NON_ASCII_LATIN_UNITS : 1;
          let afterLower = cls !== UPPER;
          for (; j < length; j++) {
            const next = classOf(text.charCodeAt(j));
            if (next === LOWER) {
              units += 1;
              afterLower = true;
            } else if (next === UPPER) {
              if (afterLower) {
                tokens += Math.ceil(units / LATIN_UNITS_PER_TOKEN);
                units = 0;
              }
              units += 1;
              afterLower = false;
            } else if (next === LATIN_EXT) {
              units += NON_ASCII_LATIN_UNITS;
              afterLower = true;
            } else if (next === MARK) {
              units += NON_ASCII_LATIN_UNITS;
            } else {
              break;
            }
          }
          tokens += Math.ceil(units / LATIN_UNITS_PER_TOKEN);
          break;
        }
        case OTHER_LETTER: {
          for (; j < length; j++) {
            const next = classOf(text.charCodeAt(j));
            if (next !== OTHER_LETTER && next !== MARK) break;
          }
          tokens += Math.ceil((j - i) / OTHER_LETTERS_PER_TOKEN);
          break;
        }
        case DIGIT: {
          while (j < length && classOf(text.charCodeAt(j)) === DIGIT) j++;
          tokens += Math.ceil((j - i) / DIGITS_PER_TOKEN);
          break;
        }
        case SPACE: {
          while (j < length && classOf(text.charCodeAt(j)) === SPACE) j++;
          if (j - i > 1 || code !== 0x20) tokens += Math.ceil((j - i) / SPACES_PER_TOKEN);
          break;
        }
        case PUNCT: {
          while (j < length && classOf(text.charCodeAt(j)) === PUNCT) j++;
          tokens += Math.ceil((j - i) / PUNCT_PER_TOKEN);
          break;
        }
        case CJK:
          cjkLetters += 1;
          break;
        case SYMBOL:
          tokens += 1;
          break;
        case RARE_SYMBOL:
          tokens += 2;
          break;
        case HIGH_SURROGATE:
          if (j < length && classOf(text.charCodeAt(j)) === LOW_SURROGATE) {
            tokens += ASTRAL_TOKENS;
            j++;
          } else {
            tokens += 1;
          }
          break;
        default:
          // Lone low surrogate, or a combining mark with no letter to attach to.
          tokens += 1;
      }
      i = j;
    }
    return tokens + Math.ceil(cjkLetters * CJK_TOKENS_PER_LETTER);
  }
}

/** Shared default tokenizer instance. */
export const defaultTokenizer: Tokenizer = Object.freeze(new HeuristicTokenizer());

// ---------------------------------------------------------------------------
// Custom tokenizers
// ---------------------------------------------------------------------------

/**
 * Validates a raw token count: it must be a finite number >= 0. Fractional
 * counts are rounded up so budgets stay conservative. The error names the
 * tokenizer but never includes the counted text.
 */
export function normalizeTokenCount(value: unknown, tokenizerName: string): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    const shown = typeof value === "number" ? String(value) : typeof value;
    throw new JevValidationError(
      `Tokenizer "${tokenizerName}" returned ${shown}; expected a finite number >= 0`,
    );
  }
  return Math.ceil(value);
}

/**
 * Wraps a counting function (for example a js-tiktoken encoder's
 * `text => enc.encode(text).length`) as a `Tokenizer`. Every result is
 * validated with `normalizeTokenCount`.
 */
export function createTokenizer(count: (text: string) => number, name = "custom"): Tokenizer {
  if (typeof count !== "function") {
    throw new JevValidationError("createTokenizer: count must be a function");
  }
  if (typeof name !== "string" || name.length === 0) {
    throw new JevValidationError("createTokenizer: name must be a non-empty string");
  }
  return Object.freeze({
    name,
    count: (text: string): number => normalizeTokenCount(count(text), name),
  });
}
