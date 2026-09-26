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
/** Han, Hiragana, Katakana, Hangul syllable. */
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
/**
 * Non-ASCII character that often costs two tokens: rarer symbols, and letters
 * that vocabularies seldom merge (halfwidth Katakana, Hangul Jamo).
 */
const RARE = 11;
const HIGH_SURROGATE = 12;
const LOW_SURROGATE = 13;

// ---------------------------------------------------------------------------
// Costs, calibrated against cl100k_base and o200k_base. English prose (~1.25-
// 1.35x: about 3.4-4 chars per estimated token, ~3 on Markdown-heavy text),
// code, JSON, CSV, logs, numeric tables, HTML, Markdown, NFC CJK and emoji
// all estimate at or above both. Known low estimates:
// - lines of numbers indented by two or three spaces ("\n  42") run up to
//   ~20% below both: BPE splits that run into three tokens;
// - accented Latin and other scripts can fall below cl100k, whose vocabulary
//   is English-heavy;
// - text dense in combining marks (pointed Hebrew, Arabic with harakat, NFD
//   Vietnamese, zalgo) can fall below o200k too, by ~10-55%;
// - decomposed (NFD) Hangul runs ~35% below both, halfwidth Katakana up to
//   ~10%;
// - random strings run below both: letter runs (DNA or protein sequences,
//   random identifiers) by ~55-60%, base64 by ~30%, printable ASCII by ~10%,
//   hex by ~5%, and uniformly random BMP code points by ~45-50%.
// The orchestrator's safety reserve and shrink-and-rechunk loop absorb such
// misses (spec 6.1, 7).
// ---------------------------------------------------------------------------

/** Letter units per token in a Latin run. Common English words are one token up to ~5 letters. */
const LATIN_UNITS_PER_TOKEN = 5;
/**
 * Units charged per non-ASCII Latin letter or combining mark: roughly one
 * token each, because English-heavy vocabularies usually split words there.
 */
const NON_ASCII_LATIN_UNITS = 5;
/**
 * Tokens per Han, Kana or Hangul syllable. cl100k spends ~1.1-1.15 per letter
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
    // Conjoining and compatibility Jamo, halfwidth Katakana and Hangul: the
    // real tokenizers spend about 2-3 tokens on each, far more than on a
    // composed syllable or fullwidth Kana.
    if (
      (code >= 0x1100 && code <= 0x11ff) ||
      (code >= 0x3130 && code <= 0x318f) ||
      (code >= 0xa960 && code <= 0xa97f) ||
      (code >= 0xd7b0 && code <= 0xd7ff) ||
      (code >= 0xff61 && code <= 0xffdc)
    ) {
      return RARE;
    }
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
  return RARE;
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
 *   the whole text and rounded up once; Jamo and halfwidth forms cost 2;
 * - other-script letter runs cost ceil(letters / 2) plus 1 per combining mark;
 * - ASCII digit runs cost ceil(length / 3);
 * - a lone U+0020 is free (it merges into the next word); any other
 *   whitespace run costs ceil(length / 4), so "\n\n" is 1;
 * - a whitespace run that ends in U+0020 right before an ASCII digit costs 1
 *   more, because BPE keeps that space out of the digit group;
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
          // A mark costs 1, as it does when stray, so that adding a letter
          // before marks never lowers the count (monotonicity).
          let marks = 0;
          for (; j < length; j++) {
            const next = classOf(text.charCodeAt(j));
            if (next === MARK) marks++;
            else if (next !== OTHER_LETTER) break;
          }
          tokens += Math.ceil((j - i - marks) / OTHER_LETTERS_PER_TOKEN) + marks;
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
          // cl100k and o200k never merge a space into the digit group after
          // it: " 1 2" is [" ", "1", " ", "2"]. A run ending in "\n" or "\t"
          // needs no extra token; that character already costs one.
          if (j < length && text.charCodeAt(j - 1) === 0x20 && classOf(text.charCodeAt(j)) === DIGIT) {
            tokens += 1;
          }
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
        case RARE:
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
