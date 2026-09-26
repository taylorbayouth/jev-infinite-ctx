/**
 * Splitting an input into overlapping chunks that fit Jev's context window.
 *
 * Token counts are estimates. Jev's tokenizer is not public, so the library
 * assumes 3 bytes of UTF-8 per token: conservative for English (common
 * tokenizers average about 4) and close for code, JSON, and CJK text. If Jev
 * still rejects a chunk as too long, decide() shrinks the chunks and replans.
 */

const BYTES_PER_TOKEN = 3;

/** Each chunk repeats about 5% of the previous one, so no sentence loses its context at a boundary. */
const OVERLAP = 0.05;

/**
 * Line breaks and sentence ends: chunks break only here, unless a sentence is too long. One
 * terminator is enough (a run like "..." still ends at the same place); matching the whole run
 * would backtrack quadratically over a long run of dots.
 */
const BREAK = /\n+|[.!?]["'”’)\]]*\s+|[。！？]+["'”’)\]]*\s*/g;

export interface Span {
  start: number;
  end: number;
  weight: number;
}

/** Estimated Jev tokens in `text`: its UTF-8 size divided by 3, rounded up. */
export function estimateTokens(text: string): number {
  return Math.ceil(utf8Length(text, 0, text.length) / BYTES_PER_TOKEN);
}

/**
 * Plans chunks of at most `maxTokens` (2 or more) estimated tokens that
 * together cover every character of `text`, in order. Each chunk's weight is
 * the share of the text no earlier chunk covered, so overlap never counts twice.
 */
export function planChunks(text: string, maxTokens: number): Span[] {
  if (text.length === 0) return [{ start: 0, end: 0, weight: 1 }];
  const limit = maxTokens * BYTES_PER_TOKEN;
  const overlap = Math.floor(limit * OVERLAP);
  // Pieces small next to the overlap keep both the overlap and the packing tight.
  const maxPiece = Math.max(4, Math.floor(overlap / 2));
  const pieces = split(text, maxPiece);
  const total = pieces.reduce((sum, piece) => sum + piece.bytes, 0);

  let plan = pack(pieces, limit, overlap);
  // Even out chunk sizes so the last chunk is not a scrap: use the smallest limit that still needs
  // the same number of chunks. Any limit of at least overlap + 2 pieces leaves room for new text.
  for (let low = overlap + 2 * maxPiece, high = limit; plan.length > 1 && low < high; ) {
    const mid = Math.floor((low + high) / 2);
    const trial = pack(pieces, mid, overlap);
    if (trial.length === plan.length) {
      plan = trial;
      high = mid;
    } else {
      low = mid + 1;
    }
  }
  return plan.map(({ first, last, fresh }) => ({
    start: pieces[first]!.start,
    end: pieces[last]!.end,
    weight: fresh / total,
  }));
}

interface Piece {
  start: number;
  end: number;
  bytes: number;
}

interface Packed {
  first: number;
  last: number;
  /** Bytes no earlier chunk covered. */
  fresh: number;
}

/** Greedily fills chunks up to `limit` bytes; each starts about `overlap` bytes before the previous one ended. */
function pack(pieces: readonly Piece[], limit: number, overlap: number): Packed[] {
  const chunks: Packed[] = [];
  let first = 0;
  let uncovered = 0;
  for (;;) {
    let last = first;
    let bytes = pieces[first]!.bytes;
    while (last + 1 < pieces.length && bytes + pieces[last + 1]!.bytes <= limit) bytes += pieces[++last]!.bytes;
    let fresh = 0;
    for (let i = uncovered; i <= last; i++) fresh += pieces[i]!.bytes;
    chunks.push({ first, last, fresh });
    if (last === pieces.length - 1) return chunks;
    uncovered = last + 1;
    // Step back to share about `overlap` bytes, always starting after this chunk's start. Pieces
    // are at most overlap / 2, so the next chunk still has room for new text and the loop advances.
    let next = last + 1;
    let shared = 0;
    while (next - 1 > first && shared < overlap) shared += pieces[--next]!.bytes;
    first = next;
  }
}

/** Splits `text` at line breaks and sentence ends into pieces of at most `maxBytes`. */
function split(text: string, maxBytes: number): Piece[] {
  const pieces: Piece[] = [];
  let start = 0;
  const add = (end: number): void => {
    if (end > start) splitLong(text, start, end, maxBytes, pieces);
    start = end;
  };
  for (const match of text.matchAll(BREAK)) add(match.index + match[0].length);
  add(text.length);
  return pieces;
}

/** Adds `[start, end)` as one piece, or splits it at spaces, and failing that anywhere. */
function splitLong(text: string, start: number, end: number, maxBytes: number, pieces: Piece[]): void {
  const bytes = utf8Length(text, start, end);
  if (bytes <= maxBytes) {
    pieces.push({ start, end, bytes });
    return;
  }
  let pieceStart = start;
  let pieceBytes = 0;
  for (let i = start; i < end; ) {
    // A word and the spaces after it.
    let j = i;
    while (j < end && !isSpace(text.charCodeAt(j))) j++;
    while (j < end && isSpace(text.charCodeAt(j))) j++;
    const wordBytes = utf8Length(text, i, j);
    if (pieceBytes > 0 && pieceBytes + wordBytes > maxBytes) {
      pieces.push({ start: pieceStart, end: i, bytes: pieceBytes });
      pieceStart = i;
      pieceBytes = 0;
    }
    if (wordBytes > maxBytes) {
      splitAnywhere(text, i, j, maxBytes, pieces);
      pieceStart = j;
    } else {
      pieceBytes += wordBytes;
    }
    i = j;
  }
  if (pieceBytes > 0) pieces.push({ start: pieceStart, end, bytes: pieceBytes });
}

/** Cuts `[start, end)` into pieces of at most `maxBytes` (at least 4), never inside a surrogate pair. */
function splitAnywhere(text: string, start: number, end: number, maxBytes: number, pieces: Piece[]): void {
  let pieceStart = start;
  let pieceBytes = 0;
  for (let i = start; i < end; ) {
    const pair = isHigh(text.charCodeAt(i)) && i + 1 < end && isLow(text.charCodeAt(i + 1));
    const bytes = pair ? 4 : charBytes(text.charCodeAt(i));
    if (pieceBytes > 0 && pieceBytes + bytes > maxBytes) {
      pieces.push({ start: pieceStart, end: i, bytes: pieceBytes });
      pieceStart = i;
      pieceBytes = 0;
    }
    pieceBytes += bytes;
    i += pair ? 2 : 1;
  }
  pieces.push({ start: pieceStart, end, bytes: pieceBytes });
}

function utf8Length(text: string, start: number, end: number): number {
  let bytes = 0;
  for (let i = start; i < end; i++) bytes += charBytes(text.charCodeAt(i));
  return bytes;
}

/** UTF-8 bytes for one UTF-16 code unit; each half of a surrogate pair counts 2, so a pair is 4. */
function charBytes(code: number): number {
  if (code < 0x80) return 1;
  if (code < 0x800 || (code >= 0xd800 && code <= 0xdfff)) return 2;
  return 3;
}

function isSpace(code: number): boolean {
  return code === 32 || (code >= 9 && code <= 13) || code === 0x3000;
}

function isHigh(code: number): boolean {
  return code >= 0xd800 && code <= 0xdbff;
}

function isLow(code: number): boolean {
  return code >= 0xdc00 && code <= 0xdfff;
}
