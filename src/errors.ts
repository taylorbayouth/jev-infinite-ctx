export type DecideErrorCode =
  /** The options or question are invalid. Nothing was sent. */
  | "invalid_request"
  /** The input is over `maxInputTokens`. Nothing was sent. */
  | "input_too_large"
  /** A chunk's request failed after retries. No partial answer is returned. */
  | "request_failed"
  /** The caller's `signal` fired. */
  | "aborted";

/** The only error `decide()` throws. */
export class DecideError extends Error {
  override name = "DecideError";
  readonly code: DecideErrorCode;
  /** 0-based index of the chunk that failed (`request_failed`). */
  readonly chunk: number | undefined;
  /** HTTP status from the provider, when there was one. */
  readonly status: number | undefined;

  constructor(
    code: DecideErrorCode,
    message: string,
    options: { chunk?: number | undefined; status?: number | undefined; cause?: unknown } = {},
  ) {
    super(message, { cause: options.cause });
    this.code = code;
    this.chunk = options.chunk;
    this.status = options.status;
  }
}
