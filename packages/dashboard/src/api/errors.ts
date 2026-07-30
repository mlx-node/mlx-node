/**
 * Typed error model for the transport-independent API layer.
 *
 * A handler signals failure by throwing an {@link ApiError} carrying a stable
 * machine-readable {@link ApiErrorCode}. The HTTP status is DERIVED from the code
 * through the single {@link STATUS_BY_CODE} table and is never stored on the
 * error, so a code and its status can never drift apart. Non-HTTP transports
 * (the Electron MessagePort bridge) carry the `code` and ignore the status
 * entirely.
 */

/**
 * The one and only code → HTTP status table. `ApiErrorCode` is derived from its
 * keys, so adding a code here is the single edit needed to introduce one.
 */
const STATUS_BY_CODE = {
  E_BAD_REQUEST: 400,
  E_FORBIDDEN: 403,
  E_NOT_FOUND: 404,
  E_METHOD_NOT_ALLOWED: 405,
  E_CONFLICT: 409,
  E_PAYLOAD_TOO_LARGE: 413,
  E_INTERNAL: 500,
  E_UNAVAILABLE: 503,
} as const;

export type ApiErrorCode = keyof typeof STATUS_BY_CODE;

/** The HTTP status an {@link ApiErrorCode} maps to. */
export function statusForCode(code: ApiErrorCode): number {
  return STATUS_BY_CODE[code];
}

/**
 * A failure a handler raises. `message` is user-facing (it is what the SPA
 * surfaces in a toast); `code` is the machine-readable discriminator.
 */
export class ApiError extends Error {
  readonly code: ApiErrorCode;

  constructor(code: ApiErrorCode, message: string) {
    super(message);
    this.name = 'ApiError';
    this.code = code;
  }

  /** Derived, never stored: the HTTP status for {@link code}. */
  get status(): number {
    return statusForCode(this.code);
  }

  static badRequest(message: string): ApiError {
    return new ApiError('E_BAD_REQUEST', message);
  }

  static forbidden(message: string): ApiError {
    return new ApiError('E_FORBIDDEN', message);
  }

  static notFound(message: string): ApiError {
    return new ApiError('E_NOT_FOUND', message);
  }

  static methodNotAllowed(message: string): ApiError {
    return new ApiError('E_METHOD_NOT_ALLOWED', message);
  }

  static conflict(message: string): ApiError {
    return new ApiError('E_CONFLICT', message);
  }

  static payloadTooLarge(message: string): ApiError {
    return new ApiError('E_PAYLOAD_TOO_LARGE', message);
  }

  static internal(message: string): ApiError {
    return new ApiError('E_INTERNAL', message);
  }

  static unavailable(message: string): ApiError {
    return new ApiError('E_UNAVAILABLE', message);
  }
}

/** A handler's value, plus the status the HTTP adapter should use for it. */
export interface ApiSuccess {
  ok: true;
  /** The route's declared success status (200 unless the route sets one). */
  status: number;
  body: unknown;
}

/** A failure, with `status` derived from `code` at construction. */
export interface ApiFailure {
  ok: false;
  code: ApiErrorCode;
  message: string;
  /** Always `statusForCode(code)`; carried so transports need not re-derive it. */
  status: number;
}

export type ApiResponse = ApiSuccess | ApiFailure;

/** Build a failure envelope, deriving its status from the code. */
export function failure(code: ApiErrorCode, message: string): ApiFailure {
  return { ok: false, code, message, status: statusForCode(code) };
}

/**
 * Normalize anything thrown by a handler into a failure envelope. An
 * {@link ApiError} keeps its code; anything else is an unexpected fault and
 * becomes `E_INTERNAL`, mirroring the old dispatcher's 500 catch-all (which
 * likewise surfaced `err.message` when present).
 */
export function toFailure(err: unknown): ApiFailure {
  if (err instanceof ApiError) return failure(err.code, err.message);
  return failure('E_INTERNAL', err instanceof Error ? err.message : 'Internal server error');
}
