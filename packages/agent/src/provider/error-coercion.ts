/**
 * Hardened coercion of arbitrary thrown values to message strings.
 *
 * Shared by the stream adapter's TurnEmitter-independent failsafe path
 * and `TurnEmitter.onError`: both receive caller-supplied error values
 * and both promise never to throw, so every read here is guarded.
 */

/**
 * Coerce an arbitrary thrown value to a message string without trusting
 * it: an `Error` whose `message` getter throws, an object with a poisoned
 * `toString` / `Symbol.toPrimitive`, a null-prototype object (where
 * `String(err)` itself throws), and a revoked Proxy — where even
 * `err instanceof Error` throws, because `instanceof` walks the prototype
 * chain through the (revoked or throwing) `getPrototypeOf` trap — all
 * land on the constant fallback instead of escaping. Circular objects are
 * fine — `String` never serializes deeply.
 */
export function coerceErrorMessage(err: unknown): string {
  try {
    // The `instanceof` check MUST live inside the guard: on a revoked
    // Proxy (or any Proxy with a throwing `getPrototypeOf` trap) the
    // check itself throws a TypeError before any property is read.
    if (err instanceof Error) {
      const { message } = err;
      if (typeof message === 'string' && message.length > 0) return message;
    }
  } catch {
    // hostile prototype walk or poisoned `message` getter — fall through
  }
  try {
    return String(err);
  } catch {
    return 'unserializable error';
  }
}
