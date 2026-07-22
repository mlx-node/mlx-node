import { describe, expect, it } from 'vite-plus/test';

import { shQuote } from '../ui/src/lib/shell.js';

// The dashboard builds a copy-pasteable `mlx agent --session <path>` resume
// command from a session's on-disk path. A path is attacker-influenceable (it
// contains the cwd the agent ran in, and pi lets a caller choose the session
// dir), so splicing it raw into a shell string is a command-injection vector:
// `$(…)`, backticks, `;`, and `&&` all execute; a space breaks the command.
// `shQuote` wraps the value in POSIX single quotes (the only fully-inert quoting
// in `sh`), escaping embedded single quotes with the canonical `'\''` idiom.
describe('shQuote', () => {
  it('wraps a plain token in single quotes', () => {
    expect(shQuote('hello')).toBe("'hello'");
  });

  it('quotes a path with spaces so it stays one argument', () => {
    expect(shQuote('/home/me/My Projects/app')).toBe("'/home/me/My Projects/app'");
  });

  it('escapes an embedded single quote with the canonical quote-out idiom', () => {
    expect(shQuote("it's")).toBe("'it'\\''s'");
    expect(shQuote("a'b'c")).toBe("'a'\\''b'\\''c'");
  });

  it('neutralizes command substitution and backticks', () => {
    expect(shQuote('$(rm -rf /)')).toBe("'$(rm -rf /)'");
    expect(shQuote('`whoami`')).toBe("'`whoami`'");
  });

  it('neutralizes command chaining metacharacters', () => {
    expect(shQuote('a; b && c | d')).toBe("'a; b && c | d'");
  });

  it('quotes the empty string to a valid empty argument', () => {
    expect(shQuote('')).toBe("''");
  });

  // Round-trip: every result is a single-quoted string whose only unquoted parts
  // are the `\'` escapes, so a POSIX shell re-reads exactly the input bytes.
  it('produces a value that decodes back to the original under POSIX rules', () => {
    for (const input of ["it's", '$(x)', '`y`', 'a b', "z'; rm -rf ~ #"]) {
      const quoted = shQuote(input);
      // Emulate the shell: strip the outer quoting layer and the `'\''` joins.
      const decoded = quoted
        .slice(1, -1) // drop the outer single quotes
        .replace(/'\\''/g, "'"); // each `'\''` was one literal single quote
      expect(decoded).toBe(input);
    }
  });
});
