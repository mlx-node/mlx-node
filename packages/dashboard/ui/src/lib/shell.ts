/**
 * POSIX single-quote a string so it is safe to splice into a copy-pasteable
 * shell command. Single quotes disable ALL expansion in `sh` (no `$`, no
 * backticks, no `~`, no globbing), so the value is passed through byte-for-byte;
 * an embedded single quote is emitted with the canonical `'\''` idiom (close the
 * quote, an escaped literal quote, reopen). Used by the dashboard's resume-command
 * builders, where the spliced value is an attacker-influenceable session path.
 */
export function shQuote(s: string): string {
  return `'${s.replace(/'/g, "'\\''")}'`;
}
