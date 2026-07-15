/** Longest detail line (in chars) shown in an approval prompt. */
const DETAIL_MAX_CHARS = 500;
/** Most detail lines shown before truncation kicks in. */
const DETAIL_MAX_LINES = 6;
const TRUNCATION_MARKER = '… [truncated]';

/**
 * Every character that must be rendered visibly instead of reaching the
 * terminal: C0 controls except `\n` and `\t`, DEL, and the C1 range
 * U+0080–U+009F (which contains the raw CSI/OSC/ST bytes U+009B, U+009D
 * and U+009C). Matched one character at a time — deliberately NOT as
 * multi-character escape "sequences": the printable bytes inside those
 * sequences still belong to the approval detail and must remain visible.
 */
// eslint-disable-next-line no-control-regex
const CONTROL_CHAR_RE = /[\u0000-\u0008\u000b-\u001f\u007f-\u009f]/g;

/** Render one control character as visible `\xNN` text (e.g. ESC → `\x1b`). */
function encodeControlChar(ch: string): string {
  return `\\x${ch.charCodeAt(0).toString(16).padStart(2, '0')}`;
}

/**
 * Sanitize untrusted text before embedding it in an approval prompt.
 *
 * Pi's TUI preserves ANSI, so every terminal control byte is encoded rather
 * than deleted. Printable text is retained verbatim, and the rendered detail
 * is capped by line and character count with a visible truncation marker.
 */
export function sanitizeApprovalDetail(text: string): string {
  let out = text.replace(CONTROL_CHAR_RE, encodeControlChar);
  let truncated = false;

  const lines = out.split('\n');
  if (lines.length > DETAIL_MAX_LINES) {
    out = lines.slice(0, DETAIL_MAX_LINES).join('\n');
    truncated = true;
  }
  if (out.length > DETAIL_MAX_CHARS) {
    out = out.slice(0, DETAIL_MAX_CHARS);
    // Do not leave a lone high surrogate behind after the hard cut.
    const last = out.charCodeAt(out.length - 1);
    if (last >= 0xd800 && last <= 0xdbff) {
      out = out.slice(0, -1);
    }
    truncated = true;
  }
  if (truncated) {
    out += ` ${TRUNCATION_MARKER}`;
  }
  if (out.trim().length === 0 && text.length > 0) {
    // Control characters always encode to visible text, so this only fires
    // for whitespace-only input.
    return '(unprintable content)';
  }
  return out;
}
