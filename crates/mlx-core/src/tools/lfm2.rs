use serde_json::Value;
use unicode_normalization::UnicodeNormalization;

use super::ToolCallResult;
use super::markup::all_positions;

// ---------------------------------------------------------------------------
// LFM2 pythonic tool calls — <|tool_call_start|>[fn(kw=literal)]<|tool_call_end|>
// ---------------------------------------------------------------------------
//
// Mirrors vLLM `tool_parsers/lfm2_tool_parser.py::Lfm2ToolParser`
// (`extract_tool_calls`, non-streaming) semantics:
//   - Every valid sentinel block is parsed in order. Between blocks, echoed
//     text through the last orphan end sentinel is dropped. A malformed first
//     block preserves the whole output; a malformed later block keeps prior
//     calls and strips its echo tail.
//   - The call body must be a non-empty bracketed list whose every element
//     is a function call `name(kw=literal, ...)`. Dotted names
//     (`a.b.c(...)`) are preserved verbatim.
//   - Keyword args only: positionals are ignored, `**kwargs` rejects the
//     block (vLLM `handle_single_tool` iterates `call.keywords` only and
//     `arguments[None]` fails `json.dumps`).
//   - Argument values are Python literals: strings (single/double/triple
//     quotes, `r`/`u`/`f` prefixes, escapes; `f` rejected when it carries a
//     `{...}` placeholder), numbers (dec/hex/oct/bin ints, floats, unary
//     +/-), True/False/None plus the JSON spellings true/false/null, lists,
//     tuples and sets (→ arrays), dicts with literal keys (non-string keys
//     are stringified like `json.dumps`). Anything else rejects the block.
//   - A FIRST-block parse failure returns the raw text verbatim as content
//     (vLLM returns `content=model_output`, `tools_called=False`). Later
//     malformed blocks keep earlier calls and apply echo suppression.
//
// The sentinels are non-special added tokens in the LFM2 tokenizers, so
// `skip_special_tokens=true` does NOT strip them — this parser sees them
// in the default decode with no flag changes needed.

pub(super) const LFM2_TOOL_CALL_START: &str = "<|tool_call_start|>";
pub(super) const LFM2_TOOL_CALL_END: &str = "<|tool_call_end|>";

/// vLLM `_strip_echo`: drop everything through the last orphan
/// `<|tool_call_end|>` in the post-call text — the echoed body is capped by
/// a second end sentinel, so the last end marks where real content resumes.
fn strip_lfm2_echo(raw_after: &str) -> &str {
    match raw_after.rfind(LFM2_TOOL_CALL_END) {
        Some(idx) => &raw_after[idx + LFM2_TOOL_CALL_END.len()..],
        None => raw_after,
    }
}

/// Recursive-descent parser for the pythonic call list. Byte-indexed
/// scanner: every syntax character we look for is ASCII, so UTF-8 string
/// contents pass through safely.
struct PyLiteralParser<'a> {
    s: &'a [u8],
    pos: usize,
    /// `parse_literal` nesting depth — containers and unary ops recurse, so
    /// pathological input (`[[[[…`, `x=----…1`) must fail closed instead of
    /// overflowing the stack (vLLM's `ast.parse` raises `RecursionError`,
    /// which its caller catches → raw text).
    depth: u32,
}

/// Hard cap on literal nesting — far beyond any real tool argument.
pub(super) const MAX_PY_LITERAL_DEPTH: u32 = 64;

/// Python reserved words that can never be a name — a call segment
/// (`for(x)`), a comprehension target (`for not in y`), or a lambda
/// parameter (`lambda True: x`) is a SyntaxError, and `None(x)` parses to
/// a `Constant` with no `.func.id` under vLLM's AST extraction. Soft
/// keywords (`match`, `case`, `type`) stay usable as names.
fn reserved_ident(id: &[u8]) -> bool {
    matches!(
        id,
        b"and"
            | b"as"
            | b"assert"
            | b"async"
            | b"await"
            | b"break"
            | b"class"
            | b"continue"
            | b"def"
            | b"del"
            | b"elif"
            | b"else"
            | b"except"
            | b"finally"
            | b"for"
            | b"from"
            | b"global"
            | b"if"
            | b"import"
            | b"in"
            | b"is"
            | b"lambda"
            | b"nonlocal"
            | b"not"
            | b"or"
            | b"pass"
            | b"raise"
            | b"return"
            | b"try"
            | b"while"
            | b"with"
            | b"yield"
            | b"True"
            | b"False"
            | b"None"
    )
}

impl<'a> PyLiteralParser<'a> {
    fn new(text: &'a str) -> Self {
        Self {
            s: text.as_bytes(),
            pos: 0,
            depth: 0,
        }
    }

    fn peek(&self) -> Option<u8> {
        self.s.get(self.pos).copied()
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.pos += 1;
        }
    }

    fn expect(&mut self, b: u8) -> Result<(), ()> {
        self.skip_ws();
        if self.peek() == Some(b) {
            self.pos += 1;
            Ok(())
        } else {
            Err(())
        }
    }

    fn eof(&mut self) -> bool {
        self.skip_ws();
        self.pos >= self.s.len()
    }

    fn ident(&mut self) -> Result<&'a str, ()> {
        self.skip_ws();
        let start = self.pos;
        self.pos = ident_end(self.s, self.pos);
        if start == self.pos || !ident_char_at(self.s, start, true) {
            return Err(());
        }
        std::str::from_utf8(&self.s[start..self.pos]).map_err(|_| ())
    }

    /// `name` or `a.b.c` — dotted attribute chains are preserved.
    fn dotted_name(&mut self) -> Result<String, ()> {
        let first = self.ident()?;
        if reserved_ident(first.as_bytes()) {
            return Err(());
        }
        let mut name: String = first.nfkc().collect();
        loop {
            self.skip_ws();
            if self.peek() != Some(b'.') {
                break;
            }
            self.pos += 1;
            let seg = self.ident()?;
            if reserved_ident(seg.as_bytes()) {
                return Err(());
            }
            name.push('.');
            name.extend(seg.nfkc());
        }
        Ok(name)
    }

    /// Skip a quoted string starting at `pos` (single or triple quoted);
    /// backslash escapes the next byte. Unterminated → Err.
    fn skip_quoted(&mut self, f_mode: bool, bytes_mode: bool, raw_mode: bool) -> Result<(), ()> {
        let quote = self.s[self.pos];
        self.pos += 1;
        let triple = self.peek() == Some(quote) && self.s.get(self.pos + 1) == Some(&quote);
        if triple {
            self.pos += 2;
        }
        while let Some(&b) = self.s.get(self.pos) {
            if bytes_mode && b >= 0x80 {
                return Err(());
            }
            if f_mode && b == b'{' {
                if self.s.get(self.pos + 1) == Some(&b) {
                    self.pos += 2;
                } else {
                    self.pos += 1;
                    self.skip_f_replacement(1)?;
                }
                continue;
            }
            if f_mode && b == b'}' {
                if self.s.get(self.pos + 1) != Some(&b) {
                    return Err(());
                }
                self.pos += 2;
                continue;
            }
            if b == b'\\' {
                if !raw_mode {
                    // Mandatory escapes must be well-formed or `ast.parse`
                    // raises and vLLM keeps the whole block verbatim:
                    // `\x` needs 2 hex digits, `\u`/`\U` need 4/8, and `\N`
                    // is rejected outright (see below). `\u`/`\U`/`\N` are
                    // NOT escapes in bytes literals, so they stay literal
                    // there (only `\x` is checked).
                    let escape = self.s.get(self.pos + 1).copied();
                    let hex_run = |start: usize, len: usize| {
                        (0..len)
                            .all(|i| self.s.get(start + i).is_some_and(|c| c.is_ascii_hexdigit()))
                    };
                    let hex_value = |start: usize, len: usize| {
                        (0..len).fold(0u32, |acc, i| {
                            acc * 16 + (self.s[start + i] as char).to_digit(16).unwrap_or(0)
                        })
                    };
                    match escape {
                        Some(b'x') if !hex_run(self.pos + 2, 2) => return Err(()),
                        Some(b'u') if !bytes_mode && !hex_run(self.pos + 2, 4) => {
                            return Err(());
                        }
                        // `\U` is also range-checked: past 0x10FFFF is
                        // "illegal Unicode character" (surrogates are fine).
                        Some(b'U')
                            if !bytes_mode
                                && (!hex_run(self.pos + 2, 8)
                                    || hex_value(self.pos + 2, 8) > 0x10FFFF) =>
                        {
                            return Err(());
                        }
                        // `\N` is a recognized escape introducer in str: a
                        // bare `\N` is a hard SyntaxError, and `\N{NAME}`
                        // needs a Unicode-name table we don't carry, so an
                        // unknown name cannot be told from a known one.
                        // Reject both rather than promote a call whose
                        // `ast.parse` failed — the same reject-on-
                        // non-representable stance as keyword `\N` and
                        // non-JSON literals. `b'\N'` is only a warning, so
                        // bytes skip this.
                        Some(b'N') if !bytes_mode => return Err(()),
                        _ => {}
                    }
                }
                self.pos += if f_mode && matches!(self.s.get(self.pos + 1), Some(b'{' | b'}')) {
                    1
                } else {
                    2
                };
                continue;
            }
            if b == quote {
                if triple {
                    if self.s.get(self.pos + 1) == Some(&quote)
                        && self.s.get(self.pos + 2) == Some(&quote)
                    {
                        self.pos += 3;
                        return Ok(());
                    }
                    self.pos += 1;
                    continue;
                }
                self.pos += 1;
                return Ok(());
            }
            self.pos += 1;
        }
        Err(())
    }

    fn validate_f_expression(&self, expression: &[u8]) -> Result<(), ()> {
        let expression = std::str::from_utf8(expression).map_err(|_| ())?.trim();
        if expression.is_empty() {
            return Err(());
        }
        let wrapped = format!("({expression}),");
        let mut parser = PyLiteralParser::new(&wrapped);
        parser.depth = self.depth;
        parser.skip_expression(false)?;
        parser.expect(b',')?;
        if parser.eof() { Ok(()) } else { Err(()) }
    }

    fn skip_f_replacement(&mut self, nesting: u32) -> Result<(), ()> {
        if nesting > 3 {
            return Err(());
        }
        let start = self.pos;
        let mut stack = Vec::new();
        loop {
            let Some(&b) = self.s.get(self.pos) else {
                return Err(());
            };
            match b {
                b'\'' | b'"' => {
                    let prefix_end = self.pos;
                    let mut prefix_start = prefix_end;
                    while prefix_start > start
                        && prefix_end - prefix_start < 2
                        && matches!(
                            self.s[prefix_start - 1],
                            b'r' | b'R' | b'u' | b'U' | b'f' | b'F' | b'b' | b'B'
                        )
                    {
                        prefix_start -= 1;
                    }
                    let prefix = &self.s[prefix_start..prefix_end];
                    let valid_prefix = match prefix.len() {
                        0 => true,
                        1 => matches!(prefix[0].to_ascii_lowercase(), b'r' | b'u' | b'b' | b'f'),
                        2 => matches!(
                            (
                                prefix[0].to_ascii_lowercase(),
                                prefix[1].to_ascii_lowercase()
                            ),
                            (b'r', b'b') | (b'b', b'r') | (b'r', b'f') | (b'f', b'r')
                        ),
                        _ => false,
                    };
                    self.skip_quoted(
                        valid_prefix && prefix.iter().any(|c| c.eq_ignore_ascii_case(&b'f')),
                        valid_prefix && prefix.iter().any(|c| c.eq_ignore_ascii_case(&b'b')),
                        valid_prefix && prefix.iter().any(|c| c.eq_ignore_ascii_case(&b'r')),
                    )?;
                }
                b'(' | b'[' | b'{' => {
                    stack.push(b);
                    self.pos += 1;
                }
                b')' | b']' | b'}' => {
                    if b == b'}' && stack.is_empty() {
                        break;
                    }
                    let Some(open) = stack.pop() else {
                        return Err(());
                    };
                    if !matches!((open, b), (b'(', b')') | (b'[', b']') | (b'{', b'}')) {
                        return Err(());
                    }
                    self.pos += 1;
                }
                b'!' if stack.is_empty() && self.s.get(self.pos + 1) != Some(&b'=') => break,
                b':' if stack.is_empty() => break,
                b'=' if stack.is_empty()
                    && self.s.get(self.pos + 1) != Some(&b'=')
                    && (self.pos == start
                        || !matches!(self.s[self.pos - 1], b'!' | b'<' | b'>' | b'=')) =>
                {
                    break;
                }
                _ => {
                    let len = utf8_len(b);
                    if self.pos + len > self.s.len() {
                        return Err(());
                    }
                    self.pos += len;
                }
            }
        }
        let end = self.pos;
        self.validate_f_expression(&self.s[start..end])?;
        if self.peek() == Some(b'=') {
            self.pos += 1;
            while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
                self.pos += 1;
            }
        }
        if self.peek() == Some(b'!') {
            self.pos += 1;
            if !matches!(self.peek(), Some(b's' | b'r' | b'a')) {
                return Err(());
            }
            self.pos += 1;
            while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
                self.pos += 1;
            }
        }
        if self.peek() == Some(b':') {
            self.pos += 1;
            self.skip_f_format_spec(nesting)?;
        }
        if self.peek() != Some(b'}') {
            return Err(());
        }
        self.pos += 1;
        Ok(())
    }

    fn skip_f_format_spec(&mut self, nesting: u32) -> Result<(), ()> {
        loop {
            let Some(&b) = self.s.get(self.pos) else {
                return Err(());
            };
            if b == b'}' {
                return Ok(());
            }
            if b == b'{' {
                self.pos += 1;
                self.skip_f_replacement(nesting + 1)?;
            } else {
                let len = utf8_len(b);
                if self.pos + len > self.s.len() {
                    return Err(());
                }
                self.pos += len;
            }
        }
    }

    /// Skip one positional argument's expression text. vLLM ignores
    /// `call.args` entirely, so the content is dropped — but the skipped
    /// text must still be a plausible Python expression: anything
    /// `ast.parse` would `SyntaxError` (`@@@`, `1 +`, `a b`, `f(,)`,
    /// unbalanced closers, bare `=`/`!`) rejects the WHOLE block verbatim
    /// instead of promoting an `ok` call built from the remaining kwargs.
    ///
    /// The scan is a small operand/operator automaton, not a full grammar:
    /// `Need` requires a value (identifier, literal, number, string,
    /// container opener, unary `- + ~ not`, `*` spread); `Have` requires a
    /// binary operator, a postfix (`(` `[` `.`), or the top-level `,`/`)`
    /// argument boundary. `Lambda` covers `lambda …:` parameter lists.
    /// Deliberate residuals (rare invalid forms still accepted, e.g.
    /// `a ~ b`, `0xg`): skipping is a boundary check, not codegen — the
    /// goal is keeping obviously-malformed text from promoting a call.
    /// `starred` marks the expression as an already-consumed `*a`
    /// unpacking — a `for` at the top level then makes it a comp
    /// element (`*a for a in b` is a SyntaxError).
    fn skip_expression(&mut self, starred: bool) -> Result<(), ()> {
        self.skip_expression_until(starred, false)
    }

    fn skip_expression_until(&mut self, starred: bool, lambda_default: bool) -> Result<(), ()> {
        self.depth += 1;
        if self.depth > MAX_PY_LITERAL_DEPTH {
            self.depth -= 1;
            return Err(());
        }
        let result = self.skip_expression_inner(starred, lambda_default);
        self.depth -= 1;
        result
    }

    fn skip_expression_inner(&mut self, starred: bool, lambda_default: bool) -> Result<(), ()> {
        #[derive(Clone, Copy, PartialEq)]
        enum St {
            Need,
            Have,
            Lambda,
        }
        #[derive(Clone, Copy, PartialEq)]
        enum NeedCtx {
            Expr,
            Inversion,
            Restricted,
        }
        #[derive(Clone, Copy, PartialEq)]
        enum LambdaMarker {
            None,
            Star,
            DoubleStar,
            VarargsDone,
            KwargsDone,
            Slash,
        }
        // Identifiers that can never appear where an operand is expected.
        fn reject_keyword(id: &[u8]) -> bool {
            matches!(
                id,
                b"and"
                    | b"or"
                    | b"in"
                    | b"is"
                    | b"if"
                    | b"else"
                    | b"for"
                    | b"elif"
                    | b"while"
                    | b"return"
                    | b"def"
                    | b"class"
                    | b"import"
                    | b"from"
                    | b"as"
                    | b"with"
                    | b"try"
                    | b"except"
                    | b"finally"
                    | b"raise"
                    | b"pass"
                    | b"break"
                    | b"continue"
                    | b"global"
                    | b"nonlocal"
                    | b"del"
                    | b"assert"
                    | b"yield"
                    | b"await"
                    | b"async"
            )
        }
        // Python string-literal prefixes: r/u/f/b alone, or the r-pairs
        // rb/br/rf/fr — anything else (`rr`, `uf`, `bu` …) is a plain
        // identifier, so a glued quote after it is a SyntaxError.
        fn string_prefix(id: &[u8]) -> bool {
            match id.len() {
                1 => matches!(id[0].to_ascii_lowercase(), b'r' | b'u' | b'b' | b'f'),
                2 => matches!(
                    (id[0].to_ascii_lowercase(), id[1].to_ascii_lowercase()),
                    (b'r', b'b') | (b'b', b'r') | (b'r', b'f') | (b'f', b'r')
                ),
                _ => false,
            }
        }
        let mut st = St::Need;
        let mut need_ctx = NeedCtx::Expr;
        let mut stack: Vec<u8> = Vec::new(); // open brackets — closers must match
        // A postfix `x[…]` opens a subscript — the only bracket where `:`
        // is slice syntax (`x[1:2]`, `x[:]`, `x[::]`). `:` inside a list
        // display, tuple, or call is a SyntaxError; inside `{}` it is a
        // dict separator that requires a value.
        const SUBSCRIPT: u8 = 0;
        // A postfix `f(` opens a call — keyword `x=1` arguments are legal
        // inside it but not inside a grouping `(`, list display, dict, or
        // subscript.
        const CALLPAREN: u8 = 1;
        // `{}` resolves to a set display on its first bare element (`,` /
        // walrus / `for`) and to a dict on a key `:` — mixing the two
        // (`{1, 2:3}`, `{1:2, 3}`) is a SyntaxError.
        const SET: u8 = 2;
        const DICT: u8 = 3;
        fn close_match(o: u8, b: u8) -> bool {
            match b {
                b')' => matches!(o, b'(' | CALLPAREN),
                b']' => matches!(o, b'[' | SUBSCRIPT),
                b'}' => matches!(o, b'{' | SET | DICT),
                _ => false,
            }
        }
        fn enter_target_receiver(
            depth: usize,
            island_stack: &mut Vec<usize>,
            target_group_depths: &mut Vec<usize>,
            target_star_depths: &mut Vec<usize>,
            target_comma_depths: &mut Vec<usize>,
            target_unassignable_depths: &mut Vec<usize>,
            target_receiver_depths: &mut Vec<usize>,
        ) {
            target_group_depths.retain(|&d| d != depth);
            target_star_depths.retain(|&d| d != depth);
            target_comma_depths.retain(|&d| d != depth);
            target_unassignable_depths.retain(|&d| d != depth);
            island_stack.push(depth - 1);
            target_receiver_depths.push(depth);
        }
        fn enter_direct_target_receiver(
            depth: usize,
            island_stack: &mut Vec<usize>,
            target_star_depths: &mut Vec<usize>,
            target_comma_depths: &mut Vec<usize>,
            target_unassignable_depths: &mut Vec<usize>,
            target_direct_receiver_depths: &mut Vec<usize>,
        ) {
            target_star_depths.retain(|&d| d != depth);
            target_comma_depths.retain(|&d| d != depth);
            target_unassignable_depths.retain(|&d| d != depth);
            island_stack.push(depth);
            target_direct_receiver_depths.push(depth);
        }
        // `not` consumed as an operator (`x not in y`): `in`/`is` may follow.
        let mut after_not_op = false;
        // Ternary `x if y else z` needs its `else`: `a if b` is a
        // SyntaxError. Comprehension `if`s (`[x for i in y if z]`) are
        // filters — no `else` — tracked per bracket depth so nested
        // comprehensions and outer clauses survive inner closes.
        let mut pending_ifs: Vec<usize> = Vec::new();
        let mut comp_depths: Vec<usize> = Vec::new();
        // Every `for` is a comprehension clause that owes an `in` at the
        // same bracket depth (`[x for x]` is a SyntaxError). An `in`
        // inside deeper brackets belongs to the target or iterable
        // expression and must not satisfy it.
        let mut for_depths: Vec<usize> = Vec::new();
        // Subscript brackets opened while a `for` target is open
        // (`for a[x + 1] in` — a subscript value is a real expression).
        // Recorded as the stack depth at which they were pushed; inside
        // them the target grammar does not apply.
        let mut island_stack: Vec<usize> = Vec::new();
        let mut target_group_depths: Vec<usize> = Vec::new();
        let mut target_star_depths: Vec<usize> = Vec::new();
        let mut target_comma_depths: Vec<usize> = Vec::new();
        let mut target_unassignable_depths: Vec<usize> = Vec::new();
        let mut target_receiver_depths: Vec<usize> = Vec::new();
        let mut target_direct_receiver_depths: Vec<usize> = Vec::new();
        // A `for` at top level is a genexpr — valid only as the sole
        // argument, so a `,` boundary after it is a SyntaxError.
        let mut top_genexpr = false;
        // Lambda parameter tracking: at a param start (entry / after
        // `,`) only a name, `*`/`/` markers, `:` (zero params), or `,`
        // after a marker are legal — `lambda 1: x`, `lambda 'a': x`,
        // `lambda ,: x`, `lambda =x: y` are all SyntaxErrors. Default
        // expressions after `=` are scanned loosely like the rest.
        let mut param_start = false;
        let mut lambda_marker = LambdaMarker::None;
        // A completed param name admits only `,` `=` or the ending `:` —
        // `lambda a b` / `lambda a.b` / `lambda a(` are SyntaxErrors.
        let mut param_done = false;
        let mut lambda_depth = 0;
        let mut lambda_default_seen = false;
        let mut lambda_keyword_only = false;
        let mut param_has_default = false;
        let mut lambda_positional_seen = false;
        let mut lambda_slash_seen = false;
        let mut lambda_bare_star_pending = false;
        // Implicit-concat / string-prefix tracking: a quote after an
        // operand is legal only directly glued to a prefix identifier
        // (`rb"x"` — one literal) or after a string literal (`"a" "b"`).
        let mut last_str = false;
        let mut prefix_end = usize::MAX;
        let mut prefix_f = false;
        let mut prefix_b = false;
        let mut prefix_r = false;
        // `*a`/`**a` consumed at an element start inside a bracket —
        // recorded as (depth, bracket). A `for` at that depth makes it a
        // comprehension element where unpacking is a SyntaxError
        // (`[*a for x in y]`, `f(*a for x in y)`); a `)` closing a `(`
        // group with no `,` is `(*a)` — also a SyntaxError.
        let mut starred_depths: Vec<(usize, u8)> = Vec::new();
        if starred {
            starred_depths.push((0, 0));
        }
        // `,` inside a bracket — a comprehension is single-element, so a
        // `for` after any element comma is a SyntaxError (`[a, x for x
        // in y]`).
        let mut comma_depths: Vec<usize> = Vec::new();
        // Depths where a dict `,` was seen — the next operand is a key
        // that owes a `:` (`{k:v, k2}` is a SyntaxError). A `}` in Have
        // state with a key pending rejects; in Need it's a legal
        // trailing comma.
        let mut dict_key_next: Vec<usize> = Vec::new();
        let mut slice_colons: Vec<(usize, u8)> = Vec::new();
        let mut await_operand = false;
        let mut yield_depths: Vec<usize> = Vec::new();
        let mut yield_from_depths: Vec<usize> = Vec::new();
        // `*a`/`**a` unpacking is legal only at an element start — after
        // `,`, a bracket open, or the expression start. After an
        // operator, `:`, `=`, or a unary prefix it is a SyntaxError
        // (`f(1 + *a)`).
        let mut elem_start = !lambda_default;
        // Argument ordering inside a nested call: once a `x=1` kwarg is
        // consumed at a CALLPAREN depth, later elements must be `*a` or
        // kwargs — a bare positional (`helper(x=1, 2)`) is a SyntaxError.
        // `kwarg_seen` persists per depth; `kwarg_elem`/`star_elem`
        // describe the element currently closing and reset at `,`.
        let mut kwarg_seen_depths: Vec<usize> = Vec::new();
        let mut kw_unpack_depths: Vec<usize> = Vec::new();
        let mut kwarg_elem_depths: Vec<usize> = Vec::new();
        let mut star_elem_depths: Vec<usize> = Vec::new();
        loop {
            let Some(&b) = self.s.get(self.pos) else {
                return Err(()); // ran out of text — unterminated
            };
            // Inside a `for` target only target grammar is legal — names,
            // `*` starred items, `(`/`[` target groups, `.`/`[` postfix,
            // `,` separators, and the terminating `in`. Binary operators,
            // literals, and calls are all SyntaxErrors (`[x for x + y
            // in z]`). A subscript island suspends the target grammar
            // until it closes (`for a[x + 1] in` is valid).
            let strict_target = for_depths.last().is_some_and(|target_depth| {
                island_stack
                    .last()
                    .is_none_or(|island_depth| target_depth > island_depth)
            });
            match b {
                b' ' | b'\t' | b'\n' | b'\r' => self.pos += 1,
                // Top-level argument boundary — valid only after an operand
                // (not inside `lambda` params, where `,` separates names).
                b')' | b',' | b':'
                    if stack.is_empty() && st != St::Lambda && (b != b':' || lambda_default) =>
                {
                    return if st == St::Have
                        && pending_ifs.is_empty()
                        && for_depths.is_empty()
                        && !(top_genexpr && (b == b',' || lambda_default))
                    {
                        Ok(())
                    } else {
                        Err(())
                    };
                }
                _ => match st {
                    St::Need => {
                        if after_not_op {
                            // A `not` after an operand only ever forms
                            // `x not in y` — anything else next (`is`, a
                            // value, a group) is a SyntaxError.
                            let id_start = self.pos;
                            self.pos = ident_end(self.s, self.pos);
                            if self.pos > id_start && &self.s[id_start..self.pos] == b"in" {
                                after_not_op = false;
                                need_ctx = NeedCtx::Restricted;
                                continue; // still Need — the operand follows
                            }
                            return Err(());
                        }
                        if await_operand
                            && !matches!(b, b'(' | b'[' | b'{' | b'\'' | b'"' | b'.')
                            && !b.is_ascii_digit()
                            && !ident_char_at(self.s, self.pos, true)
                        {
                            return Err(());
                        }
                        match b {
                            // A `for` target admits only names, `*` starred
                            // items, and `(`/`[` target groups — literals,
                            // unary ops, `{}`, and every other first are
                            // SyntaxErrors.
                            _ if strict_target => match b {
                                _ if !target_group_depths.contains(&stack.len())
                                    && (matches!(b, b'\'' | b'"' | b'{' | b'.')
                                        || b.is_ascii_digit()) =>
                                {
                                    enter_direct_target_receiver(
                                        stack.len(),
                                        &mut island_stack,
                                        &mut target_star_depths,
                                        &mut target_comma_depths,
                                        &mut target_unassignable_depths,
                                        &mut target_direct_receiver_depths,
                                    );
                                    need_ctx = NeedCtx::Expr;
                                    continue;
                                }
                                _ if target_group_depths.contains(&stack.len())
                                    && !matches!(b, b'*' | b'(' | b'[' | b')' | b']')
                                    && !ident_char_at(self.s, self.pos, true) =>
                                {
                                    enter_target_receiver(
                                        stack.len(),
                                        &mut island_stack,
                                        &mut target_group_depths,
                                        &mut target_star_depths,
                                        &mut target_comma_depths,
                                        &mut target_unassignable_depths,
                                        &mut target_receiver_depths,
                                    );
                                    need_ctx = NeedCtx::Expr;
                                    continue;
                                }
                                b'*' => {
                                    if self.s.get(self.pos + 1) == Some(&b'*') {
                                        return Err(()); // `**a` is no target
                                    }
                                    target_star_depths.push(stack.len());
                                    self.pos += 1;
                                }
                                b'(' | b'[' => {
                                    stack.push(b);
                                    target_group_depths.push(stack.len());
                                    self.pos += 1;
                                }
                                b')' | b']' => {
                                    let depth = stack.len();
                                    let Some(&open) = stack.last() else {
                                        return Err(());
                                    };
                                    let mut previous = self.pos;
                                    while previous > 0
                                        && matches!(
                                            self.s[previous - 1],
                                            b' ' | b'\t' | b'\n' | b'\r'
                                        )
                                    {
                                        previous -= 1;
                                    }
                                    let empty = previous > 0
                                        && matches!(
                                            (open, self.s[previous - 1]),
                                            (b'(', b'(') | (b'[', b'[')
                                        );
                                    if !target_group_depths.contains(&depth)
                                        || !close_match(open, b)
                                        || (!empty && !target_comma_depths.contains(&depth))
                                    {
                                        return Err(());
                                    }
                                    stack.pop();
                                    self.pos += 1;
                                    target_group_depths.retain(|&d| d != depth);
                                    target_star_depths.retain(|&d| d != depth);
                                    target_comma_depths.retain(|&d| d != depth);
                                    target_unassignable_depths.retain(|&d| d != depth);
                                    target_unassignable_depths.retain(|&d| d != stack.len());
                                    st = St::Have;
                                    last_str = false;
                                    prefix_end = usize::MAX;
                                }
                                _ if ident_char_at(self.s, self.pos, true) => {
                                    let id_start = self.pos;
                                    self.pos = ident_end(self.s, self.pos);
                                    let id = &self.s[id_start..self.pos];
                                    let depth = stack.len();
                                    if id == b"in"
                                        && for_depths.last() == Some(&depth)
                                        && target_comma_depths.contains(&depth)
                                    {
                                        for_depths.pop();
                                        target_group_depths.retain(|&d| d < depth);
                                        target_star_depths.retain(|&d| d < depth);
                                        target_comma_depths.retain(|&d| d < depth);
                                        target_unassignable_depths.retain(|&d| d < depth);
                                        st = St::Need;
                                        need_ctx = NeedCtx::Restricted;
                                    } else {
                                        let direct_receiver = !target_group_depths.contains(&depth)
                                            && (matches!(id, b"True" | b"False" | b"None")
                                                || (string_prefix(id)
                                                    && matches!(self.peek(), Some(b'\'' | b'"'))));
                                        if direct_receiver {
                                            self.pos = id_start;
                                            enter_direct_target_receiver(
                                                depth,
                                                &mut island_stack,
                                                &mut target_star_depths,
                                                &mut target_comma_depths,
                                                &mut target_unassignable_depths,
                                                &mut target_direct_receiver_depths,
                                            );
                                            need_ctx = NeedCtx::Expr;
                                            continue;
                                        }
                                        if reserved_ident(id) {
                                            if !target_group_depths.contains(&depth) {
                                                return Err(());
                                            }
                                            self.pos = id_start;
                                            enter_target_receiver(
                                                depth,
                                                &mut island_stack,
                                                &mut target_group_depths,
                                                &mut target_star_depths,
                                                &mut target_comma_depths,
                                                &mut target_unassignable_depths,
                                                &mut target_receiver_depths,
                                            );
                                            need_ctx = NeedCtx::Expr;
                                            continue;
                                        }
                                        target_unassignable_depths.retain(|&d| d != depth);
                                        st = St::Have;
                                        last_str = false;
                                        prefix_end = usize::MAX;
                                    }
                                }
                                _ => return Err(()),
                            },
                            b'(' | b'[' | b'{' => {
                                stack.push(b);
                                self.pos += 1;
                                await_operand = false;
                                elem_start = true;
                                need_ctx = NeedCtx::Expr;
                            }
                            // Empty container (`()` `[]` `{}`) or a close after
                            // `,`/`:` inside brackets (`(a,)` `a[1:]`) — never
                            // after an operator (`(a +)` is a SyntaxError).
                            b')' | b']' | b'}' => {
                                let mut p = self.pos;
                                while p > 0 && matches!(self.s[p - 1], b' ' | b'\t' | b'\n' | b'\r')
                                {
                                    p -= 1;
                                }
                                let trailing_ok = p > 0
                                    && match self.s[p - 1] {
                                        b'(' | b'[' | b'{' | b',' => true,
                                        // `x[1:]` — an open slice end is
                                        // legal only in a subscript; `{k:}`
                                        // is missing its dict value.
                                        b':' => stack.last() == Some(&SUBSCRIPT),
                                        _ => false,
                                    };
                                if !trailing_ok
                                    || (b == b']'
                                        && stack.last() == Some(&SUBSCRIPT)
                                        && self.s[p - 1] == b'[')
                                {
                                    return Err(());
                                }
                                match stack.pop() {
                                    Some(o) if close_match(o, b) => {
                                        // `[x for a,]` — closing the comp
                                        // bracket while its `for` still
                                        // owes `in` is a SyntaxError.
                                        if for_depths.last() == Some(&(stack.len() + 1)) {
                                            return Err(());
                                        }
                                        let closed_depth = stack.len() + 1;
                                        if target_receiver_depths.contains(&closed_depth)
                                            && island_stack.last() == Some(&stack.len())
                                        {
                                            island_stack.pop();
                                            target_receiver_depths.retain(|&d| d != closed_depth);
                                            if !target_unassignable_depths.contains(&stack.len()) {
                                                target_unassignable_depths.push(stack.len());
                                            }
                                        } else if matches!(o, SUBSCRIPT | CALLPAREN)
                                            && target_direct_receiver_depths.contains(&stack.len())
                                            && island_stack.last() == Some(&stack.len())
                                        {
                                            if o == SUBSCRIPT {
                                                island_stack.pop();
                                                target_direct_receiver_depths
                                                    .retain(|&d| d != stack.len());
                                            }
                                        } else if matches!(o, SUBSCRIPT | CALLPAREN)
                                            && island_stack.last() == Some(&stack.len())
                                        {
                                            island_stack.pop();
                                            if o == CALLPAREN {
                                                if !target_unassignable_depths
                                                    .contains(&stack.len())
                                                {
                                                    target_unassignable_depths.push(stack.len());
                                                }
                                            } else {
                                                target_unassignable_depths
                                                    .retain(|&d| d != stack.len());
                                            }
                                        }
                                        self.pos += 1;
                                        st = St::Have;
                                        last_str = false;
                                        prefix_end = usize::MAX;
                                        comp_depths.retain(|&d| d <= stack.len());
                                        for_depths.retain(|&d| d <= stack.len());
                                        starred_depths.retain(|&(d, _)| d <= stack.len());
                                        comma_depths.retain(|&d| d <= stack.len());
                                        dict_key_next.retain(|&d| d <= stack.len());
                                        slice_colons.retain(|(d, _)| *d <= stack.len());
                                        yield_depths.retain(|&d| d <= stack.len());
                                        yield_from_depths.retain(|&d| d <= stack.len());
                                        kwarg_seen_depths.retain(|&d| d <= stack.len());
                                        kw_unpack_depths.retain(|&d| d <= stack.len());
                                        kwarg_elem_depths.retain(|&d| d <= stack.len());
                                        star_elem_depths.retain(|&d| d <= stack.len());
                                    }
                                    _ => return Err(()),
                                }
                            }
                            b'\'' | b'"' => {
                                self.skip_quoted(false, false, false)?;
                                st = St::Have;
                                await_operand = false;
                                last_str = true;
                                prefix_end = usize::MAX;
                            }
                            b'-' | b'+' | b'~' => {
                                self.pos += 1;
                                elem_start = false; // `-*a` is a SyntaxError
                                need_ctx = NeedCtx::Restricted;
                            }
                            // `*a`/`**a` unpacking — legal only at an
                            // element start (`f(*a)`, `[*a]`, `{*a}`,
                            // `(*a,)`), never after an operator (`1 + *a`)
                            // or inside a `:`/`=` continuation.
                            b'*' => {
                                if !elem_start {
                                    return Err(());
                                }
                                let is_kw_unpack = self.s.get(self.pos + 1) == Some(&b'*');
                                let depth = stack.len();
                                match stack.last().copied() {
                                    Some(CALLPAREN) => {
                                        if is_kw_unpack {
                                            kwarg_seen_depths.push(depth);
                                            kw_unpack_depths.push(depth);
                                        } else if kw_unpack_depths.contains(&depth) {
                                            return Err(());
                                        }
                                    }
                                    Some(b'{') if is_kw_unpack => {
                                        *stack.last_mut().ok_or(())? = DICT;
                                    }
                                    Some(DICT) if is_kw_unpack => {
                                        dict_key_next.retain(|&d| d != depth);
                                    }
                                    Some(DICT) => return Err(()),
                                    Some(SET) if is_kw_unpack => return Err(()),
                                    _ if is_kw_unpack => return Err(()),
                                    _ => {}
                                }
                                if let Some(&top) = stack.last() {
                                    starred_depths.push((stack.len(), top));
                                    star_elem_depths.push(stack.len());
                                }
                                self.pos += 1;
                                if is_kw_unpack {
                                    self.pos += 1; // `**a`
                                }
                                elem_start = false;
                                need_ctx = NeedCtx::Restricted;
                            }
                            b'.' => {
                                // `...` ellipsis or `.5` float — nothing else.
                                if self.s.get(self.pos + 1) == Some(&b'.')
                                    && self.s.get(self.pos + 2) == Some(&b'.')
                                {
                                    self.pos += 3;
                                    st = St::Have;
                                } else if self
                                    .s
                                    .get(self.pos + 1)
                                    .is_some_and(|c| c.is_ascii_digit())
                                {
                                    let num_start = self.pos;
                                    self.pos += 2;
                                    while self
                                        .s
                                        .get(self.pos)
                                        .is_some_and(|c| c.is_ascii_digit() || *c == b'_')
                                    {
                                        self.pos += 1;
                                    }
                                    if matches!(
                                        self.s.get(self.pos),
                                        Some(c) if matches!(c, b'e' | b'E')
                                    ) {
                                        self.pos += 1;
                                        if matches!(
                                            self.s.get(self.pos),
                                            Some(c) if matches!(c, b'+' | b'-')
                                        ) {
                                            self.pos += 1;
                                        }
                                        let exp_start = self.pos;
                                        while self
                                            .s
                                            .get(self.pos)
                                            .is_some_and(|c| c.is_ascii_digit() || *c == b'_')
                                        {
                                            self.pos += 1;
                                        }
                                        if self.pos == exp_start {
                                            return Err(());
                                        }
                                    }
                                    if matches!(
                                        self.s.get(self.pos),
                                        Some(c) if matches!(c, b'j' | b'J')
                                    ) {
                                        self.pos += 1;
                                    }
                                    if !Self::valid_numeric_underscores(
                                        &self.s[num_start..self.pos],
                                        |c| c.is_ascii_digit(),
                                        false,
                                    ) {
                                        return Err(());
                                    }
                                    st = St::Have;
                                } else {
                                    return Err(());
                                }
                                await_operand = false;
                                last_str = false;
                                prefix_end = usize::MAX;
                            }
                            // `x[:2]` / `x[::]` — `:` where an operand is
                            // expected is slice syntax, only in a subscript.
                            b':' if stack.last() == Some(&SUBSCRIPT) => {
                                let depth = stack.len();
                                match slice_colons.iter_mut().find(|(d, _)| *d == depth) {
                                    Some((_, count)) if *count >= 2 => return Err(()),
                                    Some((_, count)) => *count += 1,
                                    None => slice_colons.push((depth, 1)),
                                }
                                self.pos += 1;
                                elem_start = false;
                                need_ctx = NeedCtx::Expr;
                            }
                            b',' if stack.last() == Some(&SUBSCRIPT)
                                && slice_colons.iter().any(|(d, _)| *d == stack.len()) =>
                            {
                                slice_colons.retain(|(d, _)| *d != stack.len());
                                self.pos += 1;
                                elem_start = true;
                                need_ctx = NeedCtx::Expr;
                            }
                            _ if ident_char_at(self.s, self.pos, true) => {
                                let id_start = self.pos;
                                self.pos = ident_end(self.s, self.pos);
                                let id = &self.s[id_start..self.pos];
                                let mut next = self.pos;
                                while matches!(self.s.get(next), Some(b' ' | b'\t' | b'\n' | b'\r'))
                                {
                                    next += 1;
                                }
                                let nested_kwarg = elem_start
                                    && stack.last() == Some(&CALLPAREN)
                                    && self.s.get(next) == Some(&b'=')
                                    && self.s.get(next + 1) != Some(&b'=');
                                if !nested_kwarg && id == b"await" {
                                    if await_operand {
                                        return Err(());
                                    }
                                    await_operand = true;
                                    elem_start = false;
                                    need_ctx = NeedCtx::Restricted;
                                    continue;
                                }
                                if !nested_kwarg && id == b"yield" {
                                    if await_operand {
                                        return Err(());
                                    }
                                    let mut before = id_start;
                                    while before > 0
                                        && matches!(
                                            self.s[before - 1],
                                            b' ' | b'\t' | b'\n' | b'\r'
                                        )
                                    {
                                        before -= 1;
                                    }
                                    if stack.last() != Some(&b'(')
                                        || before == 0
                                        || self.s[before - 1] != b'('
                                    {
                                        return Err(());
                                    }
                                    let mut next = self.pos;
                                    while matches!(
                                        self.s.get(next),
                                        Some(b' ' | b'\t' | b'\n' | b'\r')
                                    ) {
                                        next += 1;
                                    }
                                    let next_end = ident_end(self.s, next);
                                    let yield_from =
                                        next_end > next && &self.s[next..next_end] == b"from";
                                    yield_depths.push(stack.len());
                                    if yield_from {
                                        self.pos = next_end;
                                        yield_from_depths.push(stack.len());
                                    }
                                    if !yield_from && self.s.get(next) == Some(&b')') {
                                        st = St::Have;
                                    } else {
                                        st = St::Need;
                                        elem_start = !yield_from;
                                        need_ctx = NeedCtx::Expr;
                                    }
                                    last_str = false;
                                    prefix_end = usize::MAX;
                                    continue;
                                }
                                if !nested_kwarg && reject_keyword(id) {
                                    return Err(());
                                }
                                match id {
                                    b"not" if !nested_kwarg => {
                                        if need_ctx == NeedCtx::Restricted {
                                            return Err(());
                                        }
                                        elem_start = false;
                                        need_ctx = NeedCtx::Inversion;
                                    } // unary — still need an operand
                                    b"lambda" if !nested_kwarg => {
                                        if need_ctx != NeedCtx::Expr
                                            || comp_depths.contains(&stack.len())
                                            || pending_ifs.contains(&stack.len())
                                        {
                                            return Err(());
                                        }
                                        st = St::Lambda;
                                        lambda_depth = stack.len();
                                        param_start = true;
                                        param_done = false;
                                        lambda_marker = LambdaMarker::None;
                                        lambda_default_seen = false;
                                        lambda_keyword_only = false;
                                        param_has_default = false;
                                        lambda_positional_seen = false;
                                        lambda_slash_seen = false;
                                        lambda_bare_star_pending = false;
                                    }
                                    _ => {
                                        st = St::Have;
                                        await_operand = false;
                                        last_str = false;
                                        let is_string_prefix = string_prefix(id);
                                        prefix_end = if is_string_prefix {
                                            self.pos
                                        } else {
                                            usize::MAX
                                        };
                                        prefix_f = is_string_prefix
                                            && id.iter().any(|c| c.eq_ignore_ascii_case(&b'f'));
                                        prefix_b = is_string_prefix
                                            && id.iter().any(|c| c.eq_ignore_ascii_case(&b'b'));
                                        prefix_r = is_string_prefix
                                            && id.iter().any(|c| c.eq_ignore_ascii_case(&b'r'));
                                    }
                                }
                            }
                            // Number operand: digits with optional `0x`/`0o`/`0b`
                            // radix, fraction, exponent, `_` separators, `j`. A
                            // letter glued straight on (`5x`) is a SyntaxError,
                            // and `_` placement follows PEP 515 (between digits,
                            // or right after the base prefix).
                            _ if b.is_ascii_digit() => {
                                if b == b'0'
                                    && matches!(
                                        self.s.get(self.pos + 1),
                                        Some(c) if matches!(c, b'x' | b'X' | b'o' | b'O' | b'b' | b'B')
                                    )
                                {
                                    let radix = match self.s[self.pos + 1].to_ascii_lowercase() {
                                        b'x' => 16,
                                        b'o' => 8,
                                        _ => 2,
                                    };
                                    self.pos += 2;
                                    let dstart = self.pos;
                                    while self
                                        .s
                                        .get(self.pos)
                                        .is_some_and(|c| (*c as char).is_digit(radix) || *c == b'_')
                                    {
                                        self.pos += 1;
                                    }
                                    if self.pos == dstart
                                        || !Self::valid_numeric_underscores(
                                            &self.s[dstart..self.pos],
                                            |c| (c as char).is_digit(radix),
                                            true,
                                        )
                                    {
                                        return Err(());
                                    }
                                } else {
                                    let num_start = self.pos;
                                    while self
                                        .s
                                        .get(self.pos)
                                        .is_some_and(|c| c.is_ascii_digit() || *c == b'_')
                                    {
                                        self.pos += 1;
                                    }
                                    if self.s.get(self.pos) == Some(&b'.') {
                                        self.pos += 1;
                                        while self
                                            .s
                                            .get(self.pos)
                                            .is_some_and(|c| c.is_ascii_digit() || *c == b'_')
                                        {
                                            self.pos += 1;
                                        }
                                    }
                                    if matches!(
                                        self.s.get(self.pos),
                                        Some(c) if matches!(c, b'e' | b'E')
                                    ) {
                                        self.pos += 1;
                                        if matches!(
                                            self.s.get(self.pos),
                                            Some(c) if matches!(c, b'+' | b'-')
                                        ) {
                                            self.pos += 1;
                                        }
                                        let exp_start = self.pos;
                                        while self
                                            .s
                                            .get(self.pos)
                                            .is_some_and(|c| c.is_ascii_digit() || *c == b'_')
                                        {
                                            self.pos += 1;
                                        }
                                        // `1e`, `1e+`, `.5e-` — an exponent
                                        // marker with no digits is a SyntaxError.
                                        if self.pos == exp_start {
                                            return Err(());
                                        }
                                    }
                                    if matches!(
                                        self.s.get(self.pos),
                                        Some(c) if matches!(c, b'j' | b'J')
                                    ) {
                                        self.pos += 1;
                                    }
                                    if !Self::valid_numeric_underscores(
                                        &self.s[num_start..self.pos],
                                        |c| c.is_ascii_digit(),
                                        false,
                                    ) {
                                        return Err(());
                                    }
                                }
                                // `5x` / `5é` — a digit-glued name is a
                                // SyntaxError.
                                if ident_char_at(self.s, self.pos, false) {
                                    return Err(());
                                }
                                st = St::Have;
                                await_operand = false;
                                last_str = false;
                                prefix_end = usize::MAX;
                            }
                            _ => return Err(()),
                        }
                    }
                    St::Have => match b {
                        _ if target_direct_receiver_depths.contains(&stack.len())
                            && !matches!(b, b'(' | b'[' | b'.' | b'\'' | b'"') =>
                        {
                            return Err(());
                        }
                        b')' | b']' | b'}' => match stack.pop() {
                            Some(o) if close_match(o, b) => {
                                let d = stack.len() + 1; // the popped bracket's inner depth
                                let target_group = target_group_depths.contains(&d);
                                let target_receiver = target_receiver_depths.contains(&d);
                                if target_group
                                    && o == b'('
                                    && target_star_depths.contains(&d)
                                    && !target_comma_depths.contains(&d)
                                {
                                    return Err(());
                                }
                                // `(a if b)` closes with the ternary's
                                // `else` still owed — a SyntaxError. So
                                // does a `for` whose `in` never arrived
                                // (`[x for x]`).
                                if pending_ifs.contains(&d) || for_depths.last() == Some(&d) {
                                    return Err(());
                                }
                                // `(*a)` — a bare starred group is a
                                // SyntaxError; `(*a,)` is a legal tuple.
                                if o == b'('
                                    && starred_depths
                                        .iter()
                                        .any(|&(sd, brk)| sd == d && brk == b'(')
                                    && !yield_depths.contains(&d)
                                    && !comma_depths.contains(&d)
                                {
                                    return Err(());
                                }
                                // `{k:v, k2}` — a dict element whose key
                                // owes `:` before the close.
                                if o == DICT && dict_key_next.contains(&d) {
                                    return Err(());
                                }
                                // `helper(x=1, 2)` — a bare positional
                                // closing a call that saw a kwarg.
                                if o == CALLPAREN
                                    && kwarg_seen_depths.contains(&d)
                                    && !kwarg_elem_depths.contains(&d)
                                    && !star_elem_depths.contains(&d)
                                {
                                    return Err(());
                                }
                                if target_receiver && island_stack.last() == Some(&stack.len()) {
                                    island_stack.pop();
                                    target_receiver_depths.retain(|&x| x != d);
                                    if !target_unassignable_depths.contains(&stack.len()) {
                                        target_unassignable_depths.push(stack.len());
                                    }
                                } else if matches!(o, SUBSCRIPT | CALLPAREN)
                                    && target_direct_receiver_depths.contains(&stack.len())
                                    && island_stack.last() == Some(&stack.len())
                                {
                                    if o == SUBSCRIPT {
                                        island_stack.pop();
                                        target_direct_receiver_depths.retain(|&x| x != stack.len());
                                    }
                                } else if matches!(o, SUBSCRIPT | CALLPAREN)
                                    && island_stack.last() == Some(&stack.len())
                                {
                                    island_stack.pop();
                                    if o == CALLPAREN {
                                        if !target_unassignable_depths.contains(&stack.len()) {
                                            target_unassignable_depths.push(stack.len());
                                        }
                                    } else {
                                        target_unassignable_depths.retain(|&x| x != stack.len());
                                    }
                                }
                                if target_group {
                                    let unassignable = target_unassignable_depths.contains(&d);
                                    target_group_depths.retain(|&x| x != d);
                                    target_star_depths.retain(|&x| x != d);
                                    target_comma_depths.retain(|&x| x != d);
                                    target_unassignable_depths.retain(|&x| x != d);
                                    if unassignable {
                                        if !target_unassignable_depths.contains(&stack.len()) {
                                            target_unassignable_depths.push(stack.len());
                                        }
                                    } else {
                                        target_unassignable_depths.retain(|&x| x != stack.len());
                                    }
                                }
                                self.pos += 1;
                                comp_depths.retain(|&x| x <= stack.len());
                                for_depths.retain(|&x| x <= stack.len());
                                starred_depths.retain(|&(x, _)| x <= stack.len());
                                comma_depths.retain(|&x| x <= stack.len());
                                dict_key_next.retain(|&x| x <= stack.len());
                                slice_colons.retain(|(x, _)| *x <= stack.len());
                                yield_depths.retain(|&x| x <= stack.len());
                                yield_from_depths.retain(|&x| x <= stack.len());
                                kwarg_seen_depths.retain(|&x| x <= stack.len());
                                kw_unpack_depths.retain(|&x| x <= stack.len());
                                kwarg_elem_depths.retain(|&x| x <= stack.len());
                                star_elem_depths.retain(|&x| x <= stack.len());
                            }
                            _ => return Err(()),
                        },
                        // Inside a `for` target only `,` separators, `.`/`[`
                        // postfix chains, and the terminating `in` are
                        // legal — operators, calls, and literals are
                        // SyntaxErrors (`[x for x + y in z]`).
                        _ if strict_target => match b {
                            b',' => {
                                if target_unassignable_depths.contains(&stack.len()) {
                                    if !target_group_depths.contains(&stack.len()) {
                                        return Err(());
                                    }
                                    enter_target_receiver(
                                        stack.len(),
                                        &mut island_stack,
                                        &mut target_group_depths,
                                        &mut target_star_depths,
                                        &mut target_comma_depths,
                                        &mut target_unassignable_depths,
                                        &mut target_receiver_depths,
                                    );
                                    continue;
                                }
                                target_comma_depths.push(stack.len());
                                self.pos += 1;
                                st = St::Need;
                            }
                            b'(' => {
                                island_stack.push(stack.len());
                                stack.push(CALLPAREN);
                                self.pos += 1;
                                st = St::Need;
                                elem_start = true;
                                need_ctx = NeedCtx::Expr;
                            }
                            b'[' => {
                                // `a[i]` — a subscript target; its value is
                                // a full expression island.
                                island_stack.push(stack.len());
                                stack.push(SUBSCRIPT);
                                self.pos += 1;
                                st = St::Need;
                                need_ctx = NeedCtx::Expr;
                            }
                            b'.' => {
                                // `a.b` attribute target.
                                self.pos += 1;
                                let id_start = self.pos;
                                if ident_char_at(self.s, self.pos, true) {
                                    self.pos = ident_end(self.s, self.pos);
                                    if reserved_ident(&self.s[id_start..self.pos]) {
                                        return Err(());
                                    }
                                    target_unassignable_depths.retain(|&d| d != stack.len());
                                } else {
                                    return Err(());
                                }
                            }
                            _ if target_group_depths.contains(&stack.len()) => {
                                enter_target_receiver(
                                    stack.len(),
                                    &mut island_stack,
                                    &mut target_group_depths,
                                    &mut target_star_depths,
                                    &mut target_comma_depths,
                                    &mut target_unassignable_depths,
                                    &mut target_receiver_depths,
                                );
                                continue;
                            }
                            _ if ident_char_at(self.s, self.pos, true) => {
                                let id_start = self.pos;
                                self.pos = ident_end(self.s, self.pos);
                                // `in` completes the target at the `for`'s
                                // own depth; anything else is a SyntaxError.
                                let depth = stack.len();
                                if &self.s[id_start..self.pos] == b"in"
                                    && for_depths.last() == Some(&depth)
                                {
                                    if target_unassignable_depths.contains(&depth) {
                                        return Err(());
                                    }
                                    for_depths.pop();
                                    target_group_depths.retain(|&d| d < depth);
                                    target_star_depths.retain(|&d| d < depth);
                                    target_comma_depths.retain(|&d| d < depth);
                                    target_unassignable_depths.retain(|&d| d < depth);
                                    st = St::Need;
                                    need_ctx = NeedCtx::Restricted;
                                } else {
                                    return Err(());
                                }
                            }
                            _ => return Err(()),
                        },
                        b'(' | b'[' => {
                            // Postfix call / index — a subscript `[` admits
                            // slice `:` inside; a call `(` admits keyword
                            // `x=1` arguments (`outer(helper(x=1), k=2)`).
                            stack.push(if b == b'[' { SUBSCRIPT } else { CALLPAREN });
                            self.pos += 1;
                            st = St::Need;
                            elem_start = true;
                            need_ctx = NeedCtx::Expr;
                        }
                        b',' if !stack.is_empty() => {
                            if yield_from_depths.contains(&stack.len()) {
                                return Err(());
                            }
                            // A comprehension is single-element — `[x for
                            // x in y, 2]` is a SyntaxError.
                            if comp_depths.contains(&stack.len()) {
                                return Err(());
                            }
                            // `helper(x=1, 2)` — after a kwarg only `*a`
                            // or another kwarg may follow; a bare
                            // positional is a SyntaxError.
                            if stack.last() == Some(&CALLPAREN)
                                && kwarg_seen_depths.contains(&stack.len())
                                && !kwarg_elem_depths.contains(&stack.len())
                                && !star_elem_depths.contains(&stack.len())
                            {
                                return Err(());
                            }
                            if stack.last() == Some(&SUBSCRIPT) {
                                slice_colons.retain(|(d, _)| *d != stack.len());
                            }
                            if stack.last() == Some(&DICT) && dict_key_next.contains(&stack.len()) {
                                return Err(());
                            }
                            match stack.last() {
                                // `{a,` — a bare element locks the
                                // display to a set.
                                Some(&b'{') => *stack.last_mut().ok_or(())? = SET,
                                // `{k:v,` — the next operand is a key
                                // that owes a `:`.
                                Some(&DICT) => dict_key_next.push(stack.len()),
                                _ => {}
                            }
                            // The element flags describe the element just
                            // closed — the next one starts fresh.
                            kwarg_elem_depths.retain(|&d| d != stack.len());
                            star_elem_depths.retain(|&d| d != stack.len());
                            comma_depths.push(stack.len());
                            self.pos += 1;
                            st = St::Need;
                            elem_start = true;
                            need_ctx = NeedCtx::Expr;
                        }
                        // `:=` walrus — the target must be a bare NAME
                        // directly after `(`/`,`/`[`/`{` or the expression
                        // start (`x:=1`, `x[a:=1]`, `{k:(v:=1)}`); inside
                        // an undecided `{` it is a bare element, locking
                        // the display to a set (`{x:=1}`). Literals,
                        // attributes, subscripts, calls, groups, and
                        // operator results can't be targets — `1:=2`,
                        // `a.b:=2`, `(x):=2`, `a+b:=2`, `x=y:=2`, and
                        // `{k: v:=1}` are all SyntaxErrors.
                        b':' if self.s.get(self.pos + 1) == Some(&b'=') => {
                            let mut p = self.pos;
                            while p > 0 && matches!(self.s[p - 1], b' ' | b'\t' | b'\n' | b'\r') {
                                p -= 1;
                            }
                            let id_end = p;
                            while p > 0 {
                                let c = self.s[p - 1];
                                if c.is_ascii_alphanumeric() || c == b'_' {
                                    p -= 1;
                                } else if c >= 0x80 && (c & 0xC0) == 0x80 {
                                    p -= 1; // UTF-8 continuation byte
                                } else if c >= 0x80 && ident_char_at(self.s, p - 1, false) {
                                    p -= 1; // XID_Continue lead byte
                                } else {
                                    break;
                                }
                            }
                            let mut q = p;
                            while q > 0 && matches!(self.s[q - 1], b' ' | b'\t' | b'\n' | b'\r') {
                                q -= 1;
                            }
                            let bare_name = p < id_end
                                && ident_char_at(self.s, p, true)
                                && !reserved_ident(&self.s[p..id_end])
                                && (q == 0 || matches!(self.s[q - 1], b'(' | b',' | b'[' | b'{'));
                            if !bare_name {
                                return Err(());
                            }
                            if stack.last() == Some(&b'{') {
                                *stack.last_mut().ok_or(())? = SET;
                            }
                            self.pos += 2;
                            st = St::Need;
                            elem_start = false;
                            need_ctx = NeedCtx::Expr;
                        }
                        b':' if !stack.is_empty() => match stack.last() {
                            // `x[1:` — slice continue; `{k:` — dict
                            // separator locking the display to a dict (a
                            // value must follow). `:` in a set, `()`,
                            // list display, or call is a SyntaxError
                            // (`[1:2]`, `{1, 2:3}`, `f(a:1)`).
                            Some(&SUBSCRIPT) => {
                                let depth = stack.len();
                                match slice_colons.iter_mut().find(|(d, _)| *d == depth) {
                                    Some((_, count)) if *count >= 2 => return Err(()),
                                    Some((_, count)) => *count += 1,
                                    None => slice_colons.push((depth, 1)),
                                }
                                self.pos += 1;
                                st = St::Need;
                                elem_start = false;
                                need_ctx = NeedCtx::Expr;
                            }
                            Some(&b'{') => {
                                // `{*a:1}` — a starred element can't be
                                // a dict key.
                                if starred_depths.iter().any(|&(d, _)| d == stack.len()) {
                                    return Err(());
                                }
                                *stack.last_mut().ok_or(())? = DICT;
                                self.pos += 1;
                                st = St::Need;
                                elem_start = false;
                                need_ctx = NeedCtx::Expr;
                            }
                            // `{k:v, k2:` — the pending key's separator.
                            Some(&DICT) if dict_key_next.contains(&stack.len()) => {
                                dict_key_next.retain(|&d| d != stack.len());
                                self.pos += 1;
                                st = St::Need;
                                elem_start = false;
                                need_ctx = NeedCtx::Expr;
                            }
                            _ => return Err(()),
                        },
                        b'.' => match self.s.get(self.pos + 1) {
                            // Attribute access `a.b` — the only legal `.`
                            // after an operand: digit-led numbers already
                            // consumed their fraction/exponent, so `.5`,
                            // `x.`, `.5.5`, `1.5.5` here are SyntaxErrors.
                            Some(_) if ident_char_at(self.s, self.pos + 1, true) => {
                                let id_start = self.pos + 1;
                                self.pos = ident_end(self.s, id_start);
                                if reject_keyword(&self.s[id_start..self.pos]) {
                                    return Err(());
                                }
                                if target_direct_receiver_depths.contains(&stack.len())
                                    && island_stack.last() == Some(&stack.len())
                                {
                                    island_stack.pop();
                                    target_direct_receiver_depths.retain(|&d| d != stack.len());
                                }
                            }
                            _ => return Err(()),
                        },
                        b'\'' | b'"' => {
                            // `rb"x"` (glued prefix → one literal) or
                            // implicit concat `"a" "b"` — nothing else.
                            if !last_str && self.pos != prefix_end {
                                return Err(());
                            }
                            let has_prefix = !last_str && self.pos == prefix_end;
                            self.skip_quoted(
                                has_prefix && prefix_f,
                                has_prefix && prefix_b,
                                has_prefix && prefix_r,
                            )?;
                            last_str = true;
                            prefix_end = usize::MAX;
                        }
                        // Two-character operators are consumed greedily so
                        // `==`/`<=`/`!=`/`**`/`//`/`<<`/`>>` stay one token.
                        b'=' if self.s.get(self.pos + 1) == Some(&b'=') => {
                            self.pos += 2;
                            st = St::Need;
                            elem_start = false;
                            need_ctx = NeedCtx::Restricted;
                        }
                        b'!' if self.s.get(self.pos + 1) == Some(&b'=') => {
                            self.pos += 2;
                            st = St::Need;
                            elem_start = false;
                            need_ctx = NeedCtx::Restricted;
                        }
                        // `=` inside a call paren is keyword-argument
                        // syntax — but only after a bare name (`g(x=1)`
                        // nested in a dropped positional). `g(5=1)`,
                        // `g(a.b=1)`, `g(*a=1)`, `g()=1` are SyntaxErrors;
                        // so is `=` anywhere else (`x[a=1]`, `{a=1}`,
                        // `(a=1)`).
                        b'=' if stack.last() == Some(&CALLPAREN) => {
                            // Walk back over whitespace + the identifier —
                            // a kwarg name is a bare ident directly after
                            // `(` or `,`.
                            let mut p = self.pos;
                            while p > 0 && matches!(self.s[p - 1], b' ' | b'\t' | b'\n' | b'\r') {
                                p -= 1;
                            }
                            let id_end = p;
                            while p > 0 {
                                let c = self.s[p - 1];
                                if c.is_ascii_alphanumeric() || c == b'_' {
                                    p -= 1;
                                } else if c >= 0x80 && (c & 0xC0) == 0x80 {
                                    p -= 1; // UTF-8 continuation byte
                                } else if c >= 0x80 && ident_char_at(self.s, p - 1, false) {
                                    p -= 1; // XID_Continue lead byte
                                } else {
                                    break;
                                }
                            }
                            let mut q = p;
                            while q > 0 && matches!(self.s[q - 1], b' ' | b'\t' | b'\n' | b'\r') {
                                q -= 1;
                            }
                            let bare_name = p < id_end
                                && ident_char_at(self.s, p, true)
                                && q > 0
                                && matches!(self.s[q - 1], b'(' | b',');
                            // `f(*a=1)` — a starred element can't take `=`
                            // (element-scoped: `helper(*a, x=1)` is legal).
                            if !bare_name || star_elem_depths.contains(&stack.len()) {
                                return Err(());
                            }
                            // A kwarg consumed — later elements at this
                            // depth must be `*a` or kwargs.
                            kwarg_seen_depths.push(stack.len());
                            kwarg_elem_depths.push(stack.len());
                            self.pos += 1;
                            st = St::Need;
                            elem_start = false;
                            need_ctx = NeedCtx::Expr;
                        }
                        b'<' | b'>' => {
                            self.pos += 1;
                            if matches!(
                                self.s.get(self.pos),
                                Some(c) if *c == b || *c == b'='
                            ) {
                                self.pos += 1;
                            }
                            st = St::Need;
                            elem_start = false;
                            need_ctx = NeedCtx::Restricted;
                        }
                        b'*' | b'/' => {
                            self.pos += 1;
                            if self.s.get(self.pos) == Some(&b) {
                                self.pos += 1; // `**` `//`
                            }
                            st = St::Need;
                            elem_start = false;
                            need_ctx = NeedCtx::Restricted;
                        }
                        b'+' | b'-' | b'%' | b'|' | b'&' | b'^' | b'@' => {
                            self.pos += 1;
                            st = St::Need;
                            elem_start = false;
                            need_ctx = NeedCtx::Restricted;
                        }
                        _ if ident_char_at(self.s, self.pos, false) => {
                            // Only word operators may follow an operand —
                            // juxtaposed operands (`a b`) are a SyntaxError.
                            let id_start = self.pos;
                            self.pos = ident_end(self.s, self.pos);
                            match &self.s[id_start..self.pos] {
                                b"async" => {
                                    let mut next = self.pos;
                                    while matches!(
                                        self.s.get(next),
                                        Some(b' ' | b'\t' | b'\n' | b'\r')
                                    ) {
                                        next += 1;
                                    }
                                    let next_end = ident_end(self.s, next);
                                    if next_end == next || &self.s[next..next_end] != b"for" {
                                        return Err(());
                                    }
                                    self.pos = next;
                                }
                                b"and" | b"or" | b"is" => {
                                    st = St::Need;
                                    elem_start = false;
                                    need_ctx = NeedCtx::Inversion;
                                }
                                b"in" => {
                                    if target_direct_receiver_depths.contains(&stack.len())
                                        && for_depths.last() == Some(&stack.len())
                                    {
                                        return Err(());
                                    }
                                    // Satisfies the innermost `for` at
                                    // this bracket depth; `in` inside
                                    // deeper brackets is a membership op.
                                    if for_depths.last() == Some(&stack.len()) {
                                        for_depths.pop();
                                    }
                                    st = St::Need;
                                    elem_start = false;
                                    need_ctx = NeedCtx::Restricted;
                                }
                                b"if" => {
                                    // Ternary needs `else`; a filter `if`
                                    // inside a comprehension does not.
                                    if !comp_depths.contains(&stack.len()) {
                                        if pending_ifs.contains(&stack.len()) {
                                            return Err(());
                                        }
                                        pending_ifs.push(stack.len());
                                    }
                                    st = St::Need;
                                    elem_start = false;
                                    need_ctx = NeedCtx::Inversion;
                                }
                                b"else" => {
                                    // `else` without an open ternary is a
                                    // SyntaxError.
                                    let Some(index) =
                                        pending_ifs.iter().rposition(|&d| d == stack.len())
                                    else {
                                        return Err(());
                                    };
                                    pending_ifs.remove(index);
                                    st = St::Need;
                                    elem_start = false;
                                    need_ctx = NeedCtx::Expr;
                                }
                                b"for" => {
                                    let d = stack.len();
                                    // `*` unpacking can't be a comp
                                    // element (`[*a for x in y]`,
                                    // `f(*a for a in b)`); a `for` can't
                                    // live in a subscript (`x[a for a in
                                    // y]`); a comp is single-element
                                    // (`[a, x for x in y]`); and a dict
                                    // key can't be a comp (`{k:v, x
                                    // for}`).
                                    if stack.last() == Some(&SUBSCRIPT)
                                        || yield_depths.contains(&d)
                                        || starred_depths.iter().any(|&(sd, _)| sd == d)
                                        || comma_depths.contains(&d)
                                        || dict_key_next.contains(&d)
                                    {
                                        return Err(());
                                    }
                                    // `for` right after `{` → set comp —
                                    // the element was a bare value, so the
                                    // display can never become a dict.
                                    if stack.last() == Some(&b'{') {
                                        *stack.last_mut().ok_or(())? = SET;
                                    }
                                    // Comprehension clauses run at the
                                    // bracket depth they appear in and
                                    // each owes an `in` at that depth; a
                                    // top-level `for` is a genexpr, valid
                                    // only as the call's sole argument.
                                    comp_depths.push(d);
                                    for_depths.push(d);
                                    top_genexpr |= stack.is_empty();
                                    st = St::Need;
                                    elem_start = false;
                                    need_ctx = NeedCtx::Restricted;
                                }
                                b"not" => {
                                    after_not_op = true;
                                    st = St::Need;
                                    elem_start = false;
                                    need_ctx = NeedCtx::Restricted;
                                }
                                _ => return Err(()),
                            }
                        }
                        _ => return Err(()),
                    },
                    St::Lambda => match b {
                        b'=' if lambda_marker == LambdaMarker::VarargsDone
                            && stack.len() == lambda_depth =>
                        {
                            return Err(());
                        }
                        _ if lambda_marker == LambdaMarker::KwargsDone
                            && stack.len() == lambda_depth
                            && b != b':'
                            && !(param_done && b == b',') =>
                        {
                            return Err(());
                        }
                        b':' if stack.len() == lambda_depth
                            && (lambda_bare_star_pending
                                || matches!(
                                    lambda_marker,
                                    LambdaMarker::Star | LambdaMarker::DoubleStar
                                )) =>
                        {
                            return Err(());
                        }
                        _ if param_done
                            && stack.len() == lambda_depth
                            && matches!(b, b',' | b':')
                            && lambda_default_seen
                            && !lambda_keyword_only
                            && !param_has_default =>
                        {
                            return Err(());
                        }
                        // After a param name only `,`, `=`, or the ending
                        // `:` is legal — `lambda a b`, `lambda a.b`, and
                        // `lambda a(` are all SyntaxErrors.
                        _ if param_done && stack.len() == lambda_depth => match b {
                            b',' => {
                                param_start = true;
                                param_done = false;
                                if lambda_marker != LambdaMarker::KwargsDone {
                                    lambda_marker = LambdaMarker::None;
                                }
                                self.pos += 1;
                            }
                            b'=' => {
                                self.pos += 1;
                                self.skip_expression_until(false, true)?;
                                lambda_default_seen = true;
                                param_has_default = true;
                            }
                            b':' => {
                                param_done = false;
                                self.pos += 1;
                                st = St::Need;
                                need_ctx = NeedCtx::Expr;
                            }
                            _ => return Err(()),
                        },
                        // The lambda's own `:` ends its parameter list.
                        b':' if stack.len() == lambda_depth => {
                            self.pos += 1;
                            st = St::Need;
                            need_ctx = NeedCtx::Expr;
                        }
                        b':' => self.pos += 1, // inside default-expr brackets
                        // `(a)`-style params are Python 2 — an opener at a
                        // param start is a SyntaxError; inside a default
                        // expression brackets are fine.
                        b'(' | b'[' | b'{' => {
                            if param_start && stack.len() == lambda_depth {
                                return Err(());
                            }
                            stack.push(b);
                            self.pos += 1;
                        }
                        b')' | b']' | b'}' if stack.len() > lambda_depth => match stack.pop() {
                            Some(o) if close_match(o, b) => {
                                self.pos += 1;
                            }
                            _ => return Err(()),
                        },
                        b'\'' | b'"' => {
                            if param_start && stack.len() == lambda_depth {
                                return Err(());
                            }
                            self.skip_quoted(false, false, false)?;
                        }
                        b',' => {
                            if stack.len() == lambda_depth {
                                // `a,,b` — a `,` is only legal right after
                                // a `*`/`/` marker or a name.
                                if param_start
                                    && matches!(
                                        lambda_marker,
                                        LambdaMarker::None | LambdaMarker::DoubleStar
                                    )
                                {
                                    return Err(());
                                }
                                lambda_bare_star_pending |= lambda_marker == LambdaMarker::Star;
                                param_start = true;
                                lambda_marker = LambdaMarker::None;
                            }
                            self.pos += 1;
                        }
                        // `=x` with no name; unary/dot bytes at a param
                        // start — all SyntaxErrors.
                        b'=' | b'-' | b'+' | b'~' | b'.'
                            if param_start && stack.len() == lambda_depth =>
                        {
                            return Err(());
                        }
                        b'=' | b'-' | b'+' | b'~' | b'.' => self.pos += 1,
                        b'*' | b'/' => {
                            if !param_start
                                || stack.len() != lambda_depth
                                || lambda_marker != LambdaMarker::None
                            {
                                return Err(());
                            }
                            if b == b'/' {
                                if !lambda_positional_seen
                                    || lambda_keyword_only
                                    || lambda_slash_seen
                                {
                                    return Err(());
                                }
                                lambda_marker = LambdaMarker::Slash;
                                lambda_slash_seen = true;
                            } else {
                                lambda_marker = if self.s.get(self.pos + 1) == Some(&b'*') {
                                    self.pos += 1;
                                    LambdaMarker::DoubleStar
                                } else {
                                    if lambda_keyword_only {
                                        return Err(());
                                    }
                                    LambdaMarker::Star
                                };
                                lambda_keyword_only = true;
                            }
                            self.pos += 1;
                        }
                        _ if ident_char_at(self.s, self.pos, false) => {
                            if param_start && stack.len() == lambda_depth {
                                if lambda_marker == LambdaMarker::Slash {
                                    return Err(());
                                }
                                // Param names start with a letter/`_` and
                                // can't be keywords — `lambda 1: x` and
                                // `lambda for: x` are both SyntaxErrors.
                                if !ident_char_at(self.s, self.pos, true) {
                                    return Err(());
                                }
                                let id_start = self.pos;
                                self.pos = ident_end(self.s, self.pos);
                                if reserved_ident(&self.s[id_start..self.pos]) {
                                    return Err(());
                                }
                                lambda_positional_seen |= !lambda_keyword_only;
                                if lambda_marker == LambdaMarker::None {
                                    lambda_bare_star_pending = false;
                                }
                                param_start = false;
                                lambda_marker = match lambda_marker {
                                    LambdaMarker::Star => LambdaMarker::VarargsDone,
                                    LambdaMarker::DoubleStar => LambdaMarker::KwargsDone,
                                    _ => LambdaMarker::None,
                                };
                                param_done = true;
                                param_has_default = false;
                            } else {
                                self.pos += 1;
                            }
                        }
                        _ => return Err(()),
                    },
                },
            }
        }
    }

    /// Parse a string literal: `'…'` `"…"` `'''…'''` `"""…"""` with
    /// optional `r`/`u`/`f`/`R`/`U`/`F` prefix. `f` is accepted only when
    /// the body has no `{`/`}` placeholder (vLLM admits JoinedStr made of
    /// pure constants). Raw control chars inside strings are accepted —
    /// our parser is lenient where `ast.parse` needed the escape rewrite.
    fn parse_string(&mut self) -> Result<String, ()> {
        let mut raw_mode = false;
        let mut f_mode = false;
        // Optional letter prefix — Python admits r/u/f/b alone and the
        // r-pairs rb/br/rf/fr only (`rr`, `uf`, `bu` … are SyntaxErrors).
        // The dispatch guard guarantees a quote within the two-byte
        // lookahead, so the prefix is at most two letters here.
        if matches!(
            self.peek(),
            Some(b'r' | b'R' | b'u' | b'U' | b'f' | b'F' | b'b' | b'B')
        ) {
            let pstart = self.pos;
            while matches!(
                self.peek(),
                Some(b'r' | b'R' | b'u' | b'U' | b'f' | b'F' | b'b' | b'B')
            ) {
                match self.s[self.pos].to_ascii_lowercase() {
                    b'r' => raw_mode = true,
                    b'f' => f_mode = true,
                    b'b' => return Err(()), // bytes are not JSON-representable
                    _ => {}
                }
                self.pos += 1;
            }
            let prefix = &self.s[pstart..self.pos];
            let valid = match prefix.len() {
                1 => true, // single r/u/f (b already returned above)
                2 => matches!(
                    (
                        prefix[0].to_ascii_lowercase(),
                        prefix[1].to_ascii_lowercase()
                    ),
                    (b'r', b'f') | (b'f', b'r') | (b'r', b'b') | (b'b', b'r')
                ),
                _ => false,
            };
            if !valid {
                return Err(());
            }
        }
        let quote = match self.peek() {
            Some(b'\'' | b'"') => self.s[self.pos],
            _ => return Err(()),
        };
        self.pos += 1;
        // Triple-quoted?
        let triple = self.peek() == Some(quote) && self.s.get(self.pos + 1) == Some(&quote);
        if triple {
            self.pos += 2;
        }
        let mut out = String::new();
        loop {
            let Some(&b) = self.s.get(self.pos) else {
                return Err(()); // unterminated
            };
            if b == quote {
                if raw_mode {
                    // Even in a raw string a backslash escapes the quote
                    // for termination (the backslash stays in the value):
                    // `r'foo\'` is unterminated in Python. An odd run of
                    // backslashes right before the quote means it cannot
                    // close the literal. Non-raw mode never reaches this —
                    // its escape arm already consumed `\'`/`\\` as pairs.
                    let mut back = self.pos;
                    while back > 0 && self.s[back - 1] == b'\\' {
                        back -= 1;
                    }
                    if (self.pos - back) % 2 == 1 {
                        out.push(quote as char);
                        self.pos += 1;
                        continue;
                    }
                }
                if triple {
                    if self.s.get(self.pos + 1) == Some(&quote)
                        && self.s.get(self.pos + 2) == Some(&quote)
                    {
                        self.pos += 3;
                        break;
                    }
                    out.push(quote as char);
                    self.pos += 1;
                    continue;
                }
                self.pos += 1;
                break;
            }
            if !triple && (b == b'\n' || b == b'\r') {
                // Single-quoted strings cannot span lines in Python — but
                // raw newlines inside args are the most common model slip
                // (vLLM fixes via escape_ctrl_chars_in_strings). Accept.
                out.push(b as char);
                self.pos += 1;
                continue;
            }
            if !raw_mode && b == b'\\' {
                self.pos += 1;
                let Some(&e) = self.s.get(self.pos) else {
                    return Err(());
                };
                self.pos += 1;
                match e {
                    b'n' => out.push('\n'),
                    b't' => out.push('\t'),
                    b'r' => out.push('\r'),
                    b'b' => out.push('\x08'),
                    b'f' => out.push('\x0C'),
                    b'a' => out.push('\x07'),
                    b'v' => out.push('\x0B'),
                    b'0'..=b'7' => {
                        // Octal escape: up to 3 digits total.
                        let mut val = (e - b'0') as u32;
                        let mut n = 1;
                        while n < 3 {
                            match self.s.get(self.pos) {
                                Some(&d @ b'0'..=b'7') => {
                                    val = val * 8 + (d - b'0') as u32;
                                    self.pos += 1;
                                    n += 1;
                                }
                                _ => break,
                            }
                        }
                        out.push(char::from_u32(val).ok_or(())?);
                    }
                    b'x' => {
                        let h = self.hex_digits(2)?;
                        out.push(char::from_u32(h).ok_or(())?);
                    }
                    b'u' => {
                        let h = self.hex_digits(4)?;
                        out.push(char::from_u32(h).ok_or(())?);
                    }
                    b'U' => {
                        let h = self.hex_digits(8)?;
                        out.push(char::from_u32(h).ok_or(())?);
                    }
                    b'\n' => {} // line continuation
                    b'N' => {
                        // `\N` is a recognized escape introducer: a bare
                        // `\N` is a hard SyntaxError, and every `\N{NAME}`
                        // needs a Unicode-name table we don't carry. Reject
                        // rather than ship the literal escape text as the
                        // argument.
                        return Err(());
                    }
                    _ => {
                        // `\'` `\"` `\\` collapse to the literal char; every
                        // OTHER unrecognized escape keeps the backslash —
                        // Python `'\d'` evaluates to `"\\d"` (backslash
                        // preserved), so `pattern='\d+'` must not lose it.
                        let e = self.s[self.pos - 1];
                        if !matches!(e, b'\'' | b'"' | b'\\') {
                            out.push('\\');
                        }
                        let rest = &self.s[self.pos - 1..];
                        let ch_len = utf8_len(rest[0]);
                        if rest.len() < ch_len {
                            return Err(());
                        }
                        out.push_str(std::str::from_utf8(&rest[..ch_len]).map_err(|_| ())?);
                        self.pos += ch_len - 1;
                    }
                }
                continue;
            }
            if f_mode && (b == b'{' || b == b'}') {
                // `{{`/`}}` are escaped literal braces — a pure-constant
                // f-string like f'{{x}}' evaluates to "{x}" (vLLM accepts;
                // JoinedStr of constants). A single brace is a real
                // placeholder → reject.
                if self.s.get(self.pos + 1) == Some(&b) {
                    out.push(b as char);
                    self.pos += 2;
                    continue;
                }
                return Err(());
            }
            // Copy one UTF-8 char verbatim.
            let rest = &self.s[self.pos..];
            let ch_len = utf8_len(rest[0]);
            if rest.len() < ch_len {
                return Err(());
            }
            out.push_str(std::str::from_utf8(&rest[..ch_len]).map_err(|_| ())?);
            self.pos += ch_len;
        }
        Ok(out)
    }

    fn hex_digits(&mut self, n: usize) -> Result<u32, ()> {
        let mut val = 0u32;
        for _ in 0..n {
            let Some(&d) = self.s.get(self.pos) else {
                return Err(());
            };
            let Some(v) = (d as char).to_digit(16) else {
                return Err(());
            };
            val = val * 16 + v;
            self.pos += 1;
        }
        Ok(val)
    }

    /// `_` separator placement per PEP 515: a separator must sit between
    /// two digits (radix digits for base-prefixed literals), except the
    /// single `_` allowed right after `0x`/`0o`/`0b` (`0x_ff` is legal).
    /// `after_prefix` marks `s[0]` as the position following a base prefix.
    fn valid_numeric_underscores(
        s: &[u8],
        is_digit: impl Fn(u8) -> bool,
        after_prefix: bool,
    ) -> bool {
        for (i, &c) in s.iter().enumerate() {
            if c != b'_' {
                continue;
            }
            let next_ok = s.get(i + 1).is_some_and(|&n| is_digit(n));
            let prev_ok = i > 0 && is_digit(s[i - 1]);
            if !next_ok || !(prev_ok || (after_prefix && i == 0)) {
                return false;
            }
        }
        true
    }

    /// Number: dec/hex/oct/bin int or float, optional leading `+`/`-` is
    /// handled by the caller (unary op). Leading zeros are tolerated
    /// (vLLM `normalize_leading_zero_ints`).
    fn parse_number(&mut self) -> Result<Value, ()> {
        let start = self.pos;
        if self.peek() == Some(b'0')
            && matches!(
                self.s.get(self.pos + 1),
                Some(b'x' | b'X' | b'o' | b'O' | b'b' | b'B')
            )
        {
            let radix = match self.s[self.pos + 1].to_ascii_lowercase() {
                b'x' => 16,
                b'o' => 8,
                _ => 2,
            };
            self.pos += 2;
            let dstart = self.pos;
            while let Some(&d) = self.s.get(self.pos) {
                if (d as char).is_digit(radix) || d == b'_' {
                    self.pos += 1;
                } else {
                    break;
                }
            }
            if !Self::valid_numeric_underscores(
                &self.s[dstart..self.pos],
                |c| (c as char).is_digit(radix),
                true,
            ) {
                return Err(());
            }
            let digits: String = std::str::from_utf8(&self.s[dstart..self.pos])
                .map_err(|_| ())?
                .chars()
                .filter(|&c| c != '_')
                .collect();
            if digits.is_empty() {
                return Err(());
            }
            let v = i128::from_str_radix(&digits, radix).map_err(|_| ())?;
            return i128_to_value(v).ok_or(());
        }
        let mut is_float = false;
        while let Some(&d) = self.s.get(self.pos) {
            match d {
                b'0'..=b'9' | b'_' => self.pos += 1,
                b'.' | b'e' | b'E' | b'+' | b'-' => {
                    // +/- only legal right after e/E.
                    if d == b'+' || d == b'-' {
                        let prev = self.s.get(self.pos.wrapping_sub(1)).copied();
                        if !matches!(prev, Some(b'e' | b'E')) {
                            break;
                        }
                    }
                    is_float = true;
                    self.pos += 1;
                }
                b'j' | b'J' => return Err(()), // complex: not JSON
                _ => break,
            }
        }
        if !Self::valid_numeric_underscores(&self.s[start..self.pos], |c| c.is_ascii_digit(), false)
        {
            return Err(());
        }
        let text: String = std::str::from_utf8(&self.s[start..self.pos])
            .map_err(|_| ())?
            .chars()
            .filter(|&c| c != '_')
            .collect();
        // Dispatch guarantees the first char is a digit or `.`, so `text`
        // can only start with those; a `.`-led scan already set `is_float`.
        if text.is_empty() || !text.bytes().any(|b| b.is_ascii_digit()) {
            return Err(());
        }
        if is_float {
            // Tolerate the Python forms Rust f64 rejects: leading-dot
            // `.5` → `0.5`, trailing-dot `1.` → `1.0`.
            let normalized = if let Some(rest) = text.strip_prefix('.') {
                format!("0.{rest}")
            } else if text.ends_with('.') {
                format!("{text}0")
            } else {
                text
            };
            match normalized.parse::<f64>() {
                // Overflow parses to ±inf and Value::from(inf) serializes
                // as null — no exact JSON form → verbatim, like huge ints.
                Ok(f) if f.is_finite() => Ok(Value::from(f)),
                _ => Err(()),
            }
        } else {
            text.parse::<i128>().ok().and_then(i128_to_value).ok_or(())
        }
    }

    /// Parse one literal value: string, number, name constant, or
    /// container. Unary +/- over a numeric operand is folded here
    /// (vLLM `get_parameter_value` UnaryOp branch).
    fn parse_literal(&mut self) -> Result<Value, ()> {
        self.depth += 1;
        if self.depth > MAX_PY_LITERAL_DEPTH {
            self.depth -= 1;
            return Err(());
        }
        let v = self.parse_literal_inner();
        self.depth -= 1;
        v
    }

    fn parse_literal_inner(&mut self) -> Result<Value, ()> {
        self.skip_ws();
        match self.peek() {
            Some(b'-' | b'+') => {
                let neg = self.s[self.pos] == b'-';
                self.pos += 1;
                let v = self.parse_literal()?;
                match v {
                    Value::Number(n) => {
                        if neg {
                            if let Some(i) = n.as_i64() {
                                // `-i64::MIN` overflows i64 (double negation
                                // like `--9223372036854775808`) — the positive
                                // fits u64 exactly.
                                Ok(match i.checked_neg() {
                                    Some(v) => Value::from(v),
                                    None => Value::from(i.unsigned_abs()),
                                })
                            } else if let Some(u) = n.as_u64() {
                                // Negating a u64: exact i64::MIN edge or a
                                // value that fits in i64 — anything larger
                                // has no exact serde_json form → reject.
                                if u == (i64::MAX as u64) + 1 {
                                    return Ok(Value::from(i64::MIN));
                                }
                                let i = i64::try_from(u).map_err(|_| ())?;
                                Ok(Value::from(-i))
                            } else {
                                n.as_f64().map(|f| Value::from(-f)).ok_or(())
                            }
                        } else {
                            Ok(Value::Number(n))
                        }
                    }
                    _ => Err(()), // unary on non-number → reject
                }
            }
            Some(b'\'' | b'"') => Ok(Value::String(self.parse_string()?)),
            Some(b'r' | b'R' | b'u' | b'U' | b'f' | b'F' | b'b' | b'B')
                if matches!(
                    self.s.get(self.pos + 1..self.pos + 3).unwrap_or(&[]),
                    [b'\'' | b'"', ..] | [b'r' | b'R' | b'u' | b'U' | b'f' | b'F', b'\'' | b'"']
                ) =>
            {
                Ok(Value::String(self.parse_string()?))
            }
            Some(b'0'..=b'9') | Some(b'.') => self.parse_number(),
            Some(b'[') => {
                self.pos += 1;
                let mut items = Vec::new();
                self.skip_ws();
                if self.peek() == Some(b']') {
                    self.pos += 1;
                    return Ok(Value::Array(items));
                }
                loop {
                    items.push(self.parse_literal()?);
                    self.skip_ws();
                    match self.peek() {
                        Some(b',') => {
                            self.pos += 1;
                            self.skip_ws();
                            if self.peek() == Some(b']') {
                                self.pos += 1;
                                return Ok(Value::Array(items));
                            }
                        }
                        Some(b']') => {
                            self.pos += 1;
                            return Ok(Value::Array(items));
                        }
                        _ => return Err(()),
                    }
                }
            }
            Some(b'(') => {
                // Tuple → JSON array. `(x)` is just x; `(x,)` is a tuple.
                self.pos += 1;
                self.skip_ws();
                if self.peek() == Some(b')') {
                    self.pos += 1;
                    return Ok(Value::Array(vec![]));
                }
                let first = self.parse_literal()?;
                self.skip_ws();
                match self.peek() {
                    Some(b',') => {
                        let mut items = vec![first];
                        while self.peek() == Some(b',') {
                            self.pos += 1;
                            self.skip_ws();
                            if self.peek() == Some(b')') {
                                break;
                            }
                            items.push(self.parse_literal()?);
                            self.skip_ws();
                        }
                        self.expect(b')')?;
                        Ok(Value::Array(items))
                    }
                    Some(b')') => {
                        self.pos += 1;
                        Ok(first) // parenthesized value
                    }
                    _ => Err(()),
                }
            }
            Some(b'{') => {
                self.pos += 1;
                self.skip_ws();
                if self.peek() == Some(b'}') {
                    self.pos += 1;
                    return Ok(Value::Object(serde_json::Map::new()));
                }
                let first = self.parse_literal()?;
                self.skip_ws();
                if self.peek() == Some(b':') {
                    // dict — keys must be literal (already parsed); Python
                    // allows any hashable; JSON needs strings.
                    self.pos += 1;
                    let mut map = serde_json::Map::new();
                    let key = literal_key_to_string(&first)?;
                    map.insert(key, self.parse_literal()?);
                    loop {
                        self.skip_ws();
                        match self.peek() {
                            Some(b',') => {
                                self.pos += 1;
                                self.skip_ws();
                                if self.peek() == Some(b'}') {
                                    self.pos += 1;
                                    return Ok(Value::Object(map));
                                }
                                let k = self.parse_literal()?;
                                self.skip_ws();
                                self.expect(b':')?;
                                map.insert(literal_key_to_string(&k)?, self.parse_literal()?);
                            }
                            Some(b'}') => {
                                self.pos += 1;
                                return Ok(Value::Object(map));
                            }
                            _ => return Err(()),
                        }
                    }
                } else {
                    // set literal `{v, v}` → array (vLLM maps sets to lists)
                    let mut items = vec![first];
                    loop {
                        self.skip_ws();
                        match self.peek() {
                            Some(b',') => {
                                self.pos += 1;
                                self.skip_ws();
                                if self.peek() == Some(b'}') {
                                    self.pos += 1;
                                    return Ok(Value::Array(items));
                                }
                                items.push(self.parse_literal()?);
                            }
                            Some(b'}') => {
                                self.pos += 1;
                                return Ok(Value::Array(items));
                            }
                            _ => return Err(()),
                        }
                    }
                }
            }
            _ => {
                let name = self.ident()?;
                match name {
                    "True" | "true" => Ok(Value::Bool(true)),
                    "False" | "false" => Ok(Value::Bool(false)),
                    "None" | "null" => Ok(Value::Null),
                    _ => Err(()), // bare name → not a literal
                }
            }
        }
    }

    /// One call element: `name(kw=literal, ...)`. Positional arguments are
    /// skipped then dropped (vLLM iterates `call.keywords` only); `**kw`
    /// rejects the call outright (vLLM fails on `arguments[None]`). A
    /// positional AFTER a keyword (`f(x=1, 5)`) mirrors `ast.parse`'s
    /// `SyntaxError`; duplicate keywords keep their final value like vLLM.
    fn parse_call(&mut self) -> Result<(String, Value), ()> {
        let name = self.dotted_name()?;
        self.expect(b'(')?;
        let mut args = serde_json::Map::new();
        let mut seen_kwarg = false;
        self.skip_ws();
        if self.peek() == Some(b')') {
            self.pos += 1;
            return Ok((name, Value::Object(args)));
        }
        loop {
            self.skip_ws();
            if self.peek() == Some(b'*') {
                // `*x` spread: ignored like other positionals (legal even
                // after keywords); `**x` → reject (vLLM's
                // `arguments[None]` failure).
                self.pos += 1;
                if self.peek() == Some(b'*') {
                    return Err(());
                }
                self.skip_expression(true)?;
            } else {
                // `kw=literal` or a positional expression. Try the
                // `ident =` lookahead first (excluding `==`); on no match
                // rewind and skip the positional, which vLLM drops.
                // Reserved-word kwargs (`from=1`) parse natively here —
                // vLLM needs its rename/restore dance, we don't.
                let save = self.pos;
                let kw: Option<String> = match self.ident() {
                    Ok(id) => {
                        let id: String = id.nfkc().collect();
                        self.skip_ws();
                        if self.peek() == Some(b'=') && self.s.get(self.pos + 1) != Some(&b'=') {
                            self.pos += 1;
                            Some(id)
                        } else {
                            self.pos = save;
                            None
                        }
                    }
                    // `ident` consumes leading digits before rejecting a
                    // digit start — REWIND so a numeric positional
                    // (`f(5)`, `f(0x10)`) still scans as an expression.
                    Err(()) => {
                        self.pos = save;
                        None
                    }
                };
                match kw {
                    Some(k) => {
                        seen_kwarg = true;
                        let v = self.parse_literal()?;
                        let _ = args.insert(k, v);
                    }
                    None => {
                        // `f(x=1, 5)` → `SyntaxError: positional argument
                        // follows keyword argument`.
                        if seen_kwarg {
                            return Err(());
                        }
                        self.skip_expression(false)?;
                    }
                }
            }
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                    self.skip_ws();
                    if self.peek() == Some(b')') {
                        self.pos += 1;
                        return Ok((name, Value::Object(args)));
                    }
                }
                Some(b')') => {
                    self.pos += 1;
                    return Ok((name, Value::Object(args)));
                }
                _ => return Err(()),
            }
        }
    }
}

fn utf8_len(first: u8) -> usize {
    if first < 0x80 {
        1
    } else if first < 0xE0 {
        2
    } else if first < 0xF0 {
        3
    } else {
        4
    }
}

/// Is the char at `s[p]` identifier syntax? ASCII `[A-Za-z_]` (or
/// `[A-Za-z0-9_]` when `start == false`) plus the Unicode XID_Start /
/// XID_Continue classes CPython uses — `café`/`变量` are names, `😀`
/// is a SyntaxError (`ast.parse` rejects it → vLLM keeps the block
/// verbatim; a bare `>= 0x80` byte gate would promote it).
fn ident_char_at(s: &[u8], p: usize, start: bool) -> bool {
    let Some(&c) = s.get(p) else {
        return false;
    };
    if c < 0x80 {
        return if start {
            c.is_ascii_alphabetic() || c == b'_'
        } else {
            c.is_ascii_alphanumeric() || c == b'_'
        };
    }
    let Some(t) = s
        .get(p..p + utf8_len(c))
        .and_then(|b| std::str::from_utf8(b).ok())
    else {
        return false;
    };
    let Some(ch) = t.chars().next() else {
        return false;
    };
    if start {
        unicode_ident::is_xid_start(ch)
    } else {
        unicode_ident::is_xid_continue(ch)
    }
}

/// End of the identifier starting at `p` — ASCII ident bytes plus
/// XID_Continue chars; everything else stops the scan.
fn ident_end(s: &[u8], mut p: usize) -> usize {
    while let Some(&c) = s.get(p) {
        if c.is_ascii_alphanumeric() || c == b'_' {
            p += 1;
        } else if c >= 0x80 && ident_char_at(s, p, false) {
            p += utf8_len(c);
        } else {
            break;
        }
    }
    p
}

fn i128_to_value(v: i128) -> Option<Value> {
    if let Ok(i) = i64::try_from(v) {
        Some(Value::from(i))
    } else if let Ok(u) = u64::try_from(v) {
        Some(Value::from(u))
    } else {
        // serde_json holds no exact representation past u64::MAX — f64
        // rounding would silently corrupt the argument (IDs, counters,
        // monetary units), so the block stays verbatim like every other
        // non-JSON-representable literal (`b'..'`, `1j`).
        None
    }
}

/// Python dict keys are any hashable; JSON keys are strings — stringify
/// non-string literal keys the way `json.dumps` does (`{True: 1}` →
/// `{"true": 1}`, `{None: 1}` → `{"null": 1}`).
fn literal_key_to_string(v: &Value) -> Result<String, ()> {
    match v {
        Value::String(s) => Ok(s.clone()),
        Value::Number(n) => Ok(n.to_string()),
        Value::Bool(b) => Ok(if *b { "true".into() } else { "false".into() }),
        Value::Null => Ok("null".into()),
        _ => Err(()), // list/dict keys are unhashable in Python too
    }
}

/// Parse the bracket-list body (`f(x=1), g(y='a')` — without the outer
/// `[]`) into `(name, args)` pairs. `None` on any parse failure, a trailing
/// element, or leftover input. A trailing comma (`[f(),]`) is legal Python
/// and accepted.
fn parse_lfm2_call_list(body: &str) -> Option<Vec<(String, Value)>> {
    let mut p = PyLiteralParser::new(body);
    let mut calls = Vec::new();
    p.skip_ws();
    p.peek()?; // `[]` — vLLM requires non-empty elts
    loop {
        let (name, args) = p.parse_call().ok()?;
        calls.push((name, args));
        p.skip_ws();
        match p.peek() {
            Some(b',') => {
                p.pos += 1;
                p.skip_ws();
                if p.peek().is_none() {
                    break; // trailing comma at end of list
                }
            }
            None => break,
            _ => return None,
        }
    }
    (p.eof() && !calls.is_empty()).then_some(calls)
}

/// vLLM `escape_nested_quotes_in_strings` (`tool_parsers/utils.py`): close a
/// broken string literal at the only closing quote that works. Models
/// emitting shell commands nest unescaped same-style quotes inside a string
/// argument (`command='sed -n '1,9p' f.py'`). A string is broken when its
/// first unescaped quote cannot syntactically close it (the next non-space
/// char is none of `,)]}:`); then every plausible closing quote is tried —
/// interior quotes escaped — and the candidate kept only when it is the
/// UNIQUE candidate that re-parses as a call list. Returns `None` on zero
/// or ambiguous candidates.
///
/// vLLM validates each candidate with `ast.parse` (any expression) and only
/// afterwards requires all-`Call` elements — so a second candidate that
/// parses as a non-call expression still counts as ambiguity there. Our
/// validator is `parse_lfm2_call_list` (all-calls up front), which makes
/// the recovery strictly more permissive on that corner: accepted here,
/// declared ambiguous upstream.
fn escape_lfm2_nested_quotes(text: &str) -> Option<String> {
    fn unescaped_quote_positions(text: &str, start: usize, quote: u8) -> Vec<usize> {
        let s = text.as_bytes();
        let mut out = Vec::new();
        let mut j = start;
        while j < s.len() {
            if s[j] == b'\\' {
                j += 2;
                continue;
            }
            if s[j] == quote {
                out.push(j);
            }
            j += 1;
        }
        out
    }
    // vLLM `_QUOTE_FOLLOWERS` — a quote closes the string only when the
    // next non-space byte is a container/argument delimiter.
    fn is_closer(text: &str, pos: usize) -> bool {
        let s = text.as_bytes();
        let mut k = pos + 1;
        while k < s.len() && s[k].is_ascii_whitespace() {
            k += 1;
        }
        k < s.len() && matches!(s[k], b',' | b')' | b']' | b'}' | b':')
    }
    // vLLM `_is_escaped`: escaped iff preceded by an ODD backslash run.
    fn is_escaped(s: &[u8], index: usize) -> bool {
        let mut n = 0usize;
        let mut j = index;
        while j > 0 && s[j - 1] == b'\\' {
            n += 1;
            j -= 1;
        }
        n % 2 == 1
    }

    let s = text.as_bytes();
    let mut index = 0usize;
    while index < s.len() {
        let quote = s[index];
        if quote != b'\'' && quote != b'"' {
            index += 1;
            continue;
        }
        let quotes = unescaped_quote_positions(text, index + 1, quote);
        let Some(&first) = quotes.first() else {
            return None; // unterminated — nothing to requote
        };
        if is_closer(text, first) {
            index = first + 1; // normal string; skip past it
            continue;
        }
        // Broken string: try each plausible close, escaping interior
        // same-quotes; keep the candidate only when exactly one parses.
        let mut winner: Option<String> = None;
        for &close in quotes.iter().filter(|&&j| is_closer(text, j)) {
            let mut candidate = String::with_capacity(text.len() + 8);
            candidate.push_str(&text[..index + 1]); // through opening quote
            for (off, ch) in text[index + 1..close].char_indices() {
                if ch == quote as char && !is_escaped(s, index + 1 + off) {
                    candidate.push('\\');
                }
                candidate.push(ch);
            }
            candidate.push(quote as char);
            candidate.push_str(&text[close + 1..]);
            // The candidate is the full bracketed list; validate by parsing
            // the bracket-free body.
            let valid = {
                let c = candidate.trim();
                c.starts_with('[')
                    && c.ends_with(']')
                    && parse_lfm2_call_list(&c[1..c.len() - 1]).is_some()
            };
            if valid {
                if winner.is_some() {
                    return None; // ambiguous — don't guess
                }
                winner = Some(candidate);
            }
        }
        return winner;
    }
    None
}

/// Parse LFM2 sentinel blocks: every `<|tool_call_start|>` …
/// `<|tool_call_end|>` pair (or end of text when the stream cut mid-call).
/// Returns (cleaned_text, calls). Parse failure returns the ORIGINAL text
/// verbatim with no calls (vLLM `content=model_output` semantics).
pub(super) fn parse_lfm2_tool_calls(text: &str) -> (String, Vec<ToolCallResult>) {
    // A sentinel inside a `<tool_call>` block is literal argument text
    // (e.g. `{"a": "<|tool_call_start|>…"}`) — not LFM2 markup. Exclude
    // every `<tool_call>` open's span (through its close, or EOF when
    // unclosed) from the sentinel search so a sentinel can never be
    // promoted to a real call out of another family's argument string.
    let protected: Vec<(usize, usize)> = all_positions(text, "<tool_call>")
        .into_iter()
        .map(|open| {
            let end = text[open..]
                .find("</tool_call>")
                .map(|c| open + c + "</tool_call>".len())
                .unwrap_or(text.len());
            (open, end)
        })
        .collect();
    let unprotected = |p: usize| !protected.iter().any(|&(s, e)| p >= s && p < e);
    let next_start = |from: usize| -> Option<usize> {
        let mut search_from = from;
        loop {
            let idx = search_from + text[search_from..].find(LFM2_TOOL_CALL_START)?;
            if unprotected(idx) {
                return Some(idx);
            }
            search_from = idx + LFM2_TOOL_CALL_START.len();
        }
    };

    // Success content strips each sentinel block VERBATIM — the streamed
    // path emits the surrounding fragments byte-for-byte, so a rewritten
    // join (e.g. vLLM's "\n") would make the done-path suffix recovery
    // see the finalized text as entirely unsent and emit it twice.
    let mut calls: Vec<ToolCallResult> = Vec::new();
    let mut cleaned = String::with_capacity(text.len());
    let mut cursor = 0;
    while let Some(start_idx) = next_start(cursor) {
        cleaned.push_str(&text[cursor..start_idx]);
        let inner_start = start_idx + LFM2_TOOL_CALL_START.len();
        let (inner, block_end) = match text[inner_start..].find(LFM2_TOOL_CALL_END) {
            Some(e) => {
                let end = inner_start + e + LFM2_TOOL_CALL_END.len();
                (&text[inner_start..inner_start + e], end)
            }
            None => (&text[inner_start..], text.len()),
        };
        let tool_text = inner.trim();
        // vLLM `TOOL_CALL_REGEX`: bracketed list only.
        let parsed = if tool_text.starts_with('[') && tool_text.ends_with(']') {
            let inner_body = &tool_text[1..tool_text.len() - 1];
            // Direct parse first; on failure, the nested-quote recovery
            // rewrites `command='sed -n '1,9p' f.py'` shapes and re-parses
            // (vLLM runs `escape_nested_quotes_in_strings` over the
            // ast.parse failure). The recovery sees the BRACKETED text —
            // the trailing `)]` is what makes the outer quote a plausible
            // close.
            parse_lfm2_call_list(inner_body).or_else(|| {
                escape_lfm2_nested_quotes(tool_text).and_then(|fixed| {
                    let f = fixed.trim();
                    if f.starts_with('[') && f.ends_with(']') {
                        parse_lfm2_call_list(&f[1..f.len() - 1])
                    } else {
                        None
                    }
                })
            })
        } else {
            None
        };
        let Some(parsed_calls) = parsed else {
            if calls.is_empty() {
                // The FIRST block failing is vLLM's `content=model_output`:
                // ast.parse on the only block it reads failed, so the whole
                // output stays verbatim with no calls.
                return (text.to_string(), Vec::new());
            }
            // A LATER block failing is out of vLLM's reach — it never
            // parses past the first end sentinel, so the malformed span
            // is echo region, not a call. Keep the calls already promoted
            // and echo-strip from this sentinel on (an orphan END caps
            // the echoed body).
            cleaned.push_str(strip_lfm2_echo(&text[start_idx..]));
            return (cleaned.trim().to_string(), calls);
        };
        let raw = &text[start_idx..block_end];
        calls.extend(
            parsed_calls
                .into_iter()
                .map(|(name, args)| ToolCallResult::ok(name, args, raw.to_string())),
        );
        // The echo region runs from this block's end to the NEXT
        // unprotected start sentinel (or EOF): drop everything through
        // the last orphan `<|tool_call_end|>` inside it, keep the prose.
        let next = next_start(block_end).unwrap_or(text.len());
        cleaned.push_str(strip_lfm2_echo(&text[block_end..next]));
        cursor = next;
    }
    if calls.is_empty() {
        return (text.to_string(), Vec::new());
    }
    cleaned.push_str(&text[cursor..]);
    (cleaned.trim().to_string(), calls)
}
