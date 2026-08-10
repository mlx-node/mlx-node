//! Streaming safety for Muse-Glimmer's ATEM surface.
//!
//! Three hazards, all structural rather than incidental:
//!   * `<atem:*>` XML is ordinary BPE text and splits across token boundaries,
//!     so a naive emitter leaks `<atem` fragments as content.
//!   * the model's first characters are ` to=<recipient><|message|>`, which is
//!     routing metadata, not content.
//!   * `<|eom|>` (200007) is deliberately not a stop, so the model can emit
//!     `<|start|>user<|message|>` and self-play a whole conversation.
//!
//! # What decides that a chunk is content
//!
//! The `to=` value, and nothing else. `to=self` is chain-of-thought, `to=user`
//! is the answer, and any other recipient is a tool call. So [`StreamGuard`]
//! runs a two-state machine — [`State::AwaitingHeader`] until `<|message|>`
//! closes the routing header, then [`State::InMessage`] with the channel that
//! header selected — and only [`Channel::Content`] text ever reaches an
//! [`GuardOutcome::Emit`]. Reasoning and tool-call bodies are consumed and
//! dropped, never buffered for later release.
//!
//! The recipient is read as the *suffix* of the header, not the first `to=` in
//! it, because that is what the checkpoint's own `response_template` does: its
//! open pattern is `to=user<\|message\|>`, so `to=` has to sit immediately
//! before the marker. A header of ` to=user to=self` therefore routes to
//! reasoning in both this guard and [`super::output_parser`]; a guard that
//! disagreed with the parser about routing would stream one channel while the
//! parser filed the text under another.
//!
//! # Why the hold-back is measured in characters
//!
//! Every literal in [`MARKER_LITERALS`] is plain text the tokenizer is free to
//! split anywhere, so the tail of the stream is always ambiguous: `<atem:fun`
//! is either the start of a tool call or nine characters of an answer, and only
//! the next token says which. The guard therefore withholds the last
//! [`HOLD_BACK_CHARS`] **characters** of unresolved content — characters, so a
//! multi-byte scalar is never cut in half — and releases everything in front of
//! them. Since `HOLD_BACK_CHARS` is at least as long as the longest literal, a
//! marker that is still incomplete at the end of the buffer always lies wholly
//! inside the withheld tail and cannot leak. A marker that *is* complete has
//! already been found by the scan, and text in front of a found marker is
//! unambiguous, so it is released in full — which is why real content does not
//! pay a permanent 24-character delay at every tool call.
//!
//! [`StreamGuard::flush`] drains the tail at end of turn. It strips a trailing
//! partial marker first: the tail is by definition the ambiguous part, and a
//! turn that stops mid-marker must not publish the fragment.

/// Held-back tail, in characters. Must be at least the longest entry of
/// [`MARKER_LITERALS`] (currently `</atem:function_calls>` and
/// `<atem:parameter name="`, both 22); the spec asks for `>= 24`.
pub const HOLD_BACK_CHARS: usize = 24;

/// Every literal that must never be split across an emit boundary, and the set
/// the hold-back length is derived from.
///
/// The scanner does not match these one by one — it matches [`ATEM_OPEN`] and
/// [`ATEM_CLOSE`], which are prefixes of every `<atem:*>` entry here, so
/// detection fires by a literal's 6th or 7th character. The full literals are
/// still the inventory that sets the bound: hold back less than the longest one
/// and a suffix of it can be released before the scan can see it whole.
/// `every_marker_literal_starts_with_a_detected_prefix` pins that relationship.
pub const MARKER_LITERALS: &[&str] = &[
    "<|start|>",
    "<|message|>",
    "<|eom|>",
    "<|eot|>",
    "<|end_of_text|>",
    "<atem:function_calls>",
    "</atem:function_calls>",
    "<atem:invoke name=\"",
    "</atem:invoke>",
    "<atem:parameter name=\"",
    "</atem:parameter>",
];

/// Closes the routing header; anywhere else it is a marker the model must not
/// be able to put into a content delta.
const MSG: &str = "<|message|>";
/// Opens a new message. The role that follows it is the self-play guard.
const START: &str = "<|start|>";
/// End of message — explicitly **not** a stop token.
const EOM: &str = "<|eom|>";
/// End of turn (200008) and end of text (200001): both terminal.
const EOT: &str = "<|eot|>";
const EOS: &str = "<|end_of_text|>";
/// Detection prefixes for the whole `<atem:*>` surface.
const ATEM_OPEN: &str = "<atem:";
const ATEM_CLOSE: &str = "</atem:";
/// The only role this turn is allowed to speak as.
const ASSISTANT: &str = "assistant";

/// What the caller may do with a chunk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GuardOutcome {
    /// Release this text as a content delta. Never contains routing metadata,
    /// reasoning, a tool-call body, or a partial marker.
    Emit(String),
    /// Nothing releasable yet. Not an error: either the tail is still
    /// ambiguous, or the text belonged to a non-content channel.
    Hold,
    /// Stop decoding. The turn hit a cap, a terminator, or a forged role.
    /// Content already resolved is **not** discarded — call
    /// [`StreamGuard::flush`] to drain it.
    EndTurn,
}

/// Which channel the current message's `to=` value selected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Channel {
    /// `to=user` — the answer, and the only channel that is ever emitted.
    Content,
    /// `to=self` — chain-of-thought.
    Reasoning,
    /// Any other recipient, including a missing or unparsable one.
    ToolCall,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// Before `<|message|>`: routing metadata. Never emitted, never dropped —
    /// the role and the `to=` value both live here and are needed whole.
    AwaitingHeader,
    /// Inside a message body. `xml` latches once `<atem:` or `</atem:` appears
    /// and suppresses the rest of the message: a tool call may follow prose in
    /// a `to=user` message, and everything from the tag on belongs to the call.
    InMessage { channel: Channel, xml: bool },
}

/// Guards one assistant turn's raw text stream.
///
/// Feed every decoded chunk to [`Self::push`] in order and call [`Self::flush`]
/// once when the turn ends. One guard per turn; it is not resettable.
#[derive(Debug)]
pub struct StreamGuard {
    state: State,
    /// Stream text that is not yet unambiguous. In `AwaitingHeader` this is the
    /// whole header; in `InMessage` it is at most `HOLD_BACK_CHARS` characters
    /// plus the current chunk.
    pending: String,
    /// Content that is unambiguous but not yet handed back, because the push
    /// that resolved it ended the turn.
    ready: String,
    max_messages: usize,
    max_tokens: usize,
    messages: usize,
    /// Counted in [`Self::push`] calls. M1 pushes once per decoded token.
    tokens: usize,
    ended: bool,
}

impl StreamGuard {
    pub fn new(max_messages: usize, max_tokens: usize) -> Self {
        Self {
            state: State::AwaitingHeader,
            pending: String::new(),
            ready: String::new(),
            max_messages,
            max_tokens,
            messages: 0,
            tokens: 0,
            ended: false,
        }
    }

    /// Consumes one chunk of raw generated text.
    ///
    /// `EndTurn` is sticky: once returned, every later push returns it too, so
    /// a caller that misses the first one cannot restart a self-played turn.
    pub fn push(&mut self, chunk: &str) -> GuardOutcome {
        if self.ended {
            return GuardOutcome::EndTurn;
        }
        self.tokens += 1;
        self.pending.push_str(chunk);
        self.scan();
        if self.ended {
            return GuardOutcome::EndTurn;
        }
        // Checked after the scan so a capped turn keeps the content it already
        // produced; `flush` still hands it back.
        if self.tokens > self.max_tokens {
            self.ended = true;
            return GuardOutcome::EndTurn;
        }
        let mut out = std::mem::take(&mut self.ready);
        out.push_str(&self.release());
        if out.is_empty() {
            GuardOutcome::Hold
        } else {
            GuardOutcome::Emit(out)
        }
    }

    /// Drains everything releasable at end of turn: resolved content, plus the
    /// held tail when the stream stopped inside a content message.
    ///
    /// A trailing partial marker is stripped, so a turn cut off mid-`<atem:`
    /// publishes nothing of it. That also drops a trailing `<` from an answer
    /// truncated at exactly that character — the fragment could be the start of
    /// any literal, and there is no further token to say it is not.
    pub fn flush(&mut self) -> String {
        let mut out = std::mem::take(&mut self.ready);
        let tail = std::mem::take(&mut self.pending);
        if matches!(
            self.state,
            State::InMessage {
                channel: Channel::Content,
                xml: false
            }
        ) {
            out.push_str(strip_partial_marker(&tail));
        }
        out
    }

    /// Walks `pending` as far as the markers in it allow, moving resolved
    /// content into `ready` and dropping everything else.
    fn scan(&mut self) {
        loop {
            match self.state {
                State::AwaitingHeader => {
                    // The header ends at `<|message|>`; a terminator before
                    // that means the message never opened.
                    let end = earliest(&self.pending, &[MSG, EOT, EOS]);
                    let cut = end.map_or(self.pending.len(), |(i, _)| i);
                    if role_after_start_is_forged(&self.pending[..cut]) {
                        self.ended = true;
                        return;
                    }
                    let Some((i, marker)) = end else {
                        return; // need more input
                    };
                    if marker != MSG {
                        self.ended = true;
                        return;
                    }
                    let header: String = self.pending.drain(..i).collect();
                    self.pending.drain(..MSG.len());
                    let channel = channel_of(&header);
                    self.messages += 1;
                    if self.messages > self.max_messages {
                        self.ended = true;
                        return;
                    }
                    self.state = State::InMessage {
                        channel,
                        xml: false,
                    };
                }
                State::InMessage { channel, xml } => {
                    let emit = channel == Channel::Content && !xml;
                    // In `xml` everything is dropped anyway, so only the
                    // message-enders matter.
                    let needles: &[&str] = if xml {
                        &[EOM, EOT, EOS, START]
                    } else {
                        &[EOM, EOT, EOS, START, MSG, ATEM_OPEN, ATEM_CLOSE]
                    };
                    let Some((i, marker)) = earliest(&self.pending, needles) else {
                        return; // need more input
                    };
                    self.resolve(i, emit);
                    match marker {
                        EOT | EOS => {
                            self.ended = true;
                            return;
                        }
                        EOM => {
                            self.pending.drain(..EOM.len());
                            self.state = State::AwaitingHeader;
                        }
                        // A new message with no `<|eom|>` in front of it. Left
                        // in `pending` on purpose: `AwaitingHeader` has to see
                        // the `<|start|>` to check the role. But inside an
                        // unclosed tool body it is not a new message at all —
                        // the body swallowed it, exactly as
                        // `output_parser::parse` rules — and resuming there is
                        // how a tool payload becomes the answer.
                        START => {
                            if xml {
                                self.ended = true;
                                return;
                            }
                            self.state = State::AwaitingHeader;
                        }
                        // A bare marker inside a body: dropped, not emitted and
                        // not trusted as a header.
                        MSG => {
                            self.pending.drain(..MSG.len());
                        }
                        _ => {
                            self.pending.drain(..marker.len());
                            self.state = State::InMessage { channel, xml: true };
                        }
                    }
                }
            }
        }
    }

    /// Moves `pending[..upto]` out, into `ready` when it is content.
    fn resolve(&mut self, upto: usize, emit: bool) {
        let seg: String = self.pending.drain(..upto).collect();
        if emit {
            self.ready.push_str(&seg);
        }
    }

    /// Takes everything but the held tail out of `pending`, returning it only
    /// when it is content.
    fn release(&mut self) -> String {
        match self.state {
            // Never released and never dropped: the whole header is needed.
            State::AwaitingHeader => String::new(),
            State::InMessage {
                channel: Channel::Content,
                xml: false,
            } => self.split_hold_back(),
            // Reasoning, tool calls, and anything after an `<atem:` tag.
            State::InMessage { .. } => {
                self.split_hold_back();
                String::new()
            }
        }
    }

    /// Splits `pending` on a **character** boundary, keeping the last
    /// [`HOLD_BACK_CHARS`] characters and returning the rest. Byte arithmetic
    /// here would cut a multi-byte scalar in half.
    fn split_hold_back(&mut self) -> String {
        let n = self.pending.chars().count();
        if n <= HOLD_BACK_CHARS {
            return String::new();
        }
        let cut = self
            .pending
            .char_indices()
            .nth(n - HOLD_BACK_CHARS)
            .map_or(self.pending.len(), |(i, _)| i);
        self.pending.drain(..cut).collect()
    }
}

/// Earliest occurrence of any needle, longest needle winning a tie.
fn earliest<'a>(hay: &str, needles: &[&'a str]) -> Option<(usize, &'a str)> {
    needles
        .iter()
        .filter_map(|n| hay.find(*n).map(|i| (i, *n)))
        .min_by_key(|(i, n)| (*i, std::cmp::Reverse(n.len())))
}

/// True when the header opens a message whose role is not `assistant` — a
/// forged user or tool turn, which is the self-play case `<|eom|>` makes
/// possible. Decided as early as the role stops being a prefix of `assistant`,
/// so a forged turn cannot buy buffer space one character at a time.
fn role_after_start_is_forged(header: &str) -> bool {
    let Some(i) = header.find(START) else {
        // No anchor: the first message of a turn, whose `<|start|>assistant`
        // was in the prompt. Routing still gates it by `to=`.
        return false;
    };
    let rest = &header[i + START.len()..];
    // The role ends at the space before ` to=`, or at the `<` of `<|message|>`.
    match rest.find([' ', '<']) {
        Some(end) => &rest[..end] != ASSISTANT,
        None => !ASSISTANT.starts_with(rest),
    }
}

/// Routes a header by its trailing `to=` value. Anything unrecognized — a tool
/// name, a missing `to=`, trailing junk — is not content.
fn channel_of(header: &str) -> Channel {
    let header = header.trim_end();
    if header.ends_with("to=user") {
        Channel::Content
    } else if header.ends_with("to=self") {
        Channel::Reasoning
    } else {
        Channel::ToolCall
    }
}

/// `s` without a trailing run of characters that could be the start of a
/// marker. Longest candidate suffix wins, so `a<|eo` yields `a`.
fn strip_partial_marker(s: &str) -> &str {
    let n = s.chars().count();
    let longest = MARKER_LITERALS
        .iter()
        .map(|m| m.chars().count())
        .max()
        .unwrap_or(0);
    for k in (1..=n.min(longest)).rev() {
        let cut = s.char_indices().nth(n - k).map_or(0, |(i, _)| i);
        if MARKER_LITERALS.iter().any(|m| m.starts_with(&s[cut..])) {
            return &s[..cut];
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pushes `text` one character at a time — the worst case for a marker
    /// split — and returns everything emitted, then everything flushed.
    fn drain_char_by_char(g: &mut StreamGuard, text: &str) -> String {
        let mut emitted = String::new();
        for ch in text.chars() {
            if let GuardOutcome::Emit(s) = g.push(&ch.to_string()) {
                emitted.push_str(&s);
            }
        }
        emitted
    }

    /// The brief pinned `HOLD_BACK_CHARS >= 24` against a hard-coded 24, which
    /// proves nothing once a longer literal appears. Derive the bound from the
    /// literals instead, and pin the current maximum so a silent shrink of the
    /// list is caught too.
    #[test]
    fn hold_back_covers_the_longest_marker_literal() {
        let longest = MARKER_LITERALS
            .iter()
            .map(|m| m.chars().count())
            .max()
            .expect("marker list is not empty");
        assert!(
            HOLD_BACK_CHARS >= longest,
            "hold-back {HOLD_BACK_CHARS} is shorter than the longest literal ({longest})"
        );
        assert_eq!(longest, 22, "longest marker literal moved");
        for m in MARKER_LITERALS {
            assert!(
                m.chars().count() <= 22,
                "{m:?} is longer than the pinned maximum"
            );
        }
    }

    /// Ties [`MARKER_LITERALS`] to what the scanner actually matches. Without
    /// this the list is decorative and the bound derived from it means nothing.
    #[test]
    fn every_marker_literal_starts_with_a_detected_prefix() {
        for m in MARKER_LITERALS {
            assert!(
                [MSG, START, EOM, EOT, EOS, ATEM_OPEN, ATEM_CLOSE]
                    .iter()
                    .any(|p| m.starts_with(p)),
                "{m:?} is in the inventory but the scanner never matches it"
            );
        }
    }

    /// Both halves bite: the guard must not leak the tag, and must not swallow
    /// the prose in front of it either — a fixed 24-character tail with no
    /// marker detection releases only `ok th` here.
    ///
    /// It is NOT a test of the hold-back's *length*: detection fires on the
    /// six-character [`ATEM_OPEN`] prefix, so this input still passes with
    /// `HOLD_BACK_CHARS` at 5 and only leaks at 4 (both measured by mutation).
    /// The length bound is pinned by
    /// `no_marker_literal_leaks_when_split_one_character_at_a_time`, where
    /// `<|message|>` and `<|end_of_text|>` need 11 and 15.
    #[test]
    fn split_atem_opener_never_leaks() {
        let mut g = StreamGuard::new(8, 1024);
        let emitted =
            drain_char_by_char(&mut g, " to=user<|message|>ok then <atem:function_calls>");
        assert!(!emitted.contains("<atem"), "leaked XML: {emitted}");
        assert!(
            emitted.contains("ok then"),
            "held back real content: {emitted}"
        );
    }

    /// Every literal in the inventory, split one character at a time, with
    /// enough content in front that the guard is already releasing when the
    /// literal starts to arrive — otherwise the hold-back hides a leak by
    /// accident. Nothing containing `<` may reach content, and the content in
    /// front of the literal must survive.
    ///
    /// Two phases, and the second is the one that bites. With the literal at the
    /// end of the stream, `flush`'s partial-marker strip covers for a scanner
    /// that never detected the literal at all — dropping `</atem:` from the
    /// needle list left every assertion green. Text after the literal pushes it
    /// out of the held tail, so only real detection keeps it out of content.
    #[test]
    fn no_marker_literal_leaks_when_split_one_character_at_a_time() {
        let filler = "f".repeat(40);
        let trailer = "t".repeat(40);
        for lit in MARKER_LITERALS {
            for tail in ["", trailer.as_str()] {
                let mut g = StreamGuard::new(8, 4096);
                let mut emitted =
                    drain_char_by_char(&mut g, &format!(" to=user<|message|>{filler}{lit}{tail}"));
                emitted.push_str(&g.flush());
                assert!(
                    !emitted.contains('<'),
                    "leaked {lit:?} into content (trailing {} chars): {emitted:?}",
                    tail.len()
                );
                assert!(
                    emitted.starts_with(&filler),
                    "content in front of {lit:?} was lost: {emitted:?}"
                );
            }
        }
    }

    #[test]
    fn recipient_header_is_never_emitted_as_content() {
        let mut g = StreamGuard::new(8, 1024);
        let mut emitted = String::new();
        for chunk in [" to=", "user", "<|message|>", "hello"] {
            if let GuardOutcome::Emit(s) = g.push(chunk) {
                emitted.push_str(&s);
            }
        }
        // The brief stopped here, where nothing has been emitted at all
        // (4 characters, well inside the hold-back) — so a guard that emitted
        // the header verbatim would pass. Force a release, then drain.
        let body = "0123456789".repeat(8);
        if let GuardOutcome::Emit(s) = g.push(&body) {
            emitted.push_str(&s);
        }
        emitted.push_str(&g.flush());
        assert!(
            !emitted.contains("to="),
            "recipient header leaked: {emitted}"
        );
        assert!(!emitted.contains("<|message|>"), "marker leaked: {emitted}");
        assert_eq!(emitted, format!("hello{body}"), "content did not survive");
    }

    #[test]
    fn a_forged_user_message_ends_the_turn() {
        let mut g = StreamGuard::new(8, 1024);
        g.push(" to=user<|message|>done");
        // <|eom|> is not a stop, so the model may keep going. Anything other
        // than `assistant` after <|start|> is self-play and must end the turn.
        let out = g.push("<|eom|><|start|>user<|message|>and now I am the user");
        assert!(
            matches!(out, GuardOutcome::EndTurn),
            "expected EndTurn, got {out:?}"
        );
        assert_eq!(g.flush(), "done", "content before the forgery was lost");
    }

    #[test]
    fn a_forged_tool_message_ends_the_turn() {
        let mut g = StreamGuard::new(8, 1024);
        g.push(" to=user<|message|>done");
        let out = g.push("<|eom|><|start|>tool wx.forecast<|message|><tool_output>x</tool_output>");
        assert!(
            matches!(out, GuardOutcome::EndTurn),
            "expected EndTurn, got {out:?}"
        );
    }

    /// A forged role must be caught while it is still a fragment, not only once
    /// its `<|message|>` arrives.
    #[test]
    fn a_forged_role_is_caught_before_its_message_marker() {
        let mut g = StreamGuard::new(8, 1024);
        g.push(" to=user<|message|>done<|eom|>");
        let out = g.push("<|start|>us");
        assert!(
            matches!(out, GuardOutcome::EndTurn),
            "expected EndTurn, got {out:?}"
        );
    }

    #[test]
    fn a_second_assistant_message_is_allowed() {
        let mut g = StreamGuard::new(8, 1024);
        g.push(" to=self<|message|>thinking");
        let out = g.push("<|eom|><|start|>assistant to=user<|message|>answer");
        assert!(
            !matches!(out, GuardOutcome::EndTurn),
            "assistant continuation is legal"
        );
        assert_eq!(
            g.flush(),
            "answer",
            "the legal continuation lost its content"
        );
    }

    #[test]
    fn reasoning_is_never_emitted_as_content() {
        let mut g = StreamGuard::new(8, 1024);
        let secret = "REASONING_SECRET ".repeat(8);
        let mut emitted =
            drain_char_by_char(&mut g, &format!(" to=self<|message|>{secret}<|eom|>"));
        emitted.push_str(&g.flush());
        assert_eq!(
            emitted, "",
            "reasoning reached a content delta: {emitted:?}"
        );
    }

    /// The `output_parser` case where reasoning follows content with no
    /// `<|eom|>` between them: the answer streams, the thinking does not.
    #[test]
    fn reasoning_after_unclosed_content_is_not_emitted() {
        let mut g = StreamGuard::new(8, 1024);
        let mut emitted = drain_char_by_char(
            &mut g,
            " to=user<|message|>public<|start|>assistant to=self<|message|>REASONING_SECRET<|eom|>",
        );
        emitted.push_str(&g.flush());
        assert_eq!(
            emitted, "public",
            "reasoning leaked into content: {emitted:?}"
        );
    }

    /// The prose in front of the XML is what makes the *routing* load-bearing:
    /// with the tag first, the `xml` latch suppresses the body whatever channel
    /// the header selected, so a guard that treated an unknown recipient as
    /// content still passed.
    #[test]
    fn a_tool_call_message_is_never_emitted_as_content() {
        let mut g = StreamGuard::new(8, 1024);
        let mut emitted = drain_char_by_char(
            &mut g,
            " to=wx.forecast<|message|>stray prose on the tool channel\n\
             <atem:function_calls>\n\
             <atem:invoke name=\"wx.forecast\">\n\
             <atem:parameter name=\"city\">Paris</atem:parameter>\n\
             </atem:invoke>\n</atem:function_calls>",
        );
        emitted.push_str(&g.flush());
        assert_eq!(
            emitted, "",
            "tool call reached a content delta: {emitted:?}"
        );
    }

    /// Prose then a tool call inside one `to=user` message — the
    /// `output_parser::content_then_tool_call_in_the_same_message` shape. The
    /// prose streams; everything from the tag on is the call, not the answer.
    #[test]
    fn content_before_a_tool_call_streams_but_the_xml_does_not() {
        let mut g = StreamGuard::new(8, 1024);
        let mut emitted = drain_char_by_char(
            &mut g,
            " to=user<|message|>on it<atem:invoke name=\"t\">\
             <atem:parameter name=\"x\">1</atem:parameter></atem:invoke>",
        );
        emitted.push_str(&g.flush());
        assert_eq!(
            emitted, "on it",
            "tool XML leaked into content: {emitted:?}"
        );
    }

    /// `output_parser::a_tool_invoke_whose_close_follows_a_new_message_is_not_trusted`
    /// as a stream: a `<|start|>` inside an unclosed tool body is not a new
    /// message, so resuming there would publish the tool payload as the answer.
    #[test]
    fn a_new_message_inside_an_unclosed_tool_body_ends_the_turn() {
        let mut g = StreamGuard::new(8, 1024);
        g.push(" to=t<|message|><atem:invoke name=\"t\"><atem:parameter name=\"q\">x");
        let out = g.push("<|start|>assistant to=user<|message|>SENT_TO_TOOL");
        assert!(
            matches!(out, GuardOutcome::EndTurn),
            "expected EndTurn, got {out:?}"
        );
        let flushed = g.flush();
        assert_eq!(flushed, "", "tool payload leaked: {flushed:?}");
    }

    #[test]
    fn message_cap_ends_the_turn() {
        let mut g = StreamGuard::new(2, 1024);
        g.push(" to=self<|message|>a<|eom|>");
        g.push("<|start|>assistant to=user<|message|>b<|eom|>");
        let out = g.push("<|start|>assistant to=user<|message|>c");
        assert!(
            matches!(out, GuardOutcome::EndTurn),
            "message cap must trip"
        );
    }

    #[test]
    fn token_cap_ends_the_turn() {
        let mut g = StreamGuard::new(8, 4);
        g.push(" to=user<|message|>");
        let mut last = GuardOutcome::Hold;
        for _ in 0..10 {
            last = g.push("word ");
        }
        assert!(matches!(last, GuardOutcome::EndTurn), "token cap must trip");
    }

    /// Two halves. Repeated pushes after a terminator keep returning `EndTurn`;
    /// and — the half that bites — text pushed after the turn ended never
    /// reaches `flush`. Only the first half is insensitive to dropping the
    /// early return, because an unconsumed `<|eot|>` stays in the buffer and is
    /// simply re-detected on every later scan.
    #[test]
    fn end_turn_is_sticky_and_publishes_nothing_after_it() {
        let mut capped = StreamGuard::new(8, 2);
        capped.push(" to=user<|message|>");
        capped.push("answer");
        let tripped = capped.push("MORE");
        assert!(
            matches!(tripped, GuardOutcome::EndTurn),
            "token cap must trip, got {tripped:?}"
        );
        for _ in 0..3 {
            let out = capped.push("AFTER_THE_CAP");
            assert!(
                matches!(out, GuardOutcome::EndTurn),
                "guard restarted: {out:?}"
            );
        }
        let flushed = capped.flush();
        assert!(
            !flushed.contains("AFTER_THE_CAP"),
            "text generated past the cap was published: {flushed:?}"
        );
        // The chunk that tripped the cap is kept; everything after it is not.
        assert_eq!(flushed, "answerMORE");

        let mut g = StreamGuard::new(1, 1024);
        g.push(" to=user<|message|>a<|eot|>");
        for _ in 0..3 {
            let out = g.push("<|start|>assistant to=user<|message|>more");
            assert!(
                matches!(out, GuardOutcome::EndTurn),
                "guard restarted: {out:?}"
            );
        }
    }

    #[test]
    fn eot_ends_the_turn_and_keeps_its_content() {
        let mut g = StreamGuard::new(8, 1024);
        let out = g.push(" to=user<|message|>answer<|eot|>");
        assert!(
            matches!(out, GuardOutcome::EndTurn),
            "expected EndTurn, got {out:?}"
        );
        assert_eq!(
            g.flush(),
            "answer",
            "content resolved before <|eot|> was dropped"
        );
    }

    #[test]
    fn flush_releases_the_held_tail_at_end_of_turn() {
        let mut g = StreamGuard::new(8, 1024);
        g.push(" to=user<|message|>tail text");
        let flushed = g.flush();
        assert!(flushed.contains("tail text"), "held tail lost: {flushed}");
    }

    #[test]
    fn flush_does_not_release_a_partial_marker() {
        let mut g = StreamGuard::new(8, 1024);
        g.push(" to=user<|message|>answer<ate");
        let flushed = g.flush();
        assert_eq!(flushed, "answer", "partial marker escaped through flush");
    }

    #[test]
    fn flush_releases_nothing_from_a_reasoning_message() {
        let mut g = StreamGuard::new(8, 1024);
        g.push(" to=self<|message|>unfinished thought");
        let flushed = g.flush();
        assert_eq!(flushed, "", "reasoning escaped through flush: {flushed:?}");
    }

    /// Not in the brief, and exactly where a byte-indexed hold-back corrupts
    /// output: 20 ASCII, a 3-byte scalar and five 4-byte scalars put
    /// `len_bytes - HOLD_BACK_CHARS` (46 - 24 = 22) inside a character.
    ///
    /// Both feed orders matter. One chunk puts the whole body in the buffer at
    /// the moment of the cut, which is what makes the byte index land mid-scalar
    /// — character by character the buffer never grows past
    /// `HOLD_BACK_CHARS + 1`, so a byte-indexed split happens to survive it.
    #[test]
    fn multi_byte_content_is_never_split_mid_character() {
        let body = format!("{}{}{}", "A".repeat(20), "日日", "😀".repeat(5));
        assert_eq!(
            body.chars().count(),
            27,
            "fixture is not past the hold-back"
        );
        assert_eq!(body.len(), 46, "fixture byte width changed");
        assert!(
            !body.is_char_boundary(body.len() - HOLD_BACK_CHARS),
            "fixture no longer straddles a character at the cut"
        );

        let mut one_chunk = StreamGuard::new(8, 1024);
        let mut emitted = match one_chunk.push(&format!(" to=user<|message|>{body}")) {
            GuardOutcome::Emit(s) => s,
            other => panic!("expected content past the hold-back, got {other:?}"),
        };
        emitted.push_str(&one_chunk.flush());
        assert_eq!(
            emitted, body,
            "multi-byte content was corrupted in one chunk"
        );

        let mut split = StreamGuard::new(8, 1024);
        let mut emitted = drain_char_by_char(&mut split, &format!(" to=user<|message|>{body}"));
        emitted.push_str(&split.flush());
        assert_eq!(
            emitted, body,
            "multi-byte content was corrupted character by character"
        );
    }

    #[test]
    fn a_multi_byte_answer_around_a_tool_call_survives() {
        let mut g = StreamGuard::new(8, 1024);
        let prose = "天気を調べます。🌤️ ".repeat(4);
        let mut emitted = drain_char_by_char(
            &mut g,
            &format!(" to=user<|message|>{prose}<atem:invoke name=\"t\"></atem:invoke>"),
        );
        emitted.push_str(&g.flush());
        assert_eq!(
            emitted, prose,
            "multi-byte content around XML was corrupted"
        );
    }

    /// A `to=` that is not the recipient — the header's routing value is its
    /// suffix, which is what the checkpoint's own open pattern requires.
    #[test]
    fn a_trailing_recipient_decides_the_channel() {
        let mut g = StreamGuard::new(8, 1024);
        let secret = "SECRET ".repeat(8);
        let mut emitted =
            drain_char_by_char(&mut g, &format!(" to=user to=self<|message|>{secret}"));
        emitted.push_str(&g.flush());
        assert_eq!(emitted, "", "a decoy to=user routed reasoning to content");
    }

    #[test]
    fn a_header_with_no_recipient_is_not_content() {
        let mut g = StreamGuard::new(8, 1024);
        let body = "BODY ".repeat(8);
        let mut emitted = drain_char_by_char(&mut g, &format!("<|message|>{body}"));
        emitted.push_str(&g.flush());
        assert_eq!(emitted, "", "a header with no to= was treated as content");
    }

    /// A bare `<|message|>` inside a body is dropped: the marker must not reach
    /// a content delta, but the prose around it is still the answer.
    #[test]
    fn a_bare_message_marker_inside_content_is_dropped() {
        let mut g = StreamGuard::new(8, 1024);
        let mut emitted = drain_char_by_char(
            &mut g,
            " to=user<|message|>first half <|message|> second half of the answer",
        );
        emitted.push_str(&g.flush());
        assert!(!emitted.contains("<|message|>"), "marker leaked: {emitted}");
        assert_eq!(emitted, "first half  second half of the answer");
    }
}
