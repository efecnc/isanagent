//! Detect repeated tool-call patterns (inspired by Hugging Face ml-intern `doom_loop.py`).

use sha2::{Digest, Sha256};

use crate::utils::{ChatMessage, ToolCallRequest};

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct ToolCallSignature {
    name: String,
    args_hash: String,
    /// Hash of the arguments with every numeric literal collapsed to a constant, so calls that
    /// differ *only* in a number (an incrementing offset/page, a single tweaked hyperparameter)
    /// share a normalized signature. `args_hash` already distinguishes any two distinct argument
    /// strings, so this field does not change exact `PartialEq`/`Hash` equality — it is consulted
    /// only by the normalized detector via [`ToolCallSignature::norm_eq`].
    norm_args_hash: String,
}

impl ToolCallSignature {
    /// Same tool, same arguments once numeric literals are collapsed.
    fn norm_eq(&self, other: &Self) -> bool {
        self.name == other.name && self.norm_args_hash == other.norm_args_hash
    }
}

fn hash_args(args: &str) -> String {
    let digest = Sha256::digest(args.as_bytes());
    let full = hex::encode(digest);
    full.chars().take(12).collect()
}

/// Collapse numeric literals (`123`, `4.5`) to a constant so a loop that only varies a number —
/// incrementing offsets, pagination, one tweaked hyperparameter — still normalizes to a repeat.
/// Operates on the raw argument string, so it catches a number whether it lives in a structured
/// arg or inside a path/query string, and needs no JSON parse.
fn normalize_args(args: &str) -> String {
    use std::sync::LazyLock;
    static NUM_RE: LazyLock<regex::Regex> = LazyLock::new(|| {
        regex::Regex::new(r"\d+(?:\.\d+)?").expect("static number regex compiles")
    });
    NUM_RE.replace_all(args, "0").into_owned()
}

fn signature(tc: &ToolCallRequest) -> ToolCallSignature {
    ToolCallSignature {
        name: tc.function.name.clone(),
        args_hash: hash_args(&tc.function.arguments),
        norm_args_hash: hash_args(&normalize_args(&tc.function.arguments)),
    }
}

fn extract_recent_tool_signatures(
    messages: &[ChatMessage],
    lookback: usize,
) -> Vec<ToolCallSignature> {
    let mut sigs = Vec::new();
    let take = messages.len().min(lookback);
    let start = messages.len().saturating_sub(take);
    for msg in messages.iter().skip(start) {
        if msg.role != "assistant" {
            continue;
        }
        let Some(ref tcs) = msg.tool_calls else {
            continue;
        };
        for tc in tcs {
            sigs.push(signature(tc));
        }
    }
    sigs
}

fn detect_identical_consecutive(
    signatures: &[ToolCallSignature],
    threshold: usize,
) -> Option<&str> {
    if signatures.len() < threshold {
        return None;
    }
    let mut count = 1usize;
    for i in 1..signatures.len() {
        if signatures[i] == signatures[i - 1] {
            count += 1;
            if count >= threshold {
                return Some(signatures[i].name.as_str());
            }
        } else {
            count = 1;
        }
    }
    None
}

/// Like [`detect_identical_consecutive`] but compares the **numeric-normalized** signature, so a
/// run that only varies a number (offset 0 → 100 → 200, page 1 → 2 → 3, a single tweaked
/// hyperparameter) counts as a repeat. Intentionally separate from the exact detector and only used
/// for the advisory nudge — never for hard-stop escalation — so legitimate bounded pagination is
/// not killed.
fn detect_identical_consecutive_normalized(
    signatures: &[ToolCallSignature],
    threshold: usize,
) -> Option<&str> {
    if signatures.len() < threshold {
        return None;
    }
    let mut count = 1usize;
    for i in 1..signatures.len() {
        if signatures[i].norm_eq(&signatures[i - 1]) {
            count += 1;
            if count >= threshold {
                return Some(signatures[i].name.as_str());
            }
        } else {
            count = 1;
        }
    }
    None
}

fn detect_repeating_sequence(signatures: &[ToolCallSignature]) -> Option<Vec<String>> {
    let n = signatures.len();
    for seq_len in 2..=5 {
        let min_required = seq_len * 2;
        if n < min_required {
            continue;
        }
        let tail = &signatures[n - min_required..];
        let pattern = &tail[..seq_len];
        let mut reps = 0usize;
        let mut start = n;
        loop {
            if start < seq_len {
                break;
            }
            start -= seq_len;
            let chunk = &signatures[start..start + seq_len];
            if chunk == pattern {
                reps += 1;
            } else {
                break;
            }
        }
        if reps >= 2 {
            return Some(pattern.iter().map(|s| s.name.clone()).collect());
        }
    }
    None
}

/// If a doom loop is detected, returns a corrective user message to append to history.
pub fn check_for_doom_loop_prompt(messages: &[ChatMessage]) -> Option<String> {
    let signatures = extract_recent_tool_signatures(messages, 30);
    if signatures.len() < 3 {
        return None;
    }
    const THRESHOLD: usize = 3;
    if let Some(tool_name) = detect_identical_consecutive(&signatures, THRESHOLD) {
        return Some(format!(
            "[SYSTEM: DOOM LOOP DETECTED] You have called '{tool_name}' with the same \
arguments multiple times in a row. STOP repeating this approach — it is not working. \
Step back and try a fundamentally different strategy: use a different tool, change \
your arguments significantly, or explain what you are stuck on and ask for guidance."
        ));
    }
    if let Some(pattern) = detect_repeating_sequence(&signatures) {
        let pattern_desc = pattern.join(" → ");
        return Some(format!(
            "[SYSTEM: DOOM LOOP DETECTED] You are stuck in a repeating cycle of tool calls: \
[{pattern_desc}]. STOP this cycle and try a fundamentally different approach."
        ));
    }
    // Softer signal: same tool called repeatedly with only a numeric argument changing. A higher
    // threshold than the exact detectors keeps this from firing on short, legitimate pagination,
    // and the wording is non-coercive because intentional pagination is valid.
    const NORM_THRESHOLD: usize = 4;
    if let Some(tool_name) =
        detect_identical_consecutive_normalized(&signatures, NORM_THRESHOLD)
    {
        return Some(format!(
            "[SYSTEM: POSSIBLE LOOP] You have called '{tool_name}' several times in a row changing \
only a numeric argument (e.g. an incrementing offset/page or one tweaked value). If you are \
intentionally paginating or sweeping, continue — but if you expected different results and are not \
making progress, STOP and try a fundamentally different approach (a different tool, a substantially \
different query, or explain what you are stuck on)."
        ));
    }
    None
}

/// True if the agent is *currently* looping at the tail of the conversation. This is distinct
/// from [`check_for_doom_loop_prompt`], which also fires for a stale identical run that merely
/// lingers in the 30-message lookback window after the model has already changed course
/// (`detect_identical_consecutive` matches a run *anywhere* in the window, not just at the end).
///
/// Escalation to a hard stop must count only genuinely-ongoing loops — otherwise a model that
/// corrects itself after one nudge would still be terminated as the stale run ages out. So the
/// escalation path gates on this tail-anchored check rather than on detection alone.
pub fn doom_loop_active_at_tail(messages: &[ChatMessage]) -> bool {
    let signatures = extract_recent_tool_signatures(messages, 30);
    let n = signatures.len();
    if n < 2 {
        return false;
    }
    // Identical loop still ongoing: the model just repeated its previous tool call byte-for-byte.
    if signatures[n - 1] == signatures[n - 2] {
        return true;
    }
    // Cyclic loop (A→B→A→B…): `detect_repeating_sequence` is tail-anchored (it inspects only the
    // most recent `2*seq_len` signatures), so a match means the cycle is still active at the tail.
    detect_repeating_sequence(&signatures).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::{ToolCallFunction, ToolCallRequest};

    fn tc(name: &str, args: &str) -> ToolCallRequest {
        ToolCallRequest {
            id: "id".into(),
            tool_type: "function".into(),
            extra_content: None,
            function: ToolCallFunction {
                name: name.into(),
                arguments: args.into(),
            },
        }
    }

    fn assistant_with_tools(calls: Vec<ToolCallRequest>) -> ChatMessage {
        ChatMessage {
            role: "assistant".into(),
            content: None,
            name: None,
            tool_calls: Some(calls),
            tool_call_id: None,
            reasoning_content: None,
            is_error: None,
        }
    }

    #[test]
    fn identical_three_triggers() {
        let t = tc("read_file", r#"{"path":"x"}"#);
        let msgs = vec![
            assistant_with_tools(vec![t.clone()]),
            assistant_with_tools(vec![t.clone()]),
            assistant_with_tools(vec![t]),
        ];
        let p = check_for_doom_loop_prompt(&msgs).expect("prompt");
        assert!(p.contains("DOOM LOOP"));
        assert!(p.contains("read_file"));
    }

    #[test]
    fn abab_pattern_triggers() {
        let a = tc("a", "{}");
        let b = tc("b", "{}");
        let msgs = vec![
            assistant_with_tools(vec![a.clone()]),
            assistant_with_tools(vec![b.clone()]),
            assistant_with_tools(vec![a.clone()]),
            assistant_with_tools(vec![b]),
        ];
        let p = check_for_doom_loop_prompt(&msgs).expect("prompt");
        assert!(p.contains("a → b"));
    }

    #[test]
    fn active_at_tail_true_while_repeating() {
        let t = tc("read_file", r#"{"path":"x"}"#);
        let msgs = vec![
            assistant_with_tools(vec![t.clone()]),
            assistant_with_tools(vec![t.clone()]),
            assistant_with_tools(vec![t]),
        ];
        assert!(doom_loop_active_at_tail(&msgs));
    }

    #[test]
    fn active_at_tail_false_after_model_corrects() {
        // The model looped (X,X,X) then changed course (Y). `check_for_doom_loop_prompt` still
        // fires on the stale X,X,X in the window, but the loop is NOT active at the tail — so
        // escalation must not count it (this is the regression the HIGH review finding caught).
        let x = tc("read_file", r#"{"path":"x"}"#);
        let y = tc("write_file", r#"{"path":"y"}"#);
        let msgs = vec![
            assistant_with_tools(vec![x.clone()]),
            assistant_with_tools(vec![x.clone()]),
            assistant_with_tools(vec![x]),
            assistant_with_tools(vec![y]),
        ];
        assert!(
            check_for_doom_loop_prompt(&msgs).is_some(),
            "stale run still detected in window"
        );
        assert!(
            !doom_loop_active_at_tail(&msgs),
            "must not be active at tail once the model has moved on"
        );
    }

    #[test]
    fn active_at_tail_true_for_tail_anchored_cycle() {
        let a = tc("a", "{}");
        let b = tc("b", "{}");
        let msgs = vec![
            assistant_with_tools(vec![a.clone()]),
            assistant_with_tools(vec![b.clone()]),
            assistant_with_tools(vec![a]),
            assistant_with_tools(vec![b]),
        ];
        assert!(doom_loop_active_at_tail(&msgs));
    }

    fn paginate(offset: i32) -> ToolCallRequest {
        tc("read_file", &format!(r#"{{"path":"x","offset":{offset}}}"#))
    }

    #[test]
    fn normalized_numeric_loop_triggers_advisory() {
        // Incrementing offset escapes the exact-hash detector but is caught by normalization.
        let msgs = vec![
            assistant_with_tools(vec![paginate(0)]),
            assistant_with_tools(vec![paginate(100)]),
            assistant_with_tools(vec![paginate(200)]),
            assistant_with_tools(vec![paginate(300)]),
        ];
        let p = check_for_doom_loop_prompt(&msgs).expect("advisory");
        assert!(p.contains("POSSIBLE LOOP"), "{p}");
        assert!(p.contains("read_file"), "{p}");
    }

    #[test]
    fn normalized_below_threshold_does_not_trigger() {
        // Only 3 numeric-varied calls — under the (more conservative) normalized threshold of 4.
        let msgs = vec![
            assistant_with_tools(vec![paginate(0)]),
            assistant_with_tools(vec![paginate(100)]),
            assistant_with_tools(vec![paginate(200)]),
        ];
        assert!(check_for_doom_loop_prompt(&msgs).is_none());
    }

    #[test]
    fn normalized_does_not_trigger_when_nonnumeric_args_differ() {
        // Different paths (no shared numeric-only variation) must not look like a loop.
        let msgs = vec![
            assistant_with_tools(vec![tc("read_file", r#"{"path":"a"}"#)]),
            assistant_with_tools(vec![tc("read_file", r#"{"path":"b"}"#)]),
            assistant_with_tools(vec![tc("read_file", r#"{"path":"c"}"#)]),
            assistant_with_tools(vec![tc("read_file", r#"{"path":"d"}"#)]),
        ];
        assert!(check_for_doom_loop_prompt(&msgs).is_none());
    }

    #[test]
    fn numeric_loop_is_not_escalated_at_tail() {
        // Escalation (hard stop) must stay exact-only so legitimate bounded pagination isn't killed:
        // a numeric-varied loop nudges but is NOT active-at-tail.
        let msgs = vec![
            assistant_with_tools(vec![paginate(0)]),
            assistant_with_tools(vec![paginate(100)]),
            assistant_with_tools(vec![paginate(200)]),
            assistant_with_tools(vec![paginate(300)]),
        ];
        assert!(check_for_doom_loop_prompt(&msgs).is_some());
        assert!(!doom_loop_active_at_tail(&msgs));
    }
}
