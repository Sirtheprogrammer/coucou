// Claude API client — the same integration as ClaudeService.swift: multi-turn
// chat with web search, and files sent as document/image/text blocks.
//
// Everything happens here rather than in the island: the API key never leaves
// the Credential Manager, and file bytes never cross the IPC boundary.

use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::secrets;

const ENDPOINT: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";
/// Server-side fallback: on a policy decline the API retries the same request on
/// a fallback model inside the same call, so the island never shows a dead end.
const FALLBACK_BETA: &str = "server-side-fallback-2026-07-01";
const MAX_TOKENS: u32 = 4096;
/// Text and code files are inlined; anything larger is skipped, as on macOS.
const MAX_INLINE_TEXT: u64 = 200_000;

pub const DEFAULT_MODEL: &str = "claude-opus-5";

const SYSTEM_PROMPT: &str = "You are Mochi, a personal AI assistant living at the top of the user's screen. \
You have web search access and can help with absolutely anything — research, coding, finding places, recommendations, tasks, questions. \
Respond in the user's language. Be thorough and complete — use as much detail as the task requires. \
No markdown formatting (no **, no ##, no bullet dashes). Use plain text with line breaks.";

#[derive(Default)]
pub struct Chat {
    /// Full multi-turn history, including tool_use / tool_result blocks.
    messages: Mutex<Vec<Value>>,
}

impl Chat {
    pub fn reset(&self) {
        self.messages.lock().unwrap().clear();
    }

    fn is_empty(&self) -> bool {
        self.messages.lock().unwrap().is_empty()
    }

    fn push(&self, message: Value) {
        self.messages.lock().unwrap().push(message);
    }

    fn pop(&self) {
        self.messages.lock().unwrap().pop();
    }

    fn snapshot(&self) -> Vec<Value> {
        self.messages.lock().unwrap().clone()
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum ChatContext {
    File { name: String, path: String },
    Window { app_name: String, title: String, url: Option<String> },
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatReply {
    pub text: String,
}

pub fn normalize_chat_endpoint(url: &str) -> String {
    let trimmed = url.trim().trim_end_matches('/');
    if trimmed.is_empty() {
        return "http://localhost:11434/v1/chat/completions".to_string();
    }
    if trimmed.ends_with("/chat/completions") {
        trimmed.to_string()
    } else if trimmed.ends_with("/v1") {
        format!("{trimmed}/chat/completions")
    } else {
        format!("{trimmed}/v1/chat/completions")
    }
}

/// One chat turn. Returns the assistant's text, or a message the island shows
/// in the note view.
pub async fn send(
    chat: &Chat,
    provider: &str,
    model: &str,
    custom_url: &str,
    query: String,
    context: Option<ChatContext>,
) -> Result<ChatReply, String> {
    match provider {
        "openai" | "google" | "deepseek" | "custom" => {
            send_openai_compatible(chat, provider, model, custom_url, query, context).await
        }
        _ => send_anthropic(chat, model, query, context).await,
    }
}

async fn send_anthropic(
    chat: &Chat,
    model: &str,
    query: String,
    context: Option<ChatContext>,
) -> Result<ChatReply, String> {
    let key = secrets::get("anthropic-api-key")
        .ok_or_else(|| "Claude API key missing. Open settings.".to_string())?;

    let mut content: Vec<Value> = Vec::new();

    // File / window context rides along with the first message only, exactly
    // like ClaudeService.chat().
    if chat.is_empty() {
        match &context {
            Some(ChatContext::File { name, path }) => {
                if let Some(block) = file_block(path) {
                    content.push(block);
                }
                content.push(json!({ "type": "text", "text": format!("File: {name}") }));
            }
            Some(ChatContext::Window { app_name, title, url }) => {
                let mut text = format!("Context — App: {app_name}, Window: {title}");
                if let Some(url) = url {
                    text.push_str(&format!(", URL: {url}"));
                }
                content.push(json!({ "type": "text", "text": text }));
            }
            None => {}
        }
    }
    content.push(json!({ "type": "text", "text": query }));

    chat.push(json!({ "role": "user", "content": content }));

    let body = json!({
        "model": model,
        "max_tokens": MAX_TOKENS,
        "system": SYSTEM_PROMPT,
        "tools": [{ "type": "web_search_20260209", "name": "web_search", "max_uses": 5 }],
        "fallbacks": "default",
        "messages": chat.snapshot(),
    });

    let response = match call(&key, &body).await {
        Ok(v) => v,
        Err(err) => {
            chat.pop(); // keep the history consistent with what the model saw
            return Err(err);
        }
    };

    // A policy decline comes back as HTTP 200 with stop_reason "refusal".
    if response.get("stop_reason").and_then(Value::as_str) == Some("refusal") {
        chat.pop();
        let why = response
            .get("stop_details")
            .and_then(|d| d.get("explanation"))
            .and_then(Value::as_str)
            .unwrap_or("Claude declined this one.");
        return Err(why.to_string());
    }

    let Some(blocks) = response.get("content").and_then(Value::as_array).cloned() else {
        chat.pop();
        return Err("Unexpected API response.".into());
    };

    // Store the whole content — tool_use / tool_result blocks included — so the
    // next turn has the right context.
    chat.push(json!({ "role": "assistant", "content": blocks.clone() }));

    let text = blocks
        .iter()
        .filter(|b| b.get("type").and_then(Value::as_str) == Some("text"))
        .filter_map(|b| b.get("text").and_then(Value::as_str))
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string();

    if text.is_empty() {
        return Err("No response text.".into());
    }
    Ok(ChatReply { text })
}

async fn send_openai_compatible(
    chat: &Chat,
    provider: &str,
    model: &str,
    custom_url: &str,
    query: String,
    context: Option<ChatContext>,
) -> Result<ChatReply, String> {
    let (key_name, provider_name, endpoint): (&str, &str, String) = match provider {
        "openai" => ("openai-api-key", "OpenAI", "https://api.openai.com/v1/chat/completions".to_string()),
        "google" => ("google-api-key", "Google Gemini", "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions".to_string()),
        "deepseek" => ("deepseek-api-key", "DeepSeek", "https://api.deepseek.com/chat/completions".to_string()),
        "custom" => ("custom-api-key", "Custom", normalize_chat_endpoint(custom_url)),
        _ => return Err(format!("Unknown provider {provider}")),
    };

    let key = secrets::get(key_name);
    if provider != "custom" && key.is_none() {
        return Err(format!("{provider_name} API key missing. Open settings."));
    }

    let mut content: Vec<Value> = Vec::new();
    let mut text_prefix = String::new();

    if chat.is_empty() {
        match &context {
            Some(ChatContext::File { name, path }) => {
                let ext = std::path::Path::new(path)
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("")
                    .to_lowercase();
                let media = match ext.as_str() {
                    "jpg" | "jpeg" => Some("image/jpeg"),
                    "png" => Some("image/png"),
                    "gif" => Some("image/gif"),
                    "webp" => Some("image/webp"),
                    _ => None,
                };

                if let Some(media_type) = media {
                    if provider != "deepseek" {
                        if let Ok(bytes) = std::fs::read(path) {
                            content.push(json!({
                                "type": "image_url",
                                "image_url": {
                                    "url": format!("data:{media_type};base64,{}", base64(&bytes))
                                }
                            }));
                        }
                    }
                    text_prefix.push_str(&format!("File: {name}\n\n"));
                } else if let Ok(text) = std::fs::read_to_string(path) {
                    if (text.len() as u64) <= MAX_INLINE_TEXT {
                        text_prefix.push_str(&format!("File: {name}\nFile contents:\n{text}\n\n"));
                    } else {
                        text_prefix.push_str(&format!("File: {name}\n\n"));
                    }
                } else {
                    text_prefix.push_str(&format!("File: {name}\n\n"));
                }
            }
            Some(ChatContext::Window { app_name, title, url }) => {
                let mut text = format!("Context — App: {app_name}, Window: {title}");
                if let Some(url) = url {
                    text.push_str(&format!(", URL: {url}"));
                }
                text_prefix.push_str(&format!("{text}\n\n"));
            }
            None => {}
        }
    }

    let full_user_text = format!("{text_prefix}{query}");
    content.push(json!({ "type": "text", "text": full_user_text }));

    chat.push(json!({ "role": "user", "content": content }));

    let msgs = to_openai_messages(SYSTEM_PROMPT, &chat.snapshot());
    let body = json!({
        "model": model,
        "max_tokens": MAX_TOKENS,
        "messages": msgs,
    });

    let response = match call_openai_compatible(&endpoint, key.as_deref(), &body).await {
        Ok(v) => v,
        Err(err) => {
            chat.pop();
            return Err(err);
        }
    };

    let text = response
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(Value::as_str)
        .ok_or_else(|| "Unexpected API response format.".to_string())?
        .trim()
        .to_string();

    if text.is_empty() {
        chat.pop();
        return Err("No response text.".into());
    }

    chat.push(json!({
        "role": "assistant",
        "content": [{ "type": "text", "text": text.clone() }]
    }));

    Ok(ChatReply { text })
}

pub fn to_openai_messages(system_prompt: &str, history: &[Value]) -> Vec<Value> {
    let mut msgs = vec![json!({
        "role": "system",
        "content": system_prompt,
    })];

    for m in history {
        let role = m.get("role").and_then(Value::as_str).unwrap_or("user");
        let content_val = m.get("content");
        match content_val {
            Some(Value::String(s)) => {
                if !s.is_empty() {
                    msgs.push(json!({ "role": role, "content": s }));
                }
            }
            Some(Value::Array(arr)) => {
                let mut text_parts = Vec::new();
                let mut other_parts = Vec::new();

                for b in arr {
                    let b_type = b.get("type").and_then(Value::as_str).unwrap_or("");
                    if b_type == "text" {
                        if let Some(t) = b.get("text").and_then(Value::as_str) {
                            text_parts.push(t);
                        }
                    } else if b_type == "image_url" {
                        other_parts.push(b.clone());
                    } else if b_type == "image" {
                        if let Some(source) = b.get("source") {
                            let media = source.get("media_type").and_then(Value::as_str).unwrap_or("image/jpeg");
                            let data = source.get("data").and_then(Value::as_str).unwrap_or("");
                            if !data.is_empty() {
                                other_parts.push(json!({
                                    "type": "image_url",
                                    "image_url": {
                                        "url": format!("data:{media};base64,{data}")
                                    }
                                }));
                            }
                        }
                    }
                }

                if !other_parts.is_empty() {
                    let mut parts = Vec::new();
                    if !text_parts.is_empty() {
                        parts.push(json!({ "type": "text", "text": text_parts.join("\n") }));
                    }
                    parts.extend(other_parts);
                    msgs.push(json!({ "role": role, "content": parts }));
                } else if !text_parts.is_empty() {
                    msgs.push(json!({ "role": role, "content": text_parts.join("\n") }));
                }
            }
            _ => {}
        }
    }
    msgs
}

async fn call_openai_compatible(endpoint: &str, key: Option<&str>, body: &Value) -> Result<Value, String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(90))
        .build()
        .map_err(|e| e.to_string())?;

    let mut req = client
        .post(endpoint)
        .header("content-type", "application/json");

    if let Some(k) = key {
        let trimmed = k.trim();
        if !trimmed.is_empty() {
            req = req.header("authorization", format!("Bearer {trimmed}"));
        }
    }

    let response = req
        .json(body)
        .send()
        .await
        .map_err(|e| format!("Network error: {e}"))?;

    let status = response.status();
    let text = response.text().await.map_err(|e| e.to_string())?;
    if !status.is_success() {
        let detail = serde_json::from_str::<Value>(&text)
            .ok()
            .and_then(|v| {
                v.get("error")
                    .and_then(|e| e.get("message"))
                    .and_then(Value::as_str)
                    .map(str::to_string)
            })
            .unwrap_or_else(|| text.chars().take(200).collect());
        return Err(format!("API error {status}: {detail}"));
    }
    serde_json::from_str(&text).map_err(|e| format!("Bad API response: {e}"))
}

async fn call(key: &str, body: &Value) -> Result<Value, String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(90))
        .build()
        .map_err(|e| e.to_string())?;

    let response = client
        .post(ENDPOINT)
        .header("x-api-key", key)
        .header("anthropic-version", ANTHROPIC_VERSION)
        .header("anthropic-beta", FALLBACK_BETA)
        .header("content-type", "application/json")
        .json(body)
        .send()
        .await
        .map_err(|e| format!("Network error: {e}"))?;

    let status = response.status();
    let text = response.text().await.map_err(|e| e.to_string())?;
    if !status.is_success() {
        // Surface the API's own message, which is what makes a bad key obvious.
        let detail = serde_json::from_str::<Value>(&text)
            .ok()
            .and_then(|v| {
                v.get("error")
                    .and_then(|e| e.get("message"))
                    .and_then(Value::as_str)
                    .map(str::to_string)
            })
            .unwrap_or_else(|| text.chars().take(200).collect());
        return Err(format!("Claude API {status}: {detail}"));
    }
    serde_json::from_str(&text).map_err(|e| format!("Bad API response: {e}"))
}

/// PDF → document block, image → image block, text/code → inline text.
/// Mirrors readFileAsBlock() in ClaudeService.swift.
fn file_block(path: &str) -> Option<Value> {
    let ext = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    let media_type = match ext.as_str() {
        "pdf" => Some(("document", "application/pdf")),
        "jpg" | "jpeg" => Some(("image", "image/jpeg")),
        "png" => Some(("image", "image/png")),
        "gif" => Some(("image", "image/gif")),
        "webp" => Some(("image", "image/webp")),
        _ => None,
    };

    if let Some((block_type, media)) = media_type {
        let bytes = std::fs::read(path).ok()?;
        return Some(json!({
            "type": block_type,
            "source": { "type": "base64", "media_type": media, "data": base64(&bytes) },
        }));
    }

    let len = std::fs::metadata(path).ok()?.len();
    if len > MAX_INLINE_TEXT {
        return None;
    }
    let text = std::fs::read_to_string(path).ok()?;
    Some(json!({ "type": "text", "text": format!("File contents:\n{text}") }))
}

/// Small standalone base64 encoder — not worth another dependency.
/// Also used for Stripe's basic auth.
pub(crate) fn base64_for(bytes: &[u8]) -> String {
    base64(bytes)
}

fn base64(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
        out.push(TABLE[(n >> 18) as usize & 63] as char);
        out.push(TABLE[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 { TABLE[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if chunk.len() > 2 { TABLE[n as usize & 63] as char } else { '=' });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::base64;

    #[test]
    fn base64_matches_rfc4648_vectors() {
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foo"), "Zm9v");
        assert_eq!(base64(b"foob"), "Zm9vYg==");
        assert_eq!(base64(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn to_openai_messages_formats_history_and_system() {
        use serde_json::json;
        let history = vec![
            json!({ "role": "user", "content": [{ "type": "text", "text": "hello" }] }),
            json!({ "role": "assistant", "content": [{ "type": "text", "text": "hi there" }] }),
        ];
        let msgs = super::to_openai_messages("system test", &history);
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[0]["content"], "system test");
        assert_eq!(msgs[1]["role"], "user");
        assert_eq!(msgs[1]["content"], "hello");
        assert_eq!(msgs[2]["role"], "assistant");
        assert_eq!(msgs[2]["content"], "hi there");
    }

    #[test]
    fn to_openai_messages_translates_image_blocks() {
        use serde_json::json;
        let history = vec![
            json!({
                "role": "user",
                "content": [
                    { "type": "image", "source": { "media_type": "image/png", "data": "abc123==" } },
                    { "type": "text", "text": "look at this" }
                ]
            }),
        ];
        let msgs = super::to_openai_messages("sys", &history);
        assert_eq!(msgs.len(), 2);
        let user_content = msgs[1]["content"].as_array().unwrap();
        assert_eq!(user_content[0]["type"], "text");
        assert_eq!(user_content[0]["text"], "look at this");
        assert_eq!(user_content[1]["type"], "image_url");
        assert_eq!(user_content[1]["image_url"]["url"], "data:image/png;base64,abc123==");
    }

    #[test]
    fn normalize_chat_endpoint_cases() {
        assert_eq!(
            super::normalize_chat_endpoint("http://localhost:11434"),
            "http://localhost:11434/v1/chat/completions"
        );
        assert_eq!(
            super::normalize_chat_endpoint("http://localhost:11434/"),
            "http://localhost:11434/v1/chat/completions"
        );
        assert_eq!(
            super::normalize_chat_endpoint("http://localhost:11434/v1"),
            "http://localhost:11434/v1/chat/completions"
        );
        assert_eq!(
            super::normalize_chat_endpoint("http://localhost:11434/v1/chat/completions"),
            "http://localhost:11434/v1/chat/completions"
        );
        assert_eq!(
            super::normalize_chat_endpoint("https://openrouter.ai/api/v1"),
            "https://openrouter.ai/api/v1/chat/completions"
        );
        assert_eq!(
            super::normalize_chat_endpoint(""),
            "http://localhost:11434/v1/chat/completions"
        );
    }
}
