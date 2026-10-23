//! Cliente da conversa do Manager (tray) com o orchestrator.
//!
//! Envia turnos escritos (source="chat") via POST /execute e consome os
//! eventos espelhados no Conversation Hub (GET /conversation/stream) para
//! alimentar a tela de chat da UI em tempo real.

use crate::state::SharedState;
use serde::Serialize;
use serde_json::Value;

/// URL base do orchestrator (mesma do tool loop; sobrescrevível via env).
pub fn orchestrator_url() -> String {
    std::env::var("AVA_ORCHESTRATOR_URL").unwrap_or_else(|_| "http://localhost:9000".to_string())
}

#[derive(Serialize)]
struct ChatExecuteRequest<'a> {
    input: &'a str,
    source: &'static str,
    #[serde(rename = "tts")]
    tts: bool,
    #[serde(rename = "tts_output")]
    tts_output: &'static str,
}

/// Envia um turno escrito da tela de chat ao orchestrator.
/// O backend espelha os eventos (delta/tool_call/done) para o Conversation
/// Hub, que a task de SSE do chat consome para alimentar a UI.
pub async fn send_chat_turn(state: &SharedState, text: &str) -> anyhow::Result<()> {
    if text.trim().is_empty() {
        anyhow::bail!("turno vazio");
    }
    let url = format!("{}/execute", orchestrator_url());
    let body = ChatExecuteRequest {
        input: text,
        source: "chat",
        // O chat do Manager é silencioso (a resposta falada é o papel do
        // pipeline de voz / TUI — aqui só texto na tela).
        tts: false,
        tts_output: "client",
    };
    let client = http_client(state).await?;
    let resp = client
        .post(&url)
        .header("Authorization", auth_header(state).await?)
        .json(&body)
        .timeout(std::time::Duration::from_secs(120))
        .send()
        .await?;
    let status = resp.status();
    if !status.is_success() {
        let detail = resp.text().await.unwrap_or_default();
        anyhow::bail!("POST /execute retornou {status}: {detail}");
    }
    Ok(())
}

/// Token interno do ambiente (mesma resolução do install.sh: env > Modules/.env).
pub async fn auth_header(state: &SharedState) -> anyhow::Result<String> {
    if let Ok(tok) = std::env::var("AVA_INTERNAL_TOKEN") {
        if !tok.trim().is_empty() {
            return Ok(format!("Bearer {}", tok.trim()));
        }
    }
    let env_file = state.project_root.join("Modules").join(".env");
    if let Ok(content) = tokio::fs::read_to_string(&env_file).await {
        for line in content.lines() {
            if let Some(rest) = line.strip_prefix("AVA_INTERNAL_TOKEN=") {
                let tok = rest.trim().trim_matches('"').trim_matches('\'');
                if !tok.is_empty() {
                    return Ok(format!("Bearer {tok}"));
                }
            }
        }
    }
    anyhow::bail!("AVA_INTERNAL_TOKEN não configurado (nem no env nem em Modules/.env)")
}

/// Cliente HTTP compartilhado (reaproveita o do estado, se exposto).
async fn http_client(state: &SharedState) -> anyhow::Result<reqwest::Client> {
    // O tray já usa reqwest para health checks — cria um client dedicado
    // (leve, sem TLS; só falamos http com o orchestrator local).
    let _ = state;
    Ok(reqwest::Client::new())
}

/// Parser de frame SSE: "id: N\nevent: name\ndata: json\n\n".
/// Retorna None se o frame não for um evento completo (ex: keepalive ":").
pub fn parse_sse_frame(frame: &str) -> Option<ConversationEvent> {
    let mut id: Option<u64> = None;
    let mut event: Option<String> = None;
    let mut data: Option<Value> = None;
    for line in frame.lines() {
        if let Some(rest) = line.strip_prefix("id: ") {
            id = rest.trim().parse().ok();
        } else if let Some(rest) = line.strip_prefix("event: ") {
            event = Some(rest.trim().to_string());
        } else if let Some(rest) = line.strip_prefix("data: ") {
            data = serde_json::from_str(rest).ok();
        }
    }
    if let (Some(id), Some(event)) = (id, event) {
        return Some(ConversationEvent {
            id,
            event,
            data: data.unwrap_or(Value::Null),
        });
    }
    None
}

/// Conecta ao SSE do Conversation Hub e repassa eventos para a UI.
/// Implementado na task de chat em main.rs (injetado via cmd_tx).
#[derive(Debug, Clone)]
pub struct ConversationEvent {
    pub id: u64,
    pub event: String,
    pub data: Value,
}

impl ConversationEvent {
    /// Texto incremental do delta (ou None se não for um delta).
    pub fn delta_text(&self) -> Option<&str> {
        if self.event == "delta" {
            return self.data.as_str();
        }
        None
    }

    /// Nome da tool chamada (evento tool_call).
    pub fn tool_name(&self) -> Option<&str> {
        if self.event == "tool_call" {
            return self.data.get("tool").and_then(Value::as_str);
        }
        None
    }

    /// Texto da pergunta (evento meta).
    pub fn input_text(&self) -> Option<&str> {
        if self.event == "meta" {
            return self.data.get("input").and_then(Value::as_str);
        }
        None
    }

    /// Indica se o turno é de voz (tts=true, evento meta). Turnos digitados
    /// (TUI/chat do Manager) chegam com tts=false.
    pub fn tts_flag(&self) -> bool {
        if self.event == "meta" {
            return self.data.get("tts").and_then(Value::as_bool).unwrap_or(false);
        }
        false
    }
}
