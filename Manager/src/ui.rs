// Alpha AI (AVA) — Interface gráfica com egui
// Copyright (C) 2026  Felipe lumazini
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as
// published by the Free Software Foundation, either version 3 of the
// License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.
//! Interface gráfica com egui.
//!
//! Tela única de status: mostra se o docker está ativo/inativo,
//! com botões para ligar/desligar.
//!
//! Comportamento de tray: fechar a janela (botão X do window manager)
//! NÃO encerra o app — só esconde a janela; o processo e o ícone da
//! bandeja continuam vivos. O encerramento real acontece só pelo menu
//! de contexto do tray ("Sair"), que envia UiCommand::Quit pelo canal
//! mpsc (a thread do tray é dona do ícone; ver main.rs).

use crate::mcp_manager;
use crate::process_manager;
use crate::state::{McpStatus, ProcStatus, SharedState, TranscriptStatus};
use crate::transcript;
use crate::UiCommand;
use eframe::egui;
use tokio::sync::mpsc;

// ── Chat do Manager (turnos voice/chat espelhados do orchestrator) ──────────

/// Tela ativa da janela do tray.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Screen {
    Status,
    Chat,
}

/// Papel de quem falou na bolha do chat.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatRole {
    User,
    Assistant,
    Tool,
}

/// Uma linha de conversa na tela de chat.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub text: String,
    /// true enquanto a resposta/turno está em andamento (sem `done` ainda).
    pub in_progress: bool,
    /// true para a tool que está sendo chamada no momento (sem step_done).
    pub tool_pending: bool,
}

pub struct TrayApp {
    state: SharedState,
    rt: tokio::runtime::Handle,
    /// Comandos vindos da thread do tray (GTK).
    cmd_rx: mpsc::UnboundedReceiver<UiCommand>,
    /// Setado pelo UiCommand::Quit — o próximo close da janela NÃO é
    /// cancelado e o run_native de fato retorna, encerrando o processo.
    quit_requested: bool,

    // Snapshot local do estado, atualizado a cada frame (lido do Mutex async
    // via try_lock para não travar a thread de UI).
    docker_status: ProcStatus,
    mcp_status: McpStatus,
    mcp_installed: bool,
    transcript_status: TranscriptStatus,
    transcript_installed: bool,

    // Mensagens de feedback transitórias (ex: "Docker iniciado com sucesso").
    feedback: Option<String>,

    // ── Tela de chat ──────────────────────────────────────────────────────
    /// Tela atual (Status ou Chat).
    screen: Screen,
    /// Histórico da sessão — SÓ do tempo em que a tela de chat ficou aberta.
    /// Escrevemos aqui tanto perguntas digitadas quanto turnos de voz que
    /// chegam do hub enquanto a tela estava aberta.
    chat_history: Vec<ChatMessage>,
    /// Buffer do campo de texto ("pergunte algo…" pequeno).
    chat_input: String,
    /// true enquanto um turno enviado pela tela (ou por voz) está em andamento.
    chat_running: bool,
    /// id do último evento lido do hub (para não reprocessar no repaint).
    chat_last_event_id: u64,
    /// Pergunta pendente de envio (setada pelo Enter, consumida pelo task tokio).
    chat_pending_submit: Option<String>,
    /// Rastreado localmente (ViewportInfo não expõe `visible` nesta versão):
    /// false quando escondemos a janela no close (X) — usado para limpar o
    /// histórico do chat conforme o requisito de sessão efêmera.
    window_visible: bool,
}

impl TrayApp {
    pub fn new(
        state: SharedState,
        rt: tokio::runtime::Handle,
        cmd_rx: mpsc::UnboundedReceiver<UiCommand>,
    ) -> Self {
        Self {
            state,
            rt,
            cmd_rx,
            quit_requested: false,
            docker_status: ProcStatus::Stopped,
            mcp_status: McpStatus::NotInstalled,
            mcp_installed: false,
            transcript_status: TranscriptStatus::NotInstalled,
            transcript_installed: false,
            feedback: None,
            screen: Screen::Status,
            chat_history: Vec::new(),
            chat_input: String::new(),
            chat_running: false,
            chat_last_event_id: 0,
            chat_pending_submit: None,
            window_visible: true,
        }
    }

    /// Lê o estado atual (não-bloqueante) para refletir na UI.
    fn refresh_snapshot(&mut self) {
        if let Ok(docker) = self.state.docker.try_lock() {
            self.docker_status = docker.status.clone();
        }
        if let Ok(mcp) = self.state.mcp.try_lock() {
            self.mcp_status = mcp.status.clone();
            self.mcp_installed = mcp.binary.exists();
        }
        if let Ok(t) = self.state.transcript.try_lock() {
            self.transcript_status = t.status.clone();
            self.transcript_installed = t.binary.exists();
        }
    }

    /// Processa os comandos pendentes da thread do tray.
    fn handle_commands(&mut self, ctx: &egui::Context) {
        while let Ok(cmd) = self.cmd_rx.try_recv() {
            tracing::debug!("UiCommand recebido: {cmd:?}");
            match cmd {
                UiCommand::Show => {
                    self.window_visible = true;
                    ctx.send_viewport_cmd(egui::ViewportCommand::Minimized(false));
                    ctx.send_viewport_cmd(egui::ViewportCommand::Visible(true));
                    ctx.send_viewport_cmd(egui::ViewportCommand::Focus);
                }
                UiCommand::ShowChat => {
                    // Wake word detectada (POST /ui/chat-open do transcribe):
                    // abre/restaura a janela JÁ na tela de chat, antes mesmo
                    // da transcrição terminar (o turno chega depois pelo hub).
                    self.window_visible = true;
                    self.screen = Screen::Chat;
                    ctx.send_viewport_cmd(egui::ViewportCommand::Minimized(false));
                    ctx.send_viewport_cmd(egui::ViewportCommand::Visible(true));
                    ctx.send_viewport_cmd(egui::ViewportCommand::Focus);
                }
                UiCommand::Quit => {
                    // Menu "Sair" com a janela oculta: nada dispara um close
                    // natural, então a própria UI pede o fechamento — sem
                    // CancelClose no handle_close_request, o run_native
                    // encerra o processo.
                    self.quit_requested = true;
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
                UiCommand::ChatEvent(ev) => self.handle_chat_event(ctx, ev),
            }
        }
    }

    /// Consome um evento do Conversation Hub e atualiza o histórico do chat.
    /// Também faz a troca automática de tela: um novo turno (`meta`) abre a
    /// aba de chat e traz a janela para frente (comportamento pedido).
    fn handle_chat_event(&mut self, ctx: &egui::Context, ev: crate::chat::ConversationEvent) {
        // Repaint imediato para o token aparecer sem esperar o ciclo de 800ms.
        ctx.request_repaint();

        match ev.event.as_str() {
            "meta" => {
                // Novo turno: registra a pergunta no histórico. A troca
                // automática de tela + foco da janela acontece SÓ para turnos
                // do transcribe (source="voice" — o /ui/chat-open já abre na
                // wake word; isso aqui é o fallback caso esse POST falhe).
                // Turnos digitados na TUI (source=tui) ou no próprio chat do
                // Manager (source=chat) chegam NUNCA abrem a tela — só
                // atualizam o histórico em segundo plano.
                let should_open = ev.source_name() == Some("voice");
                let changed_screen = self.screen != Screen::Chat;
                if let Some(input) = ev.input_text() {
                    self.chat_history.push(ChatMessage {
                        role: ChatRole::User,
                        text: input.to_string(),
                        in_progress: false,
                        tool_pending: false,
                    });
                    // Bolha da resposta entra vazia e em progresso.
                    self.chat_history.push(ChatMessage {
                        role: ChatRole::Assistant,
                        text: String::new(),
                        in_progress: true,
                        tool_pending: false,
                    });
                    self.chat_running = true;
                }
                // Traz a janela para frente SÓ quando um novo turno do
                // transcribe começa (source=voice). Para turnos de texto a
                // tela do chat é atualizada silenciosamente em segundo plano.
                if should_open && changed_screen {
                    tracing::info!("ui: novo turno do transcribe (source=voice) — abrindo tela de chat");
                    self.screen = Screen::Chat;
                    self.window_visible = true;
                    ctx.send_viewport_cmd(egui::ViewportCommand::Minimized(false));
                    ctx.send_viewport_cmd(egui::ViewportCommand::Visible(true));
                    ctx.send_viewport_cmd(egui::ViewportCommand::Focus);
                }
            }
            "delta" => {
                if let Some(piece) = ev.delta_text() {
                    // Anexa à última bolha de assistente EM PROGRESSO (busca
                    // de trás para frente: após um tool_call a última do
                    // histórico é a da tool — os deltas pertencem à resposta
                    // do modelo, não à tool). Sem assistente em progresso
                    // (resposta pós-tool), cria uma bolha nova no fim.
                    match self
                        .chat_history
                        .iter_mut()
                        .rev()
                        .find(|m| m.role == ChatRole::Assistant && m.in_progress)
                    {
                        Some(a) => a.text.push_str(piece),
                        None if !piece.is_empty() => {
                            self.chat_history.push(ChatMessage {
                                role: ChatRole::Assistant,
                                text: piece.to_string(),
                                in_progress: true,
                                tool_pending: false,
                            });
                        }
                        None => {}
                    }
                }
            }
            "tool_call" => {
                if let Some(tool) = ev.tool_name() {
                    // Fecha a bolha do assistente em progresso: o preâmbulo
                    // termina aqui; o texto que vier depois dos tools vira
                    // bolha nova (tratado pelo handler de delta acima).
                    for msg in self.chat_history.iter_mut() {
                        if msg.role == ChatRole::Assistant {
                            msg.in_progress = false;
                        }
                    }
                    self.chat_history.push(ChatMessage {
                        role: ChatRole::Tool,
                        text: format!("chamando {tool}…"),
                        in_progress: true,
                        tool_pending: true,
                    });
                }
            }
            "step_done" => {
                // Fecha a tool pendente mais recente com sucesso/falha.
                let success = ev
                    .data
                    .get("success")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false);
                for msg in self.chat_history.iter_mut().rev() {
                    if msg.role == ChatRole::Tool && msg.tool_pending {
                        msg.tool_pending = false;
                        msg.in_progress = false;
                        msg.text = format!(
                            "{} {}",
                            if success { "✓" } else { "✗" },
                            msg.text.trim_end_matches('…').trim()
                        );
                        break;
                    }
                }
            }
            "done" => {
                // Finaliza o turno: cola a resposta final (cura o texto visível
                // com final_response) e fecha as bolhas em progresso.
                if let Some(final_response) =
                    ev.data.get("final_response").and_then(serde_json::Value::as_str)
                {
                    // Cura a última bolha de assistente em progresso (não é
                    // necessariamente a última do histórico após um tool_call).
                    match self
                        .chat_history
                        .iter_mut()
                        .rev()
                        .find(|m| m.role == ChatRole::Assistant && m.in_progress)
                    {
                        Some(a) => a.text = final_response.to_string(),
                        // Turno terminou sem delta final (ex.: force_finish):
                        // cria a bolha com a resposta pronta.
                        None if !final_response.is_empty() => {
                            self.chat_history.push(ChatMessage {
                                role: ChatRole::Assistant,
                                text: final_response.to_string(),
                                in_progress: false,
                                tool_pending: false,
                            });
                        }
                        None => {}
                    }
                }
                for msg in self.chat_history.iter_mut() {
                    msg.in_progress = false;
                }
                self.chat_running = false;
            }
            "error" => {
                let err_text = ev
                    .data
                    .get("error")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("erro desconhecido");
                // Mesma busca do done: a bolha do assistente pode não ser a
                // última do histórico (tools podem ter vindo depois dela).
                if let Some(a) = self
                    .chat_history
                    .iter_mut()
                    .rev()
                    .find(|m| m.role == ChatRole::Assistant && m.in_progress)
                {
                    a.in_progress = false;
                    a.text = format!("✗ erro: {err_text}");
                } else {
                    self.chat_history.push(ChatMessage {
                        role: ChatRole::Assistant,
                        text: format!("✗ erro: {err_text}"),
                        in_progress: false,
                        tool_pending: false,
                    });
                }
                self.chat_running = false;
            }
            _ => {
                // eventos não usados na UI (audio, audio_start, audio_end, result)
                // são ignorados — a resposta falada é papel do pipeline de voz.
            }
        }
    }

    /// Intercepta o fechamento da janela (X do window manager): sem
    /// `quit_requested`, cancela o close e apenas esconde a janela — o app
    /// continua na bandeja. Com `quit_requested` (menu "Sair" do tray), NÃO
    /// envia CancelClose e o `run_native` de fato retorna, encerrando o
    /// processo de forma limpa.
    fn handle_close_request(&mut self, ctx: &egui::Context) {
        let close_requested = ctx.input(|i| i.viewport().close_requested());
        if close_requested && !self.quit_requested {
            tracing::debug!("handle_close_request: escondendo janela");
            self.window_visible = false;
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
            // Visible(false) sozinho não esconde neste ambiente; minimizar
            // é respeitado pelo gerenciador de janelas.
            ctx.send_viewport_cmd(egui::ViewportCommand::Minimized(true));
            ctx.send_viewport_cmd(egui::ViewportCommand::Visible(false));
        }
    }
}

impl eframe::App for TrayApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.refresh_snapshot();
        self.handle_commands(ctx);
        self.handle_close_request(ctx);

        // Histórico do chat só existe enquanto a tela está visível:
        // ao esconder a janela (Minimized/Visible(false) no handle_close_request)
        // limpamos a sessão — as perguntas enviadas DURANTE a visibilidade é o
        // que o usuário pediu para manter; ao reabrir, recomeçamos do zero.
        let window_hidden = !self.window_visible;
        if window_hidden && self.screen == Screen::Chat && !self.chat_history.is_empty() {
            tracing::debug!("ui: janela oculta — limpando histórico de chat da sessão");
            self.chat_history.clear();
        }

        // Repaint periódico para refletir mudanças de estado vindas da API
        // (ex: outro processo chamando /docker/stop) mesmo sem interação do usuário.
        ctx.request_repaint_after(std::time::Duration::from_millis(800));

        egui::CentralPanel::default().show(ctx, |ui| {
            // ── Barra superior de troca de tela ────────────────────────────
            ui.horizontal(|ui| {
                let status_label = egui::RichText::new("Status");
                let chat_label = egui::RichText::new("Chat");

                if ui
                    .selectable_label(self.screen == Screen::Status, status_label)
                    .clicked()
                {
                    self.screen = Screen::Status;
                }
                ui.separator();
                if ui
                    .selectable_label(self.screen == Screen::Chat, chat_label)
                    .clicked()
                {
                    self.screen = Screen::Chat;
                }
            });
            ui.add_space(4.0);
            ui.separator();
            ui.add_space(4.0);

            match self.screen {
                Screen::Status => self.draw_status_screen(ui),
                Screen::Chat => self.draw_chat_screen(ui),
            }
        });

        // Dispara o envio pendente (fora do closure de paint — precisa de &mut self total).
        self.dispatch_pending_submit();
    }
}

impl TrayApp {
    // ── Chat (Manager ↔ orchestrator) ─────────────────────────────────────

    /// Consome o buffer de envio pendente e dispara o turno no orchestrator.
    /// Chamado FORA do closure de paint (precisa de &mut self total).
    fn dispatch_pending_submit(&mut self) {
        if let Some(text) = self.chat_pending_submit.take() {
            let text = text.trim().to_string();
            if text.is_empty() {
                return;
            }
            // Bolha do usuário entra imediatamente (feedback de que foi enviado).
            self.chat_history.push(ChatMessage {
                role: ChatRole::User,
                text: text.clone(),
                in_progress: false,
                tool_pending: false,
            });
            // Bolha da Alpha entra vazia e em progresso (tokens chegam em deltas).
            self.chat_history.push(ChatMessage {
                role: ChatRole::Assistant,
                text: String::new(),
                in_progress: true,
                tool_pending: false,
            });
            self.chat_running = true;

            // Envio assíncrono via runtime tokio: POST /execute com
            // source="chat". O turno real chega pelos eventos espelhados
            // no Conversation Hub (ver chat_task em main.rs / chat.rs).
            let state = self.state.clone();
            self.rt.spawn(async move {
                if let Err(e) = crate::chat::send_chat_turn(&state, &text).await {
                    tracing::error!("ui: falha ao enviar turno de chat: {e}");
                }
            });
        }
    }

    /// Renderiza a tela de chat (simplificada — turnos voice + campo pequeno).
    fn draw_chat_screen(&mut self, ui: &mut egui::Ui) {
        ui.add_space(8.0);
        ui.heading("AVA — Chat");
        ui.add_space(8.0);

        // Área de scroll com o histórico.
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .stick_to_bottom(true)
            .show(ui, |ui| {
                if self.chat_history.is_empty() {
                    ui.label(
                        egui::RichText::new(
                            "Fale com a Alpha pelo microfone (pipeline de voz) ou digite abaixo.\n\
                             O histórico desta sessão aparece aqui e é limpo quando a janela fecha.",
                        )
                        .weak()
                        .small(),
                    );
                }
                for msg in &self.chat_history {
                    match msg.role {
                        ChatRole::User => {
                            ui.add_space(4.0);
                            egui::Frame::group(ui.style()).show(ui, |ui| {
                                ui.set_width(ui.available_width() * 0.8);
                                ui.label(egui::RichText::new("Você").strong().small());
                                ui.label(&msg.text);
                            });
                        }
                        ChatRole::Assistant => {
                            ui.add_space(4.0);
                            egui::Frame::group(ui.style()).show(ui, |ui| {
                                ui.set_width(ui.available_width());
                                ui.label(egui::RichText::new("Alpha").strong().small());
                                if msg.text.is_empty() {
                                    ui.label(
                                        egui::RichText::new("gerando resposta...")
                                            .weak()
                                            .small(),
                                    );
                                } else {
                                    ui.label(&msg.text);
                                }
                                if msg.in_progress {
                                    ui.label(egui::RichText::new("… (em andamento)").weak().small());
                                }
                            });
                        }
                        ChatRole::Tool => {
                            ui.label(egui::RichText::new(format!("🔧 {}", msg.text)).weak().small());
                        }
                    }
                }
                ui.add_space(8.0);
            });

        ui.add_space(4.0);
        ui.separator();

        // ── Campo pequeno de envio de texto ────────────────────────────────
        ui.horizontal(|ui| {
            let send_clicked = ui.button("Enviar").clicked();
            let input = egui::TextEdit::singleline(&mut self.chat_input)
                .hint_text("Pergunte algo… (Enter para enviar)")
                .desired_width(ui.available_width());
            let response = ui.add(input);
            let enter_pressed =
                response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
            if enter_pressed || send_clicked {
                let text = self.chat_input.clone();
                self.chat_input.clear();
                self.chat_pending_submit = Some(text);
                response.request_focus();
            }
        });
    }

    fn draw_status_screen(&mut self, ui: &mut egui::Ui) {
        ui.add_space(8.0);
        ui.heading("AVA — Gerenciador de Processos");
        ui.add_space(12.0);

        if let Some(msg) = &self.feedback {
            ui.colored_label(egui::Color32::from_rgb(120, 200, 120), msg);
            ui.add_space(8.0);
        }

        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                status_dot(ui, &self.docker_status);
                ui.vertical(|ui| {
                    ui.strong("Docker (perfil vulkan)");
                    ui.label(self.docker_status.label());
                });
            });

            ui.add_space(6.0);
            ui.horizontal(|ui| {
                let running = self.docker_status == ProcStatus::Running;

                if ui
                    .add_enabled(!running, egui::Button::new("Ligar"))
                    .clicked()
                {
                    self.spawn_start_docker();
                }

                if ui
                    .add_enabled(running, egui::Button::new("Desligar"))
                    .clicked()
                {
                    self.spawn_stop_docker();
                }
            });
        });

        ui.add_space(8.0);
        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                mcp_status_dot(ui, &self.mcp_status);
                ui.vertical(|ui| {
                    ui.strong("MCP Manager (host)");
                    ui.label(self.mcp_status.label());
                });
            });

            ui.add_space(6.0);
            ui.horizontal(|ui| {
                let running = self.mcp_status == McpStatus::Running;

                if ui
                    .add_enabled(self.mcp_installed && !running, egui::Button::new("Ligar"))
                    .clicked()
                {
                    self.spawn_start_mcp();
                }

                if ui
                    .add_enabled(self.mcp_installed && running, egui::Button::new("Desligar"))
                    .clicked()
                {
                    self.spawn_stop_mcp();
                }
            });

            if !self.mcp_installed {
                ui.label(
                    egui::RichText::new(
                        "executável não encontrado — coloque mcp_manager (port Rust) \
                         na raiz do projeto ou defina AVA_MCP_MANAGER_BIN",
                    )
                    .weak()
                    .small(),
                );
            }
        });

        ui.add_space(8.0);
        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                transcript_status_dot(ui, &self.transcript_status);
                ui.vertical(|ui| {
                    ui.strong("Transcript (voz)");
                    ui.label(self.transcript_status.label());
                });
            });

            ui.add_space(6.0);
            ui.horizontal(|ui| {
                let running = self.transcript_status == TranscriptStatus::Running;

                if ui
                    .add_enabled(
                        self.transcript_installed && !running,
                        egui::Button::new("Ligar"),
                    )
                    .clicked()
                {
                    self.spawn_start_transcript();
                }

                if ui
                    .add_enabled(
                        self.transcript_installed && running,
                        egui::Button::new("Desligar"),
                    )
                    .clicked()
                {
                    self.spawn_stop_transcript();
                }
            });

            if !self.transcript_installed {
                ui.label(
                    egui::RichText::new(
                        "executável não encontrado — compile o pipeline de voz \
                         (Modules/transcribe) ou defina AVA_TRANSCRIPT_BIN",
                    )
                    .weak()
                    .small(),
                );
            }
        });

        ui.add_space(16.0);
        ui.separator();
        ui.add_space(8.0);
        ui.label(
            egui::RichText::new("API REST ativa em http://localhost:9001")
                .weak()
                .small(),
        );
        ui.label(
            egui::RichText::new("docker cai após 45 min sem uso")
                .weak()
                .small(),
        );
    }

    // ── Disparo de ações assíncronas a partir da UI síncrona ──────────────
    // egui roda em uma thread síncrona; para chamar funções async do
    // process_manager, usamos o handle do runtime tokio guardado em `self.rt`.

    fn spawn_start_docker(&self) {
        let state = self.state.clone();
        self.rt.spawn(async move {
            if let Err(e) = process_manager::start_docker(&state).await {
                tracing::error!("Erro ao iniciar docker: {e}");
            }
        });
    }

    fn spawn_stop_docker(&self) {
        let state = self.state.clone();
        self.rt.spawn(async move {
            if let Err(e) = process_manager::stop_docker(&state).await {
                tracing::error!("Erro ao parar docker: {e}");
            }
        });
    }

    fn spawn_start_mcp(&self) {
        let state = self.state.clone();
        self.rt.spawn(async move {
            match mcp_manager::start(&state).await {
                Ok(()) => tracing::info!("MCP manager iniciado (UI do tray)."),
                Err(e) => tracing::error!("Erro ao iniciar MCP manager: {e:#}"),
            }
        });
    }

    fn spawn_stop_mcp(&self) {
        let state = self.state.clone();
        self.rt.spawn(async move {
            if let Err(e) = mcp_manager::stop(&state).await {
                tracing::error!("Erro ao parar MCP manager: {e:#}");
            }
        });
    }

    fn spawn_start_transcript(&self) {
        let state = self.state.clone();
        self.rt.spawn(async move {
            match transcript::start(&state).await {
                Ok(()) => tracing::info!("Transcript iniciado (UI do tray)."),
                Err(e) => tracing::error!("Erro ao iniciar transcript: {e:#}"),
            }
        });
    }

    fn spawn_stop_transcript(&self) {
        let state = self.state.clone();
        self.rt.spawn(async move {
            match transcript::stop(&state).await {
                Ok(()) => tracing::info!("Transcript encerrado (UI do tray)."),
                Err(e) => tracing::error!("Erro ao parar transcript: {e:#}"),
            }
        });
    }
}

fn status_dot(ui: &mut egui::Ui, status: &ProcStatus) {
    let color = match status {
        ProcStatus::Running => egui::Color32::from_rgb(90, 200, 100),
        ProcStatus::Stopped => egui::Color32::from_rgb(150, 150, 150),
        ProcStatus::Starting | ProcStatus::Stopping => egui::Color32::from_rgb(230, 180, 60),
    };
    let (rect, _) = ui.allocate_exact_size(egui::vec2(12.0, 12.0), egui::Sense::hover());
    ui.painter().circle_filled(rect.center(), 5.0, color);
    ui.add_space(6.0);
}

fn mcp_status_dot(ui: &mut egui::Ui, status: &McpStatus) {
    let color = match status {
        McpStatus::Running => egui::Color32::from_rgb(90, 200, 100),
        McpStatus::Stopped | McpStatus::NotInstalled => egui::Color32::from_rgb(150, 150, 150),
        McpStatus::Starting | McpStatus::Stopping => egui::Color32::from_rgb(230, 180, 60),
        McpStatus::Crashed => egui::Color32::from_rgb(224, 80, 60),
    };
    let (rect, _) = ui.allocate_exact_size(egui::vec2(12.0, 12.0), egui::Sense::hover());
    ui.painter().circle_filled(rect.center(), 5.0, color);
    ui.add_space(6.0);
}

fn transcript_status_dot(ui: &mut egui::Ui, status: &TranscriptStatus) {
    let color = match status {
        TranscriptStatus::Running => egui::Color32::from_rgb(90, 200, 100),
        TranscriptStatus::Stopped | TranscriptStatus::NotInstalled => {
            egui::Color32::from_rgb(150, 150, 150)
        }
        TranscriptStatus::Starting | TranscriptStatus::Stopping => {
            egui::Color32::from_rgb(230, 180, 60)
        }
        TranscriptStatus::Crashed => egui::Color32::from_rgb(224, 80, 60),
    };
    let (rect, _) = ui.allocate_exact_size(egui::vec2(12.0, 12.0), egui::Sense::hover());
    ui.painter().circle_filled(rect.center(), 5.0, color);
    ui.add_space(6.0);
}