//! Interface gráfica com egui.
//!
//! Duas "telas" dentro da mesma janela:
//!   1. Status: mostra se llama-server e docker estão ativos/inativos,
//!      com botões para ligar/desligar.
//!   2. Seleção de modelo: lista os .gguf disponíveis (com/sem mmproj)
//!      para o usuário escolher qual carregar.

use crate::models::scan_models;
use crate::process_manager;
use crate::state::{LlamaMode, ModelInfo, ProcStatus, SharedState, VisionConfig};
use eframe::egui;

#[derive(PartialEq)]
enum Screen {
    Status,
    SelectModel,
    VisionConfig,
}

pub struct TrayApp {
    state: SharedState,
    rt: tokio::runtime::Handle,
    screen: Screen,

    // Snapshot local do estado, atualizado a cada frame (lido do Mutex async
    // via try_lock para não travar a thread de UI).
    llama_status: ProcStatus,
    llama_model: Option<String>,
    llama_mode: LlamaMode,
    docker_status: ProcStatus,

    // Cache da lista de modelos (não escaneamos o disco a cada frame).
    available_models: Vec<ModelInfo>,
    models_error: Option<String>,

    // Mensagens de feedback transitórias (ex: "Modelo iniciado com sucesso").
    feedback: Option<String>,

    // ── Configuração de visão (tela VisionConfig) ──────────────────────
    // Snapshot editável localmente; só é enviada para o AppState quando
    // o usuário clica em "Salvar".
    vision_use_main_model: bool,
    vision_dedicated_model: Option<String>,
    vision_config_error: Option<String>,
}

impl TrayApp {
    pub fn new(state: SharedState, rt: tokio::runtime::Handle) -> Self {
        Self {
            state,
            rt,
            screen: Screen::Status,
            llama_status: ProcStatus::Stopped,
            llama_model: None,
            llama_mode: LlamaMode::Text,
            docker_status: ProcStatus::Stopped,
            available_models: Vec::new(),
            models_error: None,
            feedback: None,
            vision_use_main_model: true,
            vision_dedicated_model: None,
            vision_config_error: None,
        }
    }

    /// Lê o estado atual (não-bloqueante) para refletir na UI.
    fn refresh_snapshot(&mut self) {
        if let Ok(llama) = self.state.llama.try_lock() {
            self.llama_status = llama.status.clone();
            self.llama_model = llama
                .model_path
                .as_ref()
                .and_then(|p| std::path::Path::new(p).file_stem())
                .map(|s| s.to_string_lossy().to_string());
            self.llama_mode = llama.mode;
        }
        if let Ok(docker) = self.state.docker.try_lock() {
            self.docker_status = docker.status.clone();
        }
    }

    fn reload_models(&mut self) {
        match scan_models(&self.state.models_dir) {
            Ok(models) => {
                self.available_models = models;
                self.models_error = None;
            }
            Err(e) => {
                self.models_error = Some(e.to_string());
            }
        }
    }

    /// Carrega a configuração de visão atual do AppState para os campos
    /// locais editáveis da tela. Chamado ao entrar na tela VisionConfig.
    fn load_vision_config(&mut self) {
        if let Ok(cfg) = self.state.vision_config.try_lock() {
            self.vision_use_main_model = cfg.use_main_model;
            self.vision_dedicated_model = cfg.dedicated_model_path.clone();
        }
        self.vision_config_error = None;
        self.reload_models();
    }
}

impl eframe::App for TrayApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.refresh_snapshot();

        // Repaint periódico para refletir mudanças de estado vindas da API
        // (ex: outro processo chamando /llama/stop) mesmo sem interação do usuário.
        ctx.request_repaint_after(std::time::Duration::from_millis(800));

        egui::CentralPanel::default().show(ctx, |ui| match self.screen {
            Screen::Status => self.draw_status_screen(ui),
            Screen::SelectModel => self.draw_select_model_screen(ui),
            Screen::VisionConfig => self.draw_vision_config_screen(ui),
        });
    }
}

impl TrayApp {
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
                status_dot(ui, &self.llama_status);
                ui.vertical(|ui| {
                    ui.strong("llama-server");
                    ui.label(self.llama_status.label());
                    if let Some(model) = &self.llama_model {
                        ui.label(egui::RichText::new(model).weak().small());
                    }
                    if self.llama_status == ProcStatus::Running {
                        let color = match self.llama_mode {
                            LlamaMode::Multimodal => egui::Color32::from_rgb(120, 170, 230),
                            LlamaMode::Text => egui::Color32::from_gray(160),
                        };
                        ui.colored_label(color, self.llama_mode.label());
                    }
                });
            });

            ui.add_space(6.0);
            ui.horizontal(|ui| {
                let running = self.llama_status == ProcStatus::Running;

                if ui
                    .add_enabled(!running, egui::Button::new("Selecionar e iniciar"))
                    .clicked()
                {
                    self.reload_models();
                    self.screen = Screen::SelectModel;
                }

                if ui
                    .add_enabled(running, egui::Button::new("Desligar"))
                    .clicked()
                {
                    self.spawn_stop_llama();
                }
            });

            // ── Troca rápida de modo (texto <-> multimodal) ────────────────
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                let is_text = self.llama_mode == LlamaMode::Text;
                let is_multimodal = self.llama_mode == LlamaMode::Multimodal;

                if ui
                    .add_enabled(!is_text, egui::Button::new("Modo texto"))
                    .clicked()
                {
                    self.spawn_switch_mode(LlamaMode::Text);
                }
                if ui
                    .add_enabled(!is_multimodal, egui::Button::new("Modo multimodal"))
                    .clicked()
                {
                    self.spawn_switch_mode(LlamaMode::Multimodal);
                }
                if ui.button("⚙ Configurar visão").clicked() {
                    self.load_vision_config();
                    self.screen = Screen::VisionConfig;
                }
            });
        });

        ui.add_space(14.0);

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

        ui.add_space(16.0);
        ui.separator();
        ui.add_space(8.0);
        ui.label(
            egui::RichText::new("API REST ativa em http://localhost:9001")
                .weak()
                .small(),
        );
        ui.label(
            egui::RichText::new("llama-server cai após 15 min sem uso · docker após 45 min")
                .weak()
                .small(),
        );
    }

    fn draw_select_model_screen(&mut self, ui: &mut egui::Ui) {
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            if ui.button("← Voltar").clicked() {
                self.screen = Screen::Status;
            }
            ui.heading("Selecionar modelo");
        });
        ui.add_space(10.0);

        if let Some(err) = &self.models_error {
            ui.colored_label(egui::Color32::from_rgb(220, 100, 100), err);
            return;
        }

        if self.available_models.is_empty() {
            ui.label("Nenhum modelo .gguf encontrado em ./Models.");
            return;
        }

        egui::ScrollArea::vertical().show(ui, |ui| {
            // Clona a lista para não brigar com &mut self dentro do closure.
            let models = self.available_models.clone();
            for model in &models {
                egui::Frame::group(ui.style()).show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.horizontal(|ui| {
                        ui.vertical(|ui| {
                            ui.strong(&model.name);
                            ui.horizontal(|ui| {
                                if model.is_multimodal {
                                    ui.colored_label(
                                        egui::Color32::from_rgb(120, 170, 230),
                                        "multimodal (mmproj)",
                                    );
                                } else {
                                    ui.label(
                                        egui::RichText::new("somente texto").weak(),
                                    );
                                }
                                ui.label(
                                    egui::RichText::new(format!("{} MB", model.size_mb))
                                        .weak()
                                        .small(),
                                );
                            });
                        });

                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.button("Iniciar").clicked() {
                                self.spawn_start_llama(model.clone());
                                self.screen = Screen::Status;
                            }
                        });
                    });
                });
                ui.add_space(6.0);
            }
        });
    }

    /// Tela de configuração: escolhe se o modo multimodal usa o mmproj do
    /// próprio modelo principal, ou um modelo dedicado separado escolhido
    /// dentre os .gguf encontrados em ./Models.
    fn draw_vision_config_screen(&mut self, ui: &mut egui::Ui) {
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            if ui.button("← Voltar").clicked() {
                self.screen = Screen::Status;
            }
            ui.heading("Configuração de visão");
        });
        ui.add_space(10.0);

        ui.label(
            egui::RichText::new(
                "Define qual modelo é carregado quando o modo multimodal é ativado \
                 (pelo botão \"Modo multimodal\" ou via API POST /llama/switch_mode).",
            )
            .weak()
            .small(),
        );
        ui.add_space(10.0);

        if let Some(err) = &self.vision_config_error {
            ui.colored_label(egui::Color32::from_rgb(220, 100, 100), err);
            ui.add_space(8.0);
        }

        ui.radio_value(
            &mut self.vision_use_main_model,
            true,
            "Usar o modelo principal como multimodal (mmproj próprio)",
        );
        ui.radio_value(
            &mut self.vision_use_main_model,
            false,
            "Usar um modelo dedicado diferente",
        );

        ui.add_space(10.0);

        if !self.vision_use_main_model {
            egui::Frame::group(ui.style()).show(ui, |ui| {
                ui.set_width(ui.available_width());

                if self.models_error.is_some() {
                    ui.colored_label(
                        egui::Color32::from_rgb(220, 100, 100),
                        self.models_error.as_deref().unwrap_or(""),
                    );
                    return;
                }

                let multimodal_models: Vec<&ModelInfo> = self
                    .available_models
                    .iter()
                    .filter(|m| m.is_multimodal)
                    .collect();

                if multimodal_models.is_empty() {
                    ui.label(
                        "Nenhum modelo multimodal (com mmproj) encontrado em ./Models.",
                    );
                    return;
                }

                egui::ScrollArea::vertical()
                    .max_height(220.0)
                    .show(ui, |ui| {
                        for model in &multimodal_models {
                            let selected =
                                self.vision_dedicated_model.as_deref() == Some(model.path.as_str());
                            ui.horizontal(|ui| {
                                if ui.radio(selected, &model.name).clicked() {
                                    self.vision_dedicated_model = Some(model.path.clone());
                                }
                                ui.label(
                                    egui::RichText::new(format!("{} MB", model.size_mb))
                                        .weak()
                                        .small(),
                                );
                            });
                        }
                    });
            });
        }

        ui.add_space(14.0);
        ui.horizontal(|ui| {
            if ui.button("Salvar").clicked() {
                self.spawn_save_vision_config();
            }
        });
    }

    // ── Disparo de ações assíncronas a partir da UI síncrona ──────────────
    // egui roda em uma thread síncrona; para chamar funções async do
    // process_manager, usamos o handle do runtime tokio guardado em `self.rt`.

    fn spawn_start_llama(&self, model: ModelInfo) {
        let state = self.state.clone();
        self.rt.spawn(async move {
            let mmproj = model.mmproj_path.as_deref();
            let mtp_draft = model.mtp_draft_path.as_deref();
            if let Err(e) = process_manager::start_llama(&state, &model.path, mmproj, mtp_draft, state.llama_log.clone()).await {
                tracing::error!("Erro ao iniciar llama-server: {e}");
            }
        });
    }

    fn spawn_stop_llama(&self) {
        let state = self.state.clone();
        self.rt.spawn(async move {
            if let Err(e) = process_manager::stop_llama(&state).await {
                tracing::error!("Erro ao parar llama-server: {e}");
            }
        });
    }

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

    /// Troca o modo do llama-server (texto <-> multimodal) usando a
    /// configuração de visão atualmente salva no AppState.
    fn spawn_switch_mode(&self, mode: LlamaMode) {
        let state = self.state.clone();
        self.rt.spawn(async move {
            if let Err(e) = process_manager::switch_llama_mode(&state, mode).await {
                tracing::error!("Erro ao trocar modo do llama-server: {e}");
            }
        });
    }

    /// Salva a configuração de visão escolhida na tela VisionConfig.
    /// A validação (mmproj presente, etc.) acontece em
    /// `process_manager::set_vision_config`; assim como as demais ações
    /// `spawn_*` desta tela, falhas são apenas logadas via `tracing`
    /// (mesmo padrão de spawn_start_llama/spawn_start_docker acima).
    fn spawn_save_vision_config(&mut self) {
        let state = self.state.clone();
        let cfg = VisionConfig {
            use_main_model: self.vision_use_main_model,
            dedicated_model_path: self.vision_dedicated_model.clone(),
        };
        self.vision_config_error = None;
        self.rt.spawn(async move {
            if let Err(e) = process_manager::set_vision_config(&state, cfg).await {
                tracing::error!("Erro ao salvar configuração de visão: {e}");
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