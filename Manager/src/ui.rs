//! Interface gráfica com egui.
//!
//! Duas "telas" dentro da mesma janela:
//!   1. Status: mostra se llama-server e docker estão ativos/inativos,
//!      com botões para ligar/desligar.
//!   2. Seleção de modelo: lista os .gguf disponíveis (com/sem mmproj)
//!      para o usuário escolher qual carregar.

use crate::models::scan_models;
use crate::process_manager;
use crate::state::{ModelInfo, ProcStatus, SharedState};
use eframe::egui;

#[derive(PartialEq)]
enum Screen {
    Status,
    SelectModel,
}

pub struct TrayApp {
    state: SharedState,
    rt: tokio::runtime::Handle,
    screen: Screen,

    // Snapshot local do estado, atualizado a cada frame (lido do Mutex async
    // via try_lock para não travar a thread de UI).
    llama_status: ProcStatus,
    llama_model: Option<String>,
    docker_status: ProcStatus,

    // Cache da lista de modelos (não escaneamos o disco a cada frame).
    available_models: Vec<ModelInfo>,
    models_error: Option<String>,

    // Mensagens de feedback transitórias (ex: "Modelo iniciado com sucesso").
    feedback: Option<String>,
}

impl TrayApp {
    pub fn new(state: SharedState, rt: tokio::runtime::Handle) -> Self {
        Self {
            state,
            rt,
            screen: Screen::Status,
            llama_status: ProcStatus::Stopped,
            llama_model: None,
            docker_status: ProcStatus::Stopped,
            available_models: Vec::new(),
            models_error: None,
            feedback: None,
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

    // ── Disparo de ações assíncronas a partir da UI síncrona ──────────────
    // egui roda em uma thread síncrona; para chamar funções async do
    // process_manager, usamos o handle do runtime tokio guardado em `self.rt`.

    fn spawn_start_llama(&self, model: ModelInfo) {
        let state = self.state.clone();
        self.rt.spawn(async move {
            let mmproj = model.mmproj_path.as_deref();
            if let Err(e) = process_manager::start_llama(&state, &model.path, mmproj, state.llama_log.clone()).await {
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