//! Interface gráfica com egui.
//!
//! Tela única de status: mostra se o docker está ativo/inativo,
//! com botões para ligar/desligar.

use crate::process_manager;
use crate::state::{ProcStatus, SharedState};
use eframe::egui;

pub struct TrayApp {
    state: SharedState,
    rt: tokio::runtime::Handle,

    // Snapshot local do estado, atualizado a cada frame (lido do Mutex async
    // via try_lock para não travar a thread de UI).
    docker_status: ProcStatus,

    // Mensagens de feedback transitórias (ex: "Docker iniciado com sucesso").
    feedback: Option<String>,
}

impl TrayApp {
    pub fn new(state: SharedState, rt: tokio::runtime::Handle) -> Self {
        Self {
            state,
            rt,
            docker_status: ProcStatus::Stopped,
            feedback: None,
        }
    }

    /// Lê o estado atual (não-bloqueante) para refletir na UI.
    fn refresh_snapshot(&mut self) {
        if let Ok(docker) = self.state.docker.try_lock() {
            self.docker_status = docker.status.clone();
        }
    }
}

impl eframe::App for TrayApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.refresh_snapshot();

        // Repaint periódico para refletir mudanças de estado vindas da API
        // (ex: outro processo chamando /docker/stop) mesmo sem interação do usuário.
        ctx.request_repaint_after(std::time::Duration::from_millis(800));

        egui::CentralPanel::default().show(ctx, |ui| {
            self.draw_status_screen(ui);
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
