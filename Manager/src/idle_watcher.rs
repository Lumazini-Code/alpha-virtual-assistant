// Alpha AI (AVA) — Monitor de inatividade
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
//! Monitor de inatividade.
//!
//! Roda em background (task tokio separada) e verifica periodicamente
//! há quanto tempo o docker está sem atividade. Se passar do limite,
//! derruba o processo automaticamente.
//!
//!   - docker: 15 min sem atividade -> stop (AVA_DOCKER_IDLE_MINUTES
//!     sobrescreve)
//!
//! "Atividade" = cada chamada a POST /docker/start (a TUI faz uma por
//! pergunta). O touch acontece em process_manager::start_docker mesmo
//! quando o docker "já estava ativo" — sem isso o timer era de VIDA,
//! não de inatividade, e matava o docker em uso contínuo.

use crate::process_manager;
use crate::state::{ProcStatus, SharedState};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

/// Timeout de inatividade. Default 15 min (perfil econômico — em máquina
/// com outras apps, o stack dorme cedo e devolve a RAM; sobe on-demand na
/// próxima pergunta); configurável via AVA_DOCKER_IDLE_MINUTES (parse uma
/// vez no primeiro uso).
///
/// "Atividade" = cada POST /docker/start (a TUI chama a cada pergunta —
/// ver process_manager::start_docker).
fn docker_idle_timeout() -> Duration {
    static TIMEOUT: OnceLock<Duration> = OnceLock::new();
    *TIMEOUT.get_or_init(|| {
        let minutes = std::env::var("AVA_DOCKER_IDLE_MINUTES")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .filter(|&m| m > 0)
            .unwrap_or(15);
        tracing::info!("Timer de inatividade do docker: {minutes} min.");
        Duration::from_secs(minutes * 60)
    })
}

/// Intervalo de checagem. Não precisa ser fino — checar a cada 30s é
/// suficiente e barato.
const CHECK_INTERVAL: Duration = Duration::from_secs(30);

/// Inicia o loop de monitoramento. Deve ser chamado uma vez, em uma task
/// tokio separada (`tokio::spawn`), e roda para sempre.
pub async fn run(state: SharedState) {
    let mut interval = tokio::time::interval(CHECK_INTERVAL);

    loop {
        interval.tick().await;

        check_docker_idle(&state).await;
    }
}

async fn check_docker_idle(state: &SharedState) {
    let should_stop = {
        let docker = state.docker.lock().await;
        is_idle(docker.status.clone(), docker.last_activity, docker_idle_timeout())
    };

    if should_stop {
        tracing::info!("Docker inativo além do limite de inatividade — encerrando automaticamente.");
        if let Err(e) = process_manager::stop_docker(state).await {
            tracing::error!("Falha ao encerrar docker por inatividade: {e}");
        }
    }
}

fn is_idle(status: ProcStatus, last_activity: Option<Instant>, timeout: Duration) -> bool {
    if status != ProcStatus::Running {
        return false;
    }
    match last_activity {
        Some(t) => t.elapsed() >= timeout,
        // Se está "Running" mas nunca teve atividade registrada (não deveria
        // acontecer, já que setamos no start), por segurança não derruba.
        None => false,
    }
}
