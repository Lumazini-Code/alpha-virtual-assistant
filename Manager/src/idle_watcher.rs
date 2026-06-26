//! Monitor de inatividade.
//!
//! Roda em background (task tokio separada) e verifica periodicamente
//! há quanto tempo cada processo está sem atividade. Se passar do limite,
//! derruba o processo automaticamente.
//!
//!   - llama-server: 15 minutos sem atividade -> stop
//!   - docker:       45 minutos sem atividade -> stop
//!
//! "Atividade" é atualizada via `touch_llama_activity` / `touch_docker_activity`
//! em process_manager.rs, que deve ser chamado pelo proxy de requisições da
//! API (toda vez que uma chamada de inferência passar pelo llama-server)
//! e/ou por ações manuais do usuário na GUI.

use crate::process_manager;
use crate::state::{ProcStatus, SharedState};
use std::time::{Duration, Instant};

const LLAMA_IDLE_TIMEOUT: Duration = Duration::from_secs(15 * 60);
const DOCKER_IDLE_TIMEOUT: Duration = Duration::from_secs(45 * 60);

/// Intervalo de checagem. Não precisa ser fino — checar a cada 30s é
/// suficiente e barato.
const CHECK_INTERVAL: Duration = Duration::from_secs(30);

/// Inicia o loop de monitoramento. Deve ser chamado uma vez, em uma task
/// tokio separada (`tokio::spawn`), e roda para sempre.
pub async fn run(state: SharedState) {
    let mut interval = tokio::time::interval(CHECK_INTERVAL);

    loop {
        interval.tick().await;

        check_llama_idle(&state).await;
        check_docker_idle(&state).await;
    }
}

async fn check_llama_idle(state: &SharedState) {
    let should_stop = {
        let llama = state.llama.lock().await;
        is_idle(llama.status.clone(), llama.last_activity, LLAMA_IDLE_TIMEOUT)
    };

    if should_stop {
        tracing::info!("llama-server inativo há mais de 15 min — encerrando automaticamente.");
        if let Err(e) = process_manager::stop_llama(state).await {
            tracing::error!("Falha ao encerrar llama-server por inatividade: {e}");
        }
    }
}

async fn check_docker_idle(state: &SharedState) {
    let should_stop = {
        let docker = state.docker.lock().await;
        is_idle(docker.status.clone(), docker.last_activity, DOCKER_IDLE_TIMEOUT)
    };

    if should_stop {
        tracing::info!("Docker inativo há mais de 45 min — encerrando automaticamente.");
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