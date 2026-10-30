// Alpha AI (AVA) — serviço do grafo embutido no Manager
// Copyright (C) 2026  Felipe lumazini
//
// This program is free software: you can redistribute it and/or modify it
// under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or (at your
// option) any later version. See <https://www.gnu.org/licenses/>.
//! Ponte entre o WebUI embutido e o `graph_activation.py` headless.
//!
//! Não existe mais servidor HTTP na porta 8777. O Manager sobe o worker
//! dentro do container via `docker exec` e consome stdout como uma linha
//! JSON por evento:
//!   {"type":"snapshot", ...}    (primeira linha)
//!   {"type":"graph_delta", ...}
//!   {"type":"activation", ...}
//!   {"type":"scope_switch", ...}
//!
//! O WebUI renderiza o grafo no próprio documento (vis-network) e recebe
//! os dados por `/graph/snapshot` + `/graph/events` (SSE) e o replay por
//! `/graph/replay`.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::{broadcast, Mutex, RwLock};

const WORKER_WAIT: Duration = Duration::from_secs(12);

pub struct GraphService {
    project_root: PathBuf,
    worker: Mutex<Worker>,
    snapshot: RwLock<Option<Value>>,
    events: broadcast::Sender<String>,
    last_error: RwLock<Option<String>>,
}

struct Worker {
    child: Option<Child>,
    stdin: Option<tokio::process::ChildStdin>,
    running: bool,
}

impl GraphService {
    pub fn new(project_root: PathBuf) -> Self {
        // O canal não guarda o snapshot; ele é mantido separado para o
        // endpoint /graph/snapshot e para o primeiro frame do SSE.
        let (events, _) = broadcast::channel(512);
        Self {
            project_root,
            worker: Mutex::new(Worker {
                child: None,
                stdin: None,
                running: false,
            }),
            snapshot: RwLock::new(None),
            events,
            last_error: RwLock::new(None),
        }
    }

    /// Garante que o worker está vivo. Pode ser chamado em paralelo: o
    /// mutex serializa as tentativas.
    async fn ensure_started(self: &Arc<Self>) -> Result<(), String> {
        let mut worker = self.worker.lock().await;
        if worker.running && worker.child.is_some() {
            let exited = {
                match worker.child.as_mut() {
                    Some(child) => !matches!(child.try_wait(), Ok(None)),
                    None => true,
                }
            };
            if !exited {
                return Ok(());
            }
            worker.child = None;
            worker.stdin = None;
            worker.running = false;
        } else if worker.child.is_some() {
            // Handle antigo já saiu; limpa para permitir respawn.
            worker.child = None;
            worker.stdin = None;
            worker.running = false;
        }

        // Snapshot antigo não vale para uma nova geração do worker.
        *self.snapshot.write().await = None;
        *self.last_error.write().await = None;

        let container = std::env::var("AVA_CONTAINER_NAME")
            .ok()
            .map(|name| name.trim().to_string())
            .filter(|name| !name.is_empty())
            .unwrap_or_else(|| "ava".to_string());
        // AVA_GRAPH_WORKER=host: roda o worker com o Python do HOST (útil
        // quando o container não está de pé — os DBs são bind mounts) em
        // vez de `docker exec`. Default: dentro do container.
        let host_worker = std::env::var("AVA_GRAPH_WORKER")
            .map(|value| value.trim().eq_ignore_ascii_case("host"))
            .unwrap_or(false)
            && cfg!(not(target_os = "windows"));

        let mut command;
        if host_worker {
            let python = std::env::var("AVA_PYTHON").unwrap_or_else(|_| "python3".to_string());
            command = Command::new(python);
            command
                .args([
                    "-u",
                    "graph_activation.py",
                    "--stdio",
                    "--watch-stdin",
                    "--memory-api",
                ])
                .current_dir(&self.project_root);
        } else {
            command = Command::new("docker");
            command.args([
                "exec",
                "-i",
                "-w",
                "/app",
                &container,
                "python3",
                "-u",
                "/app/graph_activation.py",
                "--stdio",
                "--watch-stdin",
                "--memory-api",
            ]);
        }
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        #[cfg(target_os = "windows")]
        {
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            command.creation_flags(CREATE_NO_WINDOW);
        }

        let mut child = command.spawn().map_err(|e| {
            let message = format!("falha ao executar docker exec: {e}");
            tracing::warn!("graph: {message}");
            message
        })?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| "worker do grafo sem stdin".to_string())?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "worker do grafo sem stdout".to_string())?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| "worker do grafo sem stderr".to_string())?;

        worker.child = Some(child);
        worker.stdin = Some(stdin);
        worker.running = true;
        drop(worker);

        let service = self.clone();
        tokio::spawn(async move { service.read_stdout(stdout).await });
        let service = self.clone();
        tokio::spawn(async move { service.read_stderr(stderr).await });

        if host_worker {
            tracing::info!("graph: worker iniciado no host (AVA_GRAPH_WORKER=host)");
        } else {
            tracing::info!("graph: worker iniciado no container {container}");
        }
        Ok(())
    }

    async fn read_stdout<R>(self: Arc<Self>, stdout: R)
    where
        R: tokio::io::AsyncRead + Unpin,
    {
        let mut lines = BufReader::new(stdout).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            let line = line.trim().to_string();
            if line.is_empty() {
                continue;
            }
            let Ok(value) = serde_json::from_str::<Value>(&line) else {
                // Diagnóstico do Python (ex.: "poll falhou") não é evento.
                tracing::info!("graph worker: {line}");
                continue;
            };
            if value.get("type").and_then(Value::as_str) == Some("snapshot") {
                // O snapshot tem canal próprio (cache + primeiro frame do SSE).
                *self.snapshot.write().await = Some(value);
                continue;
            }
            let _ = self.events.send(line);
        }

        let mut worker = self.worker.lock().await;
        worker.child = None;
        worker.stdin = None;
        worker.running = false;
        drop(worker);
        let _ = self
            .events
            .send(json!({ "type": "worker_down" }).to_string());
        tracing::warn!("graph: worker encerrou a saída (EOF)");
    }

    async fn read_stderr<R>(self: Arc<Self>, stderr: R)
    where
        R: tokio::io::AsyncRead + Unpin,
    {
        let mut lines = BufReader::new(stderr).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            let line = line.trim().to_string();
            if line.is_empty() {
                continue;
            }
            tracing::warn!("graph worker stderr: {line}");
            *self.last_error.write().await = Some(line);
        }
    }

    /// Snapshot atual. Sobe o worker se necessário e espera o primeiro frame.
    pub async fn snapshot(self: &Arc<Self>) -> Result<Value, String> {
        if let Some(snapshot) = self.snapshot.read().await.clone() {
            return Ok(snapshot);
        }
        self.ensure_started().await?;

        let deadline = tokio::time::Instant::now() + WORKER_WAIT;
        loop {
            if let Some(snapshot) = self.snapshot.read().await.clone() {
                return Ok(snapshot);
            }
            if !self.alive().await {
                let error =
                    self.last_error.read().await.clone().unwrap_or_else(|| {
                        "worker do grafo encerrou antes do snapshot".to_string()
                    });
                return Err(error);
            }
            if tokio::time::Instant::now() >= deadline {
                return Err("timeout esperando o snapshot do grafo".to_string());
            }
            tokio::time::sleep(Duration::from_millis(120)).await;
        }
    }

    pub fn subscribe(&self) -> broadcast::Receiver<String> {
        self.events.subscribe()
    }

    /// Estado para o WebUI (não sobe o worker).
    pub async fn status(&self) -> Value {
        let alive = self.alive().await;
        let has_snapshot = self.snapshot.read().await.is_some();
        // stderr do worker também carrega diagnóstico normal ("Grafo
        // carregado..."); só expõe como erro quando o worker NÃO está vivo.
        let error = if alive {
            None
        } else {
            self.last_error.read().await.clone()
        };
        json!({
            "online": alive,
            "running": alive,
            "has_snapshot": has_snapshot,
            "error": error,
        })
    }

    async fn alive(&self) -> bool {
        let mut worker = self.worker.lock().await;
        if worker.child.is_none() {
            return false;
        }
        let exited = match worker.child.as_mut() {
            Some(child) => !matches!(child.try_wait(), Ok(None)),
            None => true,
        };
        if exited {
            worker.child = None;
            worker.running = false;
        }
        !exited
    }

    pub async fn shutdown(&self) {
        let mut worker = self.worker.lock().await;
        // Fecha o pipe primeiro: o worker sai sozinho ao ver EOF no stdin.
        worker.stdin.take();
        if let Some(mut child) = worker.child.take() {
            let _ = child.kill().await;
            let _ = child.wait().await;
        }
        worker.running = false;
    }
}

/// Replay dos activation_log.jsonl de todos os escopos (root + projetos),
/// ordenado por ts. Lê direto do bind mount; funciona com o worker parado.
pub fn replay(project_root: &Path, n: usize) -> Value {
    let n = n.clamp(1, 500);
    let db_root = project_root.join("Modules").join("memory").join("db");
    let mut logs: Vec<(String, PathBuf)> =
        vec![("general".to_string(), db_root.join("activation_log.jsonl"))];

    if let Ok(entries) = std::fs::read_dir(&db_root) {
        let mut projects: Vec<(String, PathBuf)> = entries
            .flatten()
            .filter_map(|entry| {
                let path = entry.path();
                let name = entry.file_name().to_string_lossy().to_string();
                if !path.is_dir() || name.starts_with("ava_") {
                    return None;
                }
                Some((name, path.join("activation_log.jsonl")))
            })
            .collect();
        projects.sort_by(|a, b| a.0.cmp(&b.0));
        logs.extend(projects);
    }

    let mut events: Vec<(f64, Value)> = Vec::new();
    for (scope, path) in logs {
        let Ok(content) = std::fs::read_to_string(&path) else {
            continue;
        };
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let Ok(value) = serde_json::from_str::<Value>(line) else {
                continue;
            };
            let Value::Object(mut object) = value else {
                continue;
            };
            object
                .entry("type")
                .or_insert_with(|| Value::String("activation".into()));
            object
                .entry("scope")
                .or_insert_with(|| Value::String(scope.clone()));
            let ts = object.get("ts").and_then(Value::as_f64).unwrap_or(0.0);
            events.push((ts, Value::Object(object)));
        }
    }

    events.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let total = events.len();
    let start = total.saturating_sub(n);
    let values: Vec<Value> = events[start..].iter().map(|(_, v)| v.clone()).collect();
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs_f64())
        .unwrap_or(0.0);

    json!({ "events": values, "ts": now })
}
