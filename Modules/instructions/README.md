# Instructions (AGENTS.md)

Descoberta e versionamento de instruções de projeto para o orchestrator.

## Fontes

- `AVA_GLOBAL_AGENTS_MD` (default: `~/.config/alpha-ai/AGENTS.md` no
  Linux/macOS, `%APPDATA%\alpha-ai\AGENTS.md` no Windows)
- `AVA_INSTRUCTION_FILES` — arquivos/globs extras (separados por `:` ou vírgula)
- `AVA_INSTRUCTIONS_DIRS` — diretórios onde procurar `AGENTS.md`; sobe até a
  raiz do projeto (`.git`). Default: raiz do repo + CWD.
- `AGENTS.md` próximos de um arquivo lido, descobertos sob demanda.

No fluxo WebUI, a descoberta usa o `mcp_manager` no host
(`POST /fs/instructions`) com o `cwd` do turno ou o cwd persistido do projeto.
Quando o cwd é a home do usuário, só o AGENTS.md global é usado.

Resposta da rota: `{cwd, default, global, files:[{path, content}]}`. O
orchestrator cai para a descoberta local se a rota não existir.

Desligar tudo: `AVA_INSTRUCTIONS=0`.

## Baseline + deltas

`InstructionState.prepare(session_id, sources)`:

1. lê as fontes e calcula um hash do valor canônico;
2. na primeira vez, grava o texto renderizado como **baseline**;
3. nas mudanças, renderiza apenas o delta (novo, alterado, removido);
4. o baseline nunca é reescrito — o prefixo do prompt continua cacheável.

O bloco entra no system prompt como `## Project instructions (AGENTS.md)`.

## AGENTS.md aninhados

Depois de uma leitura bem-sucedida (`read`, `*__read_file`, etc.), o
orchestrator procura `AGENTS.md` no diretório lido e ancestrais. Quando o
caminho é acessível ao processo, o conteúdo vira uma mensagem synthetic
durável, deduplicada por sessão/caminho.

Em Docker, caminhos do host não são legíveis pelo orchestrator; use um caminho
montado em `AVA_INSTRUCTIONS_DIRS`.

## Endpoints

- `GET /session/{id}/instructions`
- `POST /session/{id}/instructions/reload`

## Testes

```bash
python -m unittest discover -s tests/instructions -v
```
