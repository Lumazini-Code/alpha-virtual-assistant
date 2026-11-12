# Code Mode

Expõe uma única tool nativa `execute` que roda JavaScript e compõe chamadas de
tools MCP. O schema nativo fica pequeno e estável; o catálogo grande vira
instrução no system prompt.

## Como funciona

1. O orchestrator monta o registry base (nativas + MCP) e o catálogo MCP.
2. `make_execute_spec()` cria a tool `execute`; o schema nativo passa a ter as
   tools nativas + `execute` (tools MCP saem do schema direto).
3. O modelo chama `execute({code})`. O worker Node executa o JS com `tools` e
   `search`; cada chamada de tool vira um round-trip JSON com o Python, que
   executa pelo mesmo `ToolSnapshot` (validação/permissões/hooks).
4. O resultado final volta como texto; chamadas usadas vão em
   `metadata.toolCalls` para o aprendizado do grafo.

## Segurança

- Node roda em subprocesso; o código não recebe `require`, `process`, fs,
  rede ou timers.
- `AVA_CODEMODE_TIMEOUT_MS` limita a execução inteira.
- `AVA_CODEMODE_MAX_TOOL_CALLS` limita chamadas de tool.
- `AVA_CODEMODE_MAX_OUTPUT_CHARS` limita a saída.

## Configuração

- `AVA_CODEMODE=1` (default). `0` volta ao schema MCP direto.
- `AVA_NODE_BIN` — caminho alternativo do Node.
- Se Node/worker não existir, Code Mode se desativa e usa o schema normal.

## Testes

```bash
python -m unittest discover -s tests/codemode -v
```
