# Tool registry

Ponto único de definição e execução de tools nativas e MCP.

## Estrutura

- `registry.py` — `ToolSpec`, `ToolRegistry`, `ToolSnapshot`, `ToolContext`, `ToolResult`.
- `validation.py` — validação de JSON Schema (type, required, properties, enum, items, additionalProperties).
- `permissions.py` — `ToolPolicy` com allow/deny por glob.
- `hooks.py` — hooks `before` (reescreve argumentos) e `after` (reescreve resultado).

## Fluxo

1. `_build_tool_snapshot()` no orchestrator registra:
   - tools nativas de `TOOLS`;
   - tools MCP do schema/dispatch carregados no turno.
2. O snapshot expõe `definitions` (OpenAI) e `execute(name, args, context)`.
3. `execute` valida, aplica política, roda hooks, chama o executor e normaliza o erro.
4. O loop usa o mesmo snapshot para executar e para decidir lanes paralelas.

## Configuração

- `AVA_TOOL_REGISTRY=1` (default) liga o caminho novo.
- `AVA_TOOL_ALLOW` / `AVA_TOOL_DENY` — padrões glob (`browser__*`, `memory_*`).
- `0` em `AVA_TOOL_REGISTRY` volta ao caminho antigo.

## Testes

```bash
python -m unittest discover -s tests/tools -v
```
