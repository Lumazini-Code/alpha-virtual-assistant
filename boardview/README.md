# boardview-mcp

Servidor MCP que permite a uma IA analisar um **boardview GenCAD (`.cad`)** — o mesmo formato `.cad` lido pelo
OpenBoardView — para ajudar a diagnosticar placas reais: rails sem tensão, curtos, caminhos de alimentação,
vizinhança física de componentes e cruzamento com medições do técnico.

* **Parser:** `boardview_mcp/gencad.py` (Python puro, ~0,8 s para 2,5 MB / 2.5k componentes / 9,9k pinos).
* **Linguagem:** Python, por ter o SDK MCP oficial, Pillow para o render e porque o parsing é texto simples.
* **Geometria validada** no arquivo real: lado BOTTOM = espelhar X, depois girar anti-horário (`ROTATION`),
  depois transladar para `PLACE`.

## Instalação

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e .            # instala mcp (<2) e pillow; comando: boardview-mcp
```

### Claude Desktop / qualquer cliente MCP (stdio)

```json
{
  "mcpServers": {
    "boardview": {
      "command": "/caminho/.venv/bin/python",
      "args": ["-m", "boardview_mcp", "--board", "/caminho/placa.cad"]
    }
  }
}
```

### Claude Code

```bash
claude mcp add boardview -- /caminho/.venv/bin/python -m boardview_mcp --board /caminho/placa.cad
```

`--board` é opcional (ou use `BOARDVIEW_FILE`); sem ele a IA chama `load_board(path)`. `path` aceita o arquivo,
a pasta que o contém (`.` = cwd do servidor, útil quando o cliente MCP já está na pasta do projeto) ou um glob
(`*.cad`); se houver mais de um boardview, a resposta lista os candidatos.
Medições ficam em `~/.boardview_mcp/` (mude com `BOARDVIEW_MCP_DATA`).
Outros transportes: `--transport streamable-http|sse`.

## Tools

`load_board`, `board_info`, `list_power_rails`, `search_parts`, `get_part`, `search_nets`, `get_net`,
`trace_connectivity`, `neighbors`, `power_rail_analysis`, `render_region`, `add_measurement`, `list_measurements`,
`check_measurements_vs_board`, `clear_measurements`. O recurso `boardview://skill` serve o `SKILL.md`.

## Limites

Boardview não tem esquemático, valores reais, camadas internas nem trilhas. Classes de componentes, tipo/tensão/estado
das redes e decodificação de valores são heurísticas por nome (calibradas em nomes Dell/Wistron/Quanta). Só GenCAD
por enquanto; para BRD/BV/FZ/ASC basta adicionar outro parser que produza o mesmo `Board` (`gencad.py`).

## Testes

```bash
pip install -e .[dev]
BOARDVIEW_TEST_FILE=/caminho/placa.cad pytest -q     # sem a variável, roda só o fixture sintético
```
