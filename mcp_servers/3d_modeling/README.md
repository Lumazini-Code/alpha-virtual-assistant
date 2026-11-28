# cadquery-3d-printing-mcp

Servidor MCP (stdio) para modelagem CAD paramétrica e preparação de peças para impressão 3D,
construído sobre [CadQuery](https://cadquery.readthedocs.io/). As ferramentas mantêm um documento
em memória de objetos nomeados, com o histórico de operações e parâmetros de cada um.

## Instalação

```bash
python3 -m venv .venv-cadquery
.venv-cadquery/bin/python -m pip install -e mcp_servers/3d_modeling
```

O instalador da Alpha faz isso automaticamente em `.venv-cadquery` e registra o servidor no
mcp_manager como `cadquery-3d-printing` (`transport: stdio`, `python -m cadquery_mcp_server`).

Python 3.10–3.12 é recomendado (compatibilidade de wheels do CadQuery). Sem `python3`/`venv`,
o instalador tenta `uv venv --seed --python 3.12`; se nada estiver disponível, a etapa é pulada
com aviso e o restante da instalação segue.

## Uso

O servidor roda por stdio. `health_check` informa se o CadQuery está disponível e qual é o
diretório de trabalho ativo. Todas as dimensões são em milímetros.

- `create_primitive`, `create_profile_extrusion`, `boolean_operation`, `make_hole`,
  `edge_treatment`, `shell_object`, `pattern`, `transform_object` — montagem declarativa.
- `inspect_object`, `measure_objects`, `validate_printability` — inspeção antes de exportar.
- `export_object` (STEP, STL, AMF, 3MF, DXF), `import_step`, `save_manifest` — entrega.

`execute_cadquery_script` aceita um script que deve terminar com `result` como `cq.Workplane`
ou `cq.Shape`; use-o só quando as operações declarativas não couberem.

## Diretório de saída

`CQ_WORKDIR` define onde `export_object`, `import_step` e `save_manifest` leem/escrevem. Se não
for definido, o servidor usa o cwd do processo — no fluxo da Alpha, a pasta do projeto selecionada
na WebUI. `export_object` recusa caminhos que saiam desse diretório.

## Playbook

A skill `cadquery-mcp` ensina o fluxo obrigatório, seletores de face/aresta e regras de tolerância:
`skills/cadquery-mcp/SKILL.md` (fonte) → `.agents/skills/cadquery-mcp/SKILL.md` (store canônico).
A cópia de `mcp_servers/3d_modeling/SKILL.md` é usada somente como referência do pacote;
mantenha as duas sincronizadas.
