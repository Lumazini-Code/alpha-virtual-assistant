---
name: cadquery-mcp
description: Modelagem CAD paramétrica e preparação de peças para impressão 3D usando o MCP CadQuery. Use para criar, editar, medir, validar, importar ou exportar modelos STEP/STL/3MF com histórico de operações reproduzível.
---

# CadQuery MCP

Use o servidor `cadquery-3d-printing` para construir sólidos paramétricos em milímetros e entregar arquivos fabricáveis. Trate cada objeto nomeado como um componente do histórico do modelo.

## Fluxo obrigatório

1. Chame `health_check`; se `cadquery_available` for falso, informe que o runtime CadQuery precisa ser instalado.
2. Mantenha nomes semânticos (`base`, `lid`, `mount_holes`, `final_part`) e consulte `list_objects` antes de sobrescrever.
3. Prefira ferramentas declarativas (`create_primitive`, `create_profile_extrusion`, `boolean_operation`, `make_hole`, `edge_treatment`, `shell_object`, `pattern`, `transform_object`). Use `execute_cadquery_script` somente para geometrias que não cabem nessas operações.
4. Após cada etapa importante, chame `inspect_object`. Use `measure_objects` para conferir encaixes e folgas.
5. Antes de exportar, chame `validate_printability`. Isso é uma heurística, não substitui o slicer, análise de overhang, tolerância da impressora ou inspeção física.
6. Exporte pelo menos STEP para edição CAD e STL ou 3MF para fatiamento. Finalize com `save_manifest`.

## Regras geométricas

- Use milímetros, origem explícita e parâmetros em `params`/`variables`.
- Construa da referência para o detalhe: volume principal, cortes, furos, cascas, chanfros/raios, padrões e acabamento.
- Para encaixes FDM, comece com folga de 0,2–0,4 mm por lado e ajuste ao processo; não invente tolerâncias quando o usuário não informou impressora/material.
- Prefira raios e chanfros modestos; evite espessuras menores que a largura de linha e valide paredes finas no slicer.
- Use `boolean_operation` para manter operações explícitas e reproduzíveis; não dependa de estado oculto em scripts.

## Scripts CadQuery

`execute_cadquery_script` recebe código confiável e exige `result` como `cq.Workplane` ou `cq.Shape` final. Exemplo:

```python
base = cq.Workplane("XY").box(80, 50, 4)
post = cq.Workplane("XY").center(0, 0).circle(8).extrude(25)
result = base.union(post)
```

Não execute código fornecido por terceiros sem isolamento. Evite loops gigantes e geometria autointersectante. Se uma operação falhar, reduza a cadeia, inspecione a última etapa válida e corrija a seleção (`>Z`, `|Z`, `all`) antes de tentar novamente.

## Seletores e exportação

- `>Z`: face mais alta; `|Z`: arestas paralelas a Z; `all`: todas as arestas.
- Use `make_hole` em faces superiores quando o furo deve ser vertical.
- Use `export_object` com nomes de arquivo relativos ao `CQ_WORKDIR`; nunca tente escrever fora dele.
- STEP preserva melhor a edição CAD. STL depende de tolerância/angular tolerance; use tolerância menor para superfícies curvas, equilibrando tamanho do arquivo.

## Diagnóstico

- `solid_count == 0`: a operação não produziu sólido fechado.
- Bounding box inesperada: confira centralização, origem e transformações.
- Falha em fillet/chamfer: reduza o raio, selecione arestas mais específicas ou aplique o tratamento antes de cortes complexos.
- Falha em shell: remova uma face simples e use espessura menor.
- Resultado booleano ruim: confira se os corpos realmente se interceptam e tente uma etapa por vez.

## Entrega

Apresente ao usuário: nomes dos objetos finais, dimensões da bounding box, volume quando disponível, alertas de `validate_printability` e links para os arquivos exportados. Não declare que a peça está pronta para impressão apenas porque a heurística passou.
