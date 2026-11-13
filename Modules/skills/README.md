# Skills

Catálogo completo de Agent Skills (`SKILL.md`) + carregamento sob demanda.

## Fluxo

1. `discover()` varre `.agents/skills`, `.agents/skills-created` e
   `~/.agents/skills`, lê o frontmatter (`name`, `description`, `autoinvoke`,
   `version`) e lista até 10 arquivos auxiliares.
2. `make_source()` transforma o catálogo em uma fonte de instrução
   (`core/skills`), com baseline + deltas por sessão.
3. O system prompt lista todas as skills em `<available_skills>`.
4. A tool `skill` carrega o playbook ou um arquivo:
   - `skill({id})`
   - `skill({id, file: "references/format.md"})`
   - `skill({id: "skill://demo/references/format.md"})`

## Compatibilidade

`use_skill` e `skill_file` continuam funcionando como aliases. `skill_file`
aceita `skill://<id>/<path>`.

## Permissões

- `AVA_SKILL_ALLOW` / `AVA_SKILL_DENY` — globs por id.
- Skill negada não aparece no catálogo e não carrega.

Desligar o catálogo e voltar à seleção semântica: `AVA_SKILL_CATALOG=0`.

## Testes

```bash
python -m unittest discover -s tests/skills -v
```
