---
name: projects-and-skills
description: Use when the user asks to create a project, separate a context ("cria um projeto pra X"), or save a procedure as a reusable skill ("transforma isso em skill", "save these steps for next time"), or asks which projects/skills exist.
---

# Creating durable things

## Projects

- `project_create(name)` creates a dedicated memory scope and ACTIVATES it immediately: from then on, long-term memories, graphs and plans belong to that project. Tell the user they are now working inside it.
- Call `project_list` first to avoid duplicates.
- Use it for a new lasting context. For loose facts, use the memory write tool instead.
- If it is unclear whether the user wants a project or just a saved fact, ask first (activation changes where everything is saved).

## Skills

- `skill_create(...)` saves a reusable playbook (SKILL.md), selectable from the NEXT user request.
- The `description` is what the semantic selector matches on: state WHAT the skill does and WHEN to use it. A vague description means the skill will never be selected.
- Call `skill_list` first to avoid duplicating an installed skill.
- Not for facts (use memory) and not for one-off instructions.

`project_list` / `skill_list` show what exists; use them when the user names a project or asks which skills you have.
