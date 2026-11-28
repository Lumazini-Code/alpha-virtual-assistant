---
name: browser-plans
description: Use when the user wants to run, repeat, check or extract something from a website they use regularly (saved browser tasks). Covers list_plans, run_plan and when to fall back to manual browser_* tools.
---

# Stored browser plans

`list_plans` and `run_plan` use saved hybrid plans: pre-recorded browser flows with fixed selectors, plus steps judged by the model. They are faster, cheaper and more reliable than driving the browser manually.

1. If the request could match a saved task, call `list_plans` first.
2. If a plan matches, call `run_plan` with its parameters. Prefer it over manual `browser_*` calls.
3. Fall back to manual `browser_*` tools only if: no plan matches, the user explicitly wants a fresh manual session, or the plan failed and recovery did not help.
4. After a plan runs, verify the returned outcome against what the user asked before reporting success; if it failed, say which step failed.
