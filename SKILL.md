---
name: avenir-web
description: Use this skill to run and improve Avenir-Web autonomous web tasks end-to-end: choose mode (headless/headed/demo), write clear task instructions, execute single or batch runs, and propose high-impact next iterations.
---

# Avenir-Web

## What this skill does

This skill operates Avenir-Web for reliable web-task execution and iteration.

Responsibilities:
- run single tasks and batch tasks
- choose mode (`headless` / `headed` / `demo`)
- improve instruction quality before execution
- analyze run outputs and recommend the next best change
- execute one atomic action without strategy/checklist overhead
- read the current page by screenshoting it and asking the main model a question

Use this skill for requests like:
- run a task on a website
- run a task list and summarize outcomes
- improve success rate with better instructions/config

## Canonical entrypoints

Single task:
```bash
python example.py --task "<instruction>" --website "<url>" --mode headless
```

Atomic action:
```bash
python scripts/atomic_action.py --action CLICK --website "<url>" --coords "500,500"
```

Read page:
```bash
python scripts/read_page.py --website "<url>" --question "<question>"
```

Batch:
```bash
cd src
python run_agent.py -c config/batch_experiment.toml
```

Prefer these scripts over ad-hoc commands.

## Quick usage example

Single task:
```bash
python example.py \
  --task "On openrouter.ai, list image-input-capable distillable models sorted by price ascending." \
  --website "https://openrouter.ai/" \
  --mode demo
```

Batch:
```bash
cd src
python run_agent.py -c config/batch_experiment.toml
```

Atomic action:
```bash
python scripts/atomic_action.py \
  --action TYPE \
  --website "https://example.com/" \
  --coords "500,420" \
  --value "hello"
```

Read page:
```bash
python scripts/read_page.py \
  --website "https://openrouter.ai/" \
  --question "What models or prices are visible on this page?"
```

## Run modes

| mode | behavior | best for |
|---|---|---|
| `headless` | no visible browser window | fast, reproducible runs and large batch jobs |
| `headed` | visible browser window | manual observation without demo overlay |
| `demo` | visible window + overlay/dashboard controls | live debugging and demonstrations |

Notes:
- if mode is missing, use `headless`
- `demo` improves observability, not model intelligence

Mode selection:
1. benchmark/batch -> `headless`
2. visual debugging -> `headed`
3. demo/control flow visibility -> `demo`

## Instruction design

`confirmed_task` should include:
1. objective
2. constraints
3. completion condition

Template:
- `On <website>, <objective>. Apply constraints: <constraints>. Finish when <observable completion state>.`

Keep it single-goal, specific, and verifiable.

## Single-task workflow

Input:
- `task`
- `website`
- optional `mode`, `task-id`, `output-dir`

Steps:
1. check environment and API key
2. validate instruction quality
3. run `example.py`
4. inspect outputs
5. report status + cause + next action

Recommended report fields:
- `task_id`
- status: `success` / `partial` / `failed`
- evidence summary
- one-line cause
- one recommended next step

## Atomic action workflow

Use `scripts/atomic_action.py` when you need exactly one browser operation and do not want strategist/checklist generation.

Typical uses:
- one click
- one type
- one goto
- one scroll

Properties:
- disables strategy generation
- disables checklist generation
- executes exactly one action
- returns structured JSON with result, URL, screenshot path, and output directory

## Read-page workflow

Use `scripts/read_page.py` when you want to inspect the current page by screenshot and ask the main model a direct question.

Properties:
- opens the page
- captures a screenshot
- sends the screenshot plus page metadata to the main model
- returns structured JSON with the answer and screenshot path

## Batch workflow

### Task file schema

```json
[
  {
    "task_id": "example_task_001",
    "confirmed_task": "Find image-input-capable distillable models sorted by price ascending.",
    "website": "https://openrouter.ai/"
  }
]
```

Required per task:
- `task_id`
- `confirmed_task`
- `website`

### Config checklist (`src/config/batch_experiment.toml`)

- `[basic].save_file_dir`
- `[experiment].task_file_path`
- `[experiment].max_op`
- `[playwright].mode`
- `[model].name`
- API key source

### Batch execution

1. validate JSON schema and config paths
2. choose mode and `max_op`
3. run batch command
4. summarize per-task outcomes
5. provide one global improvement recommendation

Recommended batch report fields:
- total/completed/failed counts
- per-task status list
- recurring issue patterns
- one highest-impact next change

## API requirements

Required credential:
- `OPENROUTER_API_KEY` (preferred)

Resolution order:
1. environment variable `OPENROUTER_API_KEY`
2. `[api_keys].openrouter_api_key` in TOML (fallback)

Rules:
- never hardcode real keys in source files
- never print full keys in logs/outputs/reports
- fail fast if key is missing with an actionable message

## Script usage rules

- script-first: use repository entrypoints before custom commands
- non-interactive CLI only
- explicit flags and paths
- deterministic behavior preferred
- clear, actionable error messages

If adding helper scripts:
1. place under `scripts/`
2. use CLI flags (no prompts)
3. return stable, parseable summaries
4. document usage in this file

## Environment checklist

Before running:
1. Python environment available
2. dependencies installed (`pip install -e src`)
3. Playwright Chromium installed (`python -m playwright install chromium`)
4. API key configured
5. config/task paths valid

## Output contract

Each run summary should include:
1. execution metadata: run type, mode, task IDs
2. outcome: status and evidence summary
3. diagnosis: root-cause hypothesis
4. next action: one highest-impact recommendation

## Boundaries

- do not claim completion without evidence
- do not skip issue summary
- avoid large refactors before instruction/config fixes
- avoid interactive prompts in core workflow

## One-line identity

Avenir-Web execution and reliability skill: mode selection + instruction design + run analysis + iteration planning.
