<img src="img/icon.png" align="left" width="140" style="margin-right: 20px; margin-bottom: 30px;">

<div style="font-size: 2.5em; font-weight: bold; margin-bottom: 5px;">Avenir-Web</div>

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE.txt)

Princeton AI for Accelerating Invention Lab  
Authors: [Aiden Yiliu Li](https://yiliu.li), Xinyue Hao, [Shilong Liu](https://lsl.zone), [Mengdi Wang](https://ece.princeton.edu/people/mengdi-wang)

<br clear="left"/>

## Abstract

Avenir-Web is an autonomous web agent framework designed for reliable execution of long-horizon tasks on complex, dynamic web interfaces. Addressing challenges in element grounding and long-term task tracking, it introduces a modular architecture combining Mixture of Grounding Experts (MoGE), Experience-Imitation Planning (EIP), and Adaptive Memory with Task-Tracking Checklists. This approach establishes a new open-source state-of-the-art on the Online-Mind2Web benchmark, bridging the performance gap with proprietary models in real-world deployments.

## Installation

Requirements:

- Python `3.11` (recommended; `>=3.9` supported)
- Playwright browsers (Chromium recommended)
- A model provider API key (OpenRouter preferred)

From the repository root:

```bash
conda create -n avenir-web python=3.11
conda activate avenir-web
pip install -e src
python -m playwright install chromium
```

## API Keys

Recommended (environment variable):

```bash
export OPENROUTER_API_KEY="your-key"
```

Or set it in `src/config/batch_experiment.toml` under `[api_keys]` (`openrouter_api_key = "..."`). Environment variables take precedence.

## Quickstart

### Reproduce the Example Batch Run

The example configuration runs a batch from `data/example.json` and writes artifacts to the directory configured by `basic.save_file_dir` in `src/config/batch_experiment.toml`.

```bash
cd src
python run_agent.py -c config/batch_experiment.toml
```

### Single-Task Convenience Script

From the repository root:

```bash
python example.py --task "Find the official API docs for X" --website "https://example.com/"
```


## Outputs and Artifacts

For each task, outputs are written under `basic.save_file_dir/<task_id>/` (configured in TOML):

- `agent.log`: per-task execution log
- `result.json`: final summary
- `config.toml`: resolved config snapshot
- `llm_records.json`: recorded LLM I/O
- `screenshots/`: `screen_<step>.png`

Runner-level logs are written under `src/logs/`.

## Configuration (TOML)

The primary configuration entry point is `src/config/batch_experiment.toml`:

- `[basic]`: output directory (`save_file_dir`)
- `[model]`: model name, temperature, and (optional) specialist models (e.g., checklist/strategist)
- `[api_keys]`: API keys (environment variables still take precedence)
- `[experiment]`: task file path, overwrite policy, max operations
- `[playwright]`: headless/headful, viewport, geolocation

## Troubleshooting

- Missing API key: set `OPENROUTER_API_KEY` (preferred) or configure `[api_keys]`
- Playwright browser not found: run `python -m playwright install chromium`
- Config paths look wrong: run from `src/` or pass an absolute path to `-c`

## Acknowledgment

This project was developed with support from Princeton AI for Accelerating Invention Lab.

## Disclaimer

This repository is provided for research use. Model outputs may be incorrect, incomplete, or unsafe; you are responsible for reviewing actions and complying with applicable laws and website terms of service when running web automation.

## Contact

- Aiden Yiliu Li: yiliu.li@outlook.com
- Shilong Liu: slongliu86@gmail.com

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE.txt](file:///Users/aidenli/Desktop/ResearchProjects/SeeReAct/LICENSE.txt) file for details.

Copyright © 2026 Princeton AI for Accelerating Invention Lab. All rights reserved.
