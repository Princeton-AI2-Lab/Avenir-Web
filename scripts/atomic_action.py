#!/usr/bin/env python3
# Copyright 2026 Princeton AI for Accelerating Invention Lab
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import json
import sys
from pathlib import Path


def _ensure_src_on_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    sys.path.insert(0, str(src_dir))
    return repo_root


def _parse_coords(raw: str | None):
    if not raw:
        return None
    x_str, y_str = [part.strip() for part in raw.split(",", 1)]
    return [int(x_str), int(y_str)]


async def main_async() -> int:
    repo_root = _ensure_src_on_path()

    from avenir_web.agent import AvenirWebAgent
    import toml

    parser = argparse.ArgumentParser(
        description="Execute exactly one browser action without strategy/checklist generation."
    )
    parser.add_argument("--config", default=str(repo_root / "src" / "config" / "batch_experiment.toml"))
    parser.add_argument("--action", required=True, help="Action name, e.g. CLICK, TYPE, GOTO, WAIT")
    parser.add_argument("--website", default=None, help="Optional initial URL to open before the action")
    parser.add_argument("--value", default="", help="Optional action value")
    parser.add_argument("--coords", default=None, help="Normalized coordinates as 'x,y' in 0-1000 space")
    parser.add_argument("--field", default="", help="Optional field name for TYPE/SELECT")
    parser.add_argument("--mode", default="headless", help="Browser mode: headless, headed, or demo")
    parser.add_argument("--task-id", default="atomic_action")
    parser.add_argument("--output-dir", default=str(repo_root / "outputs_atomic"))
    args = parser.parse_args()

    config = toml.load(str(Path(args.config).resolve()))
    config.setdefault("agent", {})
    config["agent"]["enable_strategy"] = False
    config["agent"]["enable_checklist"] = False

    agent = AvenirWebAgent(
        config=config,
        save_file_dir=str(Path(args.output_dir).resolve()),
        default_task="Execute one atomic browser action.",
        default_website=args.website or "about:blank",
        max_auto_op=1,
        mode=args.mode,
        task_id=args.task_id,
        create_timestamp_dir=False,
    )

    try:
        await agent.start(website=args.website)
        result_text = await agent.perform_action(
            action_name=str(args.action).upper(),
            value=args.value,
            target_coordinates=_parse_coords(args.coords),
            field_name=args.field or None,
        )
        await agent.take_screenshot()
        payload = {
            "task_id": args.task_id,
            "action": str(args.action).upper(),
            "value": args.value,
            "coordinates": _parse_coords(args.coords),
            "result": result_text,
            "current_url": agent.page.url if agent.page else None,
            "screenshot_path": agent.screenshot_path,
            "output_dir": agent.main_path,
        }
        print(json.dumps(payload, ensure_ascii=True, indent=2))
        return 0
    finally:
        await agent.stop()


def main() -> None:
    raise SystemExit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()
