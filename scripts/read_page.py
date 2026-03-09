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


async def main_async() -> int:
    repo_root = _ensure_src_on_path()

    from avenir_web.agent import AvenirWebAgent
    import toml

    parser = argparse.ArgumentParser(
        description="Read the current page by taking a screenshot and asking the main model a question."
    )
    parser.add_argument("--config", default=str(repo_root / "src" / "config" / "batch_experiment.toml"))
    parser.add_argument("--website", required=True, help="Initial URL to open")
    parser.add_argument("--question", required=True, help="Question to ask about the current page")
    parser.add_argument("--mode", default="headless", help="Browser mode: headless, headed, or demo")
    parser.add_argument("--task-id", default="read_page")
    parser.add_argument("--output-dir", default=str(repo_root / "outputs_read_page"))
    args = parser.parse_args()

    config = toml.load(str(Path(args.config).resolve()))
    config.setdefault("agent", {})
    config["agent"]["enable_strategy"] = False
    config["agent"]["enable_checklist"] = False

    agent = AvenirWebAgent(
        config=config,
        save_file_dir=str(Path(args.output_dir).resolve()),
        default_task="Read the current page and answer a question from a screenshot.",
        default_website=args.website,
        max_auto_op=1,
        mode=args.mode,
        task_id=args.task_id,
        create_timestamp_dir=False,
    )

    try:
        await agent.start(website=args.website)
        await agent.take_screenshot()
        page_title = ""
        try:
            page_title = await agent.page.title() if agent.page else ""
        except Exception:
            page_title = ""

        system_prompt = (
            "You answer questions about a webpage screenshot. "
            "Use only visible evidence from the screenshot and the provided page metadata. "
            "If the answer is uncertain, say so explicitly."
        )
        user_prompt = (
            f"URL: {agent.page.url if agent.page else args.website}\n"
            f"Title: {page_title}\n"
            f"Question: {args.question}\n"
            "Answer concisely."
        )
        answer = await agent.engine.generate(
            prompt=[system_prompt, user_prompt, ""],
            image_path=agent.screenshot_path,
            temperature=0,
            model=agent.model,
            turn_number=0,
        )

        payload = {
            "task_id": args.task_id,
            "website": args.website,
            "question": args.question,
            "current_url": agent.page.url if agent.page else args.website,
            "page_title": page_title,
            "screenshot_path": agent.screenshot_path,
            "answer": answer,
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
