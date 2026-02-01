# Copyright 2026 Princeton AI for Accelerating Invention Lab
# Author: Aiden Yiliu Li
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# See LICENSE.txt for the full license text.

import argparse
import asyncio
import os
import sys
from pathlib import Path


def _ensure_src_on_path() -> Path:
    repo_root = Path(__file__).resolve().parent
    src_dir = repo_root / "src"
    sys.path.insert(0, str(src_dir))
    return repo_root


async def main_async() -> int:
    repo_root = _ensure_src_on_path()

    from avenir_web.runner import AvenirWebRunner

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(repo_root / "src" / "config" / "batch_experiment.toml"))
    parser.add_argument("--task", required=True)
    parser.add_argument("--website", required=True)
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--output-dir", default=str(repo_root / "outputs"))
    parser.add_argument("--headless", default=None)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    runner = AvenirWebRunner.from_toml(config_path)

    if isinstance(runner.config, dict):
        runner.config.setdefault("basic", {})
        runner.config["basic"]["save_file_dir"] = str(Path(args.output_dir).resolve())

        if args.headless is not None:
            headless_str = str(args.headless).strip().lower()
            headless_bool = headless_str in ("1", "true", "yes", "y", "t")
            runner.config.setdefault("playwright", {})
            runner.config["playwright"]["headless"] = headless_bool

    if not os.getenv("OPENROUTER_API_KEY"):
        raise SystemExit("Missing OPENROUTER_API_KEY in environment")

    result = await runner(
        confirmed_task=args.task,
        website=args.website,
        task_id=args.task_id,
    )
    print(result)
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()
