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

import asyncio
import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import toml

from .agent import AvenirWebAgent
from .runtime.llm_engine import is_blank_or_placeholder_api_key


class AvenirWebRunner:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._configure_api_keys()

    @classmethod
    def from_toml(cls, config_path: Union[str, Path]) -> "AvenirWebRunner":
        p = Path(config_path)
        config = toml.load(str(p))
        return cls(config=config)

    def _configure_api_keys(self) -> None:
        api_keys_config = self.config.get("api_keys", {}) if isinstance(self.config, dict) else {}
        if "OPENROUTER_API_KEY" not in os.environ:
            openrouter_key = api_keys_config.get("openrouter_api_key")
            if not is_blank_or_placeholder_api_key(openrouter_key):
                os.environ["OPENROUTER_API_KEY"] = openrouter_key

        if "GEMINI_API_KEY" not in os.environ:
            gemini_key = api_keys_config.get("gemini_api_key")
            if not is_blank_or_placeholder_api_key(gemini_key):
                os.environ["GEMINI_API_KEY"] = gemini_key

    async def __call__(
        self,
        confirmed_task: str,
        website: str,
        task_id: Optional[str] = None,
        max_total_execution_time_s: int = 3600,
        max_consecutive_errors: int = 3,
    ) -> Dict[str, Any]:
        if not task_id:
            task_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        task_dict = {"task_id": task_id, "confirmed_task": confirmed_task, "website": website}
        return await self.run_task(
            task_dict,
            max_total_execution_time_s=max_total_execution_time_s,
            max_consecutive_errors=max_consecutive_errors,
        )

    async def run_task(
        self,
        task_dict: Dict[str, Any],
        max_total_execution_time_s: int = 3600,
        max_consecutive_errors: int = 3,
    ) -> Dict[str, Any]:
        model_config = self.config.get("model", {}) if isinstance(self.config, dict) else {}

        model_name = model_config.get("name", "openrouter/qwen/qwen-2.5-72b-instruct")
        temperature = model_config.get("temperature", 1.0)

        task_id = str(task_dict.get("task_id") or datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        confirmed_task = str(task_dict.get("confirmed_task") or "")
        website = str(task_dict.get("website") or "")

        save_dir = self.config.get("basic", {}).get("save_file_dir", "avenir_web_agent_files")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.isabs(save_dir):
            save_dir = os.path.join(script_dir, save_dir)
        os.makedirs(save_dir, exist_ok=True)

        agent_config = {
            "task_id": task_id,
            "default_task": confirmed_task,
            "default_website": website,
            "save_file_dir": save_dir,
            "max_auto_op": self.config.get("experiment", {}).get("max_op", 30),
            "highlight": self.config.get("experiment", {}).get("highlight", self.config.get("agent", {}).get("highlight", False)),
            "headless": self.config.get("playwright", {}).get("headless", self.config.get("browser", {}).get("headless", True)),
            "viewport": self.config.get("playwright", {}).get("viewport", self.config.get("browser", {}).get("viewport", {"width": 1200, "height": 1080})),
            "model": model_name,
            "temperature": temperature,
            "create_timestamp_dir": False,
        }

        agent = AvenirWebAgent(config=self.config, **agent_config)
        result: Dict[str, Any] = {
            "task_id": task_id,
            "status": "failed",
            "error": None,
            "operations_count": 0,
            "execution_time": 0,
            "output_dir": agent.main_path,
            "result_path": os.path.join(agent.main_path, "result.json"),
        }

        start_time = asyncio.get_event_loop().time()
        consecutive_errors = 0

        try:
            await agent.start(website=website)
            while not agent.complete_flag:
                elapsed = asyncio.get_event_loop().time() - start_time
                if max_total_execution_time_s and elapsed > max_total_execution_time_s:
                    agent.complete_flag = True
                    result["status"] = "timeout"
                    result["error"] = f"Exceeded max_total_execution_time_s={max_total_execution_time_s}"
                    break

                try:
                    prediction_dict = await agent.predict()
                    await agent.execute(prediction_dict)
                    consecutive_errors = 0
                except Exception as e:
                    consecutive_errors += 1
                    logging.getLogger(__name__).warning(f"Task {task_id} step error: {e}", exc_info=True)
                    if consecutive_errors >= max_consecutive_errors:
                        agent.complete_flag = True
                        result["status"] = "error"
                        result["error"] = str(e)
                        break

            result["operations_count"] = getattr(agent, "valid_op", 0)
            if result["status"] == "failed":
                result["status"] = "completed" if agent.complete_flag else "failed"
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            logging.getLogger(__name__).error(f"Task {task_id} failed: {e}", exc_info=True)
        finally:
            try:
                await agent.stop()
            except Exception as stop_e:
                logging.getLogger(__name__).warning(f"Agent stop failed for task {task_id}: {stop_e}", exc_info=True)
            result["execution_time"] = asyncio.get_event_loop().time() - start_time
        return result

    async def run_task_file(
        self,
        task_file_path: Union[str, Path],
        max_total_execution_time_s: int = 3600,
        max_consecutive_errors: int = 3,
    ) -> Dict[str, Any]:
        p = Path(task_file_path)
        tasks: List[Dict[str, Any]] = json.loads(p.read_text(encoding="utf-8"))
        summary = {"total_tasks": len(tasks), "completed": 0, "failed": 0, "results": []}
        for t in tasks:
            r = await self.run_task(
                t,
                max_total_execution_time_s=max_total_execution_time_s,
                max_consecutive_errors=max_consecutive_errors,
            )
            summary["results"].append(r)
            if r.get("status") == "completed":
                summary["completed"] += 1
            else:
                summary["failed"] += 1
        return summary
