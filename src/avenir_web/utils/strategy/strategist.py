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

"""
Reasoning utilities for task planning using advanced models.
"""

import litellm
from ...runtime.llm_engine import add_llm_io_record, load_openrouter_api_key, get_openrouter_base_url
from ...prompting.prompts import (
    build_task_reasoning_system_prompt,
    build_task_reasoning_user_prompt
)


async def generate_task_strategy(task_description: str, website: str = None,
                                  model: str = "anthropic/claude-sonnet-4.5", 
                                  enable_thinking: bool = True, enable_online: bool = True,
                                  use_web_search: bool = False,
                                  temperature: float = 1.0, logger=None,
                                  policy_constraints: str = "",
                                  plugins: list = None,
                                  task_id: str = None) -> dict:
    """
    Generate strategic plan for a web automation task.
    
    Args:
        task_description: The task to accomplish
        website: Target website URL (optional but recommended)
        model: Model to use (with or without openrouter/ prefix)
        enable_thinking: Reserved for future thinking-mode models
        enable_online: Reserved for future online-search models
        temperature: Sampling temperature
        logger: Optional logger
        
    Returns:
        dict with keys: strategy (str), success (bool), error (str or None)
    """
    try:
        # Build web-automation specific prompt
        system_prompt = build_task_reasoning_system_prompt()

        # Build user prompt with website context
        user_prompt = build_task_reasoning_user_prompt(website, task_description, policy_constraints)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Ensure openrouter/ prefix
        api_model = model if model.startswith("openrouter/") else f"openrouter/{model}"
        if enable_online and api_model.startswith("openrouter/") and not api_model.endswith(":online"):
            api_model = f"{api_model}:online"
        
        # Enforce GPT-5 usage and forbid o1-family
        if "o1" in api_model.lower():
            raise ValueError("O1 models are not permitted for strategy. Please configure appropriate model instead.")
        
        if logger:
            logger.info(f"🧠 Generating task strategy with {api_model}")
        
        # Do not send tools or extra_body for OpenRouter – match other calls
        
        # Call LiteLLM
        api_key = load_openrouter_api_key()
        call_params = {
            "model": api_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 300,
            "api_key": api_key,
            "base_url": get_openrouter_base_url(),
        }
        if enable_online and plugins:
            call_params["extra_body"] = {"plugins": plugins}
        
        # Keep request minimal to match other OpenRouter usage in repo
        
        response = await litellm.acompletion(**call_params)
        strategy_text = response.choices[0].message.content or ""
        strategy_text = strategy_text.strip()
        def extract_plan_block(text: str):
            if not isinstance(text, str):
                return None
            lower = text.lower()
            open_tag = "<plan>"
            close_tag = "</plan>"
            start = lower.find(open_tag)
            if start < 0:
                return None
            end = lower.find(close_tag, start + len(open_tag))
            if end < 0:
                return None
            return text[start + len(open_tag) : end].strip()

        plan = extract_plan_block(strategy_text)
        if not plan:
            repair_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt + "\n\nReturn ONLY <plan>...</plan> with 2–4 imperative action sentences. No commentary."}
            ]
            try:
                repair_params = dict(call_params)
                repair_params["messages"] = repair_messages
                repair_params["temperature"] = 0.0
                repair_params["max_tokens"] = 220
                repair_resp = await litellm.acompletion(**repair_params)
                repair_text = (repair_resp.choices[0].message.content or "").strip()
                plan2 = extract_plan_block(repair_text)
                if plan2:
                    strategy_text = plan2
            except Exception:
                pass
        else:
            strategy_text = plan
        if logger:
            logger.info(f"✅ Generated strategy: {strategy_text}")
        try:
            add_llm_io_record({
                "model": api_model,
                "turn_number": 0,
                "messages": messages,
                "image_paths": None,
                "output": strategy_text,
                "context": "task_strategy",
                "task_id": task_id
            })
        except Exception:
            pass
        return {
            "strategy": strategy_text,
            "success": bool(strategy_text),
            "error": None if strategy_text else "Empty response"
        }
            
    except Exception as e:
        if logger:
            logger.error(f"❌ Strategy generation failed: {e}")
        return {
            "strategy": "",
            "success": False,
            "error": str(e)
        }
