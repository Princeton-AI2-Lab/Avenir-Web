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
import json
import logging
import os
import random
import time
from datetime import datetime

import toml
from playwright.async_api import async_playwright, Locator

from avenir_web.utils.infra.repetition_utils import analyze_repetitive_patterns
from avenir_web.prompting.prompts import build_grounding_system_prompt
from avenir_web.runtime.browser import normal_launch_async, normal_new_context_async, \
    saveconfig
from avenir_web.runtime.llm_engine import engine_factory, is_blank_or_placeholder_api_key, load_openrouter_api_key, get_openrouter_base_url
from avenir_web.runtime.llm_engine import LLM_IO_RECORDS, add_llm_io_record
from avenir_web.managers.checklist_manager import ChecklistManager
from avenir_web.utils.strategy.strategist import generate_task_strategy
from avenir_web.utils.agent.summary_utils import (
    llm_summarize_actions,
    llm_update_history_summary
)
from avenir_web.managers.action_manager import ActionHistoryManager
from avenir_web.managers.action_execution_manager import ActionExecutionManager
from avenir_web.managers.step_executor import StepExecutor
from avenir_web.utils.infra.logging_utils import setup_logger
from avenir_web.utils.browser import coordinate_utils


class AvenirWebAgent:
    def __init__(self,
                 config_path=None,
                 config=None,  # Add config parameter
                 save_file_dir="avenir_web_agent_files",
                 default_task='Find the pdf of the paper "GPT-4V(ision) is a Generalist Web Agent, if Grounded"',
                 default_website="https://www.google.com/",
                 crawler_mode=False,
                 max_auto_op=30,
                 highlight=False,
                 headless=False,
                 args=[],
                 browser_app="chrome",
                 viewport={
                     "width": 1280,
                     "height": 720
                 },
                 user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36 Edg/142.0.0.0",
                 model="openrouter/qwen/qwen-2.5-72b-instruct",
                 temperature=1.0,
                 create_timestamp_dir=True,
                 task_id=None  # Add task_id parameter
                 ):

        try:
            if config is not None:
                if not isinstance(config, dict):
                    config = {}
                config.setdefault("basic", {})
                config.setdefault("agent", {})
                config.setdefault("model", {})
                config["basic"]["save_file_dir"] = save_file_dir
                config["basic"].setdefault("default_task", default_task)
                config["basic"].setdefault("default_website", default_website)
                config["basic"].setdefault("crawler_mode", crawler_mode)
                config["agent"]["max_auto_op"] = max_auto_op
                config["agent"]["highlight"] = highlight
                config["model"]["name"] = model
                config["model"]["temperature"] = temperature
            elif config_path is not None:
                with open(config_path,
                          'r') as config_file:
                    logging.getLogger(__name__).debug(f"Configuration file loaded: {config_path}")
                    config = toml.load(config_file)
            else:
                config = {
                    "basic": {
                        "save_file_dir": save_file_dir,
                        "default_task": default_task,
                        "default_website": default_website,
                        "crawler_mode": crawler_mode,
                    },
                    "agent": {
                        "max_auto_op": max_auto_op,
                        "highlight": highlight
                    },
                    "model": {
                        "name": model,
                        "temperature": temperature
                    }
                }
            config.update({
                "browser": {
                    "headless": headless,
                    "args": args,
                    "browser_app": browser_app,
                    "viewport": viewport,
                    # Simple anti-detection settings
                    "user_agent": user_agent
                }
            })

        except FileNotFoundError:
            logging.getLogger(__name__).error(f"Config file not found: {os.path.abspath(config_path)}")
            raise
        except toml.TomlDecodeError:
            logging.getLogger(__name__).error(f"Invalid TOML config file: {os.path.abspath(config_path)}")
            raise

        self.config = config
        self.complete_flag = False
        self.session_control = {
            'active_page': None,
            'context': None,
            'browser': None
        }
        self.default_task = default_task
        self.tasks = [self.default_task]
        
        if task_id is None:
            raise ValueError("task_id must be provided and cannot be None")
        self.task_id = task_id

        if create_timestamp_dir:
            base_dir = os.path.join(self.config["basic"]["save_file_dir"], datetime.now().strftime("%Y%m%d_%H%M%S"))
        else:
            base_dir = self.config["basic"]["save_file_dir"]
        
        self.main_path = os.path.join(base_dir, self.task_id)
        os.makedirs(self.main_path, exist_ok=True)
        self.action_space = ["CLICK", "KEYBOARD", "PRESS ENTER", "WAIT", "HOVER", "SCROLL UP", "SCROLL DOWN", "SCROLL TOP", "SCROLL BOTTOM", "NEW TAB", "CLOSE TAB",
                             "GO BACK", "GO FORWARD",
                             "TERMINATE", "SELECT", "TYPE", "GOTO", "MEMORIZE", "NONE"]  # Define the list of actions here

        self.no_value_op = ["CLICK", "PRESS ENTER", "WAIT", "HOVER", "SCROLL UP", "SCROLL DOWN", "SCROLL TOP", "SCROLL BOTTOM", "NEW TAB", "CLOSE TAB",
                            "PRESS HOME", "PRESS END", "PRESS PAGEUP", "PRESS PAGEDOWN",
                            "GO BACK", "GO FORWARD", "TERMINATE", "NONE"]

        self.with_value_op = ["SELECT", "TYPE", "KEYBOARD", "GOTO", "MEMORIZE", "SAY"]
        self.last_click_coordinates = None
        self.last_click_viewport_coords = None
        self.initial_frame_saved = False
        self._current_coordinates_type = 'normalized'

        self.no_element_op = ["PRESS ENTER", "WAIT", "KEYBOARD", "SCROLL UP", "SCROLL DOWN", "SCROLL TOP", "SCROLL BOTTOM", "NEW TAB", "CLOSE TAB", "GO BACK", "GOTO",
                              "PRESS HOME", "PRESS END", "PRESS PAGEUP", "PRESS PAGEDOWN",
                              "GO FORWARD",
                              "TERMINATE", "NONE", "MEMORIZE", "SAY"]

        # Initialize the primary logger and the developer logger with error handling
        try:
            self.logger = setup_logger(self.task_id, self.main_path, redirect_to_dev_log=False)
        except Exception as e:
            self.logger = logging.getLogger(f"{self.task_id}_fallback")
            self.logger.setLevel(logging.INFO)
            self.logger.handlers.clear()
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter('%(asctime)s - %(message)s')
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
            self.logger.propagate = False
            try:
                self.logger.warning(f"Using fallback console-only logger due to file logger creation failure: {e}")
            except Exception:
                pass

        try:
            model_config = self.config.get('model', {})
            api_keys_config = self.config.get('api_keys', {})
            
            engine_params = {}
            
            engine_params['model'] = model_config.get('name', 'openrouter/qwen/qwen-2.5-72b-instruct')
            
            engine_params['temperature'] = model_config.get('temperature', 1)
            engine_params['rate_limit'] = self.config.get('experiment', {}).get('rate_limit', -1)
            
            api_key = None
            model_name = engine_params['model'].lower()

            def pick_api_key(*candidates):
                for candidate in candidates:
                    if not is_blank_or_placeholder_api_key(candidate):
                        return candidate
                return None
            
            if 'claude' in model_name:
                api_key = pick_api_key(api_keys_config.get('openrouter_api_key'), os.getenv('OPENROUTER_API_KEY'))
            elif 'gemini' in model_name:
                api_key = pick_api_key(
                    api_keys_config.get('openrouter_api_key'),
                    os.getenv('OPENROUTER_API_KEY'),
                    api_keys_config.get('gemini_api_key'),
                    os.getenv('GEMINI_API_KEY'),
                )
            else:
                api_key = pick_api_key(api_keys_config.get('openrouter_api_key'), os.getenv('OPENROUTER_API_KEY'))
            
            if api_key:
                engine_params['api_key'] = api_key
            
            self.engine = engine_factory(**engine_params)
            try:
                setattr(self.engine, 'task_id', self.task_id)
            except Exception:
                pass
            
            # Store model and temperature as instance attributes for predict() method
            self.model = engine_params.get('model', 'openrouter/qwen/qwen-2.5-72b-instruct')
            self.temperature = engine_params.get('temperature', 1.0)
            
            self.logger.info("Engine initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize engine: {e}")
            self.logger.warning("Agent will continue with limited functionality")
            self.engine = None
            self.model = 'openrouter/qwen/qwen-2.5-72b-instruct'
            self.temperature = 1.0
        
        # Initialize checklist engine and ChecklistManager
        try:
            checklist_model = model_config.get('checklist_model', 'openrouter/qwen/qwen3-vl-8b-instruct')
            checklist_engine_params = {'model': checklist_model, 'temperature': 0.7}
            if api_key:
                checklist_engine_params['api_key'] = api_key
            checklist_engine = engine_factory(**checklist_engine_params)
            self.logger.info(f"Checklist engine initialized with model: {checklist_model}")
        except Exception as e:
            self.logger.warning(f"Failed to initialize checklist engine: {e}, using main engine")
            checklist_engine = self.engine
        
        self.checklist_manager = ChecklistManager(
            engine=self.engine,
            checklist_engine=checklist_engine,
            logger=self.logger
        )
        
        self.taken_actions = []
        self.action_history = []
        self.action_summaries = []
        
        self.action_manager = ActionHistoryManager(logger=self.logger, max_stack_size=5, max_action_history=50)

        self.action_execution_manager = ActionExecutionManager(
            config=self.config,
            logger=self.logger,
            engine=self.engine,
            action_manager=self.action_manager,
            coordinate_utils=coordinate_utils
        )
        self.step_executor = StepExecutor(logger=self.logger)
        
        self.task_strategy = ""
        
        self.query_generation_count = 0
        self.max_query_generations = 5
        
        self.checklist_generated = False

        self.time_step = 0
        self.valid_op = 0
        self.predictions = []
        self.visited_links = []
        self._page = None
        
        self._screenshot_path = None

        self.history_summary_interval = 5
        self.history_recent_window = self.config.get('agent', {}).get('history_recent_window', 5)
        self.llm_history_summary_text = ""
        self.llm_summary_covered_steps = 0
        self.is_stopping = False
        from avenir_web.utils.browser.page_event_handlers import PageEventHandlers
        self.page_event_handlers = PageEventHandlers(self)

    async def _generate_task_strategy(self):
        """
        Generate strategic plan about the task using a strategist model.
        Called at the start of task execution before browser launches.
        """
        model_config = self.config.get('model', {})
        strategist_model = model_config.get('strategist_model', self.model)
        strategist_temp = model_config.get('strategist_temperature', 1.0)
        enable_online = model_config.get('strategist_enable_online', True)
        
        # Get current task and website
        current_task = self.tasks[-1] if self.tasks else self.default_task
        # Use actual_website if available (set in start()), otherwise use config default
        current_website = getattr(self, 'actual_website', None) or self.config.get("basic", {}).get("default_website")
        
        self.logger.info("="*70)
        self.logger.info("🧠 TASK STRATEGY PHASE")
        self.logger.info("="*70)
        self.logger.info(f"Task: {current_task}")
        self.logger.info(f"Website: {current_website}")
        self.logger.info(f"Strategist Model: {strategist_model}")
        
        # Build soft constraints for strategy
        try:
            from urllib.parse import urlparse
            allowed_domain = urlparse(current_website).hostname or ""
        except Exception:
            allowed_domain = ""
        from .prompting.prompts import build_task_constraints_prompt
        constraints_text = build_task_constraints_prompt(
            allowed_domain=allowed_domain,
            disallow_login=True,
            disallow_offsite=True,
            extra_rules=""
        )
        plugins_payload = None
        result = await generate_task_strategy(
            task_description=current_task,
            website=current_website,
            model=strategist_model,
            enable_online=enable_online,
            temperature=strategist_temp,
            logger=self.logger,
            policy_constraints=constraints_text,
            plugins=plugins_payload,
            task_id=self.task_id
        )
        
        if result['success']:
            self.task_strategy = result['strategy']
            self.logger.info("="*70)
            self.logger.info("✅ Strategy generated successfully")
            self.logger.info("="*70)
        else:
            self.logger.warning(f"⚠️ Failed to generate strategy: {result['error']}")
            self.task_strategy = ""  # Continue without strategy
        
    async def generate_task_checklist(self, task_description):
        """Generate a checklist for the task using the reasoning model."""
        result = await self.checklist_manager.generate_task_checklist(task_description)
        self.checklist_generated = self.checklist_manager.checklist_generated
        return result

    async def _update_checklist_after_action(self, action_data):
        """Delegate to ChecklistManager."""
        if not self.checklist_manager.task_checklist:
            return
            
        # Get current page context
        current_url = self.page.url if hasattr(self, 'page') and self.page else "Unknown"
        page_title = ""
        try:
            if hasattr(self, 'page') and self.page:
                page_title = await self.page.title()
        except:
            page_title = "Unknown"
        
        # Get page state
        page_state = {}
        try:
            if hasattr(self, 'page') and self.page:
                page_state = await self._capture_page_state()
        except Exception:
            pass
        
        # Collect latest two screenshots for multimodal checklist update
        image_paths = []
        try:
            if self.screenshot_path and os.path.exists(self.screenshot_path):
                image_paths.append(self.screenshot_path)
            prev_path = os.path.join(self.main_path, 'screenshots', f'screen_{max(0, self.time_step-1)}.png')
            if os.path.exists(prev_path):
                image_paths.append(prev_path)
        except Exception:
            pass

        # Delegate to checklist manager
        await self.checklist_manager.update_checklist_after_action(
            action_data,
            current_url,
            page_title,
            page_state,
            self.action_history,
            image_paths=image_paths
        )

    def save_action_history(self, filename="action_history.txt"):
        """Save the history of taken actions to a file in the main path."""
        history_path = os.path.join(self.main_path, filename)
        with open(history_path, 'w') as f:
            for action in self.taken_actions:
                f.write(action + '\n')
        self.logger.info(f"Action history saved to: {history_path}")

    async def _is_page_blocked_or_blank(self):
        try:
            return await self.action_execution_manager._is_page_blocked_or_blank(self.page)
        except Exception:
            return True

    async def start(self, headless=None, args=None, website=None):
        self.actual_website = website if website is not None else self.config.get("basic", {}).get("default_website", "Unknown website")
        
        await self._generate_task_strategy()
        
        self.playwright = await async_playwright().start()
        self.session_control = {}
        self.session_control['browser'] = await normal_launch_async(self.playwright,
                                                                    headless=self.config['browser'][
                                                                        'headless'] if headless is None else headless,
                                                                    args=self.config['browser']['args'] if args is None else args,
                                                                    channel=self.config['browser'].get('browser_app', 'chrome'))
        geo_cfg = (
            self.config.get('browser', {}).get('geolocation') or
            self.config.get('playwright', {}).get('geolocation')
        )
        self.session_control['context'] = await normal_new_context_async(
            self.session_control['browser'],
            viewport=self.config['browser']['viewport'],
            user_agent=self.config['browser']['user_agent'],
            geolocation=geo_cfg
        )

        self.session_control['context'].on("page", self.page_event_handlers.on_open)
        page = await self.session_control['context'].new_page()
        await self.page_event_handlers.on_open(page)

        if self.config["basic"].get("crawler_mode", False) is True:
            await self.session_control['context'].tracing.start(screenshots=True, snapshots=True)

        if website is not None:
            try:
                await self.page.goto(website, wait_until="load")
                self.logger.info(f"Loaded website: {website}")
                
                if await self._is_page_blocked_or_blank():
                    self.logger.error("⛔ Website is blocked, blank, or failed to load properly")
                    self.logger.error("Continuing anyway (only TERMINATE should exit)")
            except Exception as e:
                self.logger.error(f"Failed to load website: {e}")
                self.logger.error("⛔ Terminating due to page load failure")
                self.logger.error("Continuing anyway (only TERMINATE should exit)")
        else:
            self.logger.info("Browser started without initial navigation. Use GOTO action to navigate to a website.")

        if self.tasks and not self.checklist_generated:
            task_description = self.tasks[-1] if isinstance(self.tasks[-1], str) else str(self.tasks[-1])
            self.logger.info(f"Generating checklist for task: {task_description}")
            try:
                await self.generate_task_checklist(task_description)  # Add await since it's now async
                self.logger.info("Checklist generation completed successfully")
            except Exception as e:
                self.logger.error(f"Checklist generation failed: {e}")
                self.checklist_manager.task_checklist = [
                    {"id": "execute", "description": f"Execute task: {task_description[:50]}...", "status": "pending"}
                ]
                self.checklist_manager.checklist_generated = True
                self.checklist_generated = True
                self.logger.info("Created fallback checklist, continuing execution")
            
    async def perform_action(self, target_element=None, action_name=None, value=None, target_coordinates=None,
                             element_repr=None, field_name=None, action_description=None, clear_first: bool = True,
                             press_enter_after: bool = False):
        """Perform action with hybrid grounding - delegated to ActionExecutionManager."""
        result = await self.action_execution_manager.perform_action(
            page=self.page,
            action_name=action_name,
            value=value,
            target_element=target_element,
            target_coordinates=target_coordinates,
            field_name=field_name,
            tasks=self.tasks,
            element_repr=element_repr,
            session_control=self.session_control,
            screenshot_path=self.screenshot_path
        )
        return result

    async def predict(self):
        """
        Generate a prediction for the next action using a unified tool-calling format.
        Single LLM call returns both reasoning and action in one response.
        Always returns a valid prediction dictionary, never None.
        """
        if not self.initial_frame_saved:
            await self.take_screenshot()
            self.initial_frame_saved = True
        self.time_step += 1
        
        # Exit only if model outputs TERMINATE or max_auto_op is reached
        try:
            max_auto_op = int(self.config.get("agent", {}).get("max_auto_op", 0))
        except Exception:
            max_auto_op = 0
        if max_auto_op > 0 and self.valid_op >= max_auto_op:
            return {"action": "TERMINATE", "value": "Max operations reached", "element": None, "coordinates": None, "field": ""}

        os.makedirs(os.path.join(self.main_path, 'screenshots'), exist_ok=True)
        await self.take_screenshot()
        if not self.screenshot_path:
            self.logger.error("Screenshot path unavailable; continuing anyway (only TERMINATE should exit)")

        try:
            # Trigger summary when backlog reaches interval size
            # This ensures that when we have accumulated 'interval' new actions, they get summarized immediately,
            # instead of waiting for a specific time_step or keeping a fixed recent window.
            backlog_size = len(self.taken_actions) - self.llm_summary_covered_steps
            if backlog_size >= self.history_summary_interval:
                # Summarize everything in the backlog up to the interval boundary (or all of it if we want batch processing)
                # To be safe and consistent, we summarize exactly 'interval' steps, or all if configured.
                # Here we choose to summarize the full chunk that triggered the threshold.
                older_end = len(self.taken_actions)
                
                # Ensure we are actually advancing
                if older_end > self.llm_summary_covered_steps:
                    delta = self.taken_actions[self.llm_summary_covered_steps:older_end]
                    if self.llm_summary_covered_steps == 0 and older_end > 0 and not self.llm_history_summary_text:
                        engine = getattr(self.checklist_manager, 'checklist_engine', self.engine)
                        summary = await llm_summarize_actions(delta, engine)
                    else:
                        engine = getattr(self.checklist_manager, 'checklist_engine', self.engine)
                        summary = await llm_update_history_summary(delta, self.llm_history_summary_text, engine)
                    self.llm_history_summary_text = summary
                    self.llm_summary_covered_steps = older_end
                    if summary:
                        self.logger.info(f"📝 History Summary: {summary}")
            
            if self.taken_actions:
                action_lines = []
                if self.llm_history_summary_text:
                    action_lines.append(f"📌 Summary: {self.llm_history_summary_text}")
                
                start_idx = self.llm_summary_covered_steps
                
                max_lookback = max(20, self.history_recent_window * 2)
                if len(self.taken_actions) - start_idx > max_lookback:
                    start_idx = max(0, len(self.taken_actions) - max_lookback)
                
                actions_to_show = self.taken_actions[start_idx:]
                
                for i, action in enumerate(actions_to_show):
                    desc = action.get('action_description', 'Unknown action')
                    success = action.get('success', True)
                    error = action.get('error', None)
                    if error:
                        status = f"❌ FAILED ({error})"
                    elif success:
                        status = "✅ SUCCESS"
                    else:
                        status = "⚠️ UNCERTAIN"
                    action_lines.append(f"Step {start_idx + i + 1}: {desc} - {status}")
                previous_actions_text = "\n".join(action_lines)
                try:
                    warnings_text = analyze_repetitive_patterns(self.taken_actions)
                    if warnings_text:
                        previous_actions_text += "\nWarnings: " + warnings_text
                except Exception:
                    pass
            else:
                previous_actions_text = "No previous actions yet."
                actions_to_show = []
            
            checklist_context = self.checklist_manager.format_checklist_for_prompt() if self.checklist_manager.task_checklist else ""
            
            from .prompting.prompts import (
                build_action_response_format,
                build_system_prompt,
                build_task_constraints_prompt,
                parse_structured_action,
                parse_tool_call,
                validate_parsed_action,
            )
            
            try:
                from urllib.parse import urlparse
                start_url = getattr(self, 'actual_website', None) or self.config.get("basic", {}).get("default_website")
                allowed_domain = urlparse(start_url).hostname or ""
            except Exception:
                allowed_domain = ""
            constraints_text = build_task_constraints_prompt(
                allowed_domain=allowed_domain,
                disallow_login=True,
                disallow_offsite=True,
                extra_rules=""
            )

            suggested_next = ""
            try:
                cs = self.checklist_manager.get_checklist_status()
                if cs.get("total", 0) > 0 and cs.get("completed", 0) >= cs.get("total", 0):
                    suggested_next = "TERMINATE"
            except Exception:
                suggested_next = ""
            model_cfg = self.config.get("model", {}) if isinstance(self.config, dict) else {}
            structured_outputs_enabled = bool(model_cfg.get("structured_outputs", True))
            system_prompt_struct, user_prompt_struct = build_system_prompt(
                task=self.tasks[-1],
                previous_actions=previous_actions_text,
                checklist_context=checklist_context,
                suggested_next_step=suggested_next,
                strategic_reasoning=self.task_strategy,
                policy_constraints=constraints_text,
                use_structured_output=True,
            )
            system_prompt_tool, user_prompt_tool = build_system_prompt(
                task=self.tasks[-1],
                previous_actions=previous_actions_text,
                checklist_context=checklist_context,
                suggested_next_step=suggested_next,
                strategic_reasoning=self.task_strategy,
                policy_constraints=constraints_text,
                use_structured_output=False,
            )
            
            self.logger.info(f"Step {self.time_step} | Task: {self.tasks[-1]}")
            
            parsed_action = None
            use_structured = structured_outputs_enabled
            response_format = build_action_response_format() if use_structured else None
            for attempt in range(3):
                if attempt == 0:
                    repair = ""
                else:
                    if use_structured:
                        repair = "Your last output was invalid. Return EXACTLY one JSON object matching the required schema."
                    else:
                        repair = "Your last output was invalid. Return EXACTLY one <tool_call> with name 'browser_use' and arguments containing 'action'. Include 'description' for all actions. Keep strings <=200 chars."
                # Guard image path existence to avoid FileNotFound errors
                screenshot_for_prediction = self.screenshot_path if (self.screenshot_path and os.path.exists(self.screenshot_path)) else None
                action_model = self.model
                try:
                    if isinstance(action_model, str) and (":online" in action_model.lower()) and ("qwen" in action_model.lower()):
                        action_model = action_model.replace(":online", "")
                except Exception:
                    pass
                base_user = user_prompt_struct if use_structured else user_prompt_tool
                base_system = system_prompt_struct if use_structured else system_prompt_tool
                user_with_repair = base_user if attempt == 0 else (base_user + "\n" + repair)
                try:
                    gen_kwargs = {
                        "prompt": [base_system, user_with_repair, ""],
                        "image_path": screenshot_for_prediction,
                        "temperature": self.temperature,
                        "model": action_model,
                        "turn_number": 0,
                    }
                    if use_structured and response_format:
                        gen_kwargs["response_format"] = response_format
                    llm_response = await self.engine.generate(**gen_kwargs)
                except Exception as e:
                    if use_structured:
                        self.logger.warning(f"Structured outputs failed, falling back: {e}")
                        use_structured = False
                        response_format = None
                        fallback_repair = "" if attempt == 0 else "Your last output was invalid. Return EXACTLY one <tool_call> with name 'browser_use' and arguments containing 'action'. Include 'description' for all actions. Keep strings <=200 chars."
                        fallback_user = user_prompt_tool if attempt == 0 else (user_prompt_tool + "\n" + fallback_repair)
                        llm_response = await self.engine.generate(
                            prompt=[system_prompt_tool, fallback_user, ""],
                            image_path=screenshot_for_prediction,
                            temperature=self.temperature,
                            model=action_model,
                            turn_number=0,
                        )
                    else:
                        raise

                parsed_action = parse_structured_action(llm_response) if use_structured else parse_tool_call(llm_response)
                self.logger.debug(f"Raw LLM response: {llm_response}")
                self.logger.debug(f"Parsed action: {parsed_action}")
                if parsed_action and isinstance(parsed_action, dict):
                    parsed_action.setdefault('value', '')
                    parsed_action.setdefault('coordinates', None)
                    parsed_action.setdefault('field', '')
                    parsed_action.setdefault('action_description', '')
                    if isinstance(parsed_action['value'], str) and len(parsed_action['value']) > 200:
                        parsed_action['value'] = parsed_action['value'][:200]
                    if isinstance(parsed_action['field'], str) and len(parsed_action['field']) > 100:
                        parsed_action['field'] = parsed_action['field'][:100]
                    if isinstance(parsed_action['action_description'], str) and len(parsed_action['action_description']) > 200:
                        parsed_action['action_description'] = parsed_action['action_description'][:200]
                    coords = parsed_action.get('coordinates')
                    if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                        try:
                            parsed_action['coordinates'] = [int(coords[0]), int(coords[1])]
                        except Exception:
                            parsed_action['coordinates'] = None
                    ok, reason = validate_parsed_action(parsed_action)
                    if ok:
                        try:
                            self.logger.info(f"Predicted: {parsed_action.get('action')} | {parsed_action.get('action_description','')}")
                        except Exception:
                            pass
                        break
                    else:
                        self.logger.error(f"Schema invalid: {reason}")
                        parsed_action = None
            
            if not parsed_action or not parsed_action.get('action'):
                self.logger.error("=" * 80)
                self.logger.error("PARSING FAILURE - LLM Response Did Not Match Expected Format")
                self.logger.error("=" * 80)
                self.logger.error(f"Full LLM Response:\n{llm_response}")
                self.logger.error("=" * 80)
                if use_structured:
                    self.logger.error("Expected format: a single JSON object matching the action schema")
                else:
                    self.logger.error("Expected format: <tool_call>{\"name\": \"browser_use\", \"arguments\": {action: ...}}</tool_call>")
                self.logger.error("=" * 80)
                
                # If can't extract anything useful, check if response was empty/malformed
                if not llm_response or not llm_response.strip():
                    self.logger.error("Critical: LLM returned empty response repeatedly.")
                    return {"action": "WAIT", "value": "1", "element": None, "coordinates": None, "field": ""}

                # If response had content but no tool call, and no other intent detected
                self.logger.warning("Returning WAIT action to avoid counting malformed response as no-op")
                return {"action": "WAIT", "value": "1", "element": None, "coordinates": None, "field": ""}
            
            # Note: Action will be added to taken_actions in execute_action()
            # with full enhanced record including success/failure status
            
            return parsed_action
                
        except Exception as e:
            self.logger.error(f"Prediction error: {e}", exc_info=True)
            return {"action": "WAIT", "value": "1", "element": None, "coordinates": None, "field": ""}

    async def enhanced_gui_click(self, coords, element_repr):
        try:
            delay = random.randint(50, 150)
            nx, ny = coordinate_utils.map_normalized_to_pixels(coords[0], coords[1], self.page, self.config)
            await self.page.mouse.click(round(nx), round(ny), delay=delay)
            self.logger.info(f"Clicked at coordinates ({nx}, {ny}): {element_repr}")
            return True
        except Exception as e:
            self.logger.info(f"GUI click failed: {e}")
            return False

    async def execute(self, prediction_dict):
        return await self.step_executor.execute_step(self, prediction_dict)

    async def stop(self):
        self.is_stopping = True
        # Always try to close browser context first
        try:
            close_context = None
            try:
                if isinstance(self.session_control, dict):
                    close_context = self.session_control.get('context', None)
                else:
                    close_context = None
            except Exception:
                close_context = None
            if close_context:
                await close_context.close()
                self.logger.info("Browser context closed.")
            # Clear stored context after attempting close
            if isinstance(self.session_control, dict):
                self.session_control['context'] = None
        except Exception as e:
            self.logger.warning(f"Error closing browser context: {e}")
        
        # Close playwright instance to ensure complete browser shutdown
        try:
            if hasattr(self, 'playwright') and self.playwright:
                await self.playwright.stop()
                self.logger.info("Playwright instance stopped.")
                self.playwright = None
        except Exception as e:
            self.logger.warning(f"Error stopping playwright instance: {e}")

        # Prepare data for saving - use safe defaults if anything fails
        action_history_for_output = []
        
        try:
            for action in self.taken_actions:
                if isinstance(action, dict):
                    action_history_for_output.append({
                        "step": action.get('step', 'N/A'),
                        "action": action.get('predicted_action', ''),
                        "value": action.get('predicted_value', ''),
                        "element": action.get('element_description', ''),
                        "error": action.get('error', ''),
                        "coordinates": action.get('coordinates'),
                    })
                else:
                    action_history_for_output.append(str(action))
        except Exception as e:
            self.logger.error(f"Error processing action history: {e}")
            action_history_for_output = [f"Error processing action history: {str(e)}"]

        final_json = {
            "confirmed_task": self.default_task if hasattr(self, 'default_task') and self.default_task else "Unknown task",
            "website": getattr(self, 'actual_website', self.config.get("basic", {}).get("default_website", "Unknown website")) if hasattr(self, 'config') and self.config else "Unknown website",
            "task_id": getattr(self, 'task_id', 'demo_task'),
            "num_step": len(self.taken_actions) if hasattr(self, 'taken_actions') else 0,
            "action_history": action_history_for_output,
        }

        def locator_serializer(obj):
            """Convert non-serializable objects to a serializable format."""
            if isinstance(obj, Locator):
                return str(obj)
            try:
                return str(obj)
            except:
                return f"<Non-serializable object: {type(obj).__name__}>"

        def _simplify_messages(msgs):
            result = {"system": [], "user": [], "assistant": []}
            try:
                for m in msgs or []:
                    role = m.get("role", "user")
                    content = m.get("content", [])
                    texts = []
                    if isinstance(content, list):
                        for c in content:
                            if isinstance(c, dict) and c.get("type") == "text":
                                texts.append(str(c.get("text", "")))
                    elif isinstance(content, str):
                        texts.append(content)
                    if role in result:
                        result[role].append("\n".join(texts))
                for k in result:
                    result[k] = "\n\n".join([t for t in result[k] if t])
            except Exception:
                return {"system": "", "user": "", "assistant": ""}
            return result

        def _format_llm_records(records):
            formatted = []
            try:
                for r in records or []:
                    msgs = r.get("messages")
                    simplified = _simplify_messages(msgs) if msgs else {"system": "", "user": "", "assistant": ""}
                    formatted.append({
                        "timestamp": r.get("timestamp"),
                        "model": r.get("model"),
                        "turn_number": r.get("turn_number", 0),
                        "input": simplified,
                        "images": r.get("image_paths") or ([r.get("image_path")] if r.get("image_path") else []),
                        "output": r.get("output"),
                        "context": r.get("context")
                    })
            except Exception:
                return []
            return formatted

        llm_records_filename = "llm_records.json"
        try:
            with open(os.path.join(self.main_path, llm_records_filename), 'w', encoding='utf-8') as f:
                try:
                    filtered_records = [r for r in LLM_IO_RECORDS if r.get('task_id') == self.task_id]
                except Exception:
                    filtered_records = []
                formatted_records = _format_llm_records(filtered_records)
                json.dump(formatted_records, f, default=locator_serializer, indent=4)
            self.logger.info(f"Successfully saved {llm_records_filename}")
        except Exception as e:
            self.logger.error(f"Failed to save {llm_records_filename}: {e}")
            try:
                with open(os.path.join(self.main_path, llm_records_filename), 'w', encoding='utf-8') as f:
                    json.dump({"error": f"Failed to save predictions: {str(e)}"}, f, indent=4)
            except Exception as fallback_e:
                self.logger.error(f"Failed to save fallback {llm_records_filename}: {fallback_e}")

        # Save result.json with error handling
        try:
            with open(os.path.join(self.main_path, 'result.json'), 'w', encoding='utf-8') as file:
                json.dump(final_json, file, indent=4)
            self.logger.info("Successfully saved result.json")
        except Exception as e:
            self.logger.error(f"Failed to save result.json: {e}")
            # Try to save a minimal version
            try:
                minimal_result = {
                    "task_id": getattr(self, 'task_id', 'demo_task'),
                    "num_step": len(self.taken_actions) if hasattr(self, 'taken_actions') else 0,
                    "error": str(e),
                    "action_history": action_history_for_output
                }
                with open(os.path.join(self.main_path, 'result.json'), 'w', encoding='utf-8') as file:
                    json.dump(minimal_result, file, indent=4)
                self.logger.info("Successfully saved minimal result.json")
            except Exception as fallback_e:
                self.logger.error(f"Failed to save minimal result.json: {fallback_e}")

        # Save config with error handling
        try:
            saveconfig(self.config, os.path.join(self.main_path, 'config.toml'))
            self.logger.info("Successfully saved config.toml")
        except Exception as e:
            self.logger.error(f"Failed to save config.toml: {e}")

        self.logger.info("Agent stopped - all save operations attempted.")

    def _emergency_save(self, error_info="Unknown error"):
        """
        Emergency save mechanism when normal save operations fail.
        Saves minimal data to ensure no information is lost.
        """
        try:
            import tempfile
            import datetime
            
            # Create emergency save directory
            emergency_dir = os.path.join(tempfile.gettempdir(), f"avenir_web_emergency_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
            os.makedirs(emergency_dir, exist_ok=True)
            
            # Save minimal result data
            emergency_result = {
                "task_id": getattr(self, 'task_id', 'unknown'),
                "emergency_save": True,
                "error_info": error_info,
                "timestamp": datetime.datetime.now().isoformat(),
                "num_actions": len(getattr(self, 'taken_actions', [])),
                "error": error_info
            }
            
            # Try to save action history if available
            try:
                if hasattr(self, 'taken_actions') and self.taken_actions:
                    emergency_result["action_summary"] = []
                    for i, action in enumerate(self.taken_actions[-5:]):  # Save last 5 actions
                        if isinstance(action, dict):
                            emergency_result["action_summary"].append({
                                "step": action.get('step', i),
                                "action": action.get('predicted_action', 'unknown'),
                                "description": action.get('action_description', '')[:100]  # Truncate long descriptions
                            })
                        else:
                            emergency_result["action_summary"].append(str(action)[:100])
            except Exception:
                emergency_result["action_summary"] = "Failed to extract action summary"
            
            # Save emergency result
            emergency_file = os.path.join(emergency_dir, 'emergency_result.json')
            with open(emergency_file, 'w', encoding='utf-8') as f:
                json.dump(emergency_result, f, indent=4)
            
            # Log emergency save location
            self.logger.error(f"EMERGENCY SAVE: Data saved to {emergency_file}")
            
            return emergency_file
            
        except Exception as emergency_e:
            # If even emergency save fails, log to console
            error_msg = f"CRITICAL: Emergency save failed: {emergency_e}. Original error: {error_info}"
            self.logger.critical(error_msg)
            return None

    def clear_action_history(self):
        """
        Clears the history of actions taken by the agent.
        """
        self.taken_actions.clear()
        self.action_history.clear()  # Also clear action_history to prevent memory leak
        self.logger.info("Cleared action history and action_history.")

    def reset_comlete_flag(self, flag=False):
        self.complete_flag = flag

    def change_task(self, new_task, clear_history=False):
        """
        Changes the task requirement for the agent.

        Parameters:
        - new_task: The new task requirement as a string.
        """
        if new_task and isinstance(new_task, str):

            self.logger.info(f"Changed task from {self.tasks[-1]} to: {new_task}")
            self.tasks.append(new_task)
            # Optionally clear action history when changing task
            if clear_history:
                self.clear_action_history()
            else:
                task_change_action = {
                    "step": self.time_step,
                    "action_description": f"Changed task from {self.tasks[-2]} to: {new_task}",
                    "action_generation_response": "",
                    "action_grounding_response": "",
                    "predicted_action": "TASK_CHANGE",
                    "predicted_value": new_task,
                    "element_description": "",
                    "success": True,
                    "error": None,
                    "http_response": {},
                    "page_content_summary": "Task changed successfully"
                }
                self.taken_actions.append(task_change_action)
                # Also add to action_history for failure analysis
                self.action_history.append(task_change_action)

        else:
            self.logger.info("Invalid new task. It must be a non-empty string.")

        # Optionally, you can save the taken_actions to a file or database for record-keeping

    # ADD no op count and op count, add limit to op

    # decompose run to predict and execute.

    async def take_screenshot(self):
        """
        Take a viewport screenshot of the current page.
        
        For the unified system: Always captures viewport-only (1000x1000 coordinate system).
        This ensures coordinates from LLM map directly to visible elements.
        
        Args:
            None
        """
        screenshot_path = os.path.join(self.main_path, "screenshots", f"screen_{self.time_step}.png")
        try:
            from avenir_web.utils.browser import screenshot_utils

            screenshot_utils.ensure_screenshots_dir(self.main_path)
            screenshot_path = screenshot_utils.build_screenshot_path(self.main_path, self.time_step)
            attempts = 0
            max_attempts = 3
            while attempts < max_attempts:
                try:
                    await screenshot_utils.capture_viewport_screenshot(self.page, screenshot_path, logger=self.logger)
                    self.screenshot_path = screenshot_path
                    if not screenshot_utils.is_uniform_image(self.screenshot_path):
                        break
                    attempts += 1
                    try:
                        await self.page.evaluate("window.scrollTo(0, 0)")
                    except Exception:
                        pass
                    try:
                        await asyncio.sleep(1)
                    except Exception:
                        pass
                    if attempts == 2:
                        try:
                            context = self.session_control.get('context') if isinstance(self.session_control, dict) else None
                            target_url = None
                            if hasattr(self, 'page') and self.page:
                                try:
                                    target_url = self.page.url
                                except Exception:
                                    target_url = None
                            if context:
                                new_page = await context.new_page()
                                await self.page_event_handlers.on_open(new_page)
                                if target_url:
                                    try:
                                        await new_page.goto(target_url, wait_until="domcontentloaded")
                                    except Exception:
                                        pass
                                self.page = new_page
                        except Exception:
                            pass
                except Exception:
                    break
            self.logger.info(f"Viewport screenshot taken: {self.screenshot_path}")
        except Exception as e:
            self.logger.warning(f"Failed to take screenshot: {e}")
            try:
                from avenir_web.utils.browser import screenshot_utils

                try:
                    await self.page.wait_for_load_state('domcontentloaded', timeout=10000)
                except Exception:
                    pass
                try:
                    await screenshot_utils.capture_viewport_screenshot(self.page, screenshot_path, logger=self.logger)
                    self.screenshot_path = screenshot_path
                    self.logger.info(f"Viewport screenshot taken: {self.screenshot_path}")
                    return
                except Exception:
                    pass
                context = self.session_control.get('context') if isinstance(self.session_control, dict) else None
                if context:
                    new_page = await context.new_page()
                    await self.page_event_handlers.on_open(new_page)
                    target_url = None
                    last_resp = self.session_control.get('last_response') if isinstance(self.session_control, dict) else None
                    if last_resp and isinstance(last_resp, dict):
                        target_url = last_resp.get('url')
                    if not target_url and hasattr(self, 'actual_website'):
                        target_url = self.actual_website
                    if not target_url:
                        try:
                            pages = context.pages
                            if pages:
                                target_url = pages[-1].url
                        except Exception:
                            target_url = None
                    if target_url:
                        try:
                            await new_page.goto(target_url, wait_until="domcontentloaded")
                        except Exception:
                            pass
                    self.page = new_page
                    if target_url:
                        await screenshot_utils.capture_viewport_screenshot(self.page, screenshot_path, logger=self.logger)
                        self.screenshot_path = screenshot_path
                        self.logger.info(f"Viewport screenshot taken: {self.screenshot_path}")
                        return
            except Exception as rec_e:
                self.logger.error(f"Screenshot capture failed: {rec_e}")
            self.screenshot_path = None

    async def _take_full_page_screenshot_with_cropping(self, target_elements=None, screenshot_path=None):
        if screenshot_path is None:
            screenshot_path = self.screenshot_path

        from avenir_web.utils.browser import screenshot_utils
        await screenshot_utils.take_full_page_screenshot_with_cropping(
            self.page,
            screenshot_path=screenshot_path,
            target_elements=target_elements,
            logger=self.logger,
            timeout_ms=20000,
        )

    async def start_playwright_tracing(self):
        if (self.session_control and isinstance(self.session_control, dict) and 
            self.session_control.get('context') and 
            hasattr(self.session_control['context'], 'tracing')):
            await self.session_control['context'].tracing.start_chunk(
                title=f'Step-{self.time_step}',
                name=f"{self.time_step}"
            )

    async def stop_playwright_tracing(self):
        if (self.session_control and isinstance(self.session_control, dict) and 
            self.session_control.get('context') and 
            hasattr(self.session_control['context'], 'tracing')):
            await self.session_control['context'].tracing.stop_chunk(path=self.trace_path)

    async def save_traces(self):
        dom_tree = await self.page.evaluate("document.documentElement.outerHTML")
        os.makedirs(os.path.join(self.main_path, 'dom'), exist_ok=True)
        with open(self.dom_tree_path, 'w', encoding='utf-8') as f:
            f.write(dom_tree)

        accessibility_tree = await self.page.accessibility.snapshot()
        os.makedirs(os.path.join(self.main_path, 'accessibility'), exist_ok=True)
        with open(self.accessibility_tree_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(accessibility_tree, indent=4))

    @property
    def page(self):
        if self._page is None:
            if (self.session_control and isinstance(self.session_control, dict) and 
                self.session_control.get('active_page')):
                self._page = self.session_control['active_page']
        return self._page

    @page.setter
    def page(self, value):
        self._page = value

    @property
    def screenshot_path(self):
        if self._screenshot_path:
            return self._screenshot_path
        return os.path.join(self.main_path, 'screenshots', f'screen_{self.time_step}.png')
    
    @screenshot_path.setter
    def screenshot_path(self, value):
        self._screenshot_path = value

    @property
    def trace_path(self):
        return os.path.join(self.main_path, 'playwright_traces', f'{self.time_step}.zip')

    @property
    def dom_tree_path(self):
        return os.path.join(self.main_path, 'dom', f'{self.time_step}.html')
    
    def encode_image(self, image_path):
        """
        Encode image to base64 for GUI grounding API with automatic compression if needed.
        This method now uses the image module to handle large images.
        """
        from .runtime.image import encode_image_with_compression
        return encode_image_with_compression(image_path)
    
    async def ground_element(self, image_path, instruction):
        """Use GUI grounding API to locate element."""
        try:
            # Validate inputs
            if not image_path or not os.path.exists(image_path):
                self.logger.error(f"Invalid image path for GUI grounding: {image_path}")
                return None
                
            if not instruction or not instruction.strip():
                self.logger.error("Empty instruction for GUI grounding")
                return None
                
            base64_image = self.encode_image(image_path)
            
            # Get viewport dimensions for coordinate range
            viewport_width = self.config.get('browser', {}).get('viewport', {}).get('width', 1280)
            viewport_height = self.config.get('browser', {}).get('viewport', {}).get('height', 720)
            
            # Use 0-1000 normalized coordinate output (main model handles positioning)
            system_content = build_grounding_system_prompt()
            
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": system_content
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": instruction},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ],
                "max_tokens": 100,
                "temperature": 0.1
            }
            
            self.logger.info(f"Sending GUI grounding request for: {instruction}")
            
            # Use LiteLLM for unified API paradigm (same as main agent)
            # LiteLLM automatically handles the openrouter/ prefix correctly
            import litellm
            response = await litellm.acompletion(
                model=self.model,
                messages=payload["messages"],
                max_tokens=100,
                temperature=0.1,
                api_key=load_openrouter_api_key(),
                base_url=get_openrouter_base_url(),
            )
            
            if response and response.choices and len(response.choices) > 0:
                choice = response.choices[0]
                if choice.message and choice.message.content:
                    content = choice.message.content.strip()
                    self.logger.info(f"Coordinate response: {content}")
                    try:
                        add_llm_io_record({
                            "model": self.model if isinstance(self.model, str) else str(self.model),
                            "turn_number": 0,
                            "messages": payload["messages"],
                            "image_paths": [image_path] if image_path else None,
                            "output": content,
                            "context": "coordinate_extraction"
                        })
                    except Exception:
                        pass
                    coords = coordinate_utils.parse_coordinates(content, self.page, self.config, self.logger)
                    if coords:
                        self.logger.info(f"Successfully parsed coordinates: {coords}")
                    else:
                        self.logger.warning(f"Failed to parse coordinates from response: {content}")
                    return coords
                else:
                    self.logger.error("Invalid message structure in coordinate response")
                    return None
            else:
                self.logger.error("Invalid response format from coordinate API")
                return None
        except Exception as e:
            self.logger.error(f"Coordinate request failed: {e}")
            return None
    
    @property
    def accessibility_tree_path(self):
        return os.path.join(self.main_path, 'accessibility', f'{self.time_step}.json')

    async def _capture_page_state(self):
        from avenir_web.utils.browser import page_state_utils

        return await page_state_utils.capture_page_state(
            self.page,
            pending_hit_test_coords=getattr(self, "_pending_hit_test_coords", None),
            logger=self.logger,
        )

    


    async def _detect_page_state_change(self, state_before, state_after, action_type):
        from avenir_web.utils.browser import page_state_utils

        action_successful, changes_detected = page_state_utils.detect_page_state_change(
            state_before,
            state_after,
            action_type,
            logger=self.logger,
        )

        if isinstance(action_type, str) and action_type in ["CLICK", "TYPE", "SELECT", "KEYBOARD", "PRESS ENTER"] and not action_successful:
            try:
                self._last_changes_detected = list(changes_detected)
                self._last_action_successful = False
            except Exception:
                pass
            self.logger.warning(f"No clear evidence of success for {action_type} action")
            return False

        page_state_utils.record_action_result(
            self,
            step=self.time_step,
            action_type=action_type,
            successful=bool(action_successful),
            changes=changes_detected,
        )
        try:
            self._last_changes_detected = list(changes_detected)
            self._last_action_successful = bool(action_successful)
        except Exception:
            pass

        return action_successful
    
    async def _analyze_previous_action_results(self):
        """
        Analyze previous action results to determine if the agent should change strategy or terminate.
        This reflection mechanism helps prevent repetitive failures and improves decision making.
        """
        try:
            if not self.taken_actions or len(self.taken_actions) < 2:
                return {"should_terminate": False, "strategy_suggestions": []}
            
            # Analyze last 5 actions for patterns
            recent_actions = self.taken_actions[-5:]
            
            # Check for repetitive failures
            failed_actions = [action for action in recent_actions 
                            if isinstance(action, dict) and not action.get('success', True)]
            
            # Check for same action repetition
            action_types = [action.get('predicted_action', '') for action in recent_actions 
                          if isinstance(action, dict)]
            
            # Check for same element targeting
            element_descriptions = [action.get('element_description', '') for action in recent_actions 
                                  if isinstance(action, dict) and action.get('element_description')]
            
            # Analyze patterns
            analysis_result = {
                "should_terminate": False,
                "reason": "",
                "strategy_suggestions": [],
                "alternative_actions": []
            }
            
            # Pattern 1: Too many consecutive failures
            if len(failed_actions) >= 3:
                analysis_result["should_terminate"] = True
                analysis_result["reason"] = f"Detected {len(failed_actions)} consecutive failures in recent actions"
                return analysis_result
            
            # Pattern 2: Repetitive action types (like clicking same element repeatedly)
            if len(action_types) >= 3:
                most_common_action = max(set(action_types), key=action_types.count)
                if action_types.count(most_common_action) >= 3:
                    # Check if these repetitive actions are failing
                    repetitive_failures = sum(1 for i, action in enumerate(recent_actions) 
                                            if (isinstance(action, dict) and 
                                                action.get('predicted_action') == most_common_action and 
                                                not action.get('success', True)))
                    
                    if repetitive_failures >= 2:
                        analysis_result["should_terminate"] = True
                        analysis_result["reason"] = f"Repeated {most_common_action} action failing {repetitive_failures} times"
                        return analysis_result
                    else:
                        # Suggest alternative strategies
                        analysis_result["strategy_suggestions"] = [
                            f"Consider alternatives to repeated {most_common_action} actions",
                            "Try using search functionality if available",
                            "Consider scrolling to find different elements",
                            "Try SELECT action if dropdown elements are available"
                        ]
            
            # Pattern 3: Same element targeting repeatedly without success
            if len(element_descriptions) >= 3:
                unique_elements = set(element_descriptions)
                if len(unique_elements) == 1:  # All targeting same element
                    last_action = recent_actions[-1]
                    if isinstance(last_action, dict) and not last_action.get('success', True):
                        analysis_result["strategy_suggestions"] = [
                            "Current element targeting is not working - try different elements",
                            "Look for alternative navigation paths",
                            "Consider using search or filter functionality",
                            "Try scrolling to reveal more options"
                        ]
            
            # Pattern 4: Check for available alternative actions not being used
            last_action = recent_actions[-1] if recent_actions else None
            if isinstance(last_action, dict):
                action_generation = last_action.get('action_generation_response', '').lower()
                
                # Check if agent is aware of search boxes but not using them
                if ('search' in action_generation and 'not relevant' in action_generation):
                    analysis_result["strategy_suggestions"].append(
                        "Reconsider using search functionality - it might be more relevant than initially thought"
                    )
                
                # Check if SELECT actions are available but not considered
                if 'select' not in action_generation and 'dropdown' in action_generation:
                    analysis_result["alternative_actions"].append("SELECT")
                    analysis_result["strategy_suggestions"].append(
                        "Consider using SELECT action for dropdown elements"
                    )
            
            # Log reflection analysis
            if analysis_result["strategy_suggestions"] or analysis_result["alternative_actions"]:
                self.logger.info("=== REFLECTION ANALYSIS ===")
                self.logger.info(f"Strategy suggestions: {analysis_result['strategy_suggestions']}")
                self.logger.info(f"Alternative actions: {analysis_result['alternative_actions']}")
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Error in reflection analysis: {e}")
            return {"should_terminate": False, "strategy_suggestions": []}
