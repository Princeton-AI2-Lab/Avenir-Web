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
import logging
import os
import traceback

from ..utils.browser import coordinate_utils
from ..utils.agent import text_utils


class StepExecutor:
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)

    async def execute_step(self, agent, prediction_dict):
        if prediction_dict is None:
            self.logger.warning("=== PREDICTION DICT IS NONE ===")
            return 1

        if not isinstance(prediction_dict, dict):
            self.logger.error(f"=== INVALID PREDICTION DICT TYPE: {type(prediction_dict)} ===")
            return 1

        required_keys = ["element", "action", "value"]
        for key in required_keys:
            if key not in prediction_dict:
                if key == "element":
                    prediction_dict[key] = None
                elif key == "action":
                    prediction_dict[key] = "NONE"
                elif key == "value":
                    prediction_dict[key] = ""

        pred_element = prediction_dict.get("element")
        pred_action = prediction_dict.get("action", "NONE")
        pred_value = prediction_dict.get("value", "")
        pred_coordinate = None
        pred_coordinate_type = prediction_dict.get("coordinates_type") or "normalized"

        if pred_action is None:
            pred_action = "NONE"
        elif not isinstance(pred_action, str):
            pred_action = str(pred_action)

        if pred_value is None:
            pred_value = ""

        pred_element_description = (
            prediction_dict.get("action_description")
            or prediction_dict.get("description")
            or ""
        )

        if "coordinates" in prediction_dict:
            pred_coordinate = prediction_dict.get("coordinates")
            try:
                if isinstance(pred_coordinate, (list, tuple)) and len(pred_coordinate) >= 2:
                    pred_coordinate = (float(pred_coordinate[0]), float(pred_coordinate[1]))
            except Exception:
                pred_coordinate = prediction_dict.get("coordinates")

        pred_field = prediction_dict.get("field")

        self.logger.info("=== EXECUTING PREDICTED ACTION ===")
        self.logger.info(f"Step: {agent.time_step}")
        self.logger.info(f"Predicted Action: {pred_action}")
        self.logger.info(f"Predicted Value: {pred_value}")
        self.logger.info(f"Element Description: {pred_element_description}")
        self.logger.info(f"Element: {pred_element}")
        self.logger.info(f"Coordinates: {pred_coordinate}")

        if pred_action not in getattr(agent.action_execution_manager, "no_element_op", []):
            if pred_element is None and pred_coordinate is None and pred_action not in ("TYPE", "SELECT"):
                pred_action = "NONE"

        element_info = pred_element_description or (pred_element.get("description") if isinstance(pred_element, dict) else "")
        coordinates_for_stack = None
        if isinstance(pred_coordinate, (list, tuple)) and len(pred_coordinate) >= 2:
            coordinates_for_stack = (pred_coordinate[0], pred_coordinate[1])
        agent.action_manager.add_action_to_stack(
            pred_action,
            element_info=element_info,
            coordinates=coordinates_for_stack,
            taken_actions_count=len(agent.taken_actions),
        )
        try:
            try:
                agent._pending_hit_test_coords = None
                if pred_action == "CLICK" and isinstance(pred_coordinate, (list, tuple)) and len(pred_coordinate) >= 2:
                    cx, cy = pred_coordinate[0], pred_coordinate[1]
                    if pred_coordinate_type == "normalized":
                        px, py = coordinate_utils.map_normalized_to_pixels(cx, cy, agent.page, agent.config)
                    else:
                        px, py = coordinate_utils.normalize_coords(cx, cy, agent.page, agent.config, agent.logger)
                    agent._pending_hit_test_coords = (int(px), int(py))
            except Exception:
                agent._pending_hit_test_coords = None

            try:
                state_before = await agent._capture_page_state()
            except Exception:
                state_before = {"error": "capture_failed_before"}

            agent._current_coordinates_type = pred_coordinate_type
            new_action = await agent.perform_action(
                pred_element,
                pred_action,
                pred_value,
                pred_coordinate,
                pred_element_description,
                pred_field,
                prediction_dict.get("action_description"),
                clear_first=prediction_dict.get("clear_first", True) if pred_action in ("TYPE", "KEYBOARD") else True,
                press_enter_after=prediction_dict.get("press_enter_after", False) if pred_action == "TYPE" else False,
            )

            if new_action is None:
                new_action = f"Action {pred_action} completed but returned None"
            elif not isinstance(new_action, str):
                new_action = str(new_action)
            if pred_action == "TERMINATE":
                agent.complete_flag = True

            await asyncio.sleep(1)
            try:
                state_after = await agent._capture_page_state()
            except Exception:
                state_after = {"error": "capture_failed_after"}

            try:
                new_action = text_utils.compose_action_description(
                    pred_action,
                    pred_value,
                    pred_field,
                    pred_element_description,
                    pred_coordinate,
                )
            except Exception:
                pass

            try:
                agent._last_changes_detected = []
                action_success = await agent._detect_page_state_change(state_before, state_after, pred_action)
            except Exception:
                action_success = True

            self.logger.info(f"Action executed: {new_action}")

            enhanced_action = {
                "step": agent.time_step,
                "action": pred_action,
                "action_description": prediction_dict.get("action_description") or new_action,
                "action_generation_response": prediction_dict.get("action_generation", ""),
                "action_grounding_response": prediction_dict.get("action_grounding", ""),
                "predicted_action": pred_action,
                "predicted_value": pred_value,
                "element_description": pred_element_description if pred_element_description else (pred_element.get("description", "") if isinstance(pred_element, dict) else ""),
                "coordinates": pred_coordinate,
                "element_center": (pred_element.get("center_point") if isinstance(pred_element, dict) else None),
                "element_box": (pred_element.get("box") if isinstance(pred_element, dict) else None),
                "http_response": agent.session_control.get("last_response", {}) if agent.session_control and isinstance(agent.session_control, dict) else {},
                "success": action_success,
                "error": None,
                "page_content_summary": "Page content summary unavailable",
                "changes_detected": getattr(agent, "_last_changes_detected", None),
            }

            agent.taken_actions.append(enhanced_action)
            agent.action_history.append(enhanced_action)
            agent.taken_actions = agent.action_manager.manage_action_history(agent.taken_actions)

            await agent._update_checklist_after_action(enhanced_action)

            if pred_action != "NONE":
                agent.valid_op += 1

            try:
                agent._pending_hit_test_coords = None
            except Exception:
                pass

            if agent.config.get("basic", {}).get("crawler_mode", False) is True:
                try:
                    await agent.stop_playwright_tracing()
                    await agent.save_traces()
                except Exception:
                    pass

            return 0
        except Exception as e:
            return await self.handle_execute_exception(
                agent,
                prediction_dict,
                pred_action,
                pred_value,
                pred_field,
                pred_element_description,
                pred_coordinate,
                e,
            )

    async def handle_execute_exception(self, agent, prediction_dict, pred_action, pred_value, pred_field, pred_element_description, pred_coordinate, exc: Exception):
        self.logger.error(f"Action execution failed: {exc}")
        traceback_info = traceback.format_exc()
        self.logger.error(traceback_info)

        try:
            base_desc = text_utils.compose_action_description(
                pred_action,
                pred_value,
                pred_field,
                pred_element_description,
                pred_coordinate,
            )
            action_desc = f"{base_desc} - failed: {exc}"
        except Exception:
            action_desc = f"Failed to perform {pred_action} with value '{pred_value}': {exc}"

        enhanced_action = {
            "step": agent.time_step,
            "action": pred_action,
            "action_description": prediction_dict.get("action_description") or action_desc,
            "action_generation_response": prediction_dict.get("action_generation", ""),
            "action_grounding_response": prediction_dict.get("action_grounding", ""),
            "predicted_action": pred_action,
            "predicted_value": pred_value,
            "element_description": pred_element_description or "",
            "coordinates": pred_coordinate,
            "success": False,
            "error": str(exc),
            "http_response": agent.session_control.get("last_response", {}) if agent.session_control and isinstance(agent.session_control, dict) else {},
        }

        agent.taken_actions.append(enhanced_action)
        agent.action_history.append(enhanced_action)
        agent.taken_actions = agent.action_manager.manage_action_history(agent.taken_actions)

        try:
            await agent.take_screenshot()
        except Exception:
            pass

        try:
            await agent._update_checklist_after_action(enhanced_action)
        except Exception:
            pass

        return 1
