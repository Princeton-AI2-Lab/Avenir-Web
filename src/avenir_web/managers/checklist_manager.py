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

import json
import logging
from ..prompting.prompts import build_checklist_prompt, build_checklist_update_prompt
from ..utils.agent.checklist import (
    get_checklist_status as _util_get_checklist_status,
    format_checklist_for_prompt as _util_format_checklist_for_prompt
)

class ChecklistManager:
    def __init__(self, engine, logger=None, max_query_generations=5, checklist_engine=None):
        """
        Initialize ChecklistManager with optional separate lightweight engine.
        
        Args:
            engine: Main agent engine (used if checklist_engine not provided)
            logger: Logger instance
            max_query_generations: Max attempts for checklist generation
            checklist_engine: Optional lightweight engine for checklist operations (e.g., 4B model)
        """
        self.engine = engine
        self.checklist_engine = checklist_engine or engine  # Use dedicated engine if provided
        self.logger = logger or logging.getLogger("ChecklistManager")
        self.task_checklist = []
        self.checklist_generated = False
        self.query_generation_count = 0
        self.max_query_generations = max_query_generations
        self.pending_updates = []  # Track background update tasks

    def _extract_json(self, text):
        """Extract JSON object from text using lightweight heuristics."""
        if not text:
            return ""
            
        text = text.strip()

        def extract_first_json_object(s: str):
            start = s.find("{")
            if start < 0:
                return ""
            depth = 0
            in_str = False
            esc = False
            for i in range(start, len(s)):
                ch = s[i]
                if in_str:
                    if esc:
                        esc = False
                        continue
                    if ch == "\\":
                        esc = True
                        continue
                    if ch == "\"":
                        in_str = False
                    continue
                if ch == "\"":
                    in_str = True
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return s[start : i + 1]
            return ""

        fence_start = text.find("```")
        if fence_start != -1:
            fence_end = text.find("```", fence_start + 3)
            if fence_end != -1:
                inner = text[fence_start + 3 : fence_end].strip()
                if inner.lower().startswith("json"):
                    inner = inner[4:].strip()
                extracted = extract_first_json_object(inner) or inner
                return extracted

        extracted = extract_first_json_object(text)
        return extracted or text

    async def generate_task_checklist(self, task_description):
        """
        Generate a dynamic checklist based on task description using LLM
        """
        # Check for loop prevention - limit query generation attempts
        if self.query_generation_count >= self.max_query_generations:
            self.logger.error(f"Maximum query generation attempts ({self.max_query_generations}) reached, checklist generation failed")
            # Return empty checklist to indicate failure
            self.task_checklist = []
            self.checklist_generated = False
            return self.task_checklist
            
        # Increment query generation count
        self.query_generation_count += 1
        self.logger.debug(f"Query generation attempt {self.query_generation_count}/{self.max_query_generations}")
        
        # Check if engine is available
        if not self.engine:
            self.logger.error("Engine not available, cannot generate checklist")
            self.task_checklist = []
            self.checklist_generated = False
            return self.task_checklist
            
        try:
            checklist_prompt = build_checklist_prompt(task_description)

            # Use lightweight checklist engine for faster generation
            response = await self.checklist_engine.generate(
                prompt=["", checklist_prompt, ""],
                temperature=0.7,  # Lower temperature for more focused checklist
                max_new_tokens=500,  # Reduced tokens for simpler checklist
                turn_number=0
            )
            
            if not response:
                self.logger.error("Invalid response from engine.generate for checklist generation")
                raise Exception("Failed to get valid response from LLM for checklist generation")
            
            # Handle different response formats from different engines
            if isinstance(response, list):
                if not response:
                    self.logger.error("Empty response list from engine.generate for checklist generation")
                    raise Exception("Failed to get valid response from LLM for checklist generation")
                if response[0] is None:
                    self.logger.error("First element in response list is None for checklist generation")
                    raise Exception("Failed to get valid response from LLM for checklist generation - None element")
                response_text = response[0].strip()
            elif isinstance(response, str):
                response_text = response.strip()
            elif hasattr(response, 'choices') and response.choices and len(response.choices) > 0 and hasattr(response.choices[0], 'message') and response.choices[0].message:
                response_text = response.choices[0].message.content.strip() if response.choices[0].message.content else ""
            else:
                self.logger.error("Unknown response format from engine.generate for checklist generation")
                raise Exception("Failed to get valid response from LLM for checklist generation")
            
            # Parse JSON response
            self.logger.debug(f"Raw checklist response from LLM: {response_text}")
            
            # Robust JSON extraction
            try:
                try:
                    # First try direct parse
                    checklist_data = json.loads(response_text)
                except json.JSONDecodeError:
                    # Try extracting from code blocks or finding the first/last brace
                    extracted_text = response_text
                    
                    extracted_text = self._extract_json(response_text)
                    
                    try:
                        checklist_data = json.loads(extracted_text)
                    except json.JSONDecodeError:
                        self.logger.error("Failed to parse checklist JSON from LLM response")
                        self.logger.error(f"Raw response that failed to parse: {response_text}")
                        raise json.JSONDecodeError("Failed to extract JSON", response_text, 0)

                self.task_checklist = checklist_data.get("checklist", [])
                self._enforce_atomic_items()
                self.checklist_generated = True
                
                self.logger.debug(f"Generated checklist with {len(self.task_checklist)} items")
                for item in self.task_checklist:
                    self.logger.debug(f"  - {item['description']}")
                    
                return self.task_checklist
            
            except json.JSONDecodeError:
                self.logger.error("Failed to parse checklist JSON from LLM response")
                self.logger.error(f"Raw response that failed to parse: {response_text}")
                fallback_desc = (task_description or "Task").strip()
                self.task_checklist = [
                    {"id": "execute", "description": f"Execute task: {fallback_desc[:50]}...", "status": "pending"},
                    {"id": "complete", "description": "Complete the task", "status": "pending"}
                ]
                self._enforce_atomic_items()
                self.checklist_generated = True
                self.logger.info("Checklist JSON parse failed; created fallback checklist")
                return self.task_checklist
                
        except Exception as e:
            self.logger.error(f"Error generating checklist: {e}")
            fallback_desc = (task_description or "Task").strip()
            self.task_checklist = [
                {"id": "execute", "description": f"Execute task: {fallback_desc[:50]}...", "status": "pending"},
                {"id": "complete", "description": "Complete the task", "status": "pending"}
            ]
            self._enforce_atomic_items()
            self.checklist_generated = True
            self.logger.info("Checklist generation error; created fallback checklist")
            return self.task_checklist

    def _enforce_atomic_items(self, max_words: int = 10):
        items = []
        seen_descriptions = set()
        for item in self.task_checklist or []:
            desc = (item.get('description', '') or '').strip()
            normalized = desc.replace(";", ",").replace(" and ", ",").replace(" And ", ",")
            parts = [p.strip() for p in normalized.split(",") if p.strip()]
            for p in parts:
                t = p.strip()
                if not t:
                    continue
                words = t.split()
                t2 = " ".join(words[:max_words])
                
                # Deduplicate items based on description
                if t2.lower() in seen_descriptions:
                    continue
                seen_descriptions.add(t2.lower())

                st = (item.get('status', 'pending') or 'pending').strip().lower().replace(' ', '_')
                if st not in ('pending', 'in_progress', 'completed', 'failed'):
                    st = 'pending'
                items.append({
                    "id": f"requirement_{len(items)+1}",
                    "description": t2,
                    "status": st
                })
        self.task_checklist = items

    def update_checklist_item(self, item_id, status, description=None):
        """Update the status of a checklist item with validation and logging"""
        if not self.task_checklist:
            self.logger.warning("Cannot update checklist item: no checklist exists")
            return False
            
        valid_statuses = ['pending', 'in_progress', 'completed', 'failed']
        normalized = (status or '').strip().lower().replace(' ', '_')
        if normalized not in valid_statuses:
            self.logger.error(f"Invalid status '{status}'. Must be one of: {valid_statuses}")
            return False
            
        for item in self.task_checklist:
            if item['id'] == item_id:
                old_status = item.get('status', 'pending')
                item['status'] = normalized
                if description:
                    item['description'] = description
                    
                self.logger.info(f"Checklist item '{item_id}' updated: {old_status} -> {normalized}")
                if description:
                    self.logger.info(f"Description updated: {description}")
                return True
                
        self.logger.warning(f"Checklist item '{item_id}' not found")
        return False

    def get_checklist_status(self):
        """Delegate to checklist utils for status computation."""
        return _util_get_checklist_status(self.task_checklist)

    def format_checklist_for_prompt(self):
        """Delegate to checklist utils for checklist formatting."""
        return _util_format_checklist_for_prompt(self.task_checklist)

    async def update_checklist_after_action(self, action_data, current_url, page_title, page_state, action_history, image_paths=None):
        """
        Update checklist items based on action execution results
        
        Args:
            action_data (dict): Enhanced action data with execution results
            current_url (str): Current page URL
            page_title (str): Current page title
            page_state (dict): Current page state dictionary
            action_history (list): Recent action history
        """
        if not self.task_checklist:
            self.logger.warning("Checklist update skipped: no checklist available")
            return
            
        action_type = action_data.get('action', '').upper()
        success = action_data.get('success', False)
        error = action_data.get('error')
        
        # Format a compact page state summary
        page_state_text = (
            f"- URL: {page_state.get('url', current_url)}\n"
            f"- Title: {page_state.get('title', page_title)}\n"
            f"- Interactive Elements: {page_state.get('interactive_elements_count', 'Unknown')}\n"
            f"- Form Fields: {page_state.get('form_fields_count', 'Unknown')}\n"
            f"- Modal Present: {page_state.get('modal_present', False)}\n"
            f"- Visible Text (first 400 chars): {page_state.get('visible_text', '')[:400]}"
        )

        # Use full action history for context
        history_text = ""
        for i, act in enumerate(action_history or []):
            history_text += f"Action {i+1}: {act.get('action', 'Unknown')} - {act.get('value', 'None')}\n"
        
        # Format current checklist
        checklist_text = self.format_checklist_for_prompt()
        
        try:
            update_prompt = build_checklist_update_prompt(
                action_type=action_type,
                success=success,
                error=error,
                history_text=history_text,
                page_state_text=page_state_text,
                checklist_text=checklist_text
            )

            response = await self.checklist_engine.generate(
                prompt=["", update_prompt, ""],
                temperature=0.0,
                max_new_tokens=300,
                turn_number=0,
                image_paths=image_paths
            )
            
            if not response:
                self._apply_fallback_update(success, error)
                return
                
            # Parse response
            if isinstance(response, list):
                response_text = response[0].strip() if response and response[0] else ""
            elif isinstance(response, str):
                response_text = response.strip()
            elif hasattr(response, 'choices') and response.choices:
                response_text = response.choices[0].message.content.strip()
            else:
                self._apply_fallback_update(success, error)
                return

            # Parse JSON
            json_text = self._extract_json(response_text)
            
            try:
                update_data = json.loads(json_text)
                updates = update_data.get("updates", [])
                
                applied = False
                for update in updates:
                    item_id = update.get("item_id")
                    new_status = update.get("new_status")
                    reason = update.get("reason")
                    
                    if item_id and new_status and not applied:
                        ok = self.update_checklist_item(item_id, new_status, description=None)
                        if ok:
                            self.logger.info(f"Checklist auto-update (atomic): {item_id} -> {new_status} ({reason})")
                            applied = True
                            break
                
                if not applied:
                    self._apply_fallback_update(success, error)
                
                if self.task_checklist:
                    snapshot = ", ".join([f"{item.get('id')}: {item.get('status','pending')}" for item in self.task_checklist])
                    self.logger.debug(f"Checklist snapshot: {snapshot}")
                
            except json.JSONDecodeError:
                self.logger.warning(f"Failed to parse checklist update JSON: {response_text}")
                self._apply_fallback_update(success, error)
                
        except Exception as e:
            self.logger.error(f"Error in checklist update: {e}")

    def _apply_fallback_update(self, success, error):
        target_idx = None
        for i, item in enumerate(self.task_checklist):
            st = item.get('status', 'pending')
            if st in ('in_progress', 'pending'):
                target_idx = i
                if st == 'in_progress':
                    break
        if target_idx is not None:
            item = self.task_checklist[target_idx]
            old_status = item.get('status', 'pending')
            if success and not error:
                new_status = 'completed' if old_status == 'in_progress' else 'in_progress'
            else:
                new_status = 'failed'
            ok = self.update_checklist_item(item.get('id'), new_status)
            if ok:
                self.logger.info(f"Checklist auto-update (fallback): {item.get('id')} -> {new_status}")
        else:
            self.logger.info("Checklist update: no applicable item found for fallback")

    def add_checklist_item(self, description, item_id=None):
        """Add a new item to the checklist"""
        if item_id is None:
            item_id = f"requirement_{len(self.task_checklist) + 1}"
            
        new_item = {
            "id": item_id,
            "description": description,
            "status": "pending"
        }
        
        self.task_checklist.append(new_item)
        self.logger.info(f"Added checklist item: {description} (ID: {item_id})")
        return item_id

    def remove_checklist_item(self, item_id):
        """Remove an item from the checklist"""
        for i, item in enumerate(self.task_checklist):
            if item['id'] == item_id:
                removed = self.task_checklist.pop(i)
                self.logger.info(f"Removed checklist item: {removed['description']}")
                return True
        return False
