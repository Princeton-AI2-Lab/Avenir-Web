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
import sys
import traceback
import json
import random
import os
from ..utils.browser.dom_ops import dom_select_option, get_all_select_elements, decide_form_element, decide_select_option, dom_select_by_selector
from ..utils.infra.evaluation_utils import looks_like_api_endpoint
from ..prompting.prompts import build_field_selection_prompt

class ActionExecutionManager:
    def __init__(self, config, logger, engine, action_manager, coordinate_utils):
        self.config = config
        self.logger = logger
        self.engine = engine
        self.action_manager = action_manager
        self.coordinate_utils = coordinate_utils
        
        self.no_element_op = ["SCROLL UP", "SCROLL DOWN", "SCROLL TOP", "SCROLL BOTTOM", "GO BACK", "GO FORWARD", "CLOSE TAB", "STOP", "TERMINATE", "Wait", "WAIT"]
        self.with_value_op = ["TYPE", "SELECT", "GOTO", "WAIT", "SAY", "MEMORIZE"]

    async def perform_action(self, page, action_name, value=None, target_element=None, 
                           target_coordinates=None, selector=None, field_name=None, 
                           tasks=None, element_repr=None, session_control=None,
                           screenshot_path=None):
        """Perform action with hybrid grounding - unified approach regardless of grounding type."""
        
        element_info = element_repr or (target_element.get('description') if target_element else None)
        coordinates = None
        if target_coordinates:
            if isinstance(target_coordinates, dict):
                coordinates = (round(target_coordinates["x"]), round(target_coordinates["y"]))
            elif isinstance(target_coordinates, (tuple, list)) and len(target_coordinates) >= 2:
                coordinates = (round(target_coordinates[0]), round(target_coordinates[1]))
        elif target_element and target_element.get('center_point'):
            coords = target_element.get('center_point')
            coordinates = (round(coords[0]), round(coords[1]))

        if target_element is not None and isinstance(target_element, dict):
            selector_obj = target_element.get('selector')
            element_repr = target_element.get('description', 'Unknown element')
        else:
            selector_obj = None

        if action_name == "CLICK":
            return await self._perform_click(target_coordinates, element_repr, page)
            
        elif action_name == "HOVER":
            return await self._perform_hover(target_coordinates, target_element, selector, element_repr, page)
            
        elif action_name == "TYPE":
            return await self._perform_type(target_coordinates, value, field_name, tasks, page)
            
        elif action_name == "SCROLL UP" or action_name == "SCROLL DOWN":
            return await self._perform_scroll(action_name, target_coordinates, page)
            
        elif action_name == "SCROLL TOP":
            await page.evaluate("window.scrollTo(0, 0)")
            self.logger.info("Scrolled to top")
            return "Scrolled to top"
            
        elif action_name == "SCROLL BOTTOM":
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            self.logger.info("Scrolled to bottom")
            return "Scrolled to bottom"
            
        elif action_name == "NEW TAB":
             if session_control and 'context' in session_control:
                 # Check page limits if implemented in agent, but here we just open
                 # We don't have max_pages check here easily unless passed.
                 # Assuming safe to open.
                 new_page = await session_control['context'].new_page()
                 self.logger.info(f"Opened a new tab (total pages: {len(session_control['context'].pages)})")
                 return f"Opened a new tab (total pages: {len(session_control['context'].pages)})"
             return "FAILED: session_control not provided"
             
        elif action_name == "CLOSE TAB":
            await page.close()
            self.logger.info("Closed the current tab")
            return "Closed the current tab"
            
        elif action_name == "GO BACK":
            await page.go_back()
            self.logger.info("Navigated back")
            return "Navigated back"
            
        elif action_name == "GO FORWARD":
            await page.go_forward()
            self.logger.info("Navigated forward")
            return "Navigated forward"
            
        elif action_name == "GOTO":
            return await self._perform_goto(value, page)
            
        elif action_name == "PRESS ENTER":
            return await self._perform_press_enter(selector, element_repr, page)
            
        elif action_name == "WAIT":
             duration = 1
             if value:
                 try:
                     if isinstance(value, (int, float)):
                        duration = max(0, float(value))
                     elif isinstance(value, str) and value.strip():
                        duration = max(0, float(value.strip()))
                 except: pass
             await asyncio.sleep(duration)
             self.logger.info(f"Waited {duration} seconds")
             return f"Waited {duration} seconds"
             
        elif action_name == "SELECT":
            return await self._perform_select(target_coordinates, value, field_name, tasks, page)
            
        elif action_name == "TERMINATE":
            if await self._is_page_blocked_or_blank(page):
                self.logger.info("Page is blocked/error - allowing termination without task completion check")
                return "Page blocked or error detected. Task cannot be completed. Terminating..."
            self.logger.info("Agent requested termination. Terminating...")
            return "Agent requested termination. Terminating..."
            
        elif action_name in ["SAY"]:
            say_text = (value or "").strip()
            self.logger.info(f"SAY (Chain of Thought): {say_text}")
            return f"SAY (thinking step): {say_text}"
            
        elif action_name in ["MEMORIZE"]:
            self.logger.info(f"Keep {value} to the action history.")
            return f"Keep {value} to the action history."
            
        elif action_name == "GUI_GROUNDING":
            self.logger.info("GUI_GROUNDING disabled: use main model actions with normalized coordinates")
            return "FAILED: GUI_GROUNDING disabled"
            
        else:
            self.logger.error(f"Unsupported action: {action_name}")
            return f"FAILED: Unsupported action - {action_name}"

    async def _perform_click(self, target_coordinates, element_repr, page):
        # 1. Try coordinate-based click first
        if target_coordinates:
            try:
                if isinstance(target_coordinates, dict):
                    llm_x, llm_y = target_coordinates["x"], target_coordinates["y"]
                else:
                    llm_x, llm_y = target_coordinates[0], target_coordinates[1]
                
                # Normalize
                scaled_x, scaled_y = self.coordinate_utils.map_normalized_to_pixels(llm_x, llm_y, page, self.config)
                self.logger.info(f"🎯 [CLICK] Coordinate Mapping: input({llm_x},{llm_y}) -> Viewport({scaled_x},{scaled_y})")

                # Perform click
                delay = random.randint(50, 150)
                try:
                    await page.mouse.click(scaled_x, scaled_y, delay=delay)
                    self.logger.info(f"✅ [CLICK] Successfully clicked at viewport coordinates ({scaled_x}, {scaled_y})")
                    return f"Clicked at coordinates ({scaled_x}, {scaled_y})"
                except Exception as click_error:
                    self.logger.error(f"❌ [CLICK] Playwright click failed: {click_error} | Trying mouse down/up fallback")
                    await page.mouse.move(scaled_x, scaled_y)
                    await page.mouse.down()
                    await asyncio.sleep(0.05)
                    await page.mouse.up()
                    self.logger.debug(f"✅ [CLICK] Fallback mouse down/up executed successfully")
                    return f"Clicked at coordinates ({scaled_x}, {scaled_y})"
                    
            except Exception as e:
                self.logger.error(f"Click failed: {e}")
                # Fallback to text handled below
        
        # 2. Try text/element-based click as FALLBACK
        if element_repr:
            self.logger.info(f"⚠️ [CLICK] Coordinate click failed/missing, attempting fallback to text match for: {element_repr}")
            found_elem, _, _ = await self._find_click_target_by_text(element_repr, page)
            if found_elem and found_elem.get('box'):
                box = found_elem['box']
                cx = box['x'] + box['width'] / 2
                cy = box['y'] + box['height'] / 2
                self.logger.info(f"✅ [CLICK] Fallback found element by text at ({cx}, {cy})")
                await page.mouse.click(cx, cy)
                return f"Clicked element found by text: {element_repr}"
        
        return "FAILED: No coordinates or element for CLICK"

    async def _perform_type(self, target_coordinates, value, field_name, tasks, page):
        field_name = field_name or 'unknown'
        self.logger.info(f"🎯 [TYPE] Starting | Field: '{field_name}' | Value: '{value}'")
        
        # 1. Coordinate click to focus
        focused = False
        if target_coordinates:
             try:
                 if isinstance(target_coordinates, dict):
                    cx, cy = target_coordinates["x"], target_coordinates["y"]
                 else:
                    cx, cy = target_coordinates[0], target_coordinates[1]
                 
                 sx, sy = self.coordinate_utils.map_normalized_to_pixels(cx, cy, page, self.config)
                 delay = random.randint(50, 150)
                 self.logger.info(f"🖱️ [TYPE] Clicking coordinates ({sx}, {sy}) to focus")
                 await page.mouse.click(sx, sy, delay=delay)
                 focused = True
             except Exception as e:
                 self.logger.warning(f"Failed to focus via coordinates: {e}")
        
        # 2. If not focused via coords, use LLM to choose field
        if not focused:
             elements = await self._extract_typeable_elements(page)
             chosen = await self._choose_field_with_llm(elements, "TYPE", value, tasks[-1] if tasks else "")
             if chosen:
                 center = chosen.get('center')
                 if center:
                     self.logger.info(f"🖱️ [TYPE] Clicking chosen element at ({center['x']}, {center['y']})")
                     await page.mouse.click(center['x'], center['y'])
        
        # 3. Type
        await self._clear_active_field(page)
        await page.keyboard.type(value)
        self.logger.info(f"Typed '{value}'")
        return f"Typed '{value}'"

    async def _perform_hover(self, target_coordinates, target_element, selector, element_repr, page):
        if target_coordinates:
            try:
                if isinstance(target_coordinates, dict):
                    nx, ny = target_coordinates["x"], target_coordinates["y"]
                else:
                    nx, ny = target_coordinates[0], target_coordinates[1]
                
                sx, sy = self.coordinate_utils.map_normalized_to_pixels(nx, ny, page, self.config)
                delay = random.randint(50, 150)
                await page.mouse.hover(sx, sy, delay=delay)
                self.logger.info(f"Hovered at coordinates ({sx}, {sy})")
                return f"SUCCESS: Hovered at coordinates ({sx}, {sy})"
            except Exception as e:
                self.logger.warning(f"Coordinate hover failed: {e}")
        
        if selector:
            try:
                await selector.hover(timeout=10000)
                self.logger.info(f"Hovered over element: {element_repr}")
                return f"SUCCESS: Hovered over element: {element_repr}"
            except Exception as e:
                self.logger.warning(f"Selector hover failed: {e}")
                
        return "FAILED: Hover failed"

    async def _perform_scroll(self, action_name, target_coordinates, page):
        try:
            vp = getattr(page, 'viewport_size', None) or self.config.get('browser', {}).get('viewport', {})
            vh = vp.get('height', 720) if isinstance(vp, dict) else (vp.height if hasattr(vp, 'height') else 720)
            delta = int(vh * 0.6)
            if action_name == "SCROLL UP":
                delta = -delta
                
            # Check for coordinates to scroll specific element
            if target_coordinates:
                if isinstance(target_coordinates, dict):
                    nx, ny = target_coordinates.get('x'), target_coordinates.get('y')
                else:
                    nx, ny = target_coordinates[0], target_coordinates[1]
                
                sx, sy = self.coordinate_utils.map_normalized_to_pixels(nx, ny, page, self.config)
                
                # Scroll element at point
                await page.evaluate(
                    "(params) => {\n"+
                    "  const { x, y, d } = params;\n"+
                    "  const el = document.elementFromPoint(x, y);\n"+
                    "  function scrollableAncestor(node){\n"+
                    "    let n = node;\n"+
                    "    while(n && n !== document.body && n !== document.documentElement){\n"+
                    "      const s = getComputedStyle(n);\n"+
                    "      const oy = s.overflowY;\n"+
                    "      if((oy === 'auto' || oy === 'scroll' || oy === 'overlay') && n.scrollHeight > n.clientHeight){\n"+
                    "        return n;\n"+
                    "      }\n"+
                    "      n = n.parentElement;\n"+
                    "    }\n"+
                    "    return window;\n"+
                    "  }\n"+
                    "  function scrollWin(d){\n"+
                    "    try{ window.scrollBy(0,d); }catch(e){}\n"+
                    "    try{ document.documentElement.scrollBy(0,d); }catch(e){}\n"+
                    "    try{ document.documentElement.scrollTop += d; }catch(e){}\n"+
                    "    try{ document.body.scrollTop += d; }catch(e){}\n"+
                    "  }\n"+
                    "  const sc = scrollableAncestor(el);\n"+
                    "  if(sc === window){ scrollWin(d); } else { try{ sc.scrollBy(0,d);}catch(e){} try{ sc.scrollTop += d;}catch(e){} }\n"+
                    "}",
                    {"x": sx, "y": sy, "d": delta}
                )
            else:
                # Window scroll
                await page.evaluate(f"window.scrollBy(0, {delta})")
                
            self.logger.info(f"{action_name} executed")
            return f"{action_name} executed"
        except Exception as e:
            self.logger.error(f"Failed to scroll: {e}")
            return f"FAILED: {action_name} - {e}"

    async def _perform_goto(self, value, page):
        try:
            if looks_like_api_endpoint(value):
                self.logger.error(f"Blocked GOTO to non-HTML endpoint: {value}")
                return f"FAILED: GOTO blocked for API/GraphQL endpoint"
            
            resp = await page.goto(value, wait_until="load")
            self.logger.info(f"Navigated to {value}")
            
            # Check content type
            ct = ""
            try:
                if resp:
                    h = resp.headers
                    ct = h.get("content-type", "").lower()
            except: pass
            
            if ct and "text/html" not in ct:
                self.logger.error(f"Non-HTML content-type '{ct}' after GOTO {value}; reverting")
                try: await page.go_back()
                except: pass
                return f"FAILED: Navigated to non-HTML endpoint ({ct})"
                
            if await self._is_page_blocked_or_blank(page):
                self.logger.error(f"⛔ Page is blocked or blank after navigating to {value}")
                return f"FAILED: Page is blocked or blank at {value}"
                
            return f"Navigated to {value}"
        except Exception as e:
            self.logger.error(f"GOTO action failed: {e}")
            return f"FAILED: Navigation to {value} failed - {e}"

    async def _perform_press_enter(self, selector, element_repr, page):
        try:
            # Try to find focused element
            focused_element = await page.evaluate("() => document.activeElement ? document.activeElement.tagName : null")
            if focused_element:
                self.logger.info(f"Pressed Enter on focused {focused_element}")
                await page.keyboard.press('Enter')
                return f"SUCCESS: Pressed Enter on focused {focused_element}"
            
            # Fallback global
            await page.keyboard.press('Enter')
            return "SUCCESS: Pressed Enter globally"
        except Exception as e:
            self.logger.error(f"PRESS ENTER failed: {e}")
            return f"FAILED: PRESS ENTER - {e}"

    async def _perform_select(self, target_coordinates, value, field_name, tasks, page):
        if target_coordinates:
            try:
                 if isinstance(target_coordinates, dict):
                    nx, ny = target_coordinates["x"], target_coordinates["y"]
                 else:
                    nx, ny = target_coordinates[0], target_coordinates[1]
                 
                 sx, sy = self.coordinate_utils.map_normalized_to_pixels(nx, ny, page, self.config)
                 success, message = await dom_select_option(page, (sx, sy), value)
                 if success:
                     return f"SUCCESS: {message} (method: pixel+DOM)"
            except: pass
            
        all_selects = await get_all_select_elements(page)
        if not all_selects:
            return "FAILED: No dropdown fields available"
            
        chosen_select = await decide_form_element(
            engine=self.engine,
            inputs=[],
            selects=all_selects,
            action_type="SELECT",
            value=value,
            task=f"Find '{field_name}' dropdown for: {tasks[-1] if tasks else 'Unknown task'}",
            logger=self.logger
        )
        
        if chosen_select:
             selector = chosen_select.get('selector')
             # Try direct select
             success, message = await dom_select_by_selector(page, selector, value)
             if success: return f"SUCCESS: {message}"
             
             # Try option match
             options = chosen_select.get('options') or []
             chosen_option = await decide_select_option(
                engine=self.engine,
                options=options,
                task=f"{tasks[-1] if tasks else 'Unknown task'}",
                field_name=field_name,
                desired_value=value,
                logger=self.logger
             )
             if chosen_option:
                 option_text = chosen_option.get('text') or chosen_option.get('value') or ""
                 success, message = await dom_select_by_selector(page, selector, option_text)
                 if success: return f"SUCCESS: {message}"
                 
        return "FAILED: SELECT methods failed"

    # ... Helpers ...
    async def _extract_typeable_elements(self, page):
        try:
            elements = await page.evaluate("""
                () => {
                    const elements = [];
                    const selectors = [
                        'input[type="text"]', 'input[type="search"]', 'input[type="email"]',
                        'input[type="tel"]', 'input[type="url"]', 'input[type="password"]',
                        'input[type="number"]', 'input:not([type])', 'textarea',
                        '[contenteditable="true"]', '[role="textbox"]'
                    ];
                    selectors.forEach(selector => {
                        document.querySelectorAll(selector).forEach((el, idx) => {
                            const rect = el.getBoundingClientRect();
                            if (rect.width > 0 && rect.height > 0) {
                                let label = '';
                                if (el.id) {
                                    const labelEl = document.querySelector(`label[for="${el.id}"]`);
                                    if (labelEl) label = labelEl.textContent.trim();
                                }
                                if (!label && el.labels && el.labels.length > 0) label = el.labels[0].textContent.trim();
                                if (!label) label = el.getAttribute('aria-label') || el.getAttribute('placeholder') || '';
                                
                                elements.push({
                                    index: elements.length,
                                    tagName: el.tagName.toLowerCase(),
                                    type: el.type || 'text',
                                    id: el.id || '',
                                    name: el.name || '',
                                    placeholder: el.placeholder || '',
                                    label: label,
                                    value: el.value || el.textContent || '',
                                    center: { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 },
                                    selector: el.id ? `${el.tagName.toLowerCase()}[id="${el.id}"]` : 
                                             el.name ? `${el.tagName.toLowerCase()}[name="${el.name}"]` :
                                             `${selector}:nth-of-type(${idx + 1})`
                                });
                            }
                        });
                    });
                    return elements;
                }
            """)
            return elements
        except Exception as e:
            self.logger.error(f"Failed to extract typeable elements: {e}")
            return []

    async def _clear_active_field(self, page):
        try:
            if sys.platform == "darwin":
                try: await page.keyboard.press("Meta+A")
                except: pass
            else:
                try: await page.keyboard.press("Control+A")
                except: pass
            try: await page.keyboard.press("Backspace")
            except: pass
        except: pass

    async def _choose_field_with_llm(self, elements, action_type, intended_value, task):
        if not elements: return None
        
        if action_type == "TYPE":
            elements_desc = "\n".join([
                f"{i}. {el.get('label') or el.get('placeholder') or el.get('name') or el.get('id') or 'Unnamed field'} "
                f"(type: {el.get('type')}, current: '{el.get('value', '')}')"
                for i, el in enumerate(elements)
            ])
            prompt = build_field_selection_prompt(intended_value, task, elements_desc, len(elements))
        else: 
            return elements[0]
            
        try:
            response = await self.engine.generate(
                prompt=[prompt, "", ""],
                image_path=None,
                temperature=0.0,
                model=self.config.get('agent', {}).get('model'), # Access model from config if possible or pass it
                turn_number=0
            )
            if isinstance(response, str):
                digits = []
                for ch in response:
                    if ch.isdigit():
                        digits.append(ch)
                    elif digits:
                        break
                if digits:
                    index = int("".join(digits))
                    if 0 <= index < len(elements):
                        return elements[index]
            return elements[0]
        except Exception as e:
            self.logger.error(f"Failed to choose field with LLM: {e}")
            return elements[0]

    async def _find_click_target_by_text(self, text, page):
        try:
            t = text if isinstance(text, str) else ""
            t = t.strip()
            if not t: return None, None, None

            loc = page.get_by_text(t, exact=True)
            if await loc.count() > 0 and await loc.first.is_visible():
                box = await loc.first.bounding_box()
                return {'box': box}, (box['x'], box['y']), 'pixel'
                
            loc = page.get_by_text(t, exact=False)
            if await loc.count() > 0 and await loc.first.is_visible():
                box = await loc.first.bounding_box()
                return {'box': box}, (box['x'], box['y']), 'pixel'
                
            return None, None, None
        except Exception:
            return None, None, None

    async def _is_page_blocked_or_blank(self, page):
        try:
            content = await page.content()
            if len(content) < 200: return True
            return False
        except: return True
