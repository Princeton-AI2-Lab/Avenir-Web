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

import time
from difflib import SequenceMatcher
from typing import Any, Dict, Optional, Sequence, Tuple


async def capture_page_state(page: Any, *, pending_hit_test_coords: Optional[Sequence[int]] = None, logger: Any = None) -> Dict:
    try:
        state = {
            "url": "",
            "title": "",
            "visible_text": "",
            "interactive_elements_count": 0,
            "form_fields_count": 0,
            "modal_present": False,
            "scroll_x": 0,
            "scroll_y": 0,
            "active_element": None,
        }

        if page and getattr(page, "is_closed", None) and page.is_closed():
            state["error"] = "page_closed"
            return state

        try:
            # Reduce timeout to 2s to prevent long hangs on unstable pages
            await page.wait_for_load_state("domcontentloaded", timeout=2000)
        except Exception:
            pass

        try:
            state["url"] = page.url
        except Exception:
            state["url"] = ""

        try:
            state["title"] = await page.title()
        except Exception:
            state["title"] = ""

        # Optimized: Combine all DOM-related captures into a single evaluate call to reduce IPC overhead
        hx, hy = (None, None)
        if isinstance(pending_hit_test_coords, (list, tuple)) and len(pending_hit_test_coords) >= 2:
            hx, hy = int(pending_hit_test_coords[0]), int(pending_hit_test_coords[1])

        dom_state = await page.evaluate(
            """
            ([hx, hy]) => {
                const results = {
                    visible_text: '',
                    interactive_elements_count: 0,
                    form_fields_count: 0,
                    modal_present: false,
                    scroll_x: window.scrollX || 0,
                    scroll_y: window.scrollY || 0,
                    active_element: null,
                    hit_target: null
                };

                const root = document.body || document.documentElement;
                if (!root) return results;

                // 1. Capture Visible Text (TreeWalker is efficient but can be slow on huge DOMs)
                try {
                    const walker = document.createTreeWalker(
                        root,
                        NodeFilter.SHOW_TEXT,
                        {
                            acceptNode: function(node) {
                                const parent = node.parentElement;
                                if (!parent) return NodeFilter.FILTER_REJECT;
                                const style = window.getComputedStyle(parent);
                                if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') {
                                    return NodeFilter.FILTER_REJECT;
                                }
                                return NodeFilter.FILTER_ACCEPT;
                            }
                        }
                    );
                    let text = '';
                    let node;
                    while ((node = walker.nextNode()) && text.length < 1000) {
                        text += node.textContent.trim() + ' ';
                    }
                    results.visible_text = text.trim().slice(0, 1000);
                } catch (e) {}

                // 2. Element Counts
                try {
                    results.interactive_elements_count = document.querySelectorAll(
                        'button, input, select, textarea, a[href], [onclick], [role="button"]'
                    ).length;
                    results.form_fields_count = document.querySelectorAll('input, select, textarea').length;
                    results.modal_present = document.querySelectorAll(
                        '[role="dialog"], .modal, .popup, .overlay, [aria-modal="true"]'
                    ).length > 0;
                } catch (e) {}

                // 3. Active Element State
                try {
                    const ae = document.activeElement;
                    if (ae && ae !== document.body) {
                        const tag = (ae.tagName || '').toUpperCase();
                        let selectedText = '';
                        let selectedIndex = null;
                        if (tag === 'SELECT') {
                            selectedIndex = ae.selectedIndex;
                            const opt = ae.selectedOptions && ae.selectedOptions[0];
                            selectedText = opt ? (opt.textContent || '') : '';
                        }
                        results.active_element = {
                            tag,
                            id: ae.id || '',
                            name: ae.getAttribute('name') || '',
                            type: ae.getAttribute('type') || '',
                            role: ae.getAttribute('role') || '',
                            placeholder: ae.getAttribute('placeholder') || '',
                            ariaLabel: ae.getAttribute('aria-label') || '',
                            value: (ae.value || '').slice(0, 200),
                            selectedText: selectedText.slice(0, 200),
                            selectedIndex
                        };
                    }
                } catch (e) {}

                // 4. Hit Target Information
                if (hx !== null && hy !== null) {
                    try {
                        const el = document.elementFromPoint(hx, hy);
                        if (el) {
                            const r = el.getBoundingClientRect();
                            results.hit_target = {
                                x: hx, y: hy,
                                tag: (el.tagName || '').toUpperCase(),
                                id: el.id || '',
                                name: el.getAttribute('name') || '',
                                className: (typeof el.className === 'string' ? el.className : '').slice(0, 200),
                                role: el.getAttribute('role') || '',
                                ariaLabel: el.getAttribute('aria-label') || '',
                                href: el.getAttribute('href') || '',
                                disabled: !!el.disabled || el.getAttribute('aria-disabled') === 'true',
                                ariaPressed: el.getAttribute('aria-pressed') || '',
                                ariaExpanded: el.getAttribute('aria-expanded') || '',
                                type: el.getAttribute('type') || '',
                                value: (typeof el.value === 'string' ? el.value : '').slice(0, 200),
                                text: (el.innerText || el.textContent || '').trim().replace(/\\s+/g, ' ').slice(0, 200),
                                rect: { left: r.left, top: r.top, width: r.width, height: r.height }
                            };
                        }
                    } catch (e) {}
                }

                return results;
            }
            """,
            [hx, hy]
        )

        state.update(dom_state)
        return state
    except Exception as e:
        if logger is not None:
            try:
                logger.warning(f"Failed to capture page state: {e}")
            except Exception:
                pass
        return {
            "error": str(e),
            "url": "",
            "title": "",
            "visible_text": "",
            "interactive_elements_count": 0,
            "form_fields_count": 0,
            "modal_present": False,
        }


def detect_page_state_change(state_before: Any, state_after: Any, action_type: Any, *, logger: Any = None) -> Tuple[bool, list]:
    try:
        if not isinstance(state_before, dict):
            state_before = {}
        if not isinstance(state_after, dict):
            state_after = {}
        if state_before.get("error") or state_after.get("error"):
            return True, []

        changes_detected = []

        vb = state_before.get("visible_text")
        va = state_after.get("visible_text")
        text_before = vb if isinstance(vb, str) else ""
        text_after = va if isinstance(va, str) else ""
        if len(text_before) > 0 and len(text_after) > 0:
            similarity = SequenceMatcher(None, text_before, text_after).ratio()
            if similarity < 0.95:
                changes_detected.append("content_changed")
                if logger is not None:
                    try:
                        logger.info(f"Content changed significantly (similarity: {similarity:.2f})")
                    except Exception:
                        pass

        ib = state_before.get("interactive_elements_count")
        ia = state_after.get("interactive_elements_count")
        try:
            ib = int(ib) if ib is not None else 0
        except Exception:
            ib = 0
        try:
            ia = int(ia) if ia is not None else 0
        except Exception:
            ia = 0
        if ib != ia:
            changes_detected.append("interactive_elements_changed")
            if logger is not None:
                try:
                    logger.info(f"Interactive elements count changed: {ib} -> {ia}")
                except Exception:
                    pass

        fb = state_before.get("form_fields_count")
        fa = state_after.get("form_fields_count")
        try:
            fb = int(fb) if fb is not None else 0
        except Exception:
            fb = 0
        try:
            fa = int(fa) if fa is not None else 0
        except Exception:
            fa = 0
        if fb != fa:
            changes_detected.append("form_fields_changed")
            if logger is not None:
                try:
                    logger.info(f"Form fields count changed: {fb} -> {fa}")
                except Exception:
                    pass

        tb = state_before.get("title") if isinstance(state_before.get("title"), str) else ""
        ta = state_after.get("title") if isinstance(state_after.get("title"), str) else ""
        if tb != ta:
            changes_detected.append("title_changed")
            if logger is not None:
                try:
                    logger.info(f"Title changed: {tb} -> {ta}")
                except Exception:
                    pass

        ub = state_before.get("url") if isinstance(state_before.get("url"), str) else ""
        ua = state_after.get("url") if isinstance(state_after.get("url"), str) else ""
        if ub != ua:
            changes_detected.append("url_changed")
            if logger is not None:
                try:
                    logger.info(f"URL changed: {ub} -> {ua} (Note: URL change is secondary to content changes)")
                except Exception:
                    pass

        mpb = bool(state_before.get("modal_present", False))
        mpa = bool(state_after.get("modal_present", False))
        if mpb != mpa:
            changes_detected.append("modal_state_changed")
            if logger is not None:
                try:
                    logger.info(f"Modal/popup state changed: {mpb} -> {mpa}")
                except Exception:
                    pass

        try:
            sbx = int(state_before.get("scroll_x", 0) or 0)
            sby = int(state_before.get("scroll_y", 0) or 0)
            sax = int(state_after.get("scroll_x", 0) or 0)
            say = int(state_after.get("scroll_y", 0) or 0)
            if abs(sax - sbx) >= 5 or abs(say - sby) >= 5:
                changes_detected.append("scroll_changed")
                if logger is not None:
                    try:
                        logger.info(f"Scroll changed: ({sbx},{sby}) -> ({sax},{say})")
                    except Exception:
                        pass
        except Exception:
            pass

        ab = state_before.get("active_element")
        aa = state_after.get("active_element")
        try:
            if isinstance(ab, dict) and isinstance(aa, dict):
                focus_keys = ("tag", "id", "name", "type", "role", "placeholder", "ariaLabel")
                if any((ab.get(k) or "") != (aa.get(k) or "") for k in focus_keys):
                    tag_after = (aa.get("tag") or "").upper()
                    role_after = (aa.get("role") or "").lower()
                    if tag_after in ("INPUT", "TEXTAREA", "SELECT") or role_after in ("textbox", "combobox"):
                        changes_detected.append("focus_to_form")
                        if logger is not None:
                            try:
                                logger.info("Active element changed (to form field)")
                            except Exception:
                                pass
                    else:
                        changes_detected.append("focus_changed")
                        if logger is not None:
                            try:
                                logger.info("Active element changed")
                            except Exception:
                                pass
            elif (ab is None) != (aa is None):
                if isinstance(ab, dict) and aa is None:
                    changes_detected.append("focus_blur")
                    if logger is not None:
                        try:
                            logger.info("Active element changed (blur)")
                        except Exception:
                            pass
                elif isinstance(aa, dict) and ab is None:
                    tag_after = (aa.get("tag") or "").upper()
                    role_after = (aa.get("role") or "").lower()
                    if tag_after in ("INPUT", "TEXTAREA", "SELECT") or role_after in ("textbox", "combobox"):
                        changes_detected.append("focus_to_form")
                        if logger is not None:
                            try:
                                logger.info("Active element changed (to form field)")
                            except Exception:
                                pass
                    else:
                        changes_detected.append("focus_changed")
                        if logger is not None:
                            try:
                                logger.info("Active element changed (None/non-None)")
                            except Exception:
                                pass
        except Exception:
            pass

        try:
            if isinstance(ab, dict) and isinstance(aa, dict):
                vb2 = ab.get("value") if isinstance(ab.get("value"), str) else ""
                va2 = aa.get("value") if isinstance(aa.get("value"), str) else ""
                if vb2 != va2:
                    changes_detected.append("active_value_changed")
                    if logger is not None:
                        try:
                            logger.info("Active element value changed")
                        except Exception:
                            pass

                sb2 = ab.get("selectedText") if isinstance(ab.get("selectedText"), str) else ""
                sa2 = aa.get("selectedText") if isinstance(aa.get("selectedText"), str) else ""
                if sb2 != sa2:
                    changes_detected.append("active_selection_changed")
                    if logger is not None:
                        try:
                            logger.info("Active element selection changed")
                        except Exception:
                            pass
        except Exception:
            pass

        hb = state_before.get("hit_target")
        ha = state_after.get("hit_target")
        try:
            if isinstance(hb, dict) and isinstance(ha, dict):
                keys = (
                    "tag",
                    "id",
                    "name",
                    "role",
                    "ariaLabel",
                    "href",
                    "disabled",
                    "ariaPressed",
                    "ariaExpanded",
                    "type",
                    "value",
                    "text",
                )
                if any((hb.get(k) or "") != (ha.get(k) or "") for k in keys):
                    changes_detected.append("hit_target_changed")
                    if logger is not None:
                        try:
                            logger.info("Hit target element changed")
                        except Exception:
                            pass
            elif (hb is None) != (ha is None):
                changes_detected.append("hit_target_changed")
                if logger is not None:
                    try:
                        logger.info("Hit target element changed (None/non-None)")
                    except Exception:
                        pass
        except Exception:
            pass

        if isinstance(action_type, str) and action_type == "CLICK":
            return True, changes_detected
        
        evidence = changes_detected
        action_successful = len(evidence) > 0

        if isinstance(action_type, str) and action_type in ["SCROLL UP", "SCROLL DOWN", "SCROLL TOP", "SCROLL BOTTOM"]:
            return True, changes_detected

        if logger is not None:
            try:
                if action_successful:
                    logger.info(f"Action '{action_type}' appears SUCCESSFUL - Changes detected: {changes_detected}")
                else:
                    logger.warning(f"Action '{action_type}' appears FAILED - No significant page changes detected")
            except Exception:
                pass

        return action_successful, changes_detected
    except Exception as e:
        if logger is not None:
            try:
                logger.warning(f"Failed to detect page state change: {e}")
            except Exception:
                pass
        return True, []


def record_action_result(container: Any, *, step: int, action_type: Any, successful: bool, changes: list) -> None:
    if not hasattr(container, "action_results"):
        container.action_results = []
    container.action_results.append(
        {
            "step": step,
            "action": action_type,
            "successful": successful,
            "changes": changes,
            "timestamp": time.time(),
        }
    )
