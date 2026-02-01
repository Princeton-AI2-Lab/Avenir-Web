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

##### Specialized Prompt Builders for Agent Operations

def build_tools_definition() -> str:
    return """
<tools>
{
  "type": "function",
  "function": {
    "name": "browser_use",
    "description": "Single-step browser interaction using pixel coordinates or visible text.",
    "parameters": {
      "type": "object",
      "required": ["action"],
      "properties": {
        "action": {
          "type": "string",
          "enum": ["left_click", "hover", "keyboard", "type", "select", "press_enter", "scroll_up", "scroll_down", "scroll_top", "scroll_bottom", "new_tab", "close_tab", "go_back", "go_forward", "wait", "terminate"]
        },
        "coordinate": {"type": "array", "description": "Normalized [x,y] in 0–1000. REQUIRED for all actions except scroll; omit for scroll only. Include to target a container."},
        "text": {"type": "string", "description": "Visible label or input text. Use 'CLEAR' for keyboard.", "maxLength": 200},
        "code": {"type": "string", "description": "KeyboardEvent.code (e.g., 'PageDown', 'ArrowDown', 'Enter')", "maxLength": 50},
        "clear_first": {"type": "boolean", "description": "Clear active field before typing (type/keyboard)"},
        "press_enter_after": {"type": "boolean", "description": "Press Enter after typing (action=type)"},
        "field": {"type": "string", "description": "Semantic field name (email/search/password/country)", "maxLength": 100},
        "time": {"type": "number", "description": "Seconds to wait"},
        "status": {"type": "string", "enum": ["success", "failure"], "description": "Task status for terminate"},
        "description": {"type": "string", "description": "Short action description (<=200 chars). REQUIRED.", "maxLength": 200}
      }
    }
  }
}
</tools>

Screen: 1000×1000, origin (0,0) top-left.

Rules:
- Do not use GOTO for URL navigation.
- For <select> elements, YOU MUST use 'select' action directly. DO NOT use 'click' to open dropdowns.
- For all actions except scroll actions (scroll_up, scroll_down, scroll_top, scroll_bottom), YOU MUST provide the 'coordinate' parameter with normalized [x,y] values in 0–1000.
- keyboard: use 'code' for keys; 'text' for typing; 'CLEAR' clears the active field.
- **IMPORTANT**: You MUST provide 'coordinate' [x,y] for every CLICK, HOVER, or TYPE action. Do NOT rely on 'text' alone.

Return strictly in <tool_call> tags:
<tool_call>
{"name": "browser_use", "arguments": {"action": "...", ...}}
</tool_call>"""

def build_checklist_prompt(task_description: str) -> str:
    """Concise checklist generation prompt with strict rules against hallucination."""
    return f"""Create 2–6 atomic outcome states based STRICTLY on the task description.

Task: {task_description}

Rules:
1) Each item is an observable goal state (not an action)
2) Max 10 words; short and specific
3) IDs: "requirement_1", "requirement_2", ...
4) Examples: "Size 'blue'", "T-shirt page", "Year: 2022-2023"
5) Status must be lowercase: pending, in_progress, completed, failed
6) DO NOT invent requirements not explicitly mentioned in the task.

Output JSON:
{{
    "checklist": [
        {{"id": "requirement_1", "description": "First outcome state", "status": "pending"}},
        {{"id": "requirement_2", "description": "Second outcome state", "status": "pending"}}
    ]
}}
"""


def build_checklist_update_prompt(action_type: str, success: bool, error: str, history_text: str,
                                  page_state_text: str, checklist_text: str) -> str:
    """Concise checklist update prompt with unchanged rules."""
    return f"""Update the checklist based on this action:

Action: {action_type} | Success: {success} | Error: {error if error else 'None'}

Recent actions:\n{history_text}
Page:\n{page_state_text[:300]}...
Checklist:\n{checklist_text}

Update rules:
• completed = fully satisfied
• in_progress = partially done
• pending = not started/reset
• failed = action failed
• Update exactly ONE item per action (most directly affected)
• new_status must be one of: pending, in_progress, completed, failed (lowercase)

Output JSON:
{{
    "updates": [
        {{"item_id": "requirement_X", "new_status": "pending", "reason": "Brief reason"}}
    ]
}}"""


def parse_tool_call(response_text: str) -> dict:
    """
    Parse Qwen3 tool-call response format (official browser_use format).
    
    Args:
        response_text: Raw response from LLM containing <tool_call> tags
    
    Returns:
        Dict with parsed action information or None if parsing fails
        
    Example input:
        <tool_call>
        {"name": "browser_use", "arguments": {"action": "left_click", "coordinate": [500, 300]}}
        </tool_call>
        
    Example output:
        {
            "action": "CLICK",
            "element": None,
            "value": None,
            "coordinates": [500, 300]
        }
    """
    import json
    
    try:
        if not isinstance(response_text, str):
            return None

        def extract_tag_blocks(text: str, tag: str):
            blocks = []
            if not text:
                return blocks
            lower = text.lower()
            open_tag = f"<{tag}>"
            close_tag = f"</{tag}>"
            idx = 0
            while True:
                start = lower.find(open_tag, idx)
                if start < 0:
                    break
                end = lower.find(close_tag, start + len(open_tag))
                if end < 0:
                    break
                content_start = start + len(open_tag)
                blocks.append(text[content_start:end])
                idx = end + len(close_tag)
            return blocks

        def extract_first_json_object(text: str):
            if not text:
                return None
            start = text.find("{")
            if start < 0:
                return None
            depth = 0
            in_str = False
            esc = False
            for i in range(start, len(text)):
                ch = text[i]
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
                        return text[start : i + 1]
            return None

        matches = extract_tag_blocks(response_text, "tool_call")
        if len(matches) != 1:
            return None
        tool_call_json = (matches[0] or "").strip()
        if tool_call_json.startswith('```'):
            tool_call_json = tool_call_json.split('\n', 1)[1] if '\n' in tool_call_json else tool_call_json
            if tool_call_json.endswith('```'):
                tool_call_json = tool_call_json[:-3]
        tool_call_json = tool_call_json.strip()
        try:
            tool_call_data = json.loads(tool_call_json)
        except Exception:
            obj = extract_first_json_object(tool_call_json)
            if not obj:
                return None
            tool_call_data = json.loads(obj)
        
        if not isinstance(tool_call_data, dict) or 'name' not in tool_call_data or 'arguments' not in tool_call_data:
            return None
        
        name = tool_call_data['name']
        if name != 'browser_use':
            return None
        
        args = tool_call_data['arguments']
        action = args.get('action', '').lower()
        if action == 'ask_strategy':
            return None
        if action == 'drag':
            return None
        
        action_mapping = {
            'left_click': 'CLICK',
            'hover': 'HOVER',
            'keyboard': 'KEYBOARD',
            'type': 'TYPE',
            'select': 'SELECT',
            'press_enter': 'PRESS ENTER',
            'scroll_up': 'SCROLL UP',
            'scroll_down': 'SCROLL DOWN',
            'scroll_top': 'SCROLL TOP',
            'scroll_bottom': 'SCROLL BOTTOM',
            'new_tab': 'NEW TAB',
            'close_tab': 'CLOSE TAB',
            'go_back': 'GO BACK',
            'go_forward': 'GO FORWARD',
            'goto': 'GOTO',
            'wait': 'WAIT',
            'terminate': 'TERMINATE'
        }
        
        mapped_action = action_mapping.get(action, action.upper())

        coord = args.get('coordinate')
        if isinstance(coord, (list, tuple)) and len(coord) >= 2:
            try:
                coord = [int(coord[0]), int(coord[1])]
            except Exception:
                coord = None
        else:
            coord = None
        if action == 'keyboard':
            if isinstance(args.get('code', ''), str) and args.get('code', '').strip():
                val_text = args.get('code', '').strip()
            else:
                val_text = args.get('text', '') if isinstance(args.get('text', ''), str) else ''
        else:
            val_text = args.get('text', '') if isinstance(args.get('text', ''), str) else ''
        fld_text = args.get('field', '') if isinstance(args.get('field', ''), str) else ''
        desc_text = args.get('description', '') if isinstance(args.get('description', ''), str) else ''
        if len(val_text) > 200:
            val_text = val_text[:200]
        if len(fld_text) > 100:
            fld_text = fld_text[:100]
        if len(desc_text) > 200:
            desc_text = desc_text[:200]
        result = {
            'action': mapped_action,
            'element': None,
            'value': val_text,
            'coordinates': coord,
            'field': fld_text,
            'action_description': desc_text
        }
        if coord is not None:
            result['coordinates_type'] = 'normalized'
        if action == 'type':
            pe_after = args.get('press_enter_after', False)
            result['press_enter_after'] = bool(pe_after)
        if not result['action_description'] or not result['action_description'].strip():
            if mapped_action == 'CLICK':
                if coord:
                    result['action_description'] = f"Click at coordinates ({coord[0]}, {coord[1]})"
                else:
                    result['action_description'] = "Click element"
            elif mapped_action == 'HOVER':
                if coord:
                    result['action_description'] = f"Hover at coordinates ({coord[0]}, {coord[1]})"
                else:
                    result['action_description'] = "Hover element"
            elif mapped_action == 'TYPE':
                base = f"Type '{val_text}'" if val_text else "Type"
                result['action_description'] = base + (f" in {fld_text}" if fld_text else "")
            elif mapped_action == 'SELECT':
                base = f"Select '{val_text}'" if val_text else "Select"
                result['action_description'] = base + (f" from {fld_text}" if fld_text else "")
            elif mapped_action == 'PRESS ENTER':
                result['action_description'] = "Press Enter"
            elif mapped_action == 'SCROLL UP':
                result['action_description'] = "Scroll up"
            elif mapped_action == 'SCROLL DOWN':
                result['action_description'] = "Scroll down"
            elif mapped_action == 'SCROLL TOP':
                result['action_description'] = "Scroll to top"
            elif mapped_action == 'SCROLL BOTTOM':
                result['action_description'] = "Scroll to bottom"
            elif mapped_action == 'GOTO':
                result['action_description'] = f"Navigate to {val_text}" if val_text else "Navigate"
            elif mapped_action == 'KEYBOARD':
                if val_text and val_text.upper() == 'CLEAR':
                    result['action_description'] = "Clear field via keyboard"
                else:
                    base = f"Type '{val_text}' via keyboard" if val_text else "Type via keyboard"
                    result['action_description'] = base
            elif mapped_action == 'NEW TAB':
                result['action_description'] = "Open new tab"
            elif mapped_action == 'CLOSE TAB':
                result['action_description'] = "Close tab"
            elif mapped_action == 'GO BACK':
                result['action_description'] = "Go back"
            elif mapped_action == 'GO FORWARD':
                result['action_description'] = "Go forward"
        # Handle clear_first for TYPE/KEYBOARD with default true
        if mapped_action in ('TYPE', 'KEYBOARD'):
            cf = args.get('clear_first', True)
            if not isinstance(cf, bool):
                cf = True
            result['clear_first'] = cf
        
        # Handle special cases
        if action == 'wait':
            result['value'] = str(args.get('time', 1))
        elif action == 'terminate':
            result['value'] = ''
            result['status'] = args.get('status', 'success')
        elif action == 'goto':
            result['value'] = args.get('text', '')  # URL
        elif action == 'select':
            result['value'] = args.get('text', '')  # Option to select
        elif action == 'type':
            result['value'] = args.get('text', '')  # Text to type
        
        return result
        
    except Exception as e:
        return None


def build_action_response_format() -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "web_action",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "CLICK",
                            "HOVER",
                            "KEYBOARD",
                            "TYPE",
                            "SELECT",
                            "PRESS ENTER",
                            "SCROLL UP",
                            "SCROLL DOWN",
                            "SCROLL TOP",
                            "SCROLL BOTTOM",
                            "NEW TAB",
                            "CLOSE TAB",
                            "GO BACK",
                            "GO FORWARD",
                            "GOTO",
                            "WAIT",
                            "TERMINATE",
                        ],
                    },
                    "value": {"type": "string", "maxLength": 200},
                    "coordinates": {
                        "type": ["array", "null"],
                        "items": {"type": "integer"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "field": {"type": "string", "maxLength": 100},
                    "action_description": {"type": "string", "maxLength": 200},
                    "coordinates_type": {"type": "string", "enum": ["normalized"]},
                    "press_enter_after": {"type": "boolean"},
                    "clear_first": {"type": "boolean"},
                    "status": {"type": "string", "enum": ["success", "failure"]},
                },
                "required": ["action", "value", "field", "action_description"],
                "additionalProperties": False,
            },
        },
    }


def parse_structured_action(response_text: str) -> dict:
    import json

    if not isinstance(response_text, str):
        return None
    text = response_text.strip()
    if not text:
        return None

    def extract_first_json_object(s: str):
        start = s.find("{")
        if start < 0:
            return None
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
        return None

    try:
        return json.loads(text)
    except Exception:
        obj = extract_first_json_object(text)
        if not obj:
            return None
        try:
            return json.loads(obj)
        except Exception:
            return None


def build_task_constraints_prompt(allowed_domain: str = None,
                                  disallow_login: bool = True,
                                  disallow_offsite: bool = True,
                                  extra_rules: str = "") -> str:
    lines = ["Task-specific soft constraints:"]
    if disallow_login:
        lines += [
            "- Do NOT attempt to log in, sign in, sign up, or provide credentials.",
            "- If a login/sign-in UI is detected (password fields, 'Sign in', 'Log in', 'Create account'), TERMINATE immediately with status 'failure' and reason 'login prohibited'."
        ]
    return "\n".join(lines)


def validate_parsed_action(parsed_action: dict) -> (bool, str):
    if not isinstance(parsed_action, dict):
        return False, "not a dict"
    a = parsed_action.get('action')
    # next field removed from schema
    desc = parsed_action.get('action_description')
    if not a:
        return False, "missing action"
    allowed = {"CLICK", "HOVER", "KEYBOARD", "TYPE", "SELECT", "PRESS ENTER", "SCROLL UP", "SCROLL DOWN", "SCROLL TOP", "SCROLL BOTTOM", "NEW TAB", "CLOSE TAB", "GO BACK", "GO FORWARD", "GOTO", "WAIT", "TERMINATE"}
    if a not in allowed:
        return False, "invalid action"
    val = parsed_action.get('value', '')
    fld = parsed_action.get('field', '')
    coords = parsed_action.get('coordinates', None)
    if isinstance(val, str) and len(val) > 200:
        return False, "value too long"
    if isinstance(fld, str) and len(fld) > 100:
        return False, "field too long"
    if isinstance(desc, str) and len(desc) > 200:
        return False, "description too long"
    if not isinstance(desc, str) or not desc.strip():
        return False, "missing description"
    if coords is not None:
        if not (isinstance(coords, (list, tuple)) and len(coords) >= 2 and all(isinstance(c, int) for c in coords[:2])):
            return False, "invalid coordinates"
    return True, ""

def build_system_prompt(task: str, previous_actions: str, checklist_context: str = "", suggested_next_step: str = "", strategic_reasoning: str = "", policy_constraints: str = "", use_structured_output: bool = False) -> tuple:
    strategic_block = "\nStrategic guidance:\n" + strategic_reasoning if strategic_reasoning else ""
    if use_structured_output:
        system_lines = [
            "Return exactly one JSON object for the next action.",
            "Use keys: action, value, coordinates, field, action_description.",
            "coordinates must be [x,y] integers in 0–1000 or null.",
            "Close/accept blocking modals, overlays, cookie banners first.",
            "Do not repeat actions unless page state visibly changed.",
            "When objectives are achieved, use action=TERMINATE and status=success.",
        ]
    else:
        system_lines = [
            "One action per turn with pixel coordinates.",
            "CLICK: provide 'coordinate' or visible 'text'.",
            "Close/accept blocking modals, overlays, cookie banners first.",
            "Do not repeat actions unless page state visibly changed.",
            "TYPE/SELECT only when target field/dropdown is visible.",
            "KEYBOARD: use 'code' for keys; 'text' for typing; 'CLEAR' to clear active field.",
            "SCROLL: omit coordinates to scroll page; include [x,y] to scroll a container.",
            "If you see potential <select> elements, MUST use 'select' action directly. DO NOT use 'click' to open dropdowns.",
            "When objectives are achieved, TERMINATE with status 'success'.",
        ]
    if use_structured_output:
        system_text = "\n".join(system_lines) + ("\n" + strategic_block if strategic_block else "") + "\nReturn ONLY a single JSON object."
    else:
        tools_definition = build_tools_definition()
        system_text = "\n".join(system_lines) + ("\n" + strategic_block if strategic_block else "") + "\n" + tools_definition

    user_lines = [
        "Task:",
        task,
        "Pre-step:",
        "Close or accept any cookie/consent banner before other actions.",
        strategic_block,
        "Constraints:",
        policy_constraints if policy_constraints else "No task-specific constraints.",
        "Previous actions:",
        previous_actions if previous_actions else "No previous actions yet.",
    ]
    if checklist_context:
        user_lines += ["", "Task progress:", checklist_context]
    if suggested_next_step:
        user_lines += ["", "Suggested next move:", f"{suggested_next_step} (suggested)"]
    user_text = "\n".join(user_lines)
    return system_text, user_text


def build_task_reasoning_system_prompt() -> str:
    return """You are a site‑specific web automation strategist for autonomous web agents.

Provide actionable guidance the agent can convert into tool calls.
Requirements:
- 2–4 short imperative sentences (each is a concrete action)
- Use the website’s visible labels in quotes where relevant (links, buttons, tabs, fields)
- Do not include generic predictions or redundant phrasing (e.g., "scroll down to see ..."); only state the action
- Ground suggestions in verifiable online information about the target website; if none found, provide at most 0–2 minimal actions or leave the plan empty
- Do not use coordinates, CSS selectors, element IDs, or numbered step sequences
- No rationale; only the action sentences

Return only:
<plan>...site‑specific action plan...</plan>"""


def build_task_reasoning_user_prompt(website: str, task_description: str, policy_constraints: str = "") -> str:
    if not website or not str(website).strip():
         raise ValueError("Website is required for action‑focused reasoning")
    
    user_prompt = f"""Website: {website}
Task: {task_description}

Provide 2–4 site‑specific action sentences the agent should attempt next.
- Prefer visible labels in quotes (e.g., "Search", "Publications", "Downloads")
- Use concrete actions (scroll/top/bottom, click, open, search, filter) without commentary about expected results
- Ground actions in online sources about this site (docs/help/blog/community); do not invent labels
- If no valid online‑sourced guidance is found, return 0–2 minimal actions or leave <plan> empty
Return ONLY <plan>...</plan>; any other content will be ignored."""

    if policy_constraints and policy_constraints.strip():
        user_prompt = user_prompt + "\n\nConstraints:\n" + policy_constraints.strip()
    
    return user_prompt


def build_confusion_replan_system_prompt() -> str:
    return """You are a site‑specific web automation strategist.

The previous approach stalled. Provide actionable guidance strictly inside <plan>...</plan>.
Requirements:
- 2–4 short imperative sentences (each is a concrete action)
- Use visible labels in quotes; include navigation actions as needed (scroll/top/bottom, click links/tabs/buttons, open search, filter) without redundant commentary
- Do not include generic predictions (e.g., "scroll down to see ..."); only specify actions
- Ground suggestions in verifiable online sources about the target website; if none found, provide at most 0–2 minimal actions or leave the plan empty
- No coordinates/selectors/IDs; avoid repeating previously failed labels
- Prefer alternative entry points when a label repeatedly fails

Return only:
<plan>...site‑specific action plan...</plan>"""


def build_confusion_replan_user_prompt(website: str, page_title: str, task_description: str, confusion: str, previous_plan: str, previous_actions_text: str, checklist_text: str, policy_constraints: str) -> str:
    if not website or not str(website).strip():
        raise ValueError("Website is required for replan")

    user_sections = [
        f"Website: {website}",
        f"Current page: {page_title} ({website})",
        f"Task: {task_description}",
        f"Confusion: {confusion}",
        f"Previous plan: {previous_plan}",
        "Executed history:",
        previous_actions_text or "None",
    ]
    if checklist_text and checklist_text.strip():
        user_sections += ["Checklist:", checklist_text]
    if policy_constraints and policy_constraints.strip():
        user_sections += ["Constraints:", policy_constraints.strip()]
    return "\n".join(user_sections)


def format_reasoning_for_prompt(reasoning: str) -> str:
    """
    Format reasoning for inclusion in agent prompts.
    
    Args:
        reasoning: The reasoning text to format
        
    Returns:
        Formatted string for prompt inclusion
    """
    if not reasoning or not reasoning.strip():
        return ""
    
    return f"""
**STRATEGIC GUIDANCE FROM REASONING MODEL**:
{reasoning}

Use this guidance to inform your action selection, but adapt based on what you actually observe on the page.
"""

def build_repetitive_action_warning(action_type, element, value, count):
    return f"CRITICAL: You've attempted the identical action '{action_type} {element} {value}' {count}+ times and it keeps failing. STOP and try a completely different approach immediately!"

def build_select_failure_warning(element, value, count):
    return f"SELECT action failing repeatedly on '{element}' with value '{value}' ({count} times). The dropdown may not contain this option or the element may not be a proper select. Try CLICKING to open dropdown first, or look for alternative elements."

def build_consecutive_failure_warning(count):
    return f"You have {count} consecutive failed actions. CHANGE STRATEGY NOW - try scrolling, different elements, or a completely different approach."

def build_high_failure_rate_warning(failed_count, total_count):
    return f"High failure rate: {failed_count}/{total_count} recent actions failed. Your current approach is not working - RECONSIDER YOUR STRATEGY completely."

def build_repeating_action_type_warning(action_type, failed_count):
    return f"You've been repeating the same failing action type '{action_type}' for 4 consecutive steps with {failed_count} failures. TRY A COMPLETELY DIFFERENT ACTION TYPE (scroll, wait, or terminate)."

def build_termination_recommendation_warning():
    return "TERMINATION RECOMMENDED: You've failed 5+ times in the last 6 actions. This suggests the task may be impossible with current approach or the target element doesn't exist. Consider terminating or trying a fundamentally different strategy."

def build_select_click_loop_warning():
    return "Detected SELECT->CLICK fallback pattern repeating. The element may not be a proper dropdown. Try a different element, use TYPE instead, or choose a different action type. Avoid repeating the same failing interaction."

def build_element_failure_warning(element, count):
    return f"Element '{element}' has failed {count} times in recent actions. Choose a different target or action type; avoid repeating the same failing interaction."

def build_general_failure_warning():
    return "Multiple recent failures detected. Choose a fundamentally different approach: different action type, different element, or terminate if no viable path. Do not repeat the same action on the same element or coordinates."

def build_repetition_detection_block(forbidden_patterns, suggestions):
    text = "\n**CRITICAL: REPETITIVE ACTION DETECTED**\n"
    text += "You have been repeating the same actions without success.\n"
    if forbidden_patterns:
        text += f"**FORBIDDEN ACTIONS**: You are PROHIBITED from using these patterns: {', '.join(forbidden_patterns)}\n"
    if suggestions:
        text += "**MANDATORY REQUIREMENTS**:\n"
        for suggestion in suggestions:
            text += f"- {suggestion}\n"
    text += "\n**YOU MUST CHOOSE A COMPLETELY DIFFERENT ACTION TYPE OR APPROACH**\n"
    text += "If you cannot find a different approach, consider using TERMINATE.\n\n"
    return text

def build_none_option_description():
    return "none. **LAST RESORT ONLY** - Select this ONLY if: (1) Target element is absolutely not in the list AND (2) You've exhausted all alternatives (scrolling, similar elements, GUI_GROUNDING). **BEFORE selecting 'none'**: Try scrolling to find the element, consider similar elements that might work, or use GUI_GROUNDING if the element is visible but not numbered."

def build_element_selection_priority_prompt():
    return "**ELEMENT SELECTION PRIORITY**: First try to find a suitable element from the numbered options above. If your target element is not listed, consider: (1) Scrolling to find it, (2) Selecting a similar element that might achieve the same goal, (3) Using GUI_GROUNDING if the element is visible, (4) Only select 'none' as an absolute last resort.\n"


def build_grounding_system_prompt() -> str:
    return (
        "You are a GUI grounding model. Given a screenshot and an instruction, "
        "output coordinates in the format: <|box_start|>(x,y)<|box_end|> where x and y are integers in [0,1000]. "
        "Do NOT output any explanation or extra text."
    )


def get_default_prompts() -> dict:
    return {
        "system_prompt": "You are a web navigation assistant.",
        "action_space": "",
        "question_description": "What action should be performed next?",
        "referring_description": "Select the best element: {multichoice_question}",
        "element_format": "Element {id}: {description}",
        "action_format": "Action: {action}",
        "value_format": "Value: {value}"
    }


def get_default_prompts_pure_vision() -> dict:
    return {
        "system_prompt": "You are a web navigation assistant.",
        "question_description": "What action should be performed next?",
        "referring_description": ""
    }


def build_action_format_prompt(actions: list) -> str:
    return f"ACTION: Choose an action from {{{', '.join(actions)}}}."


LLM_RESPONSE_PATTERNS_TO_REMOVE = [
    # Choice selection patterns
    "The uppercase letter of your choice. Choose one of the following elements if it matches the target element based on your analysis:\n\n",
    "The uppercase letter of your choice. Choose one of the following elements if it matches the target element based on your analysis:\n",
    "The uppercase letter of your choice. Choose one of the following elements if it matches the target element based on your analysis:",
    
    # Analysis-based choice patterns
    "The uppercase letter of your choice based on your analysis is:\n\n",
    "The uppercase letter of your choice based on your analysis is:\n", 
    "The uppercase letter of your choice based on your analysis is:",
    "The uppercase letter of your choice based on the analysis is:\n\n",
    "The uppercase letter of your choice based on the analysis is:\n",
    "The uppercase letter of your choice based on the analysis is:",
    "The uppercase letter of your choice based on the analysis is ",
    "The uppercase letter of your choice based on my analysis is:\n\n",
    "The uppercase letter of your choice based on my analysis is:\n",
    "The uppercase letter of your choice based on my analysis is:",
    
    # My choice patterns
    "The uppercase letter of my choice is \n\n",
    "The uppercase letter of my choice is \n",
    "The uppercase letter of my choice is ",
    "The uppercase letter of my choice is:\n\n",
    "The uppercase letter of my choice is:\n",
    "The uppercase letter of my choice is:",
    "The uppercase letter of my choice based on the analysis is:\n\n",
    "The uppercase letter of my choice based on the analysis is:\n",
    "The uppercase letter of my choice based on the analysis is:",
    "The uppercase letter of my choice based on the analysis is ",
    
    # Your choice patterns
    "The uppercase letter of your choice is \n\n",
    "The uppercase letter of your choice is \n",
    "The uppercase letter of your choice is ",
    "The uppercase letter of your choice.\n\n",
    "The uppercase letter of your choice.\n",
    "The uppercase letter of your choice.",
    
    # Correct choice patterns
    "The correct choice based on the analysis would be:\n\n",
    "The correct choice based on the analysis would be:\n",
    "The correct choice based on the analysis would be :",
    "The correct choice based on the analysis would be ",
    "The correct element to select would be:\n\n",
    "The correct element to select would be:\n",
    "The correct element to select would be:",
    "The correct element to select would be ",
    
    # Action instruction patterns
    "Choose an action from {CLICK, TYPE, SELECT}.\n\n",
    "Choose an action from {CLICK, TYPE, SELECT}.\n",
    "Choose an action from {CLICK, TYPE, SELECT}.",
    "Provide additional input based on ACTION.\n\n",
    "Provide additional input based on ACTION.\n",
    "Provide additional input based on ACTION.",
]


def build_history_summary_prompt(summary_source: str) -> str:
    return (
        "Summarize the earlier actions precisely and briefly. "
        "Cover what was done and outcomes. "
        "Do not include planned next steps. "
        "Limit to 3-5 sentences.\n\n" + summary_source
    )


def build_history_update_prompt(current_summary: str, new_actions: str) -> str:
    return (
        "Update the earlier-actions summary concisely (2-3 sentences). "
        "Incorporate these new actions and outcomes. "
        "Do not include planned next steps.\n\n"
        f"Current summary:\n{current_summary}\n\nNew actions:\n{new_actions}"
    )


def build_field_selection_prompt(intended_value: str, task: str, elements_desc: str, num_elements: int) -> str:
    return f"""You need to TYPE "{intended_value}" for the task: "{task}"

Available input fields:
{elements_desc}

Which field should be used? Respond with ONLY the number (0-{num_elements-1}).
"""


def build_dom_type_prompt(value: str, task: str, elements_list: str) -> str:
    return f"""You need to TYPE the text "{value}" into an input field.

**Current Task**: {task}

**Available Input Fields**:
{elements_list if elements_list else "  No input fields found"}

**Your Decision**:
Which input field should receive this text? Respond with ONLY the number in brackets [N].
If none of the fields are appropriate, respond with "NONE".

Example response: "0" or "2" or "NONE"
"""


def build_dom_select_prompt(value: str, task: str, elements_list: str) -> str:
    return f"""You need to SELECT the option "{value}" from a dropdown.

**Current Task**: {task}

**Available Dropdowns**:
{elements_list if elements_list else "  No dropdown fields found"}

**Your Decision**:
Which dropdown should be used for selecting "{value}"? Respond with ONLY the number in brackets [N].
If none of the dropdowns are appropriate, respond with "NONE".

Example response: "0" or "1" or "NONE"
"""


def build_dom_option_prompt(task: str, field_name: str, desired_value: str, options_list: str) -> str:
    desired_text = desired_value if desired_value is not None else ""
    return f"""You need to SELECT the best option for a dropdown.

**Current Task**: {task}
**Target Field**: {field_name or 'unknown'}
**Desired Option**: {desired_text if desired_text else 'Not specified'}

**Available Options**:
{options_list if options_list else "  No options found"}

**Your Decision**:
Which option should be selected? Respond with ONLY the number in brackets [N].
If none of the options are appropriate, respond with "NONE".

Example response: "0" or "2" or "NONE"
"""
