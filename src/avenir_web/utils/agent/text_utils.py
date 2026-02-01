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

def generate_action_description(parsed_action, logger=None):
    """
    Generate a human-readable description from parsed action.
    
    Args:
        parsed_action: Dict with 'action', optional 'coordinates', 'text', 'value', 'field'
        logger: Optional logger for error reporting
        
    Returns:
        str: Human-readable action description
    """
    try:
        action = parsed_action.get('action', 'UNKNOWN')
        coords = parsed_action.get('coordinates', [])
        value = parsed_action.get('value', '')
        field = parsed_action.get('field', '')
        
        # Format based on action type
        if action == "CLICK":
            if coords:
                return f"Click at ({coords[0]}, {coords[1]})"
            return "Click"
        
        elif action == "TYPE":
            desc = f"Type '{value}'"
            if field:
                desc += f" in {field} field"
            return desc
        
        elif action == "SELECT":
            desc = f"Select '{value}'"
            if field:
                desc += f" from {field} dropdown"
            return desc
        
        elif action == "HOVER":
            if coords:
                return f"Hover at ({coords[0]}, {coords[1]})"
            return "Hover"
        
        elif action == "SCROLL UP":
            return "Scroll up"
        
        elif action == "SCROLL DOWN":
            return "Scroll down"
        
        elif action == "SCROLL TOP":
            return "Scroll to top"
        
        elif action == "SCROLL BOTTOM":
            return "Scroll to bottom"
        
        elif action == "PRESS ENTER":
            return "Press Enter"
        
        elif action == "GOTO":
            return f"Navigate to {value}"
        
        elif action == "NEW TAB":
            return "Open new tab"
        
        elif action == "CLOSE TAB":
            return "Close tab"
        
        elif action == "GO BACK":
            return "Go to previous tab"
        
        elif action == "WAIT":
            return f"Wait {value} seconds"
        
        elif action == "TERMINATE":
            return f"Terminate: {value}" if value else "Terminate task"
        
        else:
            return f"{action}" + (f": {value}" if value else "")
            
    except Exception as e:
        if logger:
            logger.error(f"Failed to generate action description: {e}")
        return f"{parsed_action.get('action', 'UNKNOWN')} action"

def compose_action_description(action, value, field, element_desc, coords=None):
    a = action or ''
    v = value or ''
    f = field or ''
    d = element_desc or ''
    if a == 'CLICK':
        return "Clicked"
    if a == 'HOVER':
        return f"Hovered on {d}" if d else "Hovered"
    if a == 'TYPE':
        if f:
            return f"Typed '{v}' in {f}"
        if d:
            return f"Typed '{v}' in {d}"
        return f"Typed '{v}'"
    if a == 'SELECT':
        if f:
            return f"Selected '{v}' from {f}"
        if d:
            return f"Selected '{v}' from {d}"
        return f"Selected '{v}'"
    if a == 'PRESS ENTER':
        return "Pressed Enter"
    if a == 'SCROLL UP':
        return "Scrolled up"
    if a == 'SCROLL DOWN':
        return "Scrolled down"
    if a == 'SCROLL TOP':
        return "Scrolled to top"
    if a == 'SCROLL BOTTOM':
        return "Scrolled to bottom"
    if a == 'GOTO':
        return f"Navigated to {v}" if v else "Navigated"
    if a == 'WAIT':
        return f"Waited {v} seconds" if v else "Waited"
    if a == 'KEYBOARD':
        if v and v.upper() == 'CLEAR':
            return "Cleared field via keyboard"
        return f"Typed '{v}' via keyboard" if v else "Typed via keyboard"
    if a == 'NEW TAB':
        return "Opened new tab"
    if a == 'CLOSE TAB':
        return "Closed tab"
    if a == 'GO BACK':
        return "Went back"
    if a == 'GO FORWARD':
        return "Went forward"
    if a == 'TERMINATE':
        return "Terminated"
    return a or "Action"

def compress_text(text: str, max_length: int) -> str:
    if not text:
        return ""
    if len(text) <= max_length:
        return text
    return text[:max_length//2] + " ... " + text[-max_length//2:]

def compress_url(url: str) -> str:
    if not url:
        return ""
    if len(url) > 50:
        return url[:30] + "..." + url[-10:]
    return url
