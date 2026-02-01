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

from ...prompting.prompts import build_history_summary_prompt, build_history_update_prompt

def generate_detailed_action_list(actions, start_idx=1) -> str:
    """Generate detailed list of actions with step numbers."""
    lines = []
    for i, action in enumerate(actions, start_idx):
        action_type = action.get('predicted_action', action.get('action', 'UNKNOWN'))
        desc = action.get('action_description', '')
        success = action.get('success', True)
        
        status = "✓" if success else "✗"
        lines.append(f"  Step {i} [{status}]: {action_type} - {desc}")
    
    return "\n".join(lines) if lines else "  No meaningful actions"

def generate_phase_summary(actions) -> str:
    """Generate condensed summary of a phase of actions."""
    if not actions:
        return "  No actions in this phase"
    
    # Count action types
    action_counts = {}
    successful_actions = 0
    failed_actions = 0
    
    for action in actions:
        action_type = action.get('predicted_action', action.get('action', 'UNKNOWN'))
        action_counts[action_type] = action_counts.get(action_type, 0) + 1
        
        if action.get('success', True):
            successful_actions += 1
        else:
            failed_actions += 1
    
    # Create summary
    action_summary = ", ".join([f"{count} {atype}" for atype, count in action_counts.items()])
    success_rate = f"{successful_actions}/{len(actions)} successful"
    
    # Extract key actions (first, last, and any significant ones)
    key_actions = []
    if len(actions) > 0:
        first_desc = actions[0].get('action_description', 'Unknown')[:60]
        key_actions.append(f"Started: {first_desc}")
    
    if len(actions) > 1:
        last_desc = actions[-1].get('action_description', 'Unknown')[:60]
        key_actions.append(f"Ended: {last_desc}")
    
    return f"  Actions: {action_summary}\n  Success Rate: {success_rate}\n  Key Events:\n    - {chr(10).join(['    - ' + ka for ka in key_actions][1:]) if len(key_actions) > 1 else key_actions[0]}"

def generate_action_journey_summary(taken_actions) -> str:
    """
    Generate a comprehensive, condensed summary of the ENTIRE action journey.
    Groups actions into logical phases and highlights key milestones.
    """
    if not taken_actions:
        return "No actions taken."
    
    total_actions = len(taken_actions)
    
    # For small action counts, show all details
    if total_actions <= 10:
        return generate_detailed_action_list(taken_actions)
    
    # For larger counts, create phased summary
    summary_lines = []
    
    # Phase 1: Initial actions (first 5)
    summary_lines.append("=== INITIAL PHASE (Steps 1-5) ===")
    summary_lines.append(generate_phase_summary(taken_actions[:5]))
    
    # Phase 2: Middle actions (condensed)
    if total_actions > 15:
        middle_start = 5
        middle_end = total_actions - 5
        summary_lines.append(f"\n=== MIDDLE PHASE (Steps {middle_start+1}-{middle_end}) ===")
        summary_lines.append(generate_phase_summary(taken_actions[middle_start:middle_end]))
    
    # Phase 3: Final actions (last 5, most critical)
    summary_lines.append(f"\n=== FINAL PHASE (Steps {total_actions-4}-{total_actions}) ===")
    summary_lines.append(generate_detailed_action_list(taken_actions[-5:], start_idx=total_actions-4))
    
    return "\n".join(summary_lines)

def generate_action_summary(actions):
    """
    Generate natural language summary of a group of actions.
    """
    if not actions:
        return ""
    
    summaries = []
    for action in actions:
        desc = action.get('action_description', '').strip()
        action_type = action.get('predicted_action', '')
        success = action.get('success', True)
        
        if not desc or desc.lower() in ['no action', 'unknown action', 'none']:
            continue
        
        if desc.startswith('Clicked at coordinates'):
            parts = desc.split('-')
            if len(parts) > 1:
                desc = parts[-1].strip()
            else:
                desc = "Clicked element"
        
        def strip_numeric_parens(s: str):
            if not s:
                return s
            out = []
            i = 0
            n = len(s)
            while i < n:
                if s[i] == "(":
                    j = s.find(")", i + 1)
                    if j != -1:
                        inner = s[i + 1 : j]
                        parts = inner.split(",")
                        if len(parts) >= 2:
                            a = parts[0].strip()
                            b = parts[1].strip()
                            if a.isdigit() and b.isdigit():
                                i = j + 1
                                continue
                out.append(s[i])
                i += 1
            return "".join(out)

        desc = strip_numeric_parens(desc).strip()
        
        if desc and desc[0].islower():
            desc = desc[0].upper() + desc[1:]
        
        if len(desc) > 3:  # Avoid very short meaningless strings
            if not success:
                desc = f"{desc} (failed)"
            summaries.append(desc)
    
    if not summaries:
        action_types = [a.get('predicted_action', '') for a in actions]
        clicks = action_types.count('CLICK')
        types = action_types.count('TYPE')
        if clicks > 0 and types > 0:
            return f"Clicked {clicks} elements and typed in {types} fields"
        elif clicks > 0:
            return f"Clicked {clicks} elements"
        elif types > 0:
            return f"Typed in {types} fields"
        return f"Performed {len(actions)} actions"
    
    if len(summaries) <= 3:
        return "; ".join(summaries)
    else:
        return f"{summaries[0]}; {summaries[1]}; ... {summaries[-1]}"

async def llm_summarize_actions(actions, engine):
    try:
        if not actions:
            return ""
        lines = []
        for a in actions:
            if isinstance(a, dict):
                desc = a.get('action_description', '')
                success = a.get('success', True)
                status = "SUCCESS" if success else "FAILED"
                lines.append(f"- {desc} ({status})")
        summary_source = "\n".join(lines)[:2000]
        prompt = build_history_summary_prompt(summary_source)
        response = await engine.generate(prompt=["", prompt, ""], temperature=0.0, max_new_tokens=120, turn_number=0)
        if isinstance(response, list):
            return (response[0] or "").strip()
        if isinstance(response, str):
            return response.strip()
        if hasattr(response, 'choices') and response.choices:
            return (response.choices[0].message.content or "").strip()
        return ""
    except Exception:
        return generate_action_summary(actions)

async def llm_update_history_summary(delta_actions, llm_history_summary_text, engine):
    try:
        if not delta_actions:
            return llm_history_summary_text
        prev = llm_history_summary_text or ""
        lines = []
        for a in delta_actions:
            if isinstance(a, dict):
                desc = a.get('action_description', '')
                success = a.get('success', True)
                status = "SUCCESS" if success else "FAILED"
                lines.append(f"- {desc} ({status})")
        delta_src = "\n".join(lines)[:2000]
        prompt = build_history_update_prompt(prev, delta_src)
        response = await engine.generate(prompt=["", prompt, ""], temperature=0.0, max_new_tokens=140, turn_number=0)
        if isinstance(response, list):
            return (response[0] or "").strip()
        if isinstance(response, str):
            return response.strip()
        if hasattr(response, 'choices') and response.choices:
            return (response.choices[0].message.content or "").strip()
        return prev
    except Exception:
        addendum = await llm_summarize_actions(delta_actions, engine)
        if prev:
            return f"{prev} {addendum}".strip()
        return addendum

def generate_recent_action_summary(taken_actions, max_actions=5) -> str:
    """Generate summary of most recent actions."""
    if not taken_actions:
        return "No recent actions."
    
    recent = taken_actions[-max_actions:]
    return generate_detailed_action_list(recent, start_idx=max(1, len(taken_actions) - max_actions + 1))

def generate_comprehensive_action_summary(taken_actions, max_actions=20, compress_old=True) -> str:
    """Generate comprehensive summary with optional compression of old actions."""
    if not taken_actions:
        return "No actions recorded."
        
    if len(taken_actions) <= max_actions or not compress_old:
        return generate_detailed_action_list(taken_actions)
        
    # Compress older actions
    return generate_action_journey_summary(taken_actions)
