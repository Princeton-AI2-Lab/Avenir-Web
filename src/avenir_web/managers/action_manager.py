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

import logging

class ActionHistoryManager:
    def __init__(self, logger=None, max_stack_size=5, max_action_history=50):
        self.logger = logger or logging.getLogger(__name__)
        self.max_stack_size = max_stack_size
        self.max_action_history = max_action_history
        self.action_stack = []
        self.forbidden_actions = set()

    def add_action_to_stack(self, action_type, element_info=None, coordinates=None, taken_actions_count=0):
        """
        Add action to the action stack for repetition detection.
        """
        try:
            action_signature = {
                'action': action_type,
                'element': element_info or '',
                'coordinates': coordinates,
                'timestamp': taken_actions_count  # Use action count as timestamp
            }
            
            self.action_stack.append(action_signature)
            
            if len(self.action_stack) > self.max_stack_size:
                self.action_stack.pop(0)  # Remove oldest action
                
            self.logger.debug(f"Added action to stack: {action_signature}")
            
        except Exception as e:
            self.logger.error(f"Error adding action to stack: {e}")

    def detect_repetitive_actions(self, taken_actions):
        """
        Detect if recent actions are repetitive and should be forbidden.
        """
        try:
            if len(self.action_stack) < 2:
                return {"has_repetition": False, "forbidden_patterns": [], "suggestions": []}
            
            result = {
                "has_repetition": False,
                "forbidden_patterns": [],
                "suggestions": [],
                "repeated_action": None
            }
            
            # Check for exact repetition (same action + element)
            recent_actions = self.action_stack[-3:] if len(self.action_stack) >= 3 else self.action_stack
            
            # Pattern 1: Same action type + element repeated
            if len(recent_actions) >= 2:
                last_action = recent_actions[-1]
                second_last = recent_actions[-2]
                
                if (last_action['action'] == second_last['action'] and 
                    last_action['element'] == second_last['element']):
                    
                    # Check if this pattern appears multiple times
                    pattern_count = 0
                    for i in range(len(recent_actions) - 1):
                        if (recent_actions[i]['action'] == last_action['action'] and
                            recent_actions[i]['element'] == last_action['element']):
                            pattern_count += 1
                    
                    if pattern_count >= 2:  # Same action+element appeared 2+ times
                        result["has_repetition"] = True
                        result["repeated_action"] = last_action['action']
                        forbidden_pattern = f"{last_action['action']}:{last_action['element']}"
                        result["forbidden_patterns"].append(forbidden_pattern)
                        self.forbidden_actions.add(forbidden_pattern)
                        
                        result["suggestions"] = [
                            f"You have repeated {last_action['action']} on the same element multiple times.",
                            "This approach is not working. You MUST try a completely different action type.",
                            "Consider: SCROLL, TYPE in search box, SELECT from dropdown, or TERMINATE if stuck."
                        ]
            
            # Pattern 2: Same coordinates clicked repeatedly
            if len(recent_actions) >= 3:
                click_coords = [a['coordinates'] for a in recent_actions if a['action'] == 'CLICK' and a['coordinates']]
                if len(click_coords) >= 2:
                    # Check if clicking same coordinates
                    if len(set(click_coords)) == 1:  # All same coordinates
                        result["has_repetition"] = True
                        result["repeated_action"] = "CLICK"
                        coord_pattern = f"CLICK:{click_coords[0]}"
                        result["forbidden_patterns"].append(coord_pattern)
                        self.forbidden_actions.add(coord_pattern)
                        
                        result["suggestions"].append(
                            f"You have clicked the same coordinates {click_coords[0]} multiple times. Try a different approach."
                        )
            
            # Pattern 3: Action type repeated too many times
            action_types = [a['action'] for a in recent_actions]
            if len(action_types) >= 3:
                most_common = max(set(action_types), key=action_types.count)
                if action_types.count(most_common) >= 3:
                    result["has_repetition"] = True
                    result["repeated_action"] = most_common
                    result["forbidden_patterns"].append(most_common)
                    self.forbidden_actions.add(most_common)
                    
                    result["suggestions"].append(
                        f"You have used {most_common} action {action_types.count(most_common)} times recently. Try a different action type."
                    )

            # Pattern 4: Same element center used repeatedly across full history
            try:
                if len(taken_actions) >= 3:
                    recent_hist = taken_actions[-5:]
                    centers = [tuple(a.get('element_center')) for a in recent_hist if a.get('element_center')]
                    coords_hist = [tuple(a.get('coordinates')) for a in recent_hist if a.get('coordinates')]
                    # If last two centers are identical and have appeared 2+ times, forbid repeating
                    if centers and len(centers) >= 2:
                        if centers[-1] == centers[-2] and centers.count(centers[-1]) >= 2:
                            result["has_repetition"] = True
                            patt = f"ELEMENT_CENTER:{centers[-1]}"
                            if patt not in result["forbidden_patterns"]:
                                result["forbidden_patterns"].append(patt)
                                self.forbidden_actions.add(patt)
                            result["suggestions"].append("You are repeating actions on the same element location. Choose a different target.")
                    # Also consider raw coordinates from parsed actions history
                    if coords_hist and len(coords_hist) >= 2:
                        if coords_hist[-1] == coords_hist[-2] and coords_hist.count(coords_hist[-1]) >= 2:
                            result["has_repetition"] = True
                            patt = f"COORD:{coords_hist[-1]}"
                            if patt not in result["forbidden_patterns"]:
                                result["forbidden_patterns"].append(patt)
                                self.forbidden_actions.add(patt)
                            result["suggestions"].append("You are targeting the same coordinates repeatedly. Try a different area.")
            except Exception:
                pass
            
            if result["has_repetition"]:
                self.logger.warning(f"Repetitive actions detected: {result['forbidden_patterns']}")
                
            return result
            
        except Exception as e:
            self.logger.error(f"Error detecting repetitive actions: {e}")
            return {"has_repetition": False, "forbidden_patterns": [], "suggestions": []}

    def is_action_forbidden(self, action_type, element_info=None, coordinates=None):
        """
        Check if a proposed action is forbidden due to repetition.
        """
        try:
            # Check against forbidden patterns
            patterns_to_check = [
                action_type,  # Just action type
                f"{action_type}:{element_info}" if element_info else None,  # Action + element
                f"{action_type}:{coordinates}" if coordinates else None  # Action + coordinates
            ]
            
            for pattern in patterns_to_check:
                if pattern and pattern in self.forbidden_actions:
                    self.logger.warning(f"Action forbidden due to repetition: {pattern}")
                    return True
                    
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking forbidden action: {e}")
            return False

    def manage_action_history(self, taken_actions):
        """
        Manage action history by limiting history length.
        Returns the (possibly trimmed) taken_actions list.
        """
        try:
            # Limit history length
            if len(taken_actions) > self.max_action_history:
                # Keep only the most recent actions
                actions_to_remove = len(taken_actions) - self.max_action_history
                trimmed_actions = taken_actions[actions_to_remove:]
                self.logger.info(f"Action history trimmed: removed {actions_to_remove} oldest actions, keeping {len(trimmed_actions)} recent actions")
                return trimmed_actions
            
            self.logger.debug(f"Action history managed: {len(taken_actions)} actions in history")
            return taken_actions
            
        except Exception as e:
            self.logger.error(f"Error managing action history: {e}")
            return taken_actions
            
    def clear_history(self):
        self.action_stack = []
        self.forbidden_actions = set()
