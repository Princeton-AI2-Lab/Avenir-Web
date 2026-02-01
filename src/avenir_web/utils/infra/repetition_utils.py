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

from ...prompting.prompts import (
    build_repetitive_action_warning,
    build_select_failure_warning,
    build_consecutive_failure_warning,
    build_high_failure_rate_warning,
    build_repeating_action_type_warning,
    build_termination_recommendation_warning,
    build_select_click_loop_warning,
    build_element_failure_warning,
    build_general_failure_warning,
)


def analyze_repetitive_patterns(previous_actions):
    if not previous_actions or len(previous_actions) < 2:
        return None

    recent_actions = previous_actions[-10:] if len(previous_actions) > 10 else previous_actions

    def is_action_failed(action):
        if not isinstance(action, dict):
            return False

        if action.get("success") is False or action.get("error"):
            return True

        return False

    action_patterns = []
    consecutive_failures = 0
    failed_actions = 0

    for action in recent_actions:
        if isinstance(action, dict):
            action_type = action.get("action", "").upper()
            element = action.get("element", "")
            value = action.get("value", "")
            failed = is_action_failed(action)

            action_patterns.append((action_type, element, value, failed))

            if failed:
                failed_actions += 1
                consecutive_failures += 1
            else:
                consecutive_failures = 0

    warnings = []

    if len(action_patterns) >= 3:
        for i in range(len(action_patterns) - 2):
            current = action_patterns[i]
            next_1 = action_patterns[i + 1]
            next_2 = action_patterns[i + 2]

            if (
                current[0] == next_1[0] == next_2[0]
                and current[1] == next_1[1] == next_2[1]
                and current[2] == next_1[2] == next_2[2]
                and current[3]
                and next_1[3]
                and next_2[3]
            ):
                warnings.append(build_repetitive_action_warning(current[0], current[1], current[2], 3))
                break

    select_failures = []
    for i, (action_type, element, value, failed) in enumerate(action_patterns):
        if action_type == "SELECT" and failed:
            select_failures.append((i, element, value))

    if len(select_failures) >= 2:
        same_select_attempts = {}
        for _, element, value in select_failures:
            key = (element, value)
            same_select_attempts[key] = same_select_attempts.get(key, 0) + 1

        for (element, value), count in same_select_attempts.items():
            if count >= 2:
                warnings.append(build_select_failure_warning(element, value, count))

    if consecutive_failures >= 2:
        warnings.append(build_consecutive_failure_warning(consecutive_failures))

    if len(recent_actions) >= 3 and failed_actions / len(recent_actions) > 0.5:
        warnings.append(build_high_failure_rate_warning(failed_actions, len(recent_actions)))

    action_types = [pattern[0] for pattern in action_patterns]
    if len(action_types) >= 4:
        last_4_actions = action_types[-4:]
        if len(set(last_4_actions)) == 1:
            failed_count = sum(1 for j in range(-4, 0) if action_patterns[j][3])
            if failed_count >= 3:
                warnings.append(build_repeating_action_type_warning(last_4_actions[0], failed_count))

    if len(recent_actions) >= 6:
        recent_failed = sum(1 for pattern in action_patterns[-6:] if pattern[3])
        if recent_failed >= 5:
            warnings.append(build_termination_recommendation_warning())

    select_click_pattern = 0
    for i in range(len(action_patterns) - 1):
        current = action_patterns[i]
        next_action = action_patterns[i + 1]
        if current[0] == "SELECT" and current[3] and next_action[0] == "CLICK" and current[1] == next_action[1]:
            select_click_pattern += 1

    if select_click_pattern >= 1:
        warnings.append(build_select_click_loop_warning())

    element_failure_count = {}
    for pattern in action_patterns[-6:]:
        if pattern[3]:
            element = pattern[1]
            if element and element != "none":
                element_failure_count[element] = element_failure_count.get(element, 0) + 1

    for element, count in element_failure_count.items():
        if count >= 2:
            warnings.append(build_element_failure_warning(element, count))

    if len(recent_actions) >= 3 and failed_actions >= 2:
        warnings.append(build_general_failure_warning())

    return " ".join(warnings) if warnings else None
