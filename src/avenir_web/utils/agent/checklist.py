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

"""
Checklist utility functions for status computation and formatting.
Only contains functions actually used by ChecklistManager.
"""


def get_checklist_status(task_checklist):
    """
    Compute checklist status and progress metrics from a list of items.
    """
    if not task_checklist:
        return {
            "total": 0,
            "completed": 0,
            "in_progress": 0,
            "pending": 0,
            "failed": 0,
            "progress": 0.0,
            "completion_rate": 0.0,
            "success_rate": 0.0
        }

    status_counts = {"pending": 0, "in_progress": 0, "completed": 0, "failed": 0}

    for item in task_checklist:
        status = (item.get('status', 'pending') or 'pending').strip().lower().replace(' ', '_')
        status_counts[status] = status_counts.get(status, 0) + 1

    total = len(task_checklist)
    completed = status_counts['completed']
    failed = status_counts['failed']

    progress = (completed / total * 100) if total > 0 else 0
    completion_rate = (completed / total) if total > 0 else 0
    success_rate = (completed / (completed + failed)) if (completed + failed) > 0 else 0.0

    return {
        "total": total,
        "completed": completed,
        "in_progress": status_counts['in_progress'],
        "pending": status_counts['pending'],
        "failed": failed,
        "progress": progress,
        "completion_rate": completion_rate,
        "success_rate": success_rate
    }


def format_checklist_for_prompt(task_checklist):
    """
    Format checklist for inclusion in LLM prompts.
    """
    if not task_checklist:
        return "No checklist available"

    status = get_checklist_status(task_checklist)

    checklist_str = "TASK CHECKLIST:\n"
    checklist_str += "=" * 50 + "\n"

    for i, item in enumerate(task_checklist, 1):
        item_status = item.get('status', 'pending')
        description = item.get('description', '')
        item_id = item.get('id', f'requirement_{i}')

        status_indicator = {
            'pending': '[PENDING]',
            'in_progress': '[IN_PROGRESS]',
            'completed': '[COMPLETED]',
            'failed': '[FAILED]'
        }.get(item_status, '[PENDING]')
        checklist_str += f"{i:2d}. {status_indicator} {item_id}: {description}\n"

    checklist_str += "=" * 50 + "\n"
    checklist_str += f"PROGRESS SUMMARY:\n"
    checklist_str += f"  Total Items: {status['total']}\n"
    checklist_str += f"  Completed: {status['completed']} ({status['progress']:.1f}%)\n"
    checklist_str += f"  In Progress: {status['in_progress']}\n"
    checklist_str += f"  Pending: {status['pending']}\n"
    checklist_str += f"  Failed: {status['failed']}\n"
    checklist_str += f"  Success Rate: {status['success_rate']:.1%}\n"
    checklist_str += "=" * 50

    return checklist_str
