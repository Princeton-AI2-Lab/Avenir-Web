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
import os

def setup_logger(task_id, main_path, redirect_to_dev_log=False):
    """Set up a logger to log to both file and console within the main_path."""
    # Use task_id as logger name to ensure each task has a unique logger
    logger_name = f"{task_id}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers to avoid conflicts
    logger.handlers.clear()
    
    # Create a file handler for writing logs to a file
    log_filename = 'agent.log'
    f_handler = logging.FileHandler(os.path.join(main_path, log_filename), encoding='utf-8')
    f_handler.setLevel(logging.INFO)

    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.INFO)

    file_formatter = logging.Formatter('%(asctime)s - %(message)s')
    console_formatter = logging.Formatter('%(message)s')

    f_handler.setFormatter(file_formatter)
    c_handler.setFormatter(console_formatter)

    logger.addHandler(f_handler)
    if not redirect_to_dev_log:  # Only add console handler if not redirecting to dev log
        logger.addHandler(c_handler)

    logger.propagate = False
    return logger
