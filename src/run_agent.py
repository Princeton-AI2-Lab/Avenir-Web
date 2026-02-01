#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
This script uses the Avenir-Web agent to perform web automation tasks.
It supports batch task mode.
Configuration is loaded from TOML files compatible with the bundled agent config format.
"""

import argparse
import asyncio
import datetime
import json
import logging
import os
import sys
from pathlib import Path

import toml
from avenir_web.runtime.llm_engine import is_blank_or_placeholder_api_key
from avenir_web.agent import AvenirWebAgent

# Setup your API Key here, or pass through environment
# os.environ["OPENAI_API_KEY"] = "Your API KEY Here"
# os.environ["GEMINI_API_KEY"] = "Your API KEY Here"


class SafeStreamHandler(logging.StreamHandler):
    def flush(self):
        try:
            super().flush()
        except (TimeoutError, OSError):
            pass


def setup_logging():
    """Setup comprehensive logging configuration for debugging and monitoring"""
    
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate timestamp for log files
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Configure root logger with minimal setup to avoid interfering with task-specific loggers
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Only add console handler to root logger to avoid conflicts
    console_handler = SafeStreamHandler(sys.stdout)
    console_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    root_logger.addHandler(console_handler)
    
    # Create separate loggers for run_agent.py specific logging
    run_logger = logging.getLogger('run_agent')
    run_handler = logging.FileHandler(
        os.path.join(log_dir, f"avenir_web_run_{timestamp}.log"),
        mode='w',
        encoding='utf-8'
    )
    run_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    run_logger.addHandler(run_handler)
    run_logger.setLevel(logging.INFO)
    
    # Error logger for critical issues
    error_logger = logging.getLogger('errors')
    error_handler = logging.FileHandler(
        os.path.join(log_dir, f"avenir_web_errors_{timestamp}.log"),
        mode='w',
        encoding='utf-8'
    )
    error_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    error_handler.setLevel(logging.ERROR)
    error_logger.addHandler(error_handler)
    error_logger.setLevel(logging.ERROR)
    
    # Task execution logger - batch processing details
    task_logger = logging.getLogger('task_execution')
    task_handler = logging.FileHandler(
        os.path.join(log_dir, f"task_execution_{timestamp}.log"),
        mode='w',
        encoding='utf-8'
    )
    task_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    )
    task_logger.addHandler(task_handler)
    task_logger.setLevel(logging.INFO)
    
    # Performance logger - timing and resource usage
    perf_logger = logging.getLogger('performance')
    perf_handler = logging.FileHandler(
        os.path.join(log_dir, f"performance_{timestamp}.log"),
        mode='w',
        encoding='utf-8'
    )
    perf_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(message)s')
    )
    perf_logger.addHandler(perf_handler)
    perf_logger.setLevel(logging.INFO)
    
    print(f"Logging configured. Log files will be saved in '{log_dir}/' directory")
    print(f"Main log: avenir_web_run_{timestamp}.log")
    print(f"Error log: avenir_web_errors_{timestamp}.log")
    print(f"Task log: task_execution_{timestamp}.log")
    print(f"Performance log: performance_{timestamp}.log")





async def run_single_task(config, task_dict):
    """Run a single task using Avenir-Web agent
    
    Returns:
        dict: Task execution result with status and details
    """
    
    # Get specialized loggers
    task_logger = logging.getLogger('task_execution')
    perf_logger = logging.getLogger('performance')
    
    task_id = task_dict['task_id']
    
    task_logger.info(f"Starting task: {task_id}")
    task_logger.info(f"Task description: {task_dict['confirmed_task']}")
    task_logger.info(f"Website: {task_dict['website']}")
    
    print(f"Starting task: {task_dict['confirmed_task']}")
    print(f"Website: {task_dict['website']}")
    
    print("\n=== Using Original Task ===")
    
    # Get model configuration
    model_config = config.get('model', {})
    
    model_name = model_config.get('name', 'openrouter/qwen/qwen-2.5-72b-instruct')
    temperature = model_config.get('temperature', 1)
    
    # API key configuration - prioritize environment variables, then new [api_keys] section
    api_keys_config = config.get('api_keys', {})
    
    # Set API keys from config if not in environment
    if 'OPENROUTER_API_KEY' not in os.environ:
        openrouter_key = api_keys_config.get('openrouter_api_key')
        if not is_blank_or_placeholder_api_key(openrouter_key):
            os.environ['OPENROUTER_API_KEY'] = openrouter_key
    
    if 'GEMINI_API_KEY' not in os.environ:
        gemini_key = api_keys_config.get('gemini_api_key')
        if not is_blank_or_placeholder_api_key(gemini_key):
            os.environ['GEMINI_API_KEY'] = gemini_key

    # Extract configuration for the agent
    # Note: save_file_dir should be the base directory; the agent will create a task_id subdirectory
    agent_config = {
        'task_id': task_dict['task_id'],
        'default_task': task_dict['confirmed_task'],
        'default_website': task_dict['website'],
        'save_file_dir': config['basic']['save_file_dir'],
        'max_auto_op': config.get('experiment', {}).get('max_op', 30),
        'highlight': config.get('experiment', {}).get('highlight', config.get('agent', {}).get('highlight', False)),
        'headless': config.get('playwright', {}).get('headless', config.get('browser', {}).get('headless', True)),
        'viewport': config.get('playwright', {}).get('viewport', config.get('browser', {}).get('viewport', {'width': 1200, 'height': 1080})),
        'model': model_name,
        'temperature': temperature,
        'create_timestamp_dir': False
    }
    
    # Create agent with configuration
    agent = AvenirWebAgent(config=config, **agent_config)
    
    # Initialize result tracking
    result = {
        'task_id': task_dict['task_id'],
        'status': 'failed',
        'error': None,
        'operations_count': 0,
        'execution_time': 0
    }
    
    import time
    start_time = time.time()
    perf_logger.info(f"Task {task_id} started at {datetime.datetime.now()}")
    
    try:
        # Start the agent
        task_logger.info(f"Initializing agent for task {task_id}")
        await agent.start(website=task_dict['website'])
        task_logger.info(f"Agent started successfully for task {task_id}")
        
        task_logger.info(f"Starting prediction loop for task {task_id}")
        
        # Initialize execution tracking
        consecutive_errors = 0
        max_consecutive_errors = 3  # Allow max 3 consecutive errors before terminating
        total_execution_time = 0
        max_total_execution_time = 3600  # 60 minutes maximum per task
        
        while not agent.complete_flag:
            # Check global timeout
            total_execution_time = time.time() - start_time
            if total_execution_time > max_total_execution_time:
                timeout_msg = f"Task exceeded maximum execution time ({max_total_execution_time/60:.1f} minutes), terminating"
                task_logger.error(f"Task {task_id} terminated due to global timeout")
                print(timeout_msg)
                logging.error(f"Task {task_dict['task_id']} terminated due to global timeout")
                agent.complete_flag = True
                result['status'] = 'global_timeout'
                break
                
            try:
                async def step():
                    prediction_dict = await agent.predict()
                    
                    if prediction_dict is None:
                        print(f"Prediction returned None despite fixes - creating fallback")
                        logging.error(f"Prediction returned None for task {task_dict['task_id']} despite fixes")
                        prediction_dict = {
                            'action': 'NONE', 
                            'value': '',
                            'element': None,
                            'action_generation': 'Fallback due to None prediction',
                            'action_grounding': 'System fallback',
                            'termination_reason': 'none_prediction_fallback'
                        }
                    
                    if not isinstance(prediction_dict, dict):
                        print(f"Prediction returned invalid type: {type(prediction_dict)}, treating as NONE action.")
                        logging.warning(f"Prediction returned invalid type for task {task_dict['task_id']}: {type(prediction_dict)}")
                        prediction_dict = {
                            'action': 'NONE', 
                            'value': None,
                            'element': None,
                            'action_generation': 'Invalid prediction type',
                            'action_grounding': 'No grounding available'
                        }
                    
                    required_keys = ['action', 'value', 'element']
                    for key in required_keys:
                        if key not in prediction_dict:
                            prediction_dict[key] = None if key != 'action' else 'NONE'
                            
                    await agent.execute(prediction_dict)

                await step()
            except asyncio.TimeoutError:
                print("Step timed out after 3 minutes, skipping to next step.")
                logging.warning(f"Step for task {task_dict['task_id']} timed out after 3 minutes.")
                try:
                    if agent.session_control.get('context') and agent.session_control['context'].pages:
                        agent.page = agent.session_control['context'].pages[-1]
                        await agent.page.bring_to_front()
                        logging.info(f"Timeout recovery: switched to existing page {agent.page.url}")
                except Exception as recovery_e:
                    logging.warning(f"Timeout recovery failed: {recovery_e}")
                    pass
            except Exception as inner_e:
                consecutive_errors += 1
                error_msg = f"Error during execution (#{consecutive_errors}), continuing: {inner_e}"
                task_logger.warning(f"Execution error in task {task_id}: {inner_e}")
                print(error_msg)
                logging.warning(f"Execution error for task {task_dict['task_id']}: {inner_e}")
                
                # Force termination after too many consecutive errors to prevent infinite loops
                if consecutive_errors >= max_consecutive_errors:
                    error_msg = f"Too many consecutive errors ({consecutive_errors}), terminating task"
                    task_logger.error(f"Task {task_id} terminated due to {consecutive_errors} consecutive errors")
                    print(error_msg)
                    logging.error(f"Task {task_dict['task_id']} terminated due to consecutive errors")
                    agent.complete_flag = True
                    result['status'] = 'error_terminated'
                    break
            else:
                # Reset consecutive error counter on successful execution
                consecutive_errors = 0

        # Record execution details
        result['operations_count'] = agent.valid_op if agent else 0
        result['execution_time'] = time.time() - start_time
        
        # Log performance metrics
        perf_logger.info(f"Task {task_id} - Operations: {result['operations_count']}, Time: {result['execution_time']:.2f}s")
        
        # If we completed normally without errors, mark as success
        if result['status'] == 'failed' and not result['error']:
            result['status'] = 'completed'
            task_logger.info(f"Task {task_id} completed successfully")
            print(f"Task completed successfully: {task_dict['task_id']}")
        
    except Exception as e:
        result['error'] = str(e)
        result['status'] = 'error'
        result['execution_time'] = time.time() - start_time
        task_logger.error(f"Task {task_id} failed with error: {e}", exc_info=True)
        print(f"Error running task {task_dict['task_id']}: {e}")
        logging.error(f"Task {task_dict['task_id']} failed: {e}")
    finally:
        # Stop the agent
        try:
            await agent.stop()
            task_logger.info(f"Agent stopped for task {task_id}")
        except Exception as stop_e:
            task_logger.warning(f"Error stopping agent for task {task_id}: {stop_e}")
            logging.warning(f"Error stopping agent for task {task_dict['task_id']}: {stop_e}")
            
            # Try emergency save if agent.stop() fails
            try:
                if hasattr(agent, '_emergency_save'):
                    emergency_file = agent._emergency_save(f"Agent stop failed: {stop_e}")
                    if emergency_file:
                        task_logger.info(f"Emergency save completed for task {task_id}: {emergency_file}")
                        print(f"Emergency save completed for task {task_dict['task_id']}: {emergency_file}")
                    else:
                        task_logger.error(f"Emergency save also failed for task {task_id}")
                        print(f"Emergency save also failed for task {task_dict['task_id']}")
            except Exception as emergency_e:
                task_logger.critical(f"Emergency save failed for task {task_id}: {emergency_e}")
                print(f"CRITICAL: Emergency save failed for task {task_dict['task_id']}: {emergency_e}")
    
    perf_logger.info(f"Task {task_id} finished at {datetime.datetime.now()}")
    return result


async def run_batch_mode(config):
    """
    Run in batch mode - multiple tasks from JSON file with automatic task switching on failure.
    
    Args:
        config: Configuration dictionary loaded from TOML file
        
    Returns:
        dict: Batch execution summary with success/failure counts
    """
    print("=== Batch Mode ===")
    
    task_file_path = config['experiment']['task_file_path']
    if not os.path.isabs(task_file_path):
        # Make path relative to script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        task_file_path = os.path.join(script_dir, task_file_path)
    
    try:
        with open(task_file_path, 'r', encoding='utf-8') as file:
            query_tasks = json.load(file)
    except FileNotFoundError:
        print(f"Error: Task file '{task_file_path}' not found.")
        return {'status': 'error', 'message': f'Task file not found: {task_file_path}'}
    except json.JSONDecodeError:
        print(f"Error: Task file '{task_file_path}' is not a valid JSON file.")
        return {'status': 'error', 'message': f'Invalid JSON in task file: {task_file_path}'}
    
    # Initialize batch execution summary
    import time
    batch_summary = {
        'total_tasks': len(query_tasks),
        'completed': 0,
        'failed': 0,
        'timeout': 0,
        'error': 0,
        'max_operations_reached': 0,
        'max_no_ops_reached': 0,
        'task_results': [],
        'start_time': time.time(),
        'end_time': None
    }
    
    print(f"\n{'='*60}")
    print(f"BATCH MODE: Processing {len(query_tasks)} tasks")
    print(f"Tasks file: {task_file_path}")
    print(f"Output directory: {config['basic']['save_file_dir']}")
    print(f"{'='*60}")
    
    overwrite = config['experiment']['overwrite']
    
    for i, task_dict in enumerate(query_tasks, 1):
        print(f"\n--- Processing Task {i}/{len(query_tasks)} ---")
        
        # Check if task already exists
        task_result_path = os.path.join(config['basic']['save_file_dir'], task_dict['task_id'])
        if os.path.exists(task_result_path) and not overwrite:
            print(f"Task {task_dict['task_id']} already exists, skipping...")
            continue
        
        try:
            # Run the task with retry logic for better error recovery
            result = await run_single_task_with_retry(config, task_dict, max_retries=2)
            
            # Update batch summary based on task result
            status = result.get('status', 'failed')
            batch_summary['task_results'].append(result)
            
            if status == 'completed':
                batch_summary['completed'] += 1
                print(f"Task {task_dict['task_id']} completed successfully")
            elif status == 'timeout':
                batch_summary['timeout'] += 1
                print(f"Task {task_dict['task_id']} timed out")
            elif status == 'error':
                batch_summary['error'] += 1
                print(f"Task {task_dict['task_id']} failed with error: {result.get('error', 'Unknown error')}")
            elif status == 'max_operations_reached':
                batch_summary['max_operations_reached'] += 1
                print(f"Task {task_dict['task_id']} reached maximum operations")
            elif status == 'max_no_ops_reached':
                batch_summary['max_no_ops_reached'] += 1
                print(f"Task {task_dict['task_id']} reached maximum no-op actions")
            else:
                batch_summary['failed'] += 1
                print(f"Task {task_dict['task_id']} failed")
            
            # Log execution details
            ops_count = result.get('operations_count', 0)
            exec_time = result.get('execution_time', 0)
            print(f"   Operations: {ops_count}, Time: {exec_time:.1f}s")
            
        except Exception as e:
            batch_summary['error'] += 1
            error_result = {
                'task_id': task_dict['task_id'],
                'status': 'error',
                'error': str(e),
                'operations_count': 0,
                'execution_time': 0
            }
            batch_summary['task_results'].append(error_result)
            print(f"Unexpected error in task {task_dict['task_id']}: {e}")
            logging.error(f"Unexpected error in task {task_dict['task_id']}: {e}", exc_info=True)
        
        if i < len(query_tasks):
            await asyncio.sleep(2)
    
    # Finalize batch summary
    batch_summary['end_time'] = time.time()
    total_time = batch_summary['end_time'] - batch_summary['start_time']
    
    # Print final summary
    print(f"\n{'='*60}")
    print(f"BATCH EXECUTION COMPLETED")
    print(f"{'='*60}")
    print(f"Total tasks: {batch_summary['total_tasks']}")
    print(f"Completed: {batch_summary['completed']}")
    print(f"Failed: {batch_summary['failed']}")
    print(f"Timeout: {batch_summary['timeout']}")
    print(f"Max operations: {batch_summary['max_operations_reached']}")
    print(f"Max no-ops: {batch_summary['max_no_ops_reached']}")
    print(f"Errors: {batch_summary['error']}")
    print(f"Total time: {total_time:.1f}s")
    print(f"{'='*60}")
    
    return batch_summary


async def run_single_task_with_retry(config, task_dict, max_retries=2):
    """
    Run a single task with retry logic for better error recovery.
    
    Args:
        config: Agent configuration dictionary
        task_dict: Task configuration dictionary
        max_retries: Maximum number of retry attempts (default: 2)
        
    Returns:
        dict: Task execution result with status and details
    """
    
    task_id = task_dict.get('task_id', 'unknown')
    
    for attempt in range(max_retries + 1):
        if attempt > 0:
            print(f"\nRetry attempt {attempt}/{max_retries} for task {task_id}")
            logging.info(f"Retrying task {task_id}, attempt {attempt}/{max_retries}")
            await asyncio.sleep(5)
        
        try:
            result = await run_single_task(config, task_dict)
            
            # Check if task completed successfully or reached expected termination
            status = result.get('status', 'failed')
            if status in ['completed', 'max_operations_reached', 'max_no_ops_reached']:
                if attempt > 0:
                    print(f"Task {task_id} succeeded on retry attempt {attempt}")
                return result
            
            # If task failed due to timeout or error, consider retry
            if status in ['timeout', 'error'] and attempt < max_retries:
                error_msg = result.get('error', 'Unknown error')
                print(f"WARNING: Task {task_id} failed (attempt {attempt + 1}): {error_msg}")
                logging.warning(f"Task {task_id} failed on attempt {attempt + 1}: {error_msg}")
                continue
            
            # If we've exhausted retries or it's a non-retryable failure
            return result
            
        except Exception as e:
            error_msg = str(e)
            print(f"Unexpected error in task {task_id} (attempt {attempt + 1}): {error_msg}")
            logging.error(f"Unexpected error in task {task_id} attempt {attempt + 1}: {error_msg}", exc_info=True)
            
            if attempt < max_retries:
                continue
            
            # Return error result if all retries exhausted
            return {
                'task_id': task_id,
                'status': 'error',
                'error': error_msg,
                'operations_count': 0,
                'execution_time': 0,
                'retry_attempts': attempt + 1
            }
    
    # This should not be reached, but just in case
    return {
        'task_id': task_id,
        'status': 'failed',
        'error': 'All retry attempts exhausted',
        'operations_count': 0,
        'execution_time': 0,
        'retry_attempts': max_retries + 1
    }


async def main():
    """Main function with enhanced error handling and retry logic"""
    parser = argparse.ArgumentParser(
        description="Run Avenir-Web web automation tasks"
    )
    parser.add_argument(
        "-c", "--config_path", 
        help="Path to the TOML configuration file.", 
        type=str, 
        metavar='config',
        default="config/batch_experiment.toml"
    )

    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Load configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = args.config_path
    if not os.path.isabs(config_path):
        config_path = os.path.join(script_dir, config_path)
    
    try:
        with open(config_path, 'r') as toml_config_file:
            config = toml.load(toml_config_file)
            print(f"Configuration File Loaded - {config_path}")
    except FileNotFoundError:
        print(f"Error: File '{config_path}' not found.")
        sys.exit(1)
    except toml.TomlDecodeError:
        print(f"Error: File '{config_path}' is not a valid TOML file.")
        sys.exit(1)
    
    # Validate API key configuration (OpenRouter/Gemini only)
    api_keys_config = config.get('api_keys', {})
    model_config = config.get('model', {})
    model_name = model_config.get('name', 'openrouter/qwen/qwen-2.5-72b-instruct')
    
    if any(k in model_name.lower() for k in ["claude", "qwen", "openrouter", "gemini"]):
        # Prefer OpenRouter; allow Gemini if configured
        openrouter_key = api_keys_config.get('openrouter_api_key') or os.getenv('OPENROUTER_API_KEY')
        gemini_key = api_keys_config.get('gemini_api_key') or os.getenv('GEMINI_API_KEY')
        if not (openrouter_key or gemini_key):
            print("Error: Missing API key. Set OPENROUTER_API_KEY (preferred) or GEMINI_API_KEY.")
            sys.exit(1)
    
    # Create save directory if it doesn't exist
    save_dir = config['basic']['save_file_dir']
    if not os.path.isabs(save_dir):
        save_dir = os.path.join(script_dir, save_dir)
    os.makedirs(save_dir, exist_ok=True)
    config['basic']['save_file_dir'] = save_dir
    
    # Batch mode only
    if 'experiment' not in config or 'task_file_path' not in config['experiment']:
        print("Error: Missing [experiment].task_file_path in config. Provide a tasks JSON file path.")
        sys.exit(1)
    await run_batch_mode(config)


if __name__ == "__main__":
    asyncio.run(main())
