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
DOM-based operations for faster and more reliable SELECT and TYPE actions.
Uses Playwright to extract all form elements and operate on them directly without coordinates.

This module provides:
1. Element extraction: get_all_input_elements(), get_all_select_elements()
2. LLM decision making: decide_form_element() to choose which element to use
3. Direct DOM operations: dom_type_by_selector(), dom_select_by_selector()
"""

from typing import List, Dict, Optional, Tuple
import logging
from ...prompting.prompts import build_dom_type_prompt, build_dom_select_prompt, build_dom_option_prompt


async def get_all_select_elements(page) -> List[Dict]:
    """
    Extract all select dropdown elements from the page using Playwright.
    
    Returns:
        List of select element info dicts:
        [
            {
                'index': 0,
                'id': 'country',
                'name': 'country',
                'label': 'Country',
                'description': 'Country dropdown with 195 options',
                'selector': 'select#country',
                'current_value': 'us',
                'current_text': 'United States',
                'options': [
                    {'value': 'us', 'text': 'United States', 'index': 0},
                    {'value': 'ca', 'text': 'Canada', 'index': 1},
                    ...
                ],
                'visible': True
            },
            ...
        ]
    """
    try:
        elements = await page.evaluate("""
            () => {
                const selects = [];
                const allSelects = document.querySelectorAll('select');
                
                allSelects.forEach((el, index) => {
                    // Check if visible
                    const isVisible = el.offsetParent !== null && 
                                     window.getComputedStyle(el).display !== 'none' &&
                                     window.getComputedStyle(el).visibility !== 'hidden';
                    
                    if (!isVisible) return;
                    
                    // Generate unique selector
                    let selector = '';
                    if (el.id) {
                        selector = `select[id="${el.id}"]`;
                    } else if (el.name) {
                        selector = `select[name="${el.name}"]`;
                    } else {
                        selector = `select:nth-of-type(${selects.length + 1})`;
                    }
                    
                    // Get all options
                    const options = Array.from(el.options).map((opt, idx) => ({
                        value: opt.value,
                        text: opt.text.trim(),
                        index: idx,
                        selected: opt.selected
                    }));
                    
                    // Build description
                    const label = el.labels?.[0]?.textContent?.trim() || 
                                 document.querySelector(`label[for="${el.id}"]`)?.textContent?.trim();
                    
                    let description = '';
                    if (label) {
                        description = `${label} dropdown with ${options.length} options`;
                    } else if (el.name) {
                        description = `${el.name} dropdown with ${options.length} options`;
                    } else {
                        description = `Dropdown with ${options.length} options`;
                    }
                    
                    const selectedOption = el.options[el.selectedIndex];
                    
                    selects.push({
                        index: selects.length,
                        id: el.id || '',
                        name: el.name || '',
                        label: label || '',
                        description: description,
                        selector: selector,
                        current_value: selectedOption ? selectedOption.value : '',
                        current_text: selectedOption ? selectedOption.text.trim() : '',
                        options: options,
                        visible: isVisible,
                        disabled: el.disabled
                    });
                });
                
                return selects;
            }
        """)
        
        return elements
        
    except Exception as e:
        logging.error(f"Failed to get select elements: {e}")
        return []


async def get_select_options(page, coordinates: Tuple[int, int]) -> Optional[Dict]:
    """
    Get all available options from a select dropdown at given coordinates.
    
    Args:
        page: Playwright page object
        coordinates: (x, y) tuple of the select element location
        
    Returns:
        Dict with select element info and available options, or None if not found
        {
            'selector': 'select#country',
            'current_value': 'us',
            'current_text': 'United States',
            'options': [
                {'value': 'us', 'text': 'United States', 'index': 0},
                {'value': 'ca', 'text': 'Canada', 'index': 1},
                ...
            ]
        }
    """
    try:
        x, y = coordinates
        
        # Find the select element at these coordinates using Playwright DOM API
        select_element = await page.evaluate(f"""
            () => {{
                const element = document.elementFromPoint({x}, {y});
                if (!element) return null;
                
                // Find closest select element (might be clicking on container)
                let selectEl = element;
                if (selectEl.tagName !== 'SELECT') {{
                    selectEl = element.closest('select');
                }}
                if (!selectEl) return null;
                
                // Get all options
                const options = Array.from(selectEl.options).map((opt, index) => {{
                    value: opt.value,
                    text: opt.text.trim(),
                    index: index,
                    selected: opt.selected
                }});
                
                // Get current selection
                const selectedOption = selectEl.options[selectEl.selectedIndex];
                
                // Generate a unique selector for this element
                let selector = selectEl.id ? `select[id="${selectEl.id}"]` : null;
                if (!selector && selectEl.name) {{
                    selector = `select[name="${{selectEl.name}}"]`;
                }}
                if (!selector) {{
                    // Fallback: use position in DOM
                    const selects = Array.from(document.querySelectorAll('select'));
                    const index = selects.indexOf(selectEl);
                    selector = `select:nth-of-type(${{index + 1}})`;
                }}
                
                return {{
                    selector: selector,
                    current_value: selectedOption ? selectedOption.value : '',
                    current_text: selectedOption ? selectedOption.text.trim() : '',
                    options: options,
                    tag_name: 'select'
                }};
            }}
        """)
        
        return select_element
        
    except Exception as e:
        logging.error(f"Failed to get select options: {e}")
        return None


async def dom_select_option(page, coordinates: Tuple[int, int], option_text: str) -> Tuple[bool, str]:
    """
    Select an option from a dropdown using DOM manipulation (no scrolling needed).
    
    Args:
        page: Playwright page object
        coordinates: (x, y) tuple of the select element
        option_text: Text of the option to select
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        # First, get available options
        select_info = await get_select_options(page, coordinates)
        
        if not select_info:
            return False, "Could not find select element at coordinates"
        
        selector = select_info['selector']
        options = select_info['options']
        
        # Find matching option (try multiple strategies)
        matched_option = None
        option_text_lower = option_text.lower().strip()
        
        # Strategy 1: Exact text match
        for opt in options:
            if opt['text'].lower() == option_text_lower:
                matched_option = opt
                break
        
        # Strategy 2: Partial text match
        if not matched_option:
            for opt in options:
                if option_text_lower in opt['text'].lower():
                    matched_option = opt
                    break
        
        # Strategy 3: Value match
        if not matched_option:
            for opt in options:
                if opt['value'].lower() == option_text_lower:
                    matched_option = opt
                    break
        
        if not matched_option:
            available_options = [opt['text'] for opt in options]
            return False, f"Option '{option_text}' not found. Available: {available_options}"
        
        # Select the option using DOM (bypasses scrolling)
        success = await page.evaluate("""
            ([selector, index]) => {
                const selectEl = document.querySelector(selector);
                if (!selectEl) return false;
                
                selectEl.selectedIndex = index;
                
                // Trigger change events for JavaScript listeners
                selectEl.dispatchEvent(new Event('change', { bubbles: true }));
                selectEl.dispatchEvent(new Event('input', { bubbles: true }));
                
                return true;
            }
        """, [selector, matched_option['index']])
        
        if success:
            return True, f"Selected '{matched_option['text']}' from dropdown"
        else:
            return False, "Failed to select option via DOM"
            
    except Exception as e:
        logging.error(f"DOM select failed: {e}")
        return False, f"Error: {str(e)}"


async def dom_select_by_selector(page, selector: str, option_text: str) -> Tuple[bool, str]:
    """
    Select an option from dropdown by selector (no coordinates needed).
    
    Args:
        page: Playwright page object
        selector: CSS selector for the select element
        option_text: Text of the option to select
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        # Get options and select
        result = await page.evaluate("""
            ([selector, optionText]) => {
                const selectEl = document.querySelector(selector);
                if (!selectEl) return { success: false, message: 'Element not found' };
                
                if (selectEl.disabled) {
                    return { success: false, message: 'Element is disabled' };
                }
                
                // Get all options
                const options = Array.from(selectEl.options);
                const optionTextLower = optionText.toLowerCase().trim();
                
                // Find matching option (try multiple strategies)
                let matchedOption = null;
                
                // Strategy 1: Exact text match
                matchedOption = options.find(opt => opt.text.toLowerCase() === optionTextLower);
                
                // Strategy 2: Partial text match
                if (!matchedOption) {
                    matchedOption = options.find(opt => opt.text.toLowerCase().includes(optionTextLower));
                }
                
                // Strategy 3: Value match
                if (!matchedOption) {
                    matchedOption = options.find(opt => opt.value.toLowerCase() === optionTextLower);
                }
                
                if (!matchedOption) {
                    const available = options.map(opt => opt.text).join(', ');
                    return { 
                        success: false, 
                        message: `Option '${optionText}' not found. Available: ${available}` 
                    };
                }
                
                // Select the option
                selectEl.value = matchedOption.value;
                selectEl.selectedIndex = options.indexOf(matchedOption);
                
                // Trigger change events
                selectEl.dispatchEvent(new Event('change', { bubbles: true }));
                selectEl.dispatchEvent(new Event('input', { bubbles: true }));
                
                return { 
                    success: true, 
                    message: `Selected '${matchedOption.text}' from dropdown` 
                };
            }
        """, [selector, option_text])
        
        return result['success'], result['message']
        
    except Exception as e:
        logging.error(f"DOM select by selector failed: {e}")
        return False, f"Error: {str(e)}"


# ============================================================================
# SMART MATCHING (Try exact match first, avoid LLM if possible)
# ============================================================================

# ============================================================================
# LLM-BASED DECISION MAKING
# ============================================================================

async def decide_form_element(engine, inputs: List[Dict], selects: List[Dict], 
                              action_type: str, value: str, task: str, logger=None) -> Optional[Dict]:
    """
    Use LLM to decide which form element to interact with.
    
    Args:
        engine: LLM inference engine
        inputs: List of available input elements
        selects: List of available select elements
        action_type: Either "TYPE" or "SELECT"
        value: The value to type or select
        task: The current task description
        logger: Optional logger instance
        
    Returns:
        Selected element dict or None if no appropriate element found
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Build prompt
    if action_type == "TYPE":
        elements_list = "\n".join([
            f"  [{i}] {inp['description']}"
            f" (type={inp['type']}, current_value='{inp['value']}')"
            for i, inp in enumerate(inputs)
        ])
        prompt = build_dom_type_prompt(value, task, elements_list)
    else:  # SELECT
        elements_list = "\n".join([
            f"  [{i}] {sel['description']}"
            f" (current='{sel['current_text']}', options={len(sel['options'])})"
            for i, sel in enumerate(selects)
        ])
        prompt = build_dom_select_prompt(value, task, elements_list)
    
    try:
        # Call LLM
        logger.info(f"🤔 Asking LLM to decide which element for {action_type} action...")
        response = await engine.generate(
            prompt=[prompt, "", ""],
            image_path=None,  # No screenshot needed for element selection
            turn_number=0
        )
        
        # Handle list response
        if isinstance(response, list):
            response = response[0] if response else ""
        
        logger.info(f"LLM decision: {response}")
        
        # Parse response
        response = response.strip()
        
        if response.upper() == "NONE" or not response:
            logger.warning(f"LLM said NONE - no appropriate element found")
            return None
        
        # Extract number
        try:
            index = int(response)
        except ValueError:
            digits = []
            for ch in response:
                if ch.isdigit():
                    digits.append(ch)
                elif digits:
                    break
            if digits:
                index = int("".join(digits))
            else:
                logger.error(f"Could not parse index from LLM response: {response}")
                return None
        
        # Get the element
        if action_type == "TYPE":
            if 0 <= index < len(inputs):
                selected = inputs[index]
                logger.info(f"✅ Selected input [{index}]: {selected['description']}")
                return selected
            else:
                logger.error(f"Index {index} out of range for inputs (0-{len(inputs)-1})")
                return None
        else:  # SELECT
            if 0 <= index < len(selects):
                selected = selects[index]
                logger.info(f"✅ Selected dropdown [{index}]: {selected['description']}")
                return selected
            else:
                logger.error(f"Index {index} out of range for selects (0-{len(selects)-1})")
                return None
        
    except Exception as e:
        logger.error(f"Error in LLM form element decision: {e}")
        return None


async def decide_select_option(engine, options: List[Dict], task: str, field_name: str, desired_value: str, logger=None) -> Optional[Dict]:
    if logger is None:
        logger = logging.getLogger(__name__)
    options_list = "\n".join([
        f"  [{i}] {opt.get('text', '')} (value='{opt.get('value', '')}')"
        for i, opt in enumerate(options)
    ])
    prompt = build_dom_option_prompt(task, field_name, desired_value, options_list)
    try:
        logger.info("🤔 Asking LLM to choose dropdown option...")
        response = await engine.generate(
            prompt=[prompt, "", ""],
            image_path=None,
            turn_number=0
        )
        if isinstance(response, list):
            response = response[0] if response else ""
        logger.info(f"LLM option decision: {response}")
        response = response.strip()
        if response.upper() == "NONE" or not response:
            logger.warning("LLM said NONE - no appropriate option found")
            return None
        try:
            index = int(response)
        except ValueError:
            digits = []
            for ch in response:
                if ch.isdigit():
                    digits.append(ch)
                elif digits:
                    break
            if digits:
                index = int("".join(digits))
            else:
                logger.error(f"Could not parse index from LLM response: {response}")
                return None
        if 0 <= index < len(options):
            selected = options[index]
            logger.info(f"✅ Selected option [{index}]: {selected.get('text')}")
            return selected
        logger.error(f"Index {index} out of range for options (0-{len(options)-1})")
        return None
    except Exception as e:
        logger.error(f"Error in LLM option decision: {e}")
        return None
