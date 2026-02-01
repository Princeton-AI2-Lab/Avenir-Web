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

import json
import logging

def get_viewport(page, config):
    viewport_width = None
    viewport_height = None
    
    if page:
        vs = getattr(page, 'viewport_size', None)
        if isinstance(vs, dict) and 'width' in vs and 'height' in vs:
            viewport_width, viewport_height = vs['width'], vs['height']
            
    if viewport_width is None or viewport_height is None:
        viewport_width = config.get('browser', {}).get('viewport', {}).get('width', 1500)
        viewport_height = config.get('browser', {}).get('viewport', {}).get('height', 1200)
        
    return viewport_width, viewport_height

def map_normalized_to_pixels(x, y, page, config):
    viewport_width, viewport_height = get_viewport(page, config)
    try:
        scaled_x = (float(x) / 1000.0) * viewport_width
        scaled_y = (float(y) / 1000.0) * viewport_height
        return (int(max(0, min(viewport_width - 1, scaled_x))), int(max(0, min(viewport_height - 1, scaled_y))))
    except Exception:
        return (viewport_width // 2, viewport_height // 2)

def normalize_coords(x, y, page, config, logger=None):
    viewport_width, viewport_height = get_viewport(page, config)
    
    try:
        # Ensure x and y are numeric
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            if logger:
                logger.error(f"Invalid coordinate types: x={type(x)}, y={type(y)}")
            return (viewport_width // 2, viewport_height // 2)
        
        scaled_x = float(x)
        scaled_y = float(y)
        
        # Clamp to viewport bounds and convert to int
        clamped_x = int(max(0, min(viewport_width - 1, scaled_x)))
        clamped_y = int(max(0, min(viewport_height - 1, scaled_y)))
        
        if logger:
            logger.debug(f"Clamped coords: input({x},{y}) -> Pixels({clamped_x},{clamped_y})[{viewport_width}x{viewport_height}]")
        
        return (clamped_x, clamped_y)
        
    except Exception as e:
        if logger:
            logger.error(f"Coordinate normalization failed: {e}", exc_info=True)
        return (viewport_width // 2, viewport_height // 2)

def parse_coordinates(text, page, config, logger=None):
    """Parse coordinates from GUI grounding response."""
    if not isinstance(text, str):
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

    def parse_int_pair_from_parens(s: str):
        l = s.find("(")
        while l != -1:
            r = s.find(")", l + 1)
            if r == -1:
                break
            inner = s[l + 1 : r]
            parts = inner.split(",")
            if len(parts) >= 2:
                try:
                    x = int(parts[0].strip())
                    y = int(parts[1].strip())
                    return x, y
                except Exception:
                    pass
            l = s.find("(", r + 1)
        return None

    # 1) Try to parse JSON
    try:
        # Try to parse the entire text as JSON first
        stripped = text.strip()
        if stripped.startswith('{') and stripped.endswith('}'):
            data = json.loads(stripped)
            if 'arguments' in data and 'coordinate' in data['arguments']:
                coords = data['arguments']['coordinate']
                if isinstance(coords, list) and len(coords) == 2:
                    x, y = int(coords[0]), int(coords[1])
                    if logger:
                        logger.debug(f"Parsed JSON coordinates: ({x}, {y})")
                    return map_normalized_to_pixels(x, y, page, config)
        else:
            obj = extract_first_json_object(text)
            if obj:
                data = json.loads(obj)
                if 'arguments' in data and 'coordinate' in data['arguments']:
                    coords = data['arguments']['coordinate']
                    if isinstance(coords, list) and len(coords) == 2:
                        x, y = int(coords[0]), int(coords[1])
                        if logger:
                            logger.debug(f"Parsed JSON coordinates: ({x}, {y})")
                        return map_normalized_to_pixels(x, y, page, config)
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        if logger:
            logger.debug(f"Failed to parse JSON format: {e}")
    
    # 2) Other formats
    box_start = text.find("<|box_start|>")
    if box_start != -1:
        box_end = text.find("<|box_end|>", box_start)
        segment = text[box_start:box_end] if box_end != -1 else text[box_start:]
        pair = parse_int_pair_from_parens(segment)
        if pair:
            x, y = pair
            return map_normalized_to_pixels(x, y, page, config)

    pair = parse_int_pair_from_parens(text)
    if pair:
        x, y = pair
        return map_normalized_to_pixels(x, y, page, config)
    return None
