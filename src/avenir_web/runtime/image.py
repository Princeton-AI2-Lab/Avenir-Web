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

import os
import base64
import logging
from PIL import Image
import io

# Claude's maximum image size limit (5MB)
MAX_IMAGE_SIZE_BYTES = 5 * 1024 * 1024  # 5MB

def get_file_size(file_path):
    """Get file size in bytes."""
    return os.path.getsize(file_path)

def get_base64_size(base64_string):
    """Get the size of a base64 encoded string in bytes."""
    return len(base64_string.encode('utf-8'))

def compress_image_to_limit(image_path, max_size_bytes=MAX_IMAGE_SIZE_BYTES, quality_start=85, min_quality=20):
    """
    Compress an image to stay under the specified size limit.
    
    Args:
        image_path (str): Path to the input image
        max_size_bytes (int): Maximum allowed size in bytes (default: 5MB)
        quality_start (int): Starting JPEG quality (default: 85)
        min_quality (int): Minimum JPEG quality to try (default: 20)
    
    Returns:
        str: Base64 encoded compressed image
        
    Raises:
        ValueError: If image cannot be compressed to fit within size limit
    """
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # First check if the original file is already under the limit
    original_size = get_file_size(image_path)
    logger.info(f"Original image size: {original_size / (1024*1024):.2f} MB")
    
    # Open and convert image
    with Image.open(image_path) as img:
        # Convert to RGB if necessary (for JPEG compatibility)
        if img.mode in ('RGBA', 'LA', 'P'):
            # Create a white background for transparent images
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Try different compression levels
        quality = quality_start
        while quality >= min_quality:
            # Compress image to bytes
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='JPEG', quality=quality, optimize=True)
            img_bytes = img_buffer.getvalue()
            
            # Encode to base64 and check size
            base64_string = base64.b64encode(img_bytes).decode('utf-8')
            base64_size = get_base64_size(base64_string)
            
            logger.info(f"Quality {quality}: {base64_size / (1024*1024):.2f} MB")
            
            if base64_size <= max_size_bytes:
                logger.info(f"Successfully compressed image to {base64_size / (1024*1024):.2f} MB at quality {quality}")
                return base64_string
            
            # Reduce quality for next iteration
            quality -= 10
        
        # If still too large, try reducing image dimensions
        logger.warning("Quality reduction insufficient, trying dimension reduction")
        
        # Try reducing dimensions by 10% each iteration
        scale_factor = 0.9
        current_img = img.copy()
        
        while scale_factor >= 0.3:  # Don't go below 30% of original size
            new_width = int(current_img.width * scale_factor)
            new_height = int(current_img.height * scale_factor)
            resized_img = current_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Try with moderate quality
            img_buffer = io.BytesIO()
            resized_img.save(img_buffer, format='JPEG', quality=70, optimize=True)
            img_bytes = img_buffer.getvalue()
            
            base64_string = base64.b64encode(img_bytes).decode('utf-8')
            base64_size = get_base64_size(base64_string)
            
            logger.info(f"Resized to {new_width}x{new_height} (scale {scale_factor:.1f}): {base64_size / (1024*1024):.2f} MB")
            
            if base64_size <= max_size_bytes:
                logger.info(f"Successfully compressed image to {base64_size / (1024*1024):.2f} MB with {scale_factor:.1f}x scale")
                return base64_string
            
            scale_factor -= 0.1
        
        # If we still can't fit it, raise an error
        raise ValueError(f"Unable to compress image to fit within {max_size_bytes / (1024*1024):.1f} MB limit")

def encode_image_with_compression(image_path, max_size_bytes=MAX_IMAGE_SIZE_BYTES):
    """
    Encode an image to base64 with automatic compression if needed.
    
    Args:
        image_path (str): Path to the image file
        max_size_bytes (int): Maximum allowed size in bytes (default: 5MB)
    
    Returns:
        str: Base64 encoded image (compressed if necessary)
    """
    logger = logging.getLogger(__name__)
    
    if image_path is None:
        raise ValueError("Image path cannot be None")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # First try the original file
    try:
        with open(image_path, "rb") as image_file:
            original_bytes = image_file.read()
            base64_string = base64.b64encode(original_bytes).decode('utf-8')
            base64_size = get_base64_size(base64_string)
            
            if base64_size <= max_size_bytes:
                logger.info(f"Original image fits within limit: {base64_size / (1024*1024):.2f} MB")
                return base64_string
            else:
                logger.info(f"Original image too large: {base64_size / (1024*1024):.2f} MB, compressing...")
                return compress_image_to_limit(image_path, max_size_bytes)
    
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
        # Fallback to compression
        return compress_image_to_limit(image_path, max_size_bytes)
