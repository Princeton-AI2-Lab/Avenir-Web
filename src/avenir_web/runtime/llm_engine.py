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
import time
import asyncio
import logging

import backoff
from dotenv import load_dotenv
import litellm

OPENROUTER_API_KEY_PLACEHOLDERS = {
    "Your API KEY Here",
    "Your OpenRouter API Key Here",
    "Your API key here",
}

LLM_IO_RECORDS = []

def get_openrouter_base_url() -> str:
    base = (
        os.getenv("OPENROUTER_BASE_URL")
        or os.getenv("OPENROUTER_API_BASE")
        or "https://openrouter.ai/api/v1"
    )
    return str(base).rstrip("/")

def is_blank_or_placeholder_api_key(value):
    if value is None:
        return True
    s = str(value).strip()
    if not s:
        return True
    lowered = s.lower()
    for placeholder in OPENROUTER_API_KEY_PLACEHOLDERS:
        if lowered == placeholder.lower():
            return True
    return False

def describe_api_key_state(value):
    if value is None:
        return "missing"
    s = str(value)
    trimmed = s.strip()
    if not trimmed:
        return "blank"
    if is_blank_or_placeholder_api_key(trimmed):
        return "placeholder"
    return f"present(len={len(trimmed)})"

def add_llm_io_record(record):
    try:
        import datetime
        r = dict(record)
        r.setdefault("timestamp", datetime.datetime.now().isoformat())
        LLM_IO_RECORDS.append(r)
    except Exception:
        pass


def load_openrouter_api_key():
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if is_blank_or_placeholder_api_key(api_key):
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set to a valid value "
            f"(state={describe_api_key_state(api_key)}). "
            "Set OPENROUTER_API_KEY in the process environment (or .env)."
        )
    return str(api_key).strip()


def encode_image(image_path):
    """
    Encode image to base64 with automatic compression if needed.
    This function now uses the image module to handle large images.
    """
    from .image import encode_image_with_compression
    return encode_image_with_compression(image_path)


def engine_factory(api_key=None, model=None, **kwargs):
    """
    Create a generic OpenRouter LLM engine.
    Works with any model accessible via OpenRouter.
    """
    if model is None:
        model = "openrouter/qwen/qwen-2.5-72b-instruct"
    
    # Ensure API key is set
    if not is_blank_or_placeholder_api_key(api_key):
        os.environ["OPENROUTER_API_KEY"] = str(api_key).strip()
    else:
        load_openrouter_api_key()
    
    # Ensure openrouter/ prefix is present for LiteLLM
    if not model.startswith("openrouter/"):
        model = f"openrouter/{model}"
    
    # Pass model name to RouterEngine (format: openrouter/<provider>/<model>)
    return RouterEngine(model=model, **kwargs)
    

class Engine:
    def __init__(
            self,
            stop=["\n\n"],
            rate_limit=-1,
            model=None,
            temperature=0,
            **kwargs,
    ) -> None:
        """
            Base class to init an engine

        Args:
            stop (list, optional): Tokens indicate stop of sequence. Defaults to ["\n"].
            rate_limit (int, optional): Max number of requests per minute. Defaults to -1.
            model (_type_, optional): Model family. Defaults to None.
            temperature (float, optional): Sampling temperature. Defaults to 0.
        """
        self.time_slots = [0]
        self.stop = stop
        self.temperature = temperature
        self.model = model
        # convert rate limit to minimum request interval
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.next_avil_time = [0] * len(self.time_slots)
        self.current_key_idx = 0
        self.request_timeout_s = kwargs.get("request_timeout_s", 120)
        logging.getLogger(__name__).debug(f"Initializing model {self.model}")        

    def tokenize(self, input):
        return self.tokenizer(input)


class RouterEngine(Engine):
    def __init__(self, **kwargs) -> None:
        """
        Init a generic engine via OpenRouter.
        Requires OPENROUTER_API_KEY set in the environment.
        """
        super().__init__(**kwargs)
        try:
            self.task_id = kwargs.get("task_id")
        except Exception:
            self.task_id = None

    @backoff.on_exception(
        backoff.expo,
        (Exception,),  # Catch all exceptions for retry
        max_tries=3,
        max_time=30,
    )
    async def generate(self, prompt: list = None, max_new_tokens=4096, temperature=None, model=None, image_path=None,
                 ouput_0=None, turn_number=0, image_paths=None, **kwargs):
        logger = logging.getLogger(__name__)
        self.current_key_idx = (self.current_key_idx + 1) % len(self.time_slots)
        start_time = time.time()
        if (
                self.request_interval > 0
                and start_time < self.next_avil_time[self.current_key_idx]
        ):
            await asyncio.sleep(self.next_avil_time[self.current_key_idx] - start_time)
        prompt0, prompt1, prompt2 = prompt

        try:
            if image_paths is not None and isinstance(image_paths, (list, tuple)) and len(image_paths) > 0:
                image_contents = []
                for p in image_paths:
                    try:
                        b64 = encode_image(p)
                        image_contents.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}})
                    except Exception:
                        continue
                if turn_number == 0:
                    prompt_input = [
                        {"role": "system", "content": [{"type": "text", "text": prompt0}]},
                        {"role": "user", "content": [{"type": "text", "text": prompt1}] + image_contents},
                    ]
                elif turn_number == 1:
                    prompt_input = [
                        {"role": "system", "content": [{"type": "text", "text": prompt0}]},
                        {"role": "user", "content": [{"type": "text", "text": prompt1}] + image_contents},
                        {"role": "assistant", "content": [{"type": "text", "text": f"\n\n{ouput_0}"}]},
                        {"role": "user", "content": [{"type": "text", "text": prompt2}]}, 
                    ]
            elif image_path is not None:
                base64_image = encode_image(image_path)
                if turn_number == 0:
                    prompt_input = [
                        {"role": "system", "content": [{"type": "text", "text": prompt0}]},
                        {"role": "user", "content": [{"type": "text", "text": prompt1}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"}}]},
                    ]
                elif turn_number == 1:
                    prompt_input = [
                        {"role": "system", "content": [{"type": "text", "text": prompt0}]},
                        {"role": "user", "content": [{"type": "text", "text": prompt1}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"}}]},
                        {"role": "assistant", "content": [{"type": "text", "text": f"\n\n{ouput_0}"}]},
                        {"role": "user", "content": [{"type": "text", "text": prompt2}]}, 
                    ]
            else:
                # Handle case when no image is provided
                if turn_number == 0:
                    prompt_input = [
                        {"role": "system", "content": [{"type": "text", "text": prompt0}]},
                        {"role": "user", "content": [{"type": "text", "text": prompt1}]},
                    ]
                elif turn_number == 1:
                    prompt_input = [
                        {"role": "system", "content": [{"type": "text", "text": prompt0}]},
                        {"role": "user", "content": [{"type": "text", "text": prompt1}]},
                        {"role": "assistant", "content": [{"type": "text", "text": f"\n\n{ouput_0}"}]},
                        {"role": "user", "content": [{"type": "text", "text": prompt2}]}, 
                    ]
            
            current_model = model if model else self.model
            
            # Use OpenRouter; LiteLLM recognizes it from the "openrouter/" prefix in the model name
            try:
                base_url = None
                if isinstance(current_model, str) and current_model.startswith("openrouter/"):
                    base_url = get_openrouter_base_url()
                response = await asyncio.wait_for(
                    litellm.acompletion(
                        model=current_model,
                        messages=prompt_input,
                        max_tokens=max_new_tokens if max_new_tokens else 4096,
                        temperature=temperature if temperature else self.temperature,
                        api_key=os.getenv("OPENROUTER_API_KEY"),
                        base_url=base_url,
                        **kwargs,
                    ),
                    timeout=self.request_timeout_s,
                )
            except asyncio.TimeoutError:
                elapsed = time.time() - start_time
                logger.warning(f"Timeout waiting for LLM response after {elapsed:.2f}s (model={current_model}, turn={turn_number})")
                raise
            finally:
                elapsed = time.time() - start_time
                if elapsed > 60:
                    logger.warning(f"Slow LLM response: {elapsed:.2f}s (model={current_model}, turn={turn_number})")
                else:
                    logger.debug(f"LLM response time: {elapsed:.2f}s (model={current_model}, turn={turn_number})")
            output_text = [choice["message"]["content"] for choice in response.choices][0]
            
            # Check for empty response and raise exception to trigger backoff retry
            if not output_text or not output_text.strip():
                choice = response.choices[0]
                raise ValueError(f"Received empty response from LLM (finish_reason={choice.get('finish_reason')})")
                
            try:
                add_llm_io_record({
                    "model": current_model,
                    "turn_number": turn_number,
                    "messages": prompt_input,
                    "image_path": image_path,
                    "image_paths": image_paths,
                    "output": output_text,
                    "task_id": getattr(self, "task_id", None)
                })
            except Exception:
                pass
            return output_text
            
        except Exception as e:
            logger.error(f"Error in OpenRouter API call: {str(e)}")
            raise
