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

from playwright.async_api import Playwright
from pathlib import Path
import toml
import os

async def normal_launch_async(playwright: Playwright, headless=False, args=None, channel=None):
    default_args = [
        "--disable-blink-features=AutomationControlled",
        "--disable-gpu",
        "--disable-dev-shm-usage",
        "--no-sandbox",
        "--disable-features=BackForwardCache,Translate",
    ]
    if args is None:
        args = default_args
    else:
        merged_extra = [a for a in default_args if a not in args]
        args = args + merged_extra
    
    browser = await playwright.chromium.launch(
        traces_dir=None,
        headless=headless,
        args=args,
        channel=channel,
    )
    return browser



async def normal_new_context_async(
        browser,
        storage_state=None,
        har_path=None,
        video_path=None,
        tracing=False,
        trace_screenshots=False,
        trace_snapshots=False,
        trace_sources=False,
        locale=None,
        geolocation=None,
        user_agent: str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36 Edg/142.0.0.0",
        viewport: dict = {"width": 1280, "height": 720},
):
    context = await browser.new_context(
        storage_state=storage_state,
        user_agent=user_agent,
        viewport=viewport,
        device_scale_factor=1,
        locale=locale,
        record_har_path=har_path,
        record_video_dir=video_path,
        geolocation=geolocation,
        extra_http_headers={
            "Referer": "https://www.google.com/",
            "Accept-Language": "en-us"
        }
    )

    if tracing:
        await context.tracing.start(screenshots=trace_screenshots, snapshots=trace_snapshots, sources=trace_sources)
    return context

#
# def persistent_launch(playwright: Playwright, user_data_dir: str = ""):
#     context = playwright.chromium.launch_persistent_context(
#         user_data_dir=user_data_dir,
#         headless=False,
#         args=["--no-default-browser-check",
#               "--no_sandbox",
#               "--disable-blink-features=AutomationControlled",
#               ],
#         ignore_default_args=ignore_args,
#         user_agent="Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
#         viewport={"width": 1280, "height": 720},
#         bypass_csp=True,
#         slow_mo=1000,
#         chromium_sandbox=True,
#         channel="chrome-dev"
#     )
#     return context

#
# async def persistent_launch_async(playwright: Playwright, user_data_dir: str = "", record_video_dir="video"):
#     context = await playwright.chromium.launch_persistent_context(
#         user_data_dir=user_data_dir,
#         headless=False,
#         args=[
#             "--disable-blink-features=AutomationControlled",
#         ],
#         ignore_default_args=ignore_args,
#         user_agent="Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
#         # viewport={"width": 1280, "height": 720},
#         record_video_dir=record_video_dir,
#         channel="chrome-dev"
#         # slow_mo=1000,
#     )
#     return context



def saveconfig(config, save_file):
    """
    config is a dictionary.
    save_path: saving path include file name.
    """


    if isinstance(save_file, str):
        save_file = Path(save_file)
    if isinstance(config, dict):
        with open(save_file, 'w') as f:
            import copy
            config_without_key = copy.deepcopy(config)
            if "api_keys" in config_without_key and "openrouter_api_key" in config_without_key["api_keys"]:
                config_without_key["api_keys"]["openrouter_api_key"] = "Your API key here"
            toml.dump(config_without_key, f)
    else:
        os.system(" ".join(["cp", str(config), str(save_file)]))
