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

import time

from avenir_web.runtime.browser import normal_new_context_async


def _build_anti_close_init_script() -> str:
    return """
        (function(){
            try{
                var origClose = window.close;
                Object.defineProperty(window, 'close', { configurable: true, writable: true, value: function(){ try{ console.log('[Automation] window.close intercepted'); }catch(e){} } });
                window.addEventListener('beforeunload', function(){ try{ console.log('[Automation] beforeunload intercepted'); }catch(e){} });
            }catch(e){}
        })();
    """


class PageEventHandlers:
    def __init__(self, agent):
        self.agent = agent

    async def on_close(self):
        agent = self.agent
        try:
            if getattr(agent, "is_stopping", False):
                return
            if not (agent.session_control and isinstance(agent.session_control, dict)):
                return
            context = agent.session_control.get("context")
            browser = agent.session_control.get("browser")
            agent.logger.info("The active tab was closed. Will recover a working page.")
            if context:
                try:
                    pages = context.pages if hasattr(context, "pages") else []
                except Exception:
                    pages = []
                if pages:
                    try:
                        agent.page = pages[-1]
                        await agent.page.bring_to_front()
                        agent.logger.info(f"Switched the active tab to: {agent.page.url}")
                        return
                    except Exception:
                        pass
                try:
                    agent.page = await context.new_page()
                except Exception as e:
                    agent.logger.warning(f"Context.new_page failed: {e}")
                    if browser:
                        try:
                            geo_cfg = (
                                agent.config.get("browser", {}).get("geolocation")
                                or agent.config.get("playwright", {}).get("geolocation")
                            )
                            agent.session_control["context"] = await normal_new_context_async(
                                browser,
                                viewport=agent.config["browser"]["viewport"],
                                user_agent=agent.config["browser"]["user_agent"],
                                geolocation=geo_cfg,
                            )
                            agent.session_control["context"].on("page", self.on_open)
                            agent.page = await agent.session_control["context"].new_page()
                        except Exception as e2:
                            agent.logger.error(f"Failed to recreate context: {e2}")
                            return
                try:
                    target_url = (
                        getattr(agent, "actual_website", None)
                        or agent.config.get("basic", {}).get("default_website")
                        or "https://www.google.com/"
                    )
                    await agent.page.goto(target_url, wait_until="load")
                    agent.logger.info(f"Switched the active tab to: {agent.page.url}")
                except Exception as e:
                    agent.logger.info(f"Failed to navigate after refresh: {e}")
            else:
                if browser:
                    try:
                        geo_cfg = (
                            agent.config.get("browser", {}).get("geolocation")
                            or agent.config.get("playwright", {}).get("geolocation")
                        )
                        agent.session_control["context"] = await normal_new_context_async(
                            browser,
                            viewport=agent.config["browser"]["viewport"],
                            user_agent=agent.config["browser"]["user_agent"],
                            geolocation=geo_cfg,
                        )
                        agent.session_control["context"].on("page", self.on_open)
                        agent.page = await agent.session_control["context"].new_page()
                        target_url = (
                            getattr(agent, "actual_website", None)
                            or agent.config.get("basic", {}).get("default_website")
                            or "https://www.google.com/"
                        )
                        await agent.page.goto(target_url, wait_until="load")
                        agent.logger.info(f"Recovered by creating new context: {agent.page.url}")
                    except Exception as e:
                        agent.logger.error(f"Failed to recover without context: {e}")
                        return
        except Exception as e:
            agent.logger.warning(f"page_on_close_handler failed: {e}")

    async def on_navigation(self, frame):
        current_url = frame.url
        current_time = time.time()

    async def on_crash(self, page):
        agent = self.agent
        try:
            if getattr(agent, "is_stopping", False):
                return
            crashed_url = None
            try:
                crashed_url = page.url
            except Exception:
                crashed_url = None
            agent.logger.error(f"Page crashed: {crashed_url or 'unknown URL'}")
            try:
                await page.close()
            except Exception:
                pass
            context = agent.session_control.get("context") if isinstance(agent.session_control, dict) else None
            if not context:
                agent.logger.error("No browser context available for crash handling")
                return
            new_page = await context.new_page()
            await self.on_open(new_page)
            target_url = None
            if crashed_url and not str(crashed_url).startswith(("about:", "data:", "chrome-error://", "blob:")):
                target_url = crashed_url
            elif getattr(agent, "actual_website", None):
                target_url = agent.actual_website
            if target_url:
                try:
                    await new_page.goto(target_url, wait_until="domcontentloaded")
                    agent.logger.info(f"Recovered to {target_url}")
                except Exception as e:
                    agent.logger.warning(f"Failed to restore to {target_url}: {e}")
            else:
                agent.logger.info("Created a fresh page without navigation target")
            agent.page = new_page
        except Exception as e:
            agent.logger.error(f"Crash handling failed: {e}")

    async def on_response(self, response):
        agent = self.agent
        if agent.session_control and isinstance(agent.session_control, dict):
            agent.session_control["last_response"] = {
                "url": response.url,
                "status": response.status,
                "status_text": response.status_text,
                "headers": dict(response.headers),
            }

    async def on_open(self, page):
        agent = self.agent
        await page.add_init_script("Object.defineProperty(navigator, 'webdriver', { get: () => undefined });")
        await page.add_init_script(_build_anti_close_init_script())
        page.on("framenavigated", self.on_navigation)
        page.on("close", self.on_close)
        page.on("crash", self.on_crash)
        page.on("response", self.on_response)
        agent.page = page
