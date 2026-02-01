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
from typing import Any, Dict, Optional, Sequence, Tuple


def ensure_screenshots_dir(main_path: str) -> str:
    screenshots_dir = os.path.join(main_path, "screenshots")
    os.makedirs(screenshots_dir, exist_ok=True)
    return screenshots_dir


def build_screenshot_path(main_path: str, time_step: int, labeled: bool = False) -> str:
    suffix = "_labeled" if labeled else ""
    return os.path.join(main_path, "screenshots", f"screen_{time_step}{suffix}.png")


def is_uniform_image(image_path: str) -> bool:
    try:
        from PIL import Image
    except Exception:
        return False

    try:
        with Image.open(image_path) as img:
            extrema = img.getextrema()
            uniform = True
            for band in extrema:
                if isinstance(band, tuple) and band[0] != band[1]:
                    uniform = False
                    break
            if uniform:
                try:
                    lum = img.convert("L")
                    hist = lum.histogram()
                    total = sum(hist) or 1
                    white_ratio = hist[-1] / float(total)
                    if white_ratio < 0.99:
                        uniform = False
                except Exception:
                    pass
            return uniform
    except Exception:
        return False


async def capture_viewport_screenshot(
    page: Any,
    screenshot_path: str,
    *,
    logger: Any = None,
    timeout_ms: int = 45000,
) -> None:
    try:
        try:
            await page.wait_for_load_state("domcontentloaded", timeout=15000)
        except Exception:
            pass
        try:
            await page.wait_for_load_state("networkidle", timeout=20000)
        except Exception:
            pass
        await page.screenshot(path=screenshot_path, timeout=timeout_ms)
    except Exception as e:
        if logger is not None:
            try:
                logger.warning(f"Viewport screenshot failed: {e}")
            except Exception:
                pass
        raise


def _compute_viewport_scale(img_size: Tuple[int, int], viewport_size: Optional[Dict[str, int]], config: Dict) -> Tuple[float, float]:
    if isinstance(viewport_size, dict) and viewport_size.get("width") and viewport_size.get("height"):
        return img_size[0] / float(viewport_size["width"]), img_size[1] / float(viewport_size["height"])
    vx = config.get("browser", {}).get("viewport", {}).get("width", img_size[0])
    vy = config.get("browser", {}).get("viewport", {}).get("height", img_size[1])
    sx = img_size[0] / float(vx) if vx else 1.0
    sy = img_size[1] / float(vy) if vy else 1.0
    return sx, sy


async def annotate_screenshot(
    image_path: str,
    *,
    page: Any,
    config: Dict,
    last_click_viewport_coords: Optional[Sequence[float]] = None,
    output_path: Optional[str] = None,
    logger: Any = None,
) -> Optional[str]:
    if not image_path or not os.path.exists(image_path):
        return None

    if is_uniform_image(image_path):
        return None

    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return None

    try:
        active = await page.evaluate(
            """
            () => {
                const ae = document.activeElement;
                if (ae && ae !== document.body) {
                    const r = ae.getBoundingClientRect();
                    return {
                        tag: ae.tagName,
                        placeholder: ae.placeholder || '',
                        id: ae.id || '',
                        rect: { left: r.left, top: r.top, width: r.width, height: r.height }
                    };
                }
                return null;
            }
            """
        )
    except Exception:
        active = None

    img = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    draw = ImageDraw.Draw(img)

    sx, sy = _compute_viewport_scale(img.size, getattr(page, "viewport_size", None), config)

    if active and isinstance(active, dict):
        rect = active.get("rect") or {}
        w = rect.get("width") or 0
        h = rect.get("height") or 0
        if w > 0 and h > 0:
            left = rect.get("left") or 0
            top = rect.get("top") or 0
            x0 = int(max(0, left * sx))
            y0 = int(max(0, top * sy))
            x1 = int(min(img.width - 1, (left + w) * sx))
            y1 = int(min(img.height - 1, (top + h) * sy))
            x1 = max(x0, x1)
            y1 = max(y0, y1)
            draw.rectangle([(x0, y0), (x1, y1)], outline="lime", width=3)
            label_text = (active.get("tag") or "") + " " + (active.get("placeholder") or active.get("id") or "")
            if label_text.strip():
                try:
                    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
                except Exception:
                    font = ImageFont.load_default()
                tb = draw.textbbox((0, 0), label_text, font=font)
                tw = tb[2] - tb[0]
                th = tb[3] - tb[1]
                tx = max(0, min(x0, img.width - max(6, tw) - 6))
                ty = max(0, min(max(0, y0 - th - 6), img.height - max(6, th) - 6))
                draw.rectangle([(tx - 3, ty - 3), (tx + tw + 3, ty + th + 3)], fill="black")
                draw.text((tx, ty), label_text, fill="lime", font=font)

    if last_click_viewport_coords:
        try:
            cx, cy = last_click_viewport_coords[0], last_click_viewport_coords[1]
        except Exception:
            cx, cy = None, None
        if cx is not None and cy is not None:
            cx = int(round(cx * sx))
            cy = int(round(cy * sy))
            ms = 16
            lw = 3
            ex0 = max(0, cx - 30)
            ey0 = max(0, cy - 30)
            ex1 = min(img.width - 1, cx + 30)
            ey1 = min(img.height - 1, cy + 30)
            ex1 = max(ex0 + 1, ex1)
            ey1 = max(ey0 + 1, ey1)
            odraw.ellipse([(ex0, ey0), (ex1, ey1)], fill=(0, 255, 0, 96))
            img = Image.alpha_composite(img, overlay)
            draw = ImageDraw.Draw(img)
            draw.line([(cx, cy - ms), (cx, cy + ms)], fill="red", width=lw)
            draw.line([(cx - ms, cy), (cx + ms, cy)], fill="red", width=lw)
            dx0 = max(0, cx - 8)
            dy0 = max(0, cy - 8)
            dx1 = min(img.width - 1, cx + 8)
            dy1 = min(img.height - 1, cy + 8)
            dx1 = max(dx0 + 1, dx1)
            dy1 = max(dy0 + 1, dy1)
            draw.ellipse([(dx0, dy0), (dx1, dy1)], outline="red", width=lw)

    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_labeled{ext or '.png'}"

    img.save(output_path)
    if logger is not None:
        try:
            logger.info(f"Labeled screenshot saved: {output_path}")
        except Exception:
            pass
    return output_path


async def take_full_page_screenshot_with_cropping(
    page: Any,
    *,
    screenshot_path: str,
    target_elements: Optional[Sequence[Any]] = None,
    logger: Any = None,
    timeout_ms: int = 20000,
) -> None:
    try:
        total_height = await page.evaluate(
            """() => {
                return Math.max(
                    document.documentElement.scrollHeight,
                    document.body.scrollHeight,
                    document.documentElement.clientHeight
                );
            }"""
        )

        viewport_size = getattr(page, "viewport_size", None)
        if viewport_size is None:
            viewport_size = {"width": 1280, "height": 720}
        total_width = viewport_size["width"]

        if target_elements:
            try:
                element_positions = []
                for element_info in target_elements:
                    if isinstance(element_info, dict) and "y" in element_info:
                        element_positions.append(element_info["y"])
                    elif hasattr(element_info, "y"):
                        element_positions.append(element_info.y)

                if element_positions:
                    height_start = min(element_positions)
                    height_end = max(element_positions)
                    clip_start = min(total_height - 1144, max(0, height_start - 200))
                    clip_height = min(total_height - clip_start, max(height_end - height_start + 200, 1144))
                    clip = {"x": 0, "y": clip_start, "width": total_width, "height": clip_height}

                    await page.screenshot(
                        path=screenshot_path,
                        clip=clip,
                        full_page=True,
                        type="png",
                        timeout=timeout_ms,
                    )
                    if logger is not None:
                        try:
                            logger.info(f"Smart cropped screenshot taken with clip: {clip}")
                        except Exception:
                            pass
                    return
            except Exception as crop_error:
                if logger is not None:
                    try:
                        logger.warning(f"Smart cropping failed: {crop_error}, falling back to full page")
                    except Exception:
                        pass

        await page.screenshot(path=screenshot_path, full_page=True, type="png", timeout=timeout_ms)
        if logger is not None:
            try:
                logger.info("Full page screenshot taken as fallback")
            except Exception:
                pass
    except Exception as e:
        await page.screenshot(path=screenshot_path)
        if logger is not None:
            try:
                logger.warning(f"Full page screenshot failed, took viewport screenshot: {e}")
            except Exception:
                pass
