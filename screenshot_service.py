#!/usr/bin/env python3
"""
WebSentinel AI - Real Website Screenshot & DOM Inspection Service
Uses headless Playwright to capture real 1280x800 browser renders and DOM metrics for any URL.
"""

import os
import time
import base64
import asyncio
from typing import Dict, Any, Optional
from playwright.async_api import async_playwright
from PIL import Image, ImageDraw

SCREENSHOT_DIR = os.path.join(os.path.dirname(__file__), "data", "screenshots")
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

class ScreenshotService:
    @staticmethod
    async def capture_website(url: str, domain_id: str) -> Dict[str, Any]:
        """
        Navigates to any real website, captures high-res screenshot, and extracts DOM metadata.
        """
        t0 = time.perf_counter()
        clean_url = url if url.startswith("http") else f"https://{url}"
        
        filepath = os.path.join(SCREENSHOT_DIR, f"{domain_id}.png")
        
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    viewport={"width": 1280, "height": 800},
                    user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 WebSentinel/1.0"
                )
                page = await context.new_page()
                
                # Navigate with domcontentloaded and wait for layout
                try:
                    response = await page.goto(clean_url, wait_until="domcontentloaded", timeout=10000)
                    await page.wait_for_timeout(1500)
                except Exception:
                    response = await page.goto(clean_url, wait_until="load", timeout=8000)
                    await page.wait_for_timeout(1000)
                
                http_status = response.status if response else 200
                title = await page.title()
                
                # Extract DOM stats
                meta_desc = await page.evaluate("""
                    () => {
                        const meta = document.querySelector('meta[name="description"]');
                        return meta ? meta.getAttribute('content') : '';
                    }
                """)
                
                # Screenshot
                screenshot_bytes = await page.screenshot(type="png", full_page=False)
                await browser.close()
                
                # Save to disk
                with open(filepath, "wb") as f:
                    f.write(screenshot_bytes)
                    
                b64_str = "data:image/png;base64," + base64.b64encode(screenshot_bytes).decode("utf-8")
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                return {
                    "success": True,
                    "http_status": http_status,
                    "title": title or clean_url,
                    "meta_description": meta_desc or "",
                    "screenshot_path": filepath,
                    "screenshot_b64": b64_str,
                    "screenshot_bytes": screenshot_bytes,
                    "render_latency_ms": round(elapsed_ms, 1)
                }

        except Exception as e:
            # Generate fallback diagnostic canvas for unreachable or failed sites
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            fallback_img = ScreenshotService._generate_fallback_canvas(clean_url, str(e))
            fallback_img.save(filepath, format="PNG")
            
            import io
            buf = io.BytesIO()
            fallback_img.save(buf, format="PNG")
            b64_str = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

            return {
                "success": False,
                "http_status": 500,
                "title": f"Unreachable: {clean_url}",
                "meta_description": f"Connection error: {str(e)}",
                "screenshot_path": filepath,
                "screenshot_b64": b64_str,
                "screenshot_bytes": buf.getvalue(),
                "render_latency_ms": round(elapsed_ms, 1),
                "error": str(e)
            }

    @staticmethod
    def _generate_fallback_canvas(url: str, err: str) -> Image.Image:
        img = Image.new("RGB", (640, 400), color=(15, 23, 42))
        draw = ImageDraw.Draw(img)
        # Header bar
        draw.rectangle([(0, 0), (640, 40)], fill=(30, 41, 59))
        draw.text((20, 12), f"WebSentinel Inspector: {url}", fill=(203, 213, 225))
        # Error block
        draw.rectangle([(40, 80), (600, 340)], fill=(2, 6, 23), outline=(239, 68, 68), width=2)
        draw.text((60, 110), "SITE OFFLINE / CONNECTION REFUSED", fill=(239, 68, 68))
        draw.text((60, 150), f"Target URL: {url}", fill=(248, 250, 252))
        draw.text((60, 185), f"Diagnostic: {err[:80]}", fill=(148, 163, 184))
        draw.text((60, 230), "Intel® OpenVINO™ Visual Uptime Engine", fill=(0, 199, 253))
        return img

if __name__ == "__main__":
    print("Testing real screenshot capture on 'https://www.intel.com'...")
    res = asyncio.run(ScreenshotService.capture_website("https://www.intel.com", "test_intel"))
    print(f"Captured: {res['title']} | Status: {res['http_status']} | Time: {res['render_latency_ms']}ms | File: {res['screenshot_path']}")
