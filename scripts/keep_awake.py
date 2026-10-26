"""Open the Streamlit app in a headless browser so Community Cloud does not put it to sleep.

If the app is already asleep, click the wake-up button and wait for it to come back.

    python scripts/keep_awake.py https://your-app.streamlit.app/
"""

import sys

from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright

WAKE_BUTTON = "button:has-text('Yes, get this app back up')"


def visit(url: str, timeout_ms: int = 60_000) -> str:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            try:
                page.locator(WAKE_BUTTON).first.click(timeout=15_000)
            except PlaywrightTimeout:
                # No wake-up button: the app is awake, and this visit counts as traffic.
                page.wait_for_timeout(5_000)
                return "awake"
            page.locator(WAKE_BUTTON).first.wait_for(state="hidden", timeout=timeout_ms)
            return "woken up"
        finally:
            browser.close()


if __name__ == "__main__":
    print(f"{sys.argv[1]}: {visit(sys.argv[1])}")
