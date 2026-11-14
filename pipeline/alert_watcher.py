"""
AtlashAI Incident Alert Watcher
Sends real-time breakdown & fallback alerts to the developer's Telegram.

Alerts are dispatched from a short-lived daemon thread with a plain synchronous
HTTP call. The previous implementation used asyncio.get_event_loop(), which
raises RuntimeError in worker threads (where the pipeline and LLM calls run),
and the error was swallowed, so alerts effectively never sent.
"""

import html
import logging
import os
import threading
from datetime import datetime

import httpx

logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()


def _esc(value, limit: int = 300) -> str:
    """Escape user/LLM/provider text before inserting it into Telegram HTML."""
    return html.escape(str(value)[:limit], quote=False)


def _send_telegram_sync(message: str) -> bool:
    token = os.getenv("TELEGRAM_BOT_TOKEN", TELEGRAM_BOT_TOKEN).strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", TELEGRAM_CHAT_ID).strip()
    if not token or not chat_id:
        logger.debug("[AlertWatcher] Telegram credentials not configured. Skipping alert.")
        return False
    try:
        resp = httpx.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=5.0,
        )
        if resp.status_code == 200:
            logger.info("[AlertWatcher] Incident alert dispatched to Telegram successfully.")
            return True
        logger.error("[AlertWatcher] Telegram API Error: %s", resp.text[:200])
        return False
    except Exception as e:
        logger.error("[AlertWatcher] Failed to send Telegram alert: %s", e)
        return False


async def send_telegram_alert_async(message: str) -> bool:
    """Async-compatible wrapper (runs the sync call in a thread)."""
    import asyncio
    return await asyncio.to_thread(_send_telegram_sync, message)


def _dispatch(text: str) -> None:
    """Fire-and-forget; safe from any thread, never raises, never blocks the caller."""
    try:
        threading.Thread(target=_send_telegram_sync, args=(text,), daemon=True).start()
    except Exception as e:  # pragma: no cover
        logger.error("[AlertWatcher] could not start alert thread: %s", e)


def notify_pipeline_error(stage: str, topic: str, error_msg: str, user_id: str = "Unknown"):
    """Fires an instant alert when a critical pipeline stage breaks"""
    time_str = datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")
    text = (
        f"<b>[AtlashAI Alert] Pipeline Breakdown Detected!</b>\n\n"
        f"<b>Stage:</b> <code>{_esc(stage, 120)}</code>\n"
        f"<b>Topic:</b> {_esc(topic, 200)}\n"
        f"<b>User:</b> <code>{_esc(user_id, 80)}</code>\n"
        f"<b>Error Details:</b>\n<pre>{_esc(error_msg, 300)}</pre>\n\n"
        f"<b>Time:</b> {time_str}"
    )
    _dispatch(text)


def notify_model_fallback(failed_model: str, next_model: str, reason: str, agent_name: str = "Agent"):
    """Fires an alert when an LLM fails and switches to backup model"""
    time_str = datetime.now().strftime("%I:%M:%S %p")
    text = (
        f"<b>[AtlashAI Watcher] LLM Failover Triggered</b>\n\n"
        f"<b>Agent:</b> <code>{_esc(agent_name, 80)}</code>\n"
        f"<b>Failed Model:</b> <code>{_esc(failed_model, 80)}</code>\n"
        f"<b>Switching To:</b> <code>{_esc(next_model, 80)}</code>\n"
        f"<b>Reason:</b> <pre>{_esc(reason, 200)}</pre>\n\n"
        f"<b>Time:</b> {time_str}"
    )
    _dispatch(text)
