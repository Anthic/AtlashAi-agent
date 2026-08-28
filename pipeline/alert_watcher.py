"""
AtlashAI Incident Alert Watcher
Asynchronously sends real-time breakdown & fallback alerts to Developer's Telegram
"""

import os
import httpx
import asyncio
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

async def send_telegram_alert_async(message: str) -> bool:
    """Non-blocking background HTTP call to Telegram Bot API"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.debug("[AlertWatcher] Telegram credentials not configured. Skipping alert.")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code == 200:
                logger.info("[AlertWatcher] Incident alert dispatched to Telegram successfully.")
                return True
            else:
                logger.error(f"[AlertWatcher] Telegram API Error: {resp.text}")
                return False
    except Exception as e:
        logger.error(f"[AlertWatcher] Failed to send Telegram alert: {e}")
        return False


def notify_pipeline_error(stage: str, topic: str, error_msg: str, user_id: str = "Unknown"):
    """Fires an instant alert when a critical pipeline stage breaks"""
    time_str = datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")
    text = (
        f" <b>[AtlashAI Alert] Pipeline Breakdown Detected!</b>\n\n"
        f"<b>Stage:</b> <code>{stage}</code>\n"
        f"<b>Topic:</b> {topic}\n"
        f"<b>User:</b> <code>{user_id}</code>\n"
        f"<b>Error Details:</b>\n<pre>{error_msg[:300]}</pre>\n\n"
        f" <b>Time:</b> {time_str}"
    )
    # Fire and forget asynchronously
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(send_telegram_alert_async(text))
        else:
            asyncio.run(send_telegram_alert_async(text))
    except Exception:
        pass


def notify_model_fallback(failed_model: str, next_model: str, reason: str, agent_name: str = "Agent"):
    """Fires an alert when an LLM fails and switches to backup model"""
    time_str = datetime.now().strftime("%I:%M:%S %p")
    text = (
        f" <b>[AtlashAI Watcher] LLM Failover Triggered</b>\n\n"
        f"<b>Agent:</b> <code>{agent_name}</code>\n"
        f"<b>Failed Model:</b>  <code>{failed_model}</code>\n"
        f"<b>Switching To:</b>  <code>{next_model}</code>\n"
        f"<b>Reason:</b> <pre>{reason[:200]}</pre>\n\n"
        f" <b>Time:</b> {time_str}"
    )
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(send_telegram_alert_async(text))
        else:
            asyncio.run(send_telegram_alert_async(text))
    except Exception:
        pass
