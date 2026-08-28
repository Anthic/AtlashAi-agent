"""
Test script for Telegram Alert Watcher
Run this script to verify that incident notifications appear in your Telegram chat.
"""

import os
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.alert_watcher import notify_model_fallback, notify_pipeline_error

def test_telegram_alerts():
    print("=" * 60)
    print("🚀 AtlashAI Telegram Alert Watcher Test")
    print("=" * 60)

    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    if not token or not chat_id:
        print("⚠️  Warning: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is missing in .env!")
        print("👉 Please add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to your .env file.")
        return

    print(f"✅ Found Telegram Config: Chat ID = {chat_id}")
    print("📤 1. Sending Simulated LLM Fallback Alert...")
    notify_model_fallback(
        failed_model="qwen/qwen3.8-27b (Groq)",
        next_model="gemini-3.5-flash-lite (Google)",
        reason="HTTP 429: Rate Limit Exceeded (Simulated Test)",
        agent_name="ParaphraserAgent"
    )

    time.sleep(2)

    print("📤 2. Sending Simulated Critical Pipeline Error Alert...")
    notify_pipeline_error(
        stage="RAG Vector Extraction (Qdrant)",
        topic="Solid State Lithium Batteries 2024",
        error_msg="ConnectionTimeout: 504 Gateway Timeout while fetching embeddings (Simulated Test)",
        user_id="user_anthi_dev"
    )

    print("⏳ Waiting 3 seconds for async delivery...")
    time.sleep(3)
    print("\n🎉 Test Complete! Check your Telegram App now for the notifications! 📱")
    print("=" * 60)

if __name__ == "__main__":
    test_telegram_alerts()
