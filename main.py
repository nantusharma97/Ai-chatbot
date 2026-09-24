import os
import telebot
from flask import Flask, request
from openai import OpenAI

# ---------- Environment variables (set these in Render dashboard) ----------
# Accepts UPPERCASE (recommended) or lowercase names.
BOT_TOKEN = os.environ.get("BOT_TOKEN") or os.environ.get("bot_token")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("hf_token")

if not BOT_TOKEN:
    raise RuntimeError("Missing environment variable: BOT_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("Missing environment variable: HF_TOKEN")

# Render sets this automatically for Web Services (e.g. https://your-app.onrender.com)
RENDER_URL = os.environ.get("RENDER_EXTERNAL_URL")
PORT = int(os.environ.get("PORT", 10000))

# ---------- Hugging Face (OpenAI-compatible) client ----------
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
)
MODEL = "meta-llama/Llama-3.1-8B-Instruct:novita"
SYSTEM_PROMPT = "You are a helpful, friendly assistant. Keep answers clear and concise."
MAX_HISTORY = 10  # number of recent messages remembered per chat

# ---------- Telegram + Flask ----------
bot = telebot.TeleBot(BOT_TOKEN, threaded=False)
app = Flask(__name__)

histories = {}  # chat_id -> list of {"role": ..., "content": ...}


def ask_llama(chat_id, user_text):
    history = histories.setdefault(chat_id, [])
    history.append({"role": "user", "content": user_text})
    del history[:-MAX_HISTORY]

    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
    )
    reply = completion.choices[0].message.content or "(empty response)"

    history.append({"role": "assistant", "content": reply})
    del history[:-MAX_HISTORY]
    return reply


def send_long(chat_id, text):
    # Telegram limit is 4096 characters per message
    for i in range(0, len(text), 4000):
        bot.send_message(chat_id, text[i:i + 4000])


@bot.message_handler(commands=["start", "help"])
def handle_start(message):
    bot.reply_to(
        message,
        "Hi! I'm an AI bot powered by Llama 3.1. Send me any message.\n"
        "Use /reset to clear our conversation.",
    )


@bot.message_handler(commands=["reset"])
def handle_reset(message):
    histories.pop(message.chat.id, None)
    bot.reply_to(message, "Conversation cleared.")


@bot.message_handler(content_types=["text"])
def handle_text(message):
    chat_id = message.chat.id
    try:
        bot.send_chat_action(chat_id, "typing")
        send_long(chat_id, ask_llama(chat_id, message.text))
    except Exception as e:
        print("Error:", e)
        bot.send_message(chat_id, "Sorry, something went wrong. Please try again.")


# ---------- Webhook routes ----------
@app.route("/", methods=["GET"])
def index():
    return "Bot is running", 200


@app.route(f"/{BOT_TOKEN}", methods=["POST"])
def webhook():
    update = telebot.types.Update.de_json(request.get_data().decode("utf-8"))
    bot.process_new_updates([update])
    return "OK", 200


if __name__ == "__main__":
    if RENDER_URL:
        bot.remove_webhook()
        bot.set_webhook(url=f"{RENDER_URL}/{BOT_TOKEN}")
        app.run(host="0.0.0.0", port=PORT)
    else:
        # Local testing fallback: long polling
        bot.remove_webhook()
        print("RENDER_EXTERNAL_URL not set - running with polling")
        bot.infinity_polling()
