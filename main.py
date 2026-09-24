import os
import re
import json
from datetime import datetime, timezone

import telebot
from flask import Flask, request
from openai import OpenAI
from ddgs import DDGS

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
BOT_NAME = "Flux"
MAX_HISTORY = 10  # number of recent messages remembered per chat

# ---------- Telegram + Flask ----------
bot = telebot.TeleBot(BOT_TOKEN, threaded=False)
app = Flask(__name__)

histories = {}  # chat_id -> list of {"role": ..., "content": ...}


def today():
    return datetime.now(timezone.utc).strftime("%A, %B %d, %Y")


def system_prompt():
    return (
        f"You are {BOT_NAME}, a helpful, friendly AI assistant on Telegram. "
        f"Your name is {BOT_NAME}. Today's date is {today()} (UTC). "
        "Keep answers clear and concise."
    )


# ---------- Web search (only when needed) ----------
ROUTER_PROMPT = f"""You decide whether a user's message needs a LIVE internet search.
Today is {{date}}.

Search ONLY when the answer depends on up-to-date or changing information: latest news,
current events, prices, exchange rates, scores, weather, new releases, who currently holds
a position, anything after your training data, or facts you are unsure about.

Do NOT search for: greetings, chit-chat, coding help, math, translation, writing,
advice, or stable general knowledge.

Reply with ONLY one JSON object, nothing else:
{{"search": true, "query": "short web search query"}}
or
{{"search": false}}"""


def needs_search(chat_id, user_text):
    """Ask the model if a web search is needed. Returns a query string or None."""
    recent = histories.get(chat_id, [])[-4:]
    convo = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in recent)
    try:
        r = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            max_tokens=60,
            messages=[
                {"role": "system", "content": ROUTER_PROMPT.format(date=today())},
                {"role": "user", "content": f"Recent conversation:\n{convo}\n\nNew message: {user_text}"},
            ],
        )
        raw = r.choices[0].message.content or ""
        match = re.search(r"\{.*\}", raw, re.S)
        if not match:
            return None
        data = json.loads(match.group(0))
        if data.get("search") and str(data.get("query", "")).strip():
            return str(data["query"]).strip()
    except Exception as e:
        print("Router error:", e)
    return None


def web_search(query):
    """Search the web (DuckDuckGo via ddgs). Returns formatted text or None."""
    lines = []
    try:
        for r in DDGS().news(query, max_results=3):
            lines.append(f"- [{r.get('date', '')[:10]}] {r.get('title')} ({r.get('source', '')}): "
                         f"{r.get('body', '')} | {r.get('url')}")
    except Exception as e:
        print("News search error:", e)
    try:
        for r in DDGS().text(query, max_results=5):
            lines.append(f"- {r.get('title')}: {r.get('body', '')} | {r.get('href')}")
    except Exception as e:
        print("Text search error:", e)
    return "\n".join(lines) if lines else None


# ---------- Chat logic ----------
def ask_flux(chat_id, user_text):
    query = needs_search(chat_id, user_text)
    system = system_prompt()

    if query:
        results = web_search(query)
        if results:
            system += (
                f"\n\nLive web search results for \"{query}\" (retrieved {today()}):\n{results}\n\n"
                "Use these results to answer with up-to-date facts. Mention the source name or "
                "link for key facts. If the results don't contain the answer, say so honestly "
                "instead of guessing."
            )
        else:
            system += (
                "\n\nA web search was attempted but returned nothing. Answer from your own "
                "knowledge and clearly say you couldn't verify the latest information."
            )

    history = histories.setdefault(chat_id, [])
    history.append({"role": "user", "content": user_text})
    del history[:-MAX_HISTORY]

    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system}] + history,
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
        f"Hi! I'm {BOT_NAME} ⚡ - your AI assistant.\n"
        "Ask me anything. When you need the latest info, I'll search the web for it.\n"
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
        send_long(chat_id, ask_flux(chat_id, message.text))
    except Exception as e:
        print("Error:", e)
        bot.send_message(chat_id, "Sorry, something went wrong. Please try again.")


# ---------- Webhook routes ----------
@app.route("/", methods=["GET"])
def index():
    return f"{BOT_NAME} is running", 200


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
