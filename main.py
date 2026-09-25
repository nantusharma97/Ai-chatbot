import os
import re
import json
import base64
import socket
import ipaddress
from datetime import datetime, timezone
from urllib.parse import urlparse, urljoin

import requests
import telebot
from bs4 import BeautifulSoup
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

# Optional: persistent memory (free Upstash Redis). Without these, memory resets on restart.
UPSTASH_URL = os.environ.get("UPSTASH_REDIS_REST_URL")
UPSTASH_TOKEN = os.environ.get("UPSTASH_REDIS_REST_TOKEN")

# Render sets this automatically for Web Services (e.g. https://your-app.onrender.com)
RENDER_URL = os.environ.get("RENDER_EXTERNAL_URL")
PORT = int(os.environ.get("PORT", 10000))

# ---------- Hugging Face (OpenAI-compatible) client ----------
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
)
MODEL = "meta-llama/Llama-3.1-8B-Instruct:novita"
# Vision model for photos. Override with a VISION_MODEL env var if this one is unavailable.
VISION_MODEL = os.environ.get("VISION_MODEL", "Qwen/Qwen3.6-27B:ovhcloud")
BOT_NAME = "Flux"
MAX_HISTORY = 10  # number of recent messages remembered per chat
HISTORY_TTL = 60 * 60 * 24 * 30  # keep saved chats for 30 days

# ---------- Telegram + Flask ----------
bot = telebot.TeleBot(BOT_TOKEN, threaded=False)
app = Flask(__name__)


def today():
    return datetime.now(timezone.utc).strftime("%A, %B %d, %Y")


def system_prompt():
    return (
        f"You are {BOT_NAME}, a helpful, friendly AI assistant on Telegram. "
        f"Your name is {BOT_NAME}. Today's date is {today()} (UTC). "
        "Keep answers clear and concise."
    )


# ---------- Persistent memory (Upstash Redis REST, falls back to RAM) ----------
_local = {}  # chat_id -> history (fallback / cache)


def _redis(*command):
    r = requests.post(
        UPSTASH_URL,
        headers={"Authorization": f"Bearer {UPSTASH_TOKEN}"},
        json=list(command),
        timeout=5,
    )
    r.raise_for_status()
    return r.json().get("result")


def load_history(chat_id):
    if UPSTASH_URL and UPSTASH_TOKEN:
        try:
            raw = _redis("GET", f"flux:hist:{chat_id}")
            return json.loads(raw) if raw else []
        except Exception as e:
            print("Memory load error:", e)
    return list(_local.get(chat_id, []))


def save_history(chat_id, history):
    history = history[-MAX_HISTORY:]
    _local[chat_id] = history
    if UPSTASH_URL and UPSTASH_TOKEN:
        try:
            _redis("SET", f"flux:hist:{chat_id}", json.dumps(history), "EX", HISTORY_TTL)
        except Exception as e:
            print("Memory save error:", e)


def clear_history(chat_id):
    _local.pop(chat_id, None)
    if UPSTASH_URL and UPSTASH_TOKEN:
        try:
            _redis("DEL", f"flux:hist:{chat_id}")
        except Exception as e:
            print("Memory clear error:", e)


# ---------- Web search (only when needed) ----------
ROUTER_PROMPT = """You decide whether a user's message needs a LIVE internet search.
Today is {date}.

Search ONLY when the answer depends on up-to-date or changing information: latest news,
current events, prices, exchange rates, scores, weather, new releases, who currently holds
a position, anything after your training data, or facts you are unsure about.

Do NOT search for: greetings, chit-chat, coding help, math, translation, writing,
advice, or stable general knowledge.

Reply with ONLY one JSON object, nothing else:
{{"search": true, "query": "short web search query"}}
or
{{"search": false}}"""


def needs_search(history, user_text):
    """Ask the model if a web search is needed. Returns a query string or None."""
    convo = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in history[-4:])
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


# ---------- Link reading ----------
URL_RE = re.compile(r"https?://[^\s<>\"']+")


def find_url(text):
    m = URL_RE.search(text or "")
    return m.group(0).rstrip(").,;!?]") if m else None


def is_public_url(url):
    """Block localhost / private network addresses (SSRF protection)."""
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        return False
    try:
        for info in socket.getaddrinfo(p.hostname, None):
            if not ipaddress.ip_address(info[4][0]).is_global:
                return False
    except Exception:
        return False
    return True


def fetch_page(url):
    """Download a web page and return its readable text (max ~15,000 chars)."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; FluxBot/1.0)"}
    for _ in range(4):  # follow up to 3 redirects, checking each one
        if not is_public_url(url):
            raise ValueError("that address isn't allowed or can't be reached")
        r = requests.get(url, headers=headers, timeout=10, allow_redirects=False, stream=True)
        if r.status_code in (301, 302, 303, 307, 308):
            url = urljoin(url, r.headers.get("Location", ""))
            continue
        break
    else:
        raise ValueError("too many redirects")

    r.raise_for_status()
    ctype = r.headers.get("Content-Type", "").split(";")[0].strip().lower()
    if ctype and "html" not in ctype and not ctype.startswith("text/"):
        raise ValueError(f"unsupported content type ({ctype})")

    html = r.raw.read(1_500_000, decode_content=True).decode("utf-8", errors="ignore")
    if "html" in ctype or not ctype:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
            tag.decompose()
        title = soup.title.get_text(strip=True) if soup.title else ""
        text = soup.get_text("\n", strip=True)
    else:
        title, text = "", html

    text = re.sub(r"\n{2,}", "\n", text)
    if len(text) < 200:
        raise ValueError("the page has too little readable text (it may need JavaScript)")
    return f"Title: {title}\n\n{text[:15000]}"


# ---------- Chat logic ----------
def chat_reply(chat_id, history, user_text, extra_system=""):
    history.append({"role": "user", "content": user_text})
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system_prompt() + extra_system}] + history[-MAX_HISTORY:],
    )
    reply = completion.choices[0].message.content or "(empty response)"
    history.append({"role": "assistant", "content": reply})
    save_history(chat_id, history)
    return reply


def ask_flux(chat_id, user_text):
    history = load_history(chat_id)
    extra = ""
    query = needs_search(history, user_text)

    if query:
        results = web_search(query)
        if results:
            extra = (
                f"\n\nLive web search results for \"{query}\" (retrieved {today()}):\n{results}\n\n"
                "Use these results to answer with up-to-date facts. Mention the source name or "
                "link for key facts. If the results don't contain the answer, say so honestly "
                "instead of guessing."
            )
        else:
            extra = (
                "\n\nA web search was attempted but returned nothing. Answer from your own "
                "knowledge and clearly say you couldn't verify the latest information."
            )
    return chat_reply(chat_id, history, user_text, extra)


def answer_link(chat_id, user_text, url):
    try:
        page = fetch_page(url)
    except Exception as e:
        return f"Sorry, I couldn't open that link: {str(e)[:150]}"

    title_line = page.split("\n", 1)[0]  # "Title: ..."
    history = load_history(chat_id)
    extra = (
        f"\n\nThe user shared this link: {url}\n"
        f"Below is the ACTUAL extracted page content (may be incomplete or contain leftover menu/"
        f"related-article text). Do not use anything you already know about this URL, this topic, "
        f"or this publication date from your training data - only use what is literally written "
        f"below. If the extracted text is boilerplate, a cookie/consent notice, a paywall message, "
        f"or otherwise doesn't contain the actual article, say exactly that instead of guessing or "
        f"summarizing from memory.\n\n{page}\n\n"
        "If the user sent only the link, give a concise summary with the key points from the text "
        "above."
    )
    reply = chat_reply(chat_id, history, user_text, extra)
    return f"📄 {title_line}\n\n{reply}"


def answer_photo(chat_id, message):
    file_info = bot.get_file(message.photo[-1].file_id)  # largest size
    data = bot.download_file(file_info.file_path)
    b64 = base64.b64encode(data).decode()
    question = message.caption or "Describe this image in detail."

    history = load_history(chat_id)
    completion = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[{"role": "system", "content": system_prompt()}]
        + history[-6:]
        + [{
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
    )
    reply = completion.choices[0].message.content or "(empty response)"
    # Save as text so follow-up questions still have context
    history.append({"role": "user", "content": f"[Photo] {question}"})
    history.append({"role": "assistant", "content": reply})
    save_history(chat_id, history)
    return reply


def send_long(chat_id, text):
    # Telegram limit is 4096 characters per message
    for i in range(0, len(text), 4000):
        bot.send_message(chat_id, text[i:i + 4000])


# ---------- Telegram handlers ----------
@bot.message_handler(commands=["start", "help"])
def handle_start(message):
    bot.reply_to(
        message,
        f"Hi! I'm {BOT_NAME} ⚡ - your AI assistant.\n\n"
        "• Ask me anything - I search the web when I need the latest info\n"
        "• Send a link and I'll summarize the page\n"
        "• Send a photo (add a caption to ask about it)\n"
        "• /reset clears our conversation",
    )


@bot.message_handler(commands=["reset"])
def handle_reset(message):
    clear_history(message.chat.id)
    bot.reply_to(message, "Conversation cleared.")


@bot.message_handler(content_types=["photo"])
def handle_photo(message):
    chat_id = message.chat.id
    try:
        bot.send_chat_action(chat_id, "typing")
        send_long(chat_id, answer_photo(chat_id, message))
    except Exception as e:
        print("Photo error:", e)
        bot.send_message(chat_id, "Sorry, I couldn't analyze that image. Please try again.")


@bot.message_handler(content_types=["text"])
def handle_text(message):
    chat_id = message.chat.id
    try:
        bot.send_chat_action(chat_id, "typing")
        url = find_url(message.text)
        reply = answer_link(chat_id, message.text, url) if url else ask_flux(chat_id, message.text)
        send_long(chat_id, reply)
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
history = load_history(chat_id)
    completion = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[{"role": "system", "content": system_prompt()}]
        + history[-6:]
        + [{
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
    )
    reply = completion.choices[0].message.content or "(empty response)"
    # Save as text so follow-up questions still have context
    history.append({"role": "user", "content": f"[Photo] {question}"})
    history.append({"role": "assistant", "content": reply})
    save_history(chat_id, history)
    return reply


def send_long(chat_id, text):
    # Telegram limit is 4096 characters per message
    for i in range(0, len(text), 4000):
        bot.send_message(chat_id, text[i:i + 4000])


# ---------- Telegram handlers ----------
@bot.message_handler(commands=["start", "help"])
def handle_start(message):
    bot.reply_to(
        message,
        f"Hi! I'm {BOT_NAME} ⚡ - your AI assistant.\n\n"
        "• Ask me anything - I search the web when I need the latest info\n"
        "• Send a link and I'll summarize the page\n"
        "• Send a photo (add a caption to ask about it)\n"
        "• /reset clears our conversation",
    )


@bot.message_handler(commands=["reset"])
def handle_reset(message):
    clear_history(message.chat.id)
    bot.reply_to(message, "Conversation cleared.")


@bot.message_handler(content_types=["photo"])
def handle_photo(message):
    chat_id = message.chat.id
    try:
        bot.send_chat_action(chat_id, "typing")
        send_long(chat_id, answer_photo(chat_id, message))
    except Exception as e:
        print("Photo error:", e)
        bot.send_message(chat_id, "Sorry, I couldn't analyze that image. Please try again.")


@bot.message_handler(content_types=["text"])
def handle_text(message):
    chat_id = message.chat.id
    try:
        bot.send_chat_action(chat_id, "typing")
        url = find_url(message.text)
        reply = answer_link(chat_id, message.text, url) if url else ask_flux(chat_id, message.text)
        send_long(chat_id, reply)
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
