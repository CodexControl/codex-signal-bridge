#!/usr/bin/env python3
"""
Codex Control - AI Council Bridge
==================================
Runs on a GitHub Actions schedule (no server, no Cloudflare). For each
configured symbol it:

  1. Pulls price + a few indicators from TwelveData.
  2. Pulls upcoming/recent economic calendar events from TwelveData's
     economic calendar endpoint (falls back to "unavailable" if your plan
     doesn't include it - the prompt tells Gemini to say so rather than guess).
  3. Sends everything to Gemini and asks for a strict-JSON verdict modeled on
     the Core Control Council spec (BUY / SELL / WAIT, one timeframe, a
     reason, a risk, and - only for BUY/SELL - entry/stop/target).
  4. Writes signals/<SYMBOL>.json into the repo. The EA polls the raw
     GitHub URL for that file (see External_Signal_URL in the EA inputs).

Nothing here places trades. It only produces a JSON opinion. Whether MT4
acts on it automatically or waits for you to click EXECUTE is controlled
entirely inside the EA (AI_Auto_Trading_Enabled input) - this script has no
say in that.

Secrets are read from environment variables (set as GitHub Actions repo
secrets, never hard-coded): GEMINI_API_KEY, TWELVEDATA_API_KEY.
"""

import json
import os
import sys
import time
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import requests

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
TWELVEDATA_API_KEY = os.environ["TWELVEDATA_API_KEY"]

# Edit this list to match your broker's actual pairs. Don't guess the names
# by hand - run the EA once with Enable_External_Signal=true and it prints a
# ready-to-paste list (see SETUP_GUIDE.md Part 0 / Codex_AI_MarketWatch_Symbols.txt),
# already normalized to plain canonical names regardless of what suffix your
# broker adds (HFM ECN Zero, cent accounts, or anything else). Keep it short -
# each symbol costs one TwelveData quote call + one Gemini call per run, and
# both have free-tier rate limits.
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "BTCUSDT"]

GEMINI_MODEL = "gemini-2.0-flash"  # swap for whichever Gemini model you have access to
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

OUT_DIR = os.path.join(os.path.dirname(__file__), "signals")

# Same free ForexFactory-sourced feed your EA's own news panel already uses.
FF_CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
FF_CACHE_PATH = os.path.join(os.path.dirname(__file__), "signals", "_econ_calendar_cache.json")

# Rough currency mapping so each symbol only gets the calendar events that
# are actually relevant to it, instead of dumping the whole week at Gemini.
CURRENCY_KEYWORDS = {
    "USD": ["USD", "US"], "EUR": ["EUR"], "GBP": ["GBP"], "JPY": ["JPY"],
    "AUD": ["AUD"], "NZD": ["NZD"], "CAD": ["CAD"], "CHF": ["CHF"], "CNY": ["CNY", "CNH"],
}


def symbol_currencies(symbol):
    s = symbol.upper()
    hits = [c for c in CURRENCY_KEYWORDS if c in s]
    if hits:
        return hits
    # Crypto and anything else: USD macro news is still the dominant driver.
    return ["USD"]


def td_get(path, **params):
    params["apikey"] = TWELVEDATA_API_KEY
    r = requests.get(f"https://api.twelvedata.com/{path}", params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and data.get("status") == "error":
        return {"error": data.get("message", "unknown TwelveData error")}
    return data


def fetch_market_snapshot(symbol):
    """Pull quote + a couple of common indicators. Any single failed call
    degrades to 'unavailable' rather than aborting the whole symbol."""
    snapshot = {"symbol": symbol}

    quote = td_get("quote", symbol=symbol)
    snapshot["quote"] = quote if "error" not in quote else "UNAVAILABLE"

    for tf in ("30min", "1h"):
        ts = td_get("time_series", symbol=symbol, interval=tf, outputsize=30)
        snapshot[f"candles_{tf}"] = ts.get("values", "UNAVAILABLE") if isinstance(ts, dict) else "UNAVAILABLE"

    rsi = td_get("rsi", symbol=symbol, interval="1h", outputsize=1)
    snapshot["rsi_1h"] = rsi.get("values", "UNAVAILABLE") if isinstance(rsi, dict) else "UNAVAILABLE"

    ema50 = td_get("ema", symbol=symbol, interval="1h", time_period=50, outputsize=1)
    snapshot["ema50_1h"] = ema50.get("values", "UNAVAILABLE") if isinstance(ema50, dict) else "UNAVAILABLE"

    return snapshot


def _parse_ff_xml(xml_bytes):
    root = ET.fromstring(xml_bytes)
    events = []
    for ev in root.findall(".//event"):
        events.append({
            "title": (ev.findtext("title") or "").strip(),
            "country": (ev.findtext("country") or "").strip(),
            "date": (ev.findtext("date") or "").strip(),
            "time": (ev.findtext("time") or "").strip(),
            "impact": (ev.findtext("impact") or "").strip(),
            "forecast": (ev.findtext("forecast") or "").strip(),
            "previous": (ev.findtext("previous") or "").strip(),
        })
    return events


def fetch_all_calendar_events():
    """Pulls the free ForexFactory weekly calendar feed. ForexFactory caps
    this feed at 2 downloads / 5 minutes *shared across everyone hitting it
    from the same IP range* - GitHub Actions runners share IPs, so a blocked
    request is expected occasionally. When that happens we fall back to the
    last successfully cached copy (committed alongside the signals) instead
    of failing the whole run."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "application/xml, text/xml, */*",
    }
    try:
        r = requests.get(FF_CALENDAR_URL, headers=headers, timeout=20)
        r.raise_for_status()
        body = r.content
        if b"Request Denied" in body or b"<!DOCTYPE html" in body[:200]:
            raise RuntimeError("ForexFactory feed rate-limited this request (shared IP cap)")
        events = _parse_ff_xml(body)
        if not events:
            raise RuntimeError("feed returned zero events - likely blocked or malformed")
        # Cache the good result for next time this happens.
        with open(FF_CACHE_PATH, "w") as f:
            json.dump({"fetched_at": datetime.now(timezone.utc).isoformat(), "events": events}, f)
        return events
    except Exception as e:
        print(f"ForexFactory calendar fetch failed, trying cache: {e}", file=sys.stderr)
        if os.path.exists(FF_CACHE_PATH):
            with open(FF_CACHE_PATH) as f:
                cached = json.load(f)
            return cached.get("events", [])
        return None  # truly nothing available


def economic_calendar_for_symbol(all_events, symbol):
    if all_events is None:
        return "UNAVAILABLE - ForexFactory feed unreachable and no cached copy exists yet"
    currencies = symbol_currencies(symbol)
    relevant = [e for e in all_events if e.get("country", "").upper() in currencies]
    # Cap it so the prompt stays small; highest-impact events first.
    impact_order = {"High": 0, "Medium": 1, "Low": 2, "": 3}
    relevant.sort(key=lambda e: impact_order.get(e.get("impact", ""), 3))
    return relevant[:20] if relevant else "NO MATCHING EVENTS THIS WEEK FOR " + ",".join(currencies)


COUNCIL_PROMPT = """You are CODEX, the orchestrator described below. Act as the
full Core Control Council: run every listed Councilor's perspective
internally (World, Forex, Trend, Market, Sentiment, Political, Country, War,
SUPRES, Entry/Exit, Timeframe Trend, Momentum, Experts, Signal Giver,
Confirmer, Assurer, Time Frame, Signal Integrity, Watcher, Recommender), then
return ONLY the final consensus as strict JSON - no prose, no markdown fences.

Rules:
- The three allowed actions are BUY, SELL, WAIT. WAIT is a completely valid
  and often correct answer - never force a trade to satisfy the user.
- Pick exactly one timeframe, "M30" or "H1", never both.
- Never fabricate a price, indicator value, or news event. If the market data
  or economic calendar below says UNAVAILABLE, say so in your reasoning
  instead of inventing a number.
- entry/stop/target are only meaningful for BUY/SELL; use "0" for WAIT.
- Keep "reason" and "risk" each to one plain-English sentence.

Return exactly this JSON shape:
{{
  "action": "BUY" | "SELL" | "WAIT",
  "timeframe": "M30" | "H1",
  "entry": "0.00000",
  "stop": "0.00000",
  "target": "0.00000",
  "strength": 0,
  "reason": "one sentence",
  "risk": "one sentence"
}}

SYMBOL: {symbol}

MARKET DATA (TwelveData):
{market_json}

ECONOMIC CALENDAR (TwelveData):
{econ_json}
"""


def ask_gemini(symbol, market_snapshot, econ_calendar):
    prompt = COUNCIL_PROMPT.format(
        symbol=symbol,
        market_json=json.dumps(market_snapshot, indent=2, default=str)[:12000],
        econ_json=json.dumps(econ_calendar, indent=2, default=str)[:6000],
    )
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"},
    }
    r = requests.post(GEMINI_URL, json=body, timeout=60)
    r.raise_for_status()
    data = r.json()
    text = data["candidates"][0]["content"]["parts"][0]["text"]
    return json.loads(text)


def build_signal_payload(symbol, verdict):
    sig_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{symbol}-{uuid.uuid4().hex[:6]}"
    return {
        "id": sig_id,
        "action": verdict.get("action", "WAIT"),
        "symbol": symbol,
        "timeframe": verdict.get("timeframe", "H1"),
        "entry": str(verdict.get("entry", "0")),
        "stop": str(verdict.get("stop", "0")),
        "target": str(verdict.get("target", "0")),
        "strength": str(verdict.get("strength", 0)),
        "reason": verdict.get("reason", ""),
        "risk": verdict.get("risk", ""),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_events = fetch_all_calendar_events()  # one fetch for the whole run, shared across symbols

    for symbol in SYMBOLS:
        try:
            snapshot = fetch_market_snapshot(symbol)
            econ_calendar = economic_calendar_for_symbol(all_events, symbol)
            verdict = ask_gemini(symbol, snapshot, econ_calendar)
            payload = build_signal_payload(symbol, verdict)
        except Exception as e:
            print(f"[{symbol}] FAILED: {e}", file=sys.stderr)
            payload = build_signal_payload(symbol, {"action": "WAIT", "reason": f"bridge error: {e}", "risk": "data pipeline failure - treat as no signal"})

        out_path = os.path.join(OUT_DIR, f"{symbol}.json")
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[{symbol}] wrote {out_path}: {payload['action']} ({payload['timeframe']})")
        time.sleep(1)  # be polite to both APIs' rate limits


if __name__ == "__main__":
    main()
