#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Professional Gold Trading Pipeline (Enhanced)
Features:
- Uses XAUUSD=X (spot ounce price) from yfinance
- Filters news with zero-shot (facebook/bart-large-mnli) if available,
  otherwise uses strict keyword filtering
- Batch sentiment via ProsusAI/finbert if available, else fallback
- Normalizes component scores to [-1,1] before weighting
- Saves daily output gold_analysis.json, appends to historical_signals.csv
- Runs a simple backtest from historical_signals.csv and saves backtest_report.json
- Robust error handling for running in CI (GitHub Actions) or local
"""
import os
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import requests

# transformers imports with safe handling
try:
    from transformers import pipeline
except Exception:
    pipeline = None

# ---------- Configuration ----------
SYMBOLS = {
    "gold": "XAUUSD=X",
    "dxy": "DX-Y.NYB",
    "vix": "^VIX",
    "treasury": "^TNX",
    "oil": "CL=F",
    "spy": "SPY"
}

LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "365"))
NEWS_DAYS = int(os.getenv("NEWS_DAYS", "2"))
SAVE_PATH = os.getenv("SAVE_PATH", ".")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")  # optional but recommended
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))

# Weights for final signal (adjustable)
WEIGHTS = {
    "trend": float(os.getenv("W_TREND", "0.4")),
    "momentum": float(os.getenv("W_MOMENTUM", "0.3")),
    "correlation": float(os.getenv("W_CORR", "0.2")),
    "news": float(os.getenv("W_NEWS", "0.1")),
}

# Thresholds for signal decision
THRESHOLDS = {
    "buy": float(os.getenv("THRESHOLD_BUY", "0.4")),
    "sell": float(os.getenv("THRESHOLD_SELL", "-0.4"))
}

HISTORICAL_SIGNALS_CSV = os.path.join(SAVE_PATH, "historical_signals.csv")
GOLD_ANALYSIS_JSON = os.path.join(SAVE_PATH, "gold_analysis.json")
BACKTEST_REPORT_JSON = os.path.join(SAVE_PATH, "backtest_report.json")

# Keyword fallback if zero-shot not available
KEYWORDS = ["gold", "xau", "bullion", "precious metal", "spot gold", "troy ounce", "inflation", "interest rate", "fed", "dollar"]

# ---------- Helper: Model loaders ----------
def load_zero_shot_model():
    if pipeline is None:
        print("⚠️ transformers.pipeline غير متوفر — وسيتم الاعتماد على فلترة الكلمات المفتاحية.")
        return None
    try:
        print("🧠 تحميل zero-shot model (bart-large-mnli)...")
        zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        print("✅ zero-shot جاهز.")
        return zs
    except Exception as e:
        print(f"⚠️ فشل تحميل zero-shot: {e}")
        return None

def load_sentiment_model():
    if pipeline is None:
        print("⚠️ transformers.pipeline غير متوفر — لن يتوفر تحليل مشاعر محترف.")
        return None
    # Try FinBERT first, else fallback
    candidates = ["ProsusAI/finbert", "yiyanghkust/finbert-tone", "distilbert-base-uncased-finetuned-sst-2-english"]
    for cand in candidates:
        try:
            print(f"🧠 محاولة تحميل نموذج مشاعر: {cand} ...")
            sent = pipeline("sentiment-analysis", model=cand)
            print(f"✅ نموذج المشاعر {cand} جاهز.")
            return sent
        except Exception as e:
            print(f"   ⚠️ لا يمكن تحميل {cand}: {e}")
            continue
    print("❌ لم يتم تحميل أي نموذج للمشاعر — سيتم تجاوز تحليل المشاعر.")
    return None

# ---------- Fetch market data ----------
def fetch_market_data(symbols: Dict[str,str], lookback_days: int = 365) -> Optional[pd.DataFrame]:
    try:
        ticker_list = list(symbols.values())
        period = f"{lookback_days}d"
        print(f"📊 جلب بيانات Yahoo Finance للفترة: {period} ...")
        df = yf.download(ticker_list, period=period, interval="1d", progress=False)
        if df.empty:
            print("❌ لم يتم جلب بيانات من Yahoo Finance.")
            return None
        print(f"... تم جلب بيانات بعدد صفوف: {len(df)}")
        return df
    except Exception as e:
        print(f"❌ خطأ أثناء جلب السوق: {e}")
        return None

# ---------- Fetch news ----------
def fetch_news(news_api_key: Optional[str], days: int = 2, page_size: int = 100) -> List[Dict[str,Any]]:
    print("📰 جلب أخبار من NewsAPI..." if news_api_key else "📰 NewsAPI مفتاح غير موجود، سيتم تخطي جلب الأخبار.")
    if not news_api_key:
        return []
    query = ('gold OR XAU OR bullion OR "precious metal" OR "gold price" '
             'OR "interest rate" OR fed OR inflation OR CPI OR NFP OR geopolitical OR dollar')
    from_date = (datetime.utcnow() - timedelta(days=days)).date()
    url = ("https://newsapi.org/v2/everything"
           f"?q={requests.utils.quote(query)}&language=en&sortBy=publishedAt&pageSize={page_size}"
           f"&from={from_date}&apiKey={news_api_key}")
    try:
        res = requests.get(url, timeout=20)
        res.raise_for_status()
        items = res.json().get("articles", [])
        simplified = []
        for a in items:
            simplified.append({
                "title": a.get("title"),
                "description": a.get("description"),
                "content": a.get("content"),
                "source": a.get("source", {}).get("name"),
                "publishedAt": a.get("publishedAt")
            })
        print(f"... تم جلب {len(simplified)} مقال من NewsAPI.")
        return simplified
    except Exception as e:
        print(f"⚠️ خطأ أثناء جلب الأخبار: {e}")
        return []

# ---------- Filter relevance ----------
def is_relevant_by_keywords(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    return any(k in t for k in KEYWORDS)

def filter_relevant_articles(articles: List[Dict[str,Any]], zero_shot_pipeline, threshold: float = 0.45) -> List[Dict[str,Any]]:
    print("🔎 فلترة المقالات لتحديد الصلة بالذهب...")
    if not articles:
        return []
    candidates = []
    labels = ["gold", "economy", "geopolitics", "other"]
    for art in articles:
        title = (art.get("title") or "") or ""
        desc = (art.get("description") or "") or ""
        text = (title + " " + desc).strip()
        if not text:
            continue
        relevant = False
        rel_score = 0.0
        if zero_shot_pipeline:
            try:
                out = zero_shot_pipeline(text, candidate_labels=labels, multi_label=False)
                if out and out.get("labels"):
                    if out["labels"][0] == "gold" and out["scores"][0] >= threshold:
                        relevant = True
                        rel_score = float(out["scores"][0])
            except Exception:
                relevant = False
        # fallback to strict keyword check
        if not zero_shot_pipeline and is_relevant_by_keywords(text):
            relevant = True
            rel_score = 0.5
        # even if zero-shot exists, still accept high-keyword matches (guard against model miss)
        if not relevant and is_relevant_by_keywords(text):
            relevant = True
            rel_score = max(rel_score, 0.35)
        if relevant:
            art["_relevance_score"] = rel_score
            candidates.append(art)
    print(f"... بعد الفلترة بقي {len(candidates)} مقالة ذات صلة.")
    # sort by relevance score desc
    return sorted(candidates, key=lambda x: x.get("_relevance_score", 0), reverse=True)

# ---------- Batch sentiment ----------
def analyze_sentiment_batch(articles: List[Dict[str,Any]], sentiment_pipeline, batch_size: int = 16) -> Dict[str,Any]:
    print("🧾 تحليل المشاعر (دفعي)...")
    if not articles or sentiment_pipeline is None:
        print("⚠️ لا توجد مقالات أو نموذج مشاعر؛ سيتم إرجاع news_score = 0.")
        return {"status":"skipped", "news_score":0.0, "headlines":[]}
    texts = []
    for a in articles:
        # combine title + description; truncate to reasonable length for transformers (e.g., 512)
        txt = ((a.get("title") or "") + ". " + (a.get("description") or "")).strip()
        texts.append(txt[:512])
    results = []
    try:
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            out = sentiment_pipeline(batch)
            # transformer pipeline returns list of dicts for batch
            results.extend(out)
            # tiny pause to be kind to remote model hosting (if any)
            time.sleep(0.1)
    except Exception as e:
        print(f"⚠️ خطأ أثناء التحليل الدفعي: {e} — سنحاول تحليل كل عنصر منفردًا.")
        results = []
        for t in texts:
            try:
                results.append(sentiment_pipeline(t)[0])
            except Exception:
                results.append({"label":"NEUTRAL", "score":0.0})
    # convert to numeric values
    numeric = []
    headlines = []
    for a, r in zip(articles, results):
        lbl = str(r.get("label","")).lower()
        sc = float(r.get("score",0.0))
        val = 0.0
        if "pos" in lbl or lbl == "positive":
            val = sc
        elif "neg" in lbl or lbl == "negative":
            val = -sc
        else:
            val = 0.0
        numeric.append(val)
        headlines.append({
            "title": a.get("title"),
            "source": a.get("source"),
            "relevance": round(a.get("_relevance_score", 0), 4),
            "sentiment": round(val, 4)
        })
    relevances = np.array([a.get("_relevance_score", 0.5) for a in articles], dtype=float)
    vals = np.array(numeric, dtype=float)
    if relevances.sum() == 0:
        news_score = float(vals.mean()) if len(vals) else 0.0
    else:
        news_score = float(np.sum(vals * relevances) / (relevances.sum()))
    news_score = max(min(news_score, 1.0), -1.0)
    news_score = round(news_score, 4)
    print(f"... news_score (weighted) = {news_score}")
    return {"status":"success", "news_score":news_score, "headlines":headlines[:20]}

# ---------- Indicators & scoring ----------
def compute_indicators_for_gold(gold_df: pd.DataFrame) -> pd.DataFrame:
    ta_strategy = ta.Strategy(name="full", ta=[
        {"kind":"sma", "length":50}, {"kind":"sma", "length":200},
        {"kind":"rsi", "length":14}, {"kind":"macd"},
        {"kind":"bbands"}, {"kind":"atr"}, {"kind":"obv"}
    ])
    gold_df.ta.strategy(ta_strategy)
    return gold_df

def normalize_component(val: float, min_val: float, max_val: float) -> float:
    # clip then scale to [-1,1]
    v = float(max(min(val, max_val), min_val))
    return 2 * (v - min_val) / (max_val - min_val) - 1

def score_components(latest_row: pd.Series, market_data_df: pd.DataFrame, news_score: float, symbols: Dict[str,str]):
    # Trend: distance from SMA200 (relative)
    trend_raw = 0.0
    try:
        close = float(latest_row["Close"])
        sma200 = float(latest_row.get("SMA_200", np.nan))
        if not np.isnan(sma200) and sma200 != 0:
            diff = (close - sma200) / sma200  # e.g., 0.02 => 2% above sma200
            trend_raw = diff * 10  # scale so that ~0.2 -> 2.0
    except Exception:
        trend_raw = 0.0
    # Momentum: MACD histogram normalized by recent avg magnitude
    momentum_raw = 0.0
    try:
        macd_hist = float(latest_row.get("MACDh_12_26_9", np.nan))
        # scale by a reasonable factor, avoid division by zero
        momentum_raw = macd_hist
    except Exception:
        momentum_raw = 0.0
    # Correlation with DXY (negative correlation supports gold)
    corr_raw = 0.0
    try:
        gold_close = market_data_df[('Close', symbols['gold'])]
        dxy_close = market_data_df[('Close', symbols['dxy'])]
        corr = gold_close.corr(dxy_close)
        corr_raw = -corr if not np.isnan(corr) else 0.0
    except Exception:
        corr_raw = 0.0
    # Normalize
    trend_norm = normalize_component(trend_raw, min_val=-2.0, max_val=2.0)
    momentum_norm = normalize_component(momentum_raw, min_val=-2.0, max_val=2.0)
    corr_norm = normalize_component(corr_raw, min_val=-1.0, max_val=1.0)
    news_norm = max(min(news_score, 1.0), -1.0)
    comps = {
        "trend_score_raw": round(trend_raw, 6),
        "momentum_score_raw": round(momentum_raw, 6),
        "correlation_score_raw": round(corr_raw, 6),
        "trend_score": round(trend_norm, 6),
        "momentum_score": round(momentum_norm, 6),
        "correlation_score": round(corr_norm, 6),
        "news_score": round(news_norm, 6)
    }
    return comps

def decide_signal(comps: Dict[str,float], weights: Dict[str,float] = WEIGHTS, thresholds: Dict[str,float] = THRESHOLDS):
    total = (comps["trend_score"] * weights["trend"] +
             comps["momentum_score"] * weights["momentum"] +
             comps["correlation_score"] * weights["correlation"] +
             comps["news_score"] * weights["news"])
    total = round(float(total), 6)
    if total >= thresholds["buy"]:
        sig = "Buy"
    elif total <= thresholds["sell"]:
        sig = "Sell"
    else:
        sig = "Hold"
    return total, sig

# ---------- Persistence ----------
def append_signal_csv(path: str, row: Dict[str,Any]):
    df_row = pd.DataFrame([row])
    if not os.path.exists(path):
        df_row.to_csv(path, index=False, encoding="utf-8")
    else:
        df_row.to_csv(path, mode="a", index=False, header=False, encoding="utf-8")
    print(f"💾 appended signal to {path}")

def save_json(path: str, data: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"💾 saved: {path}")

# ---------- Backtest ----------
def backtest_from_signals(signals_csv: str, symbol: str):
    print("📈 بدء backtest من سجل الإشارات...")
    if not os.path.exists(signals_csv):
        print("⚠️ لا يوجد ملف إشارات للتشغيل backtest.")
        return {"status":"error", "error":"no signals file"}
    try:
        sigs = pd.read_csv(signals_csv, parse_dates=["timestamp_utc"])
        sigs.sort_values("timestamp_utc", inplace=True)
        sigs.reset_index(drop=True, inplace=True)
        # define date range for price fetch
        start = sigs["timestamp_utc"].dt.date.min()
        end = sigs["timestamp_utc"].dt.date.max() + pd.Timedelta(days=1)
        prices = yf.download(symbol, start=start.isoformat(), end=end.isoformat(), interval="1d", progress=False)
        if prices.empty:
            return {"status":"error", "error":"no price series fetched"}
        close = prices["Close"].ffill().dropna()
        close.index = pd.to_datetime(close.index)
        sigs["date"] = pd.to_datetime(sigs["timestamp_utc"]).dt.normalize()
        # create daily signals aligned with price dates (ffill last signal)
        daily_sig = sigs.groupby("date").last().reindex(close.index, method="ffill").fillna(method="ffill")
        # position logic: Buy -> 1, Sell -> 0, Hold -> previous
        position = []
        pos = 0
        for idx, row in daily_sig.iterrows():
            s = row["signal"]
            if s == "Buy":
                pos = 1
            elif s == "Sell":
                pos = 0
            position.append(pos)
        pf = pd.DataFrame({"close": close, "position": position}, index=close.index)
        pf["pct_change"] = pf["close"].pct_change().fillna(0)
        pf["strategy_return"] = pf["position"].shift(1).fillna(0) * pf["pct_change"]
        pf["cum_return"] = (1 + pf["strategy_return"]).cumprod()
        pf["buyhold_cum"] = (1 + pf["pct_change"]).cumprod()
        total_return = float(pf["cum_return"].iloc[-1] - 1)
        bh_return = float(pf["buyhold_cum"].iloc[-1] - 1)
        days = (pf.index[-1] - pf.index[0]).days or 1
        years = days / 365.25
        cagr = (pf["cum_return"].iloc[-1]) ** (1 / years) - 1 if years > 0 else 0
        bh_cagr = (pf["buyhold_cum"].iloc[-1]) ** (1 / years) - 1 if years > 0 else 0
        roll_max = pf["cum_return"].cummax()
        drawdown = pf["cum_return"] / roll_max - 1
        max_dd = float(drawdown.min())
        trades = int(((daily_sig["signal"] == "Buy") | (daily_sig["signal"] == "Sell")).sum())
        report = {
            "period_start": str(pf.index[0].date()),
            "period_end": str(pf.index[-1].date()),
            "days": int(days),
            "total_return": round(total_return, 6),
            "cagr": round(cagr, 6),
            "buyhold_return": round(bh_return, 6),
            "buyhold_cagr": round(bh_cagr, 6),
            "max_drawdown": round(max_dd, 6),
            "trades": trades
        }
        save_json(BACKTEST_REPORT_JSON, report)
        return {"status":"success", "report":report}
    except Exception as e:
        return {"status":"error", "error": str(e)}

# ---------- Main pipeline ----------
def run_pipeline():
    # load models
    zs = load_zero_shot_model()
    sent = load_sentiment_model()
    # fetch market data
    market_df = fetch_market_data(SYMBOLS, LOOKBACK_DAYS)
    if market_df is None:
        return {"status":"error", "error":"market fetch failed"}
    # prepare gold dataframe
    gold_ticker = SYMBOLS["gold"]
    try:
        gold_df = pd.DataFrame({
            "Open": market_df[("Open", gold_ticker)],
            "High": market_df[("High", gold_ticker)],
            "Low": market_df[("Low", gold_ticker)],
            "Close": market_df[("Close", gold_ticker)],
            "Volume": market_df[("Volume", gold_ticker)]
        }).dropna()
    except Exception as e:
        return {"status":"error", "error": f"error preparing gold df: {e}"}
    if gold_df.empty:
        return {"status":"error", "error":"no gold data"}
    # compute indicators
    gold_df = compute_indicators_for_gold(gold_df)
    gold_df.dropna(inplace=True)
    if gold_df.empty:
        return {"status":"error", "error":"not enough data for indicators"}
    # fetch & filter news
    raw_news = fetch_news(NEWS_API_KEY, NEWS_DAYS, page_size=100)
    relevant = filter_relevant_articles(raw_news, zs)
    news_result = analyze_sentiment_batch(relevant, sent, BATCH_SIZE)
    # latest row
    latest = gold_df.iloc[-1]
    comps = score_components(latest, market_df, news_result.get("news_score", 0.0), SYMBOLS)
    total_score, signal = decide_signal(comps)
    # prepare final result
    result = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "signal": signal,
        "total_score": total_score,
        "components": comps,
        "market_data": {
            "gold_price": float(round(latest["Close"], 6)),
            "dxy": float(round(market_df[('Close', SYMBOLS['dxy'])].iloc[-1], 6)) if ('Close', SYMBOLS['dxy']) in market_df else None,
            "vix": float(round(market_df[('Close', SYMBOLS['vix'])].iloc[-1], 6)) if ('Close', SYMBOLS['vix']) in market_df else None
        },
        "news_analysis": news_result
    }
    # save json
    save_json(GOLD_ANALYSIS_JSON, result)
    # append signal to historical CSV
    row = {
        "timestamp_utc": result["timestamp_utc"],
        "signal": signal,
        "total_score": total_score,
        "gold_price": result["market_data"]["gold_price"],
        "news_score": comps["news_score"],
        "trend_score": comps["trend_score"],
        "momentum_score": comps["momentum_score"],
        "correlation_score": comps["correlation_score"]
    }
    append_signal_csv(HISTORICAL_SIGNALS_CSV, row)
    # run backtest if enough data
    bt = backtest_from_signals(HISTORICAL_SIGNALS_CSV, SYMBOLS["gold"])
    return {"status":"success", "result": result, "backtest": bt}

if __name__ == "__main__":
    out = run_pipeline()
    print("\n--- Pipeline output ---")
    print(json.dumps(out, indent=2, ensure_ascii=False))