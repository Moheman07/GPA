#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gold Trading – Technical + Fundamental (News) unified script
تشغيل داخل GitHub Actions
"""

# --------------------------- IMPORTS ---------------------------
import os
import json
import logging
from datetime import datetime
import pytz

import pandas as pd
import yfinance as yf
import pandas_ta as ta
import requests
from dotenv import load_dotenv
from transformers import pipeline
from tqdm import tqdm

# --------------------------- CONFIG LOADER ---------------------------
DEFAULT_CONFIG = {
    "tickers": {"gold": "XAUUSD=X", "vix": "^VIX", "yield": "^TNX", "eurusd": "EURUSD=X"},
    "period": "2y",
    "interval": "1d",
    "technical": {
        "sma_fast": 50,
        "sma_slow": 200,
        "rsi_length": 14,
        "rsi_upper": 60,
        "rsi_lower": 40,
        "atr_length": 14,
        "atr_multiplier_sl": 1.8,
        "risk_reward_ratio": 2.0,
        "obv_sma": 20,
    },
    "signal_mapping": {"Strong Buy": 4, "Buy": 3, "Neutral": 2, "Sell": 1, "Strong Sell": 0},
    "sentiment": {
        "keywords": (
            "gold OR XAU OR gold price OR gold spot OR gold futures OR "
            "federal reserve OR Fed OR inflation OR CPI OR "
            "interest rates OR non-farm payrolls OR NFP OR geopolitical OR dollar OR DXY"
        ),
        "sources": "reuters,bloomberg,associated-press,the-wall-street-journal",
        "sources_weights": {
            "Reuters": 1.2,
            "Bloomberg": 1.1,
            "Associated Press": 1.0,
            "The Wall Street Journal": 1.3,
        },
        "max_pages": 3,
        "page_size": 100,
        "positive_threshold": 0.30,
        "negative_threshold": -0.30,
    },
    "output": {
        "technical_json": "technical_analysis.json",
        "news_json": "news_analysis.json",
        "combined_json": "combined_signal.json",
        "history_csv": "historical_signals.csv",
    },
    "logging": {"log_file": "gold_trading.log", "level": "INFO"},
    "run_backtest": False,
}


def load_config() -> dict:
    """
    1️⃣ تحميل ملف config.json إذا كان موجودًا
    2️⃣ إرجاع القيم الافتراضية إذا لم يُوجد الملف
    """
    cfg_path = "config.json"
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            logging.info("تم تحميل config.json")
            # دمج القيم الافتراضية مع ما تم قراءته لتفادي نقص المفاتيح
            merged = DEFAULT_CONFIG.copy()
            merged.update(cfg)
            return merged
        except Exception as e:
            logging.error(f"خطأ في قراءة config.json: {e}")
    return DEFAULT_CONFIG


def setup_logging(cfg: dict) -> None:
    log_file = cfg["logging"]["log_file"]
    level = getattr(logging, cfg["logging"]["level"].upper(), logging.INFO)
    logging.basicConfig(
        filename=log_file,
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # إظهار السجلات على الـ console في GitHub Actions
    logging.getLogger().addHandler(logging.StreamHandler())


# --------------------------- MARKET DATA ---------------------------
def download_market_data(tickers: dict, period: str, interval: str) -> pd.DataFrame:
    logging.info("جاري تحميل بيانات السوق من Yahoo Finance …")
    data = yf.download(
        list(tickers.values()),
        period=period,
        interval=interval,
        auto_adjust=False,
        threads=True,
        progress=False,
    )
    if data.empty:
        raise ValueError("البيانات المستلمة فارغة.")
    gold_key = ("Close", tickers["gold"])
    data.dropna(subset=[gold_key], inplace=True)
    if data.empty:
        raise ValueError("لا توجد بيانات صالحة للذهب بعد التنظيف.")
    logging.info(f"تم تحميل {len(data)} صفًا.")
    return data


# --------------------------- TECHNICAL ---------------------------
def prepare_gold_dataframe(data: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    t = cfg["tickers"]["gold"]
    gold = pd.DataFrame(
        {
            "Open": data[("Open", t)],
            "High": data[("High", t)],
            "Low": data[("Low", t)],
            "Close": data[("Close", t)],
            "Volume": data[("Volume", t)],
        }
    )

    tech_cfg = cfg["technical"]
    gold.ta.sma(length=tech_cfg["sma_fast"], append=True, col_names=("SMA_fast",))
    gold.ta.sma(length=tech_cfg["sma_slow"], append=True, col_names=("SMA_slow",))
    gold.ta.rsi(length=tech_cfg["rsi_length"], append=True)               # RSI_14
    gold.ta.macd(append=True)                                            # MACD_12_26_9, MACDs_12_26_9, MACDh_12_26_9
    gold.ta.bbands(append=True)                                          # BBU_20_2.0, BBM_20_2.0, BBL_20_2.0
    gold.ta.atr(length=tech_cfg["atr_length"], append=True)              # ATRr_14
    gold.ta.obv(append=True)                                             # OBV
    gold["OBV_SMA_20"] = ta.sma(gold["OBV"], length=tech_cfg["obv_sma"])

    # ---------- Pivot points ----------
    gold["Prev_High"] = gold["High"].shift(1)
    gold["Prev_Low"] = gold["Low"].shift(1)
    gold["Prev_Close"] = gold["Close"].shift(1)
    gold["Pivot"] = (gold["Prev_High"] + gold["Prev_Low"] + gold["Prev_Close"]) / 3
    gold["R1"] = 2 * gold["Pivot"] - gold["Prev_Low"]
    gold["S1"] = 2 * gold["Pivot"] - gold["Prev_High"]
    gold["R2"] = gold["Pivot"] + (gold["Prev_High"] - gold["Prev_Low"])
    gold["S2"] = gold["Pivot"] - (gold["Prev_High"] - gold["Prev_Low"])

    gold.dropna(inplace=True)
    logging.info("تم حساب جميع المؤشرات الفنية.")
    return gold


def evaluate_technical_signal(row: pd.Series, cfg: dict) -> dict:
    price = row["Close"]
    sma_fast = row["SMA_fast"]
    sma_slow = row["SMA_slow"]
    rsi = row["RSI_14"]
    macd_line = row["MACD_12_26_9"]
    macd_signal = row["MACDs_12_26_9"]
    atr = row["ATRr_14"]
    obv = row["OBV"]
    obv_sma = row["OBV_SMA_20"]
    upper_bb = row.get("BBU_20_2.0")
    lower_bb = row.get("BBL_20_2.0")
    pivot = row["Pivot"]
    r1 = row["R1"]
    s1 = row["S1"]

    # الاتجاه
    is_uptrend = price > sma_slow and sma_fast > sma_slow
    is_downtrend = price < sma_slow and sma_fast < sma_slow

    # MACD
    macd_bullish = macd_line > macd_signal
    macd_bearish = macd_line < macd_signal

    # RSI
    rsi_up = rsi >= cfg["technical"]["rsi_upper"]
    rsi_down = rsi <= cfg["technical"]["rsi_lower"]

    # OBV
    obv_confirm = obv > obv_sma

    # Bollinger
    price_overbought_bb = price > upper_bb if pd.notna(upper_bb) else False
    price_oversold_bb = price < lower_bb if pd.notna(lower_bb) else False

    # Pivot level checks
    price_above_r1 = price > r1
    price_below_s1 = price < s1

    # بناء الإشارة
    signal = "Neutral"
    if is_uptrend and macd_bullish and obv_confirm:
        signal = "Buy"
        if not rsi_up and not price_overbought_bb:
            signal = "Strong Buy"
    elif is_downtrend and macd_bearish:
        signal = "Sell"
        if rsi_down:
            signal = "Strong Sell"

    return {
        "signal": signal,
        "price": price,
        "sma_fast": sma_fast,
        "sma_slow": sma_slow,
        "rsi": rsi,
        "rsi_up": rsi_up,
        "rsi_down": rsi_down,
        "macd_bullish": macd_bullish,
        "macd_bearish": macd_bearish,
        "obv_confirm": obv_confirm,
        "price_overbought_bb": price_overbought_bb,
        "price_oversold_bb": price_oversold_bb,
        "price_above_r1": price_above_r1,
        "price_below_s1": price_below_s1,
        "atr": atr,
        "is_uptrend": is_uptrend,
        "is_downtrend": is_downtrend,
    }


def calculate_risk_management(price: float, atr: float, signal: str, cfg: dict) -> dict:
    mult = cfg["technical"]["atr_multiplier_sl"]
    rr = cfg["technical"]["risk_reward_ratio"]
    if signal in ("Buy", "Strong Buy"):
        sl = price - atr * mult
        risk = price - sl
        tp = price + risk * rr
    elif signal in ("Sell", "Strong Sell"):
        sl = price + atr * mult
        risk = sl - price
        tp = price - risk * rr
    else:
        sl, tp = None, None
    return {"stop_loss": round(sl, 2) if sl else None,
            "take_profit": round(tp, 2) if tp else None}


# --------------------------- NEWS & SENTIMENT ---------------------------
def fetch_all_news(cfg: dict) -> list:
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        logging.error("متغيّر البيئة NEWS_API_KEY غير موجود.")
        return []

    all_articles = []
    for page in range(1, cfg["sentiment"]["max_pages"] + 1):
        params = {
            "q": cfg["sentiment"]["keywords"],
            "sources": cfg["sentiment"]["sources"],
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": cfg["sentiment"]["page_size"],
            "page": page,
            "apiKey": api_key,
        }

        try:
            resp = requests.get("https://newsapi.org/v2/everything", params=params, timeout=12)
            resp.raise_for_status()
            data = resp.json()
            articles = data.get("articles", [])
            if not articles:
                logging.info(f"لا توجد مقالات في الصفحة {page}. إيقاف الجلب.")
                break
            for a in articles:
                all_articles.append({
                    "source": a["source"]["name"],
                    "title": a["title"],
                    "description": a["description"] or "",
                    "url": a["url"],
                    "publishedAt": a["publishedAt"]
                })
            logging.info(f"جلب {len(articles)} مقالات (صفحة {page}).")
        except Exception as e:
            logging.error(f"خطأ أثناء جلب الأخبار (صفحة {page}): {e}")
            break
    return all_articles


def analyze_sentiment(articles: list) -> list:
    if not articles:
        return []

    try:
        pipeline_sent = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    except Exception as e:
        logging.error(f"فشل تحميل نموذج FinBERT: {e}")
        return articles

    analyzed = []
    for art in tqdm(articles, desc="تحليل المشاعر"):
        text = art["description"] if art["description"] else art["title"]
        if not text:
            continue
        try:
            result = pipeline_sent(text[:512])[0]   # قص النص لتجنب حدّ الطول
            art["sentiment_label"] = result["label"].capitalize()   # Positive/Negative/Neutral
            art["sentiment_score"] = round(float(result["score"]), 4)
            analyzed.append(art)
        except Exception as e:
            logging.error(f"خطأ في تحليل {art['title']}: {e}")
    logging.info(f"تم تحليل مشاعر {len(analyzed)} مقالات.")
    return analyzed


def compute_weighted_sentiment(articles: list, cfg: dict) -> dict:
    if not articles:
        return {
            "overall_sentiment_score": 0.0,
            "sentiment_label": "Neutral",
            "article_counts": {"positive": 0, "negative": 0, "neutral": 0}
        }

    src_weights = cfg["sentiment"]["sources_weights"]
    total_w = 0.0
    cum_score = 0.0
    counts = {"positive": 0, "negative": 0, "neutral": 0}
    for art in articles:
        label = art.get("sentiment_label", "Neutral").lower()
        score = art.get("sentiment_score", 0.0)
        src_w = src_weights.get(art["source"], 1.0)
        # وزن يعتمد على طول النص (كل 100 حرف ≈ وزن 1)
        length_factor = max(len(art.get("description", "")) or len(art.get("title", "")), 1) / 100.0
        w = src_w * length_factor
        total_w += w
        if label == "positive":
            cum_score += w * score
            counts["positive"] += 1
        elif label == "negative":
            cum_score -= w * score
            counts["negative"] += 1
        else:
            counts["neutral"] += 1

    overall = cum_score / total_w if total_w != 0 else 0.0
    # تصنيف نهائي
    pos_th = cfg["sentiment"]["positive_threshold"]
    neg_th = cfg["sentiment"]["negative_threshold"]
    if overall >= pos_th:
        final_label = "Positive"
    elif overall <= neg_th:
        final_label = "Negative"
    else:
        final_label = "Neutral"

    return {
        "overall_sentiment_score": round(overall, 4),
        "sentiment_label": final_label,
        "article_counts": counts
    }


# --------------------------- COMBINATION ---------------------------
def combine_signals(tech: dict, sentiment: dict, cfg: dict) -> dict:
    tech_score = cfg["signal_mapping"].get(tech["signal"], 2)       # 2 = Neutral
    # نضرب الـ sentiment في 2 لتقريب النطاق إلى -2…+2
    sentiment_factor = sentiment["overall_sentiment_score"] * 2
    combined_score = tech_score + sentiment_factor

    if combined_score >= 4.5:
        final_decision = "Strong Buy"
    elif combined_score >= 3.5:
        final_decision = "Buy"
    elif combined_score <= 1.5:
        final_decision = "Strong Sell"
    elif combined_score <= 2.5:
        final_decision = "Sell"
    else:
        final_decision = "Neutral"

    return {
        "tech_score": tech_score,
        "sentiment_factor": round(sentiment_factor, 2),
        "combined_score": round(combined_score, 2),
        "final_decision": final_decision
    }


# --------------------------- HISTORY LOG ---------------------------
def append_history(record: dict, cfg: dict) -> None:
    csv_path = cfg["output"]["history_csv"]
    df = pd.DataFrame([record])
    header = not os.path.isfile(csv_path)
    df.to_csv(csv_path, mode="a", header=header, index=False, encoding="utf-8")
    logging.info(f"تم تحديث سجل التاريخ → {csv_path}")


# --------------------------- OPTIONAL BACK‑TEST (vectorbt) ---------------------------
def run_backtest(gold_df: pd.DataFrame, cfg: dict):
    try:
        import vectorbt as vbt
    except Exception:
        logging.warning("مكتبة vectorbt غير مثبتة → تخطي back‑testing.")
        return None

    gold_df["tech_signal"] = gold_df.apply(lambda r: evaluate_technical_signal(r, cfg)["signal"], axis=1)
    entries = gold_df["tech_signal"].isin(["Buy", "Strong Buy"])
    exits = gold_df["tech_signal"].isin(["Sell", "Strong Sell"])

    atr = gold_df["ATRr_14"]
    sl_series = gold_df["Close"] - atr * cfg["technical"]["atr_multiplier_sl"]

    pf = vbt.Portfolio.from_signals(
        gold_df["Close"], entries, exits,
        init_cash=100_000,
        fees=0.0005,
        sl_stop=sl_series,
        sl_stop_pct=False,
        direction="both"
    )
    logging.info("نتائج back‑testing:")
    logging.info(pf.stats())
    return pf


# --------------------------- MAIN ---------------------------
def main():
    # 0️⃣ التحميل الأولي
    cfg = load_config()
    setup_logging(cfg)

    logging.info("===== بدء تنفيذ تحليل الذهب =====")
    # 1️⃣ جلب بيانات السوق
    data = download_market_data(cfg["tickers"], cfg["period"], cfg["interval"])

    # 2️⃣ حساب المؤشرات الفنية
    gold_df = prepare_gold_dataframe(data, cfg)

    # 3️⃣ إشارة فنية لآخر صف (اليوم/التاريخ الأخير)
    latest = gold_df.iloc[-1]
    tech_signal = evaluate_technical_signal(latest, cfg)

    # 4️⃣ حساب وقف الخسارة / هدف الربح
    risk = calculate_risk_management(
        price=tech_signal["price"],
        atr=tech_signal["atr"],
        signal=tech_signal["signal"],
        cfg=cfg
    )

    # 5️⃣ جلب أخبار وتحليل المشاعر
    articles = fetch_all_news(cfg)
    articles_analyzed = analyze_sentiment(articles)
    sentiment = compute_weighted_sentiment(articles_analyzed, cfg)

    # 6️⃣ دمج الإشارتين
    combined = combine_signals(tech_signal, sentiment, cfg)

    # 7️⃣ تحضير الخرج العام
    now_est = datetime.now(pytz.timezone("America/New_York")).strftime("%Y-%m-%d %H:%M:%S")
    result = {
        "timestamp_est": now_est,
        "price_usd": round(tech_signal["price"], 2),
        "technical": {
            "signal": tech_signal["signal"],
            "details": {
                "trend": "Uptrend" if tech_signal["is_uptrend"] else ("Downtrend" if tech_signal["is_downtrend"] else "Sideways"),
                "macd": "Bullish" if tech_signal["macd_bullish"] else ("Bearish" if tech_signal["macd_bearish"] else "Neutral"),
                "rsi": round(tech_signal["rsi"], 2),
                "rsi_status": ("Overbought" if tech_signal["rsi_up"] else ("Oversold" if tech_signal["rsi_down"] else "Neutral")),
                "obv_confirm": tech_signal["obv_confirm"]
            },
            "risk_management": risk,
            "indicators": {
                "SMA_50": round(tech_signal["sma_fast"], 2) if not pd.isna(tech_signal["sma_fast"]) else None,
                "SMA_200": round(tech_signal["sma_slow"], 2) if not pd.isna(tech_signal["sma_slow"]) else None,
                "ATR": round(tech_signal["atr"], 2) if not pd.isna(tech_signal["atr"]) else None
            }
        },
        "sentiment": {
            "overall_score": sentiment["overall_sentiment_score"],
            "sentiment_label": sentiment["sentiment_label"],
            "article_counts": sentiment["article_counts"]
        },
        "combined_signal": {
            "final_decision": combined["final_decision"],
            "combined_score": combined["combined_score"]
        },
        "market_sentiment": {
            "vix": round(data[("Close", cfg["tickers"]["vix"])].iloc[-1], 2),
            "yield_10y": round(data[("Close", cfg["tickers"]["yield"])].iloc[-1], 2),
            "eur_usd": round(data[("Close", cfg["tickers"]["eurusd"])].iloc[-1], 4)
        }
    }

    # 8️⃣ حفظ الملفات
    out = cfg["output"]
    with open(out["technical_json"], "w", encoding="utf-8") as f:
        json.dump(result["technical"], f, ensure_ascii=False, indent=4)
    with open(out["news_json"], "w", encoding="utf-8") as f:
        json.dump(articles_analyzed, f, ensure_ascii=False, indent=2)
    with open(out["combined_json"], "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    logging.info("✅ حفظ جميع المخرجات ✅")

    # 9️⃣ تحديث سجل التاريخ (CSV)
    rec = {
        "timestamp_est": now_est,
        "price_usd": result["price_usd"],
        "technical_signal": tech_signal["signal"],
        "sentiment_label": sentiment["sentiment_label"],
        "overall_sentiment_score": sentiment["overall_sentiment_score"],
        "combined_score": combined["combined_score"],
        "final_decision": combined["final_decision"],
        "stop_loss": risk["stop_loss"],
        "take_profit": risk["take_profit"]
    }
    append_history(rec, cfg)

    # 🔟 (اختياري) تشغيل back‑testing
    if cfg.get("run_backtest", False):
        logging.info("🚀 تشغيل back‑testing …")
        run_backtest(gold_df, cfg)

    # 🖨️ عرض ملخص سريع
    summary = f"""
===== ملخص التحليل (الذهب) =====
التوقيت (EST)   : {now_est}
السعر الحالي     : ${result["price_usd"]}
الإشارة الفنية   : {tech_signal["signal"]}
الإشارة الخبرية : {sentiment["sentiment_label"]} (score {sentiment["overall_sentiment_score"]})
الإشارة المدمجة   : {combined["final_decision"]} (score {combined["combined_score"]})

وقف الخسارة      : {risk["stop_loss"] if risk["stop_loss"] else "N/A"}
هدف الربح        : {risk["take_profit"] if risk["take_profit"] else "N/A"}
---------------------------------
"""
    print(summary)
    logging.info("===== انتهى تشغيل النظام =====")


if __name__ == "__main__":
    # تحميل المتغيّرات البيئية (المفتاح NewsAPI)
    load_dotenv()
    main()
