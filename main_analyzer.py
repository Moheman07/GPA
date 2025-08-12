#!/usr/bin/env python3
"""
Professional Gold Analyzer – نسخة محسّنة

التحسينات المضمّنة (9 نقاط):
1️⃣  تحسين أداء جلب البيانات (caching + threads)
2️⃣  حساب المؤشرات التقنية باستخدام مكتبة `ta`
3️⃣  استخراج بيانات اقتصادية حقيقية عبر FRED (أو محاكاة إذا لم يتوفر المفتاح)
4️⃣  تحسين تحليل الأخبار باستخدام VADER + fallback عربي
5️⃣  توليد إشارات مُحسّنة مع وزن نقاط واضح
6️⃣  اختبار رجعي بسيط (Back‑testing) ومقاييس أداء
7️⃣  تقرير بصري (مخطط سعر + SMA/EMA + Bollinger)
8️⃣  تسجيل logging مفصّل بدلاً من `print`
9️⃣  تحسينات استقرار (معالجة NaN، أخطاء مدمجة، توثيق)

المتطلبات (pip install -r requirements.txt):
    yfinance pandas numpy requests python-dotenv joblib ta fredapi vaderSentiment matplotlib
"""

import os
import json
import warnings
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

import yfinance as yf
import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv
from joblib import Memory
from ta.trend import SMAIndicator, EMAIndicator, MACD, IchimokuIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from fredapi import Fred
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# --------------------------------------------------------------
# إعداد الـ logging
# --------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("GoldAnalyzer")

# قراءة المتغيّرات من .env (مفيد في GitHub Actions)
load_dotenv()

# مجلد التخزين المؤقت
CACHE_DIR = "./cache"
_memory = Memory(location=CACHE_DIR, verbose=0)


# --------------------------------------------------------------
# استثناء مخصص
# --------------------------------------------------------------
class GoldAnalyzerError(Exception):
    """Raised when a critical step of the analysis fails."""


# --------------------------------------------------------------
# الكلاس الأساسي (مستمر في الاستخدام كما هو)
# --------------------------------------------------------------
class ProfessionalGoldAnalyzer:
    """
    كلاس يدمج كل الخطوات:
        • جلب البيانات (متعدد الأطر)
        • استخراج بيانات الذهب
        • حساب مؤشرات تقنية (ta)
        • حساب مستويات الفيبوناتشي، الدعم/المقاومة
        • تحليل حجم التداول
        • تحليل ارتباطات الأصول الأخرى
        • جلب بيانات اقتصادية من FRED / محاكاة
        • جلب وتحليل أخبار (VADER + Arabic fallback)
        • توليد إشارة نهائية مع إدارة المخاطر
        • اختبار رجعي بسيط
        • تقرير بصري
        • حفظ نتائج JSON + تقرير نصي
    """

    def __init__(self) -> None:
        # رموز yfinance
        self.symbols = {
            "gold": "GC=F",
            "gold_etf": "GLD",
            "dxy": "DX-Y.NYB",
            "vix": "^VIX",
            "treasury": "^TNX",
            "oil": "CL=F",
            "spy": "SPY",
            "usdeur": "EURUSD=X",
            "silver": "SI=F",
        }

        # مفاتيح API
        self.news_api_key = os.getenv("NEWS_API_KEY")
        self.fred_api_key = os.getenv("FRED_API_KEY")
        self.fred = Fred(api_key=self.fred_api_key) if self.fred_api_key else None

        # محلل المشاعر (إنجليزي)
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

    # ------------------------------------------------------------------
    # 1️⃣ جلب البيانات (مع caching)
    # ------------------------------------------------------------------
    @_memory.cache
    def _download(self, symbols: List[str], period: str, interval: str) -> pd.DataFrame:
        """تنزيل بيانات yfinance وتخزينها مؤقتاً."""
        logger.info(
            "Downloading symbols=%s period=%s interval=%s", symbols, period, interval
        )
        df = yf.download(
            tickers=symbols,
            period=period,
            interval=interval,
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        if df.empty:
            raise GoldAnalyzerError("yfinance returned empty DataFrame")
        return df

    def fetch_multi_timeframe_data(self) -> Optional[Dict[str, pd.DataFrame]]:
        """جلب البيانات اليومية لسنة كاملة + بيانات الذهب على فاصل 1‑ساعة للشهر الأخير."""
        logger.info("📊 جلب بيانات متعددة الأطر الزمنية...")
        try:
            daily = self._download(list(self.symbols.values()), period="1y", interval="1d")
            hourly = self._download([self.symbols["gold"]], period="1mo", interval="1h")
            return {"daily": daily, "hourly": hourly}
        except Exception as exc:
            logger.exception("Failed to fetch market data")
            return None

    # ------------------------------------------------------------------
    # 2️⃣ استخراج بيانات الذهب
    # ------------------------------------------------------------------
    def extract_gold_data(self, market_data: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
        logger.info("🔍 استخراج بيانات الذهب...")
        try:
            daily = market_data["daily"]
            gold_sym = self.symbols["gold"]

            # إذا لم توجد بيانات العقود الفورية نلجئ للـ ETF
            if gold_sym not in daily.columns.levels[0] or daily[gold_sym]["Close"].dropna().empty:
                gold_sym = self.symbols["gold_etf"]
                if gold_sym not in daily.columns.levels[0] or daily[gold_sym]["Close"].dropna().empty:
                    raise GoldAnalyzerError("Gold data not found")

            gold = daily[gold_sym].copy()
            gold.dropna(subset=["Close"], inplace=True)

            if len(gold) < 200:
                raise GoldAnalyzerError("Insufficient gold history (<200 rows)")

            logger.info("✅ استخراج %d صف من بيانات الذهب", len(gold))
            return gold
        except Exception as exc:
            logger.exception("Failed to extract gold data")
            return None

    # ------------------------------------------------------------------
    # 3️⃣ حساب المؤشرات التقنية (باستخدام مكتبة `ta`)
    # ------------------------------------------------------------------
    def calculate_technical_indicators(self, gold_data: pd.DataFrame) -> pd.DataFrame:
        """إضافة جميع المؤشرات التقنية الضرورية."""
        logger.info("📈 حساب المؤشرات التقنية...")
        df = gold_data.copy()

        # ------------------- المتوسطات المتحركة -------------------
        df["SMA_10"] = SMAIndicator(df["Close"], window=10).sma_indicator()
        df["SMA_20"] = SMAIndicator(df["Close"], window=20).sma_indicator()
        df["SMA_50"] = SMAIndicator(df["Close"], window=50).sma_indicator()
        df["SMA_100"] = SMAIndicator(df["Close"], window=100).sma_indicator()
        df["SMA_200"] = SMAIndicator(df["Close"], window=200).sma_indicator()

        df["EMA_9"] = EMAIndicator(df["Close"], window=9).ema_indicator()
        df["EMA_21"] = EMAIndicator(df["Close"], window=21).ema_indicator()

        # ------------------- تقاطعات الذهب/الموت -------------------
        df["Golden_Cross"] = (df["SMA_50"] > df["SMA_200"]).astype(int)
        df["Death_Cross"] = (df["SMA_50"] < df["SMA_200"]).astype(int)

        # ------------------- RSI -------------------
        df["RSI"] = RSIIndicator(df["Close"], window=14).rsi()
        df["RSI_MA"] = df["RSI"].rolling(window=5).mean()

        # ------------------- MACD -------------------
        macd = MACD(df["Close"], window_slow=26, window_fast=12, window_sign=9)
        df["MACD"] = macd.macd()
        df["MACD_Signal"] = macd.macd_signal()
        df["MACD_Histogram"] = macd.macd_diff()
        df["MACD_Cross"] = np.where(df["MACD"] > df["MACD_Signal"], 1, -1)

        # ------------------- Bollinger Bands -------------------
        bb = BollingerBands(df["Close"], window=20, window_dev=2)
        df["BB_Upper"] = bb.bollinger_hband()
        df["BB_Lower"] = bb.bollinger_lband()
        df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["SMA_20"] * 100
        df["BB_Position"] = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"])

        # ------------------- ATR -------------------
        df["ATR"] = AverageTrueRange(df["High"], df["Low"], df["Close"], window=14).average_true_range()
        df["ATR_Percent"] = df["ATR"] / df["Close"] * 100

        # ------------------- حجم التداول -------------------
        df["Volume_SMA"] = df["Volume"].rolling(window=20).mean()
        df["Volume_Ratio"] = df["Volume"] / df["Volume_SMA"]

        # OBV – الاتجاه +1 إذا ارتفع السعر، -1 إذا هبط
        price_diff = df["Close"].diff()
        direction = np.sign(price_diff).replace(0, 1)  # treat flat as +1 لتجنب 0
        df["OBV"] = (df["Volume"] * direction).cumsum()
        df["Volume_Price_Trend"] = (df["Close"].pct_change().fillna(0) * df["Volume"]).cumsum()

        # ------------------- إضافية -------------------
        df["ROC"] = df["Close"].pct_change(periods=14) * 100
        df["Williams_R"] = WilliamsRIndicator(
            high=df["High"], low=df["Low"], close=df["Close"], lbp=14
        ).williams_r()

        # Stochastic
        stoch = StochasticOscillator(
            high=df["High"], low=df["Low"], close=df["Close"], window=14, smooth_window=3
        )
        df["Stoch_K"] = stoch.stoch()
        df["Stoch_D"] = stoch.stoch_signal()

        # Ichimoku Cloud
        ich = IchimokuIndicator(
            high=df["High"], low=df["Low"], window1=9, window2=26, window3=52
        )
        df["Tenkan_sen"] = ich.ichimoku_conversion_line()
        df["Kijun_sen"] = ich.ichimoku_base_line()
        df["Senkou_Span_A"] = ich.ichimoku_a()
        df["Senkou_Span_B"] = ich.ichimoku_b()

        logger.info("✅ جميع المؤشرات التقنية أُضيفت")
        return df.dropna()

    # ------------------------------------------------------------------
    # 4️⃣ مستويات الدعم والمقاومة
    # ------------------------------------------------------------------
    def calculate_support_resistance(
        self, data: pd.DataFrame, window: int = 20
    ) -> Dict[str, Any]:
        """حساب مستويات الدعم/المقاومة الديناميكية."""
        try:
            recent = data.tail(window * 3)

            # القمم والقيعان (نافذة 5 أيام)
            highs = recent["High"].rolling(5, center=True).max() == recent["High"]
            lows = recent["Low"].rolling(5, center=True).min() == recent["Low"]

            resistance_levels = recent.loc[highs, "High"].nlargest(3).astype(float).tolist()
            support_levels = recent.loc[lows, "Low"].nsmallest(3).astype(float).tolist()

            price = data["Close"].iloc[-1]

            # أقرب مستويات
            nearest_res = min([r for r in resistance_levels if r > price], default=None)
            nearest_sup = max([s for s in support_levels if s < price], default=None)

            return {
                "resistance_levels": [round(r, 2) for r in resistance_levels],
                "support_levels": [round(s, 2) for s in support_levels],
                "nearest_resistance": round(nearest_res, 2) if nearest_res else None,
                "nearest_support": round(nearest_sup, 2) if nearest_sup else None,
                "price_to_resistance": round(((nearest_res - price) / price * 100), 2)
                if nearest_res
                else None,
                "price_to_support": round(((price - nearest_sup) / price * 100), 2)
                if nearest_sup
                else None,
            }
        except Exception as exc:
            logger.exception("Support/Resistance calculation failed")
            return {}

    # ------------------------------------------------------------------
    # 5️⃣ مستويات فيبوناتشي
    # ------------------------------------------------------------------
    def calculate_fibonacci_levels(self, data: pd.DataFrame, periods: int = 50) -> Dict[str, Any]:
        """حساب مستويات الفيبوناتشي مع تحليل موضع السعر."""
        try:
            recent = data.tail(periods)
            high, low = recent["High"].max(), recent["Low"].min()
            diff = high - low
            price = data["Close"].iloc[-1]

            fibs = {
                "high": round(high, 2),
                "low": round(low, 2),
                "fib_23_6": round(high - diff * 0.236, 2),
                "fib_38_2": round(high - diff * 0.382, 2),
                "fib_50_0": round(high - diff * 0.5, 2),
                "fib_61_8": round(high - diff * 0.618, 2),
                "fib_78_6": round(high - diff * 0.786, 2),
            }

            # تحليل سريع للموضع
            if price > fibs["fib_23_6"]:
                analysis = "السعر فوق 23.6% - اتجاه صاعد قوي"
            elif price > fibs["fib_38_2"]:
                analysis = "السعر فوق 38.2% - اتجاه صاعد معتدل"
            elif price > fibs["fib_50_0"]:
                analysis = "السعر فوق 50% - منطقة محايدة"
            elif price > fibs["fib_61_8"]:
                analysis = "السعر فوق 61.8% - ضعف نسبي"
            else:
                analysis = "السعر تحت 61.8% - اتجاه هابط محتمل"

            fibs["analysis"] = analysis
            fibs["current_position"] = round(((price - low) / diff * 100), 2)
            return fibs
        except Exception as exc:
            logger.exception("Fibonacci calculation failed")
            return {}

    # ------------------------------------------------------------------
    # 6️⃣ البيانات الاقتصادية من FRED (أو محاكاة)
    # ------------------------------------------------------------------
    def fetch_economic_data(self) -> Dict[str, Any]:
        logger.info("📊 جلب البيانات الاقتصادية...")
        econ = {"status": "simulated", "last_update": datetime.now().isoformat(), "indicators": {}}
        if not self.fred:
            logger.warning("FRED API key غير موجود → إرجاع بيانات محاكاة")
            # بيانات محاكاة (نفسها في النسخة الأصلية)
            econ["indicators"] = {
                "US_CPI": {
                    "value": 3.2,
                    "previous": 3.4,
                    "impact": "إيجابي للذهب - تضخم منخفض",
                    "next_release": "2025-02-12",
                },
                "US_Interest_Rate": {
                    "value": 4.5,
                    "previous": 4.75,
                    "impact": "إيجابي للذهب - خفض الفائدة",
                    "next_release": "2025-01-29 FOMC",
                },
                "US_NFP": {
                    "value": 256000,
                    "previous": 227000,
                    "impact": "سلبي للذهب - سوق عمل قوي",
                    "next_release": "2025-02-07",
                },
                "DXY_Index": {
                    "value": 108.5,
                    "trend": "هابط",
                    "impact": "إيجابي للذهب - ضعف الدولار",
                },
                "Geopolitical_Risk": {
                    "level": "متوسط",
                    "events": ["توترات تجارية", "قلق من التضخم"],
                    "impact": "محايد إلى إيجابي للذهب",
                },
            }
        else:
            try:
                # CPI (المؤشر العام للمستهلكين)
                cpi_series = self.fred.get_series(
                    "CPILFESL", observation_start=datetime.now() - timedelta(days=60)
                )
                cpi = round(cpi_series.iloc[-1], 2)
                cpi_prev = round(cpi_series.iloc[-2], 2)

                # معدل الفائدة الفيدرالية
                fed_series = self.fred.get_series(
                    "FEDFUNDS", observation_start=datetime.now() - timedelta(days=60)
                )
                fed = round(fed_series.iloc[-1], 2)
                fed_prev = round(fed_series.iloc[-2], 2)

                # توظيف غير زراعي (NFP)
                nfp_series = self.fred.get_series(
                    "PAYEMS", observation_start=datetime.now() - timedelta(days=30)
                )
                nfp = int(nfp_series.iloc[-1])
                nfp_prev = int(nfp_series.iloc[-2])

                econ["status"] = "real"
                econ["indicators"] = {
                    "US_CPI": {
                        "value": cpi,
                        "previous": cpi_prev,
                        "impact": "إيجابي للذهب - تضخم منخفض"
                        if cpi < cpi_prev
                        else "سلبي للذهب - تضخم مرتفع",
                        "next_release": "2025-02-12",
                    },
                    "US_Interest_Rate": {
                        "value": fed,
                        "previous": fed_prev,
                        "impact": "إيجابي للذهب - خفض الفائدة"
                        if fed < fed_prev
                        else "سلبي للذهب - رفع الفائدة",
                        "next_release": "2025-01-29 FOMC",
                    },
                    "US_NFP": {
                        "value": nfp,
                        "previous": nfp_prev,
                        "impact": "إيجابي للذهب - انخفاض التوظيف"
                        if nfp < nfp_prev
                        else "سلبي للذهب - سوق عمل قوي",
                        "next_release": "2025-02-07",
                    },
                    "DXY_Index": {
                        "value": 108.5,
                        "trend": "هابط",
                        "impact": "إيجابي للذهب - ضعف الدولار",
                    },
                }

                # حساب التأثير الإجمالي
                pos = sum(
                    1 for i in econ["indicators"].values() if "إيجابي" in i["impact"]
                )
                neg = sum(
                    1 for i in econ["indicators"].values() if "سلبي" in i["impact"]
                )
                if pos > neg:
                    econ["overall_impact"] = "إيجابي للذهب"
                    econ["score"] = pos - neg
                elif neg > pos:
                    econ["overall_impact"] = "سلبي للذهب"
                    econ["score"] = pos - neg
                else:
                    econ["overall_impact"] = "محايد"
                    econ["score"] = 0

            except Exception as exc:
                logger.exception("Failed to fetch real economic data")
                econ["status"] = "error"
                econ["error"] = str(exc)

        return econ

    # ------------------------------------------------------------------
    # 7️⃣ جلب الأخبار وتحليل المشاعر
    # ------------------------------------------------------------------
    def fetch_news(self) -> Dict[str, Any]:
        logger.info("📰 جلب وتحليل أخبار الذهب...")
        if not self.news_api_key:
            logger.warning("NEWS_API_KEY غير مُعرَّف → تخطي جلب الأخبار")
            return {"status": "no_api_key", "message": "مفتاح NewsAPI غير موجود"}

        try:
            keywords = (
                '"gold price" OR "XAU/USD" OR "federal reserve interest" '
                'OR "US inflation" OR "FOMC meeting"'
            )
            url = (
                f"https://newsapi.org/v2/everything?"
                f"q={keywords}&language=en&sortBy=publishedAt&pageSize=30&apiKey={self.news_api_key}"
            )
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            articles = resp.json().get("articles", [])

            # كلمات المفتاح لتصنيف الأثر
            high_kw = [
                "federal reserve",
                "fed",
                "interest rate",
                "fomc",
                "inflation",
                "cpi",
                "employment",
                "nfp",
                "rate decision",
                "policy",
            ]
            med_kw = ["dollar", "dxy", "treasury", "geopolitical", "crisis", "war"]
            gold_kw = ["gold", "xau", "precious metal", "bullion"]

            categorized = {"critical": [], "high_impact": [], "medium_impact": [], "gold_specific": []}

            for art in articles:
                title = art.get("title", "")
                if not title:
                    continue
                description = art.get("description") or ""
                txt = f"{title} {description}".lower()

                item = {
                    "title": title[:150],
                    "source": art.get("source", {}).get("name", "Unknown"),
                    "published": art.get("publishedAt", ""),
                    "url": art.get("url", ""),
                    "impact": None,
                    "sentiment": None,
                }

                # ------------------- تقييم الأثر -------------------
                if any(kw in txt for kw in ["rate decision", "fomc decision", "emergency"]):
                    cat = "critical"
                elif any(kw in txt for kw in high_kw):
                    cat = "high_impact"
                elif any(kw in txt for kw in med_kw):
                    cat = "medium_impact"
                elif any(kw in txt for kw in gold_kw):
                    cat = "gold_specific"
                else:
                    cat = "medium_impact"

                # ------------------- تحليل المشاعر -------------------
                # VADER للإنجليزية
                vader_score = self.sentiment_analyzer.polarity_scores(txt)["compound"]
                if vader_score >= 0.05:
                    sentiment = "إيجابي"
                elif vader_score <= -0.05:
                    sentiment = "سلبي"
                else:
                    sentiment = "محايد"

                # fallback عربى (قائمة بسيطة)
                if not sentiment:
                    pos_ar = ["ارتفاع", "زيادة", "نمو", "قوة"]
                    neg_ar = ["هبوط", "انخفاض", "ضعف", "خسارة"]
                    pos_cnt = sum(p in txt for p in pos_ar)
                    neg_cnt = sum(n in txt for n in neg_ar)
                    sentiment = "إيجابي" if pos_cnt > neg_cnt else ("سلبي" if neg_cnt > pos_cnt else "محايد")

                item["impact"] = cat.replace("_", " ")
                item["sentiment"] = sentiment
                categorized[cat].append(item)

            # خلاصة الأخبار
            total = sum(len(v) for v in categorized.values())
            summary = {
                "total_relevant_news": total,
                "critical_count": len(categorized["critical"]),
                "high_impact_count": len(categorized["high_impact"]),
                "overall_sentiment": self._aggregate_news_sentiment(categorized),
            }

            return {
                "status": "success",
                "summary": summary,
                "categorized_news": {k: v[:3] for k, v in categorized.items() if v},
            }

        except Exception as exc:
            logger.exception("News fetching failed")
            return {"status": "error", "message": str(exc)}

    def _aggregate_news_sentiment(self, grouped: Dict[str, List[Dict[str, Any]]]) -> str:
        """مجموع المشاعر من جميع الأخبار المصنفة."""
        sentiments = [
            n["sentiment"]
            for lst in grouped.values()
            for n in lst
            if n.get("sentiment")
        ]
        if not sentiments:
            return "محايد"
        pos = sentiments.count("إيجابي")
        neg = sentiments.count("سلبي")
        if pos > neg:
            return "إيجابي للذهب"
        if neg > pos:
            return "سلبي للذهب"
        return "محايد"

    # ------------------------------------------------------------------
    # 8️⃣ تحليل الارتباطات مع الأصول
    # ------------------------------------------------------------------
    def analyze_correlations(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        logger.info("🔗 تحليل الارتباطات المتقدم...")
        try:
            daily = market_data["daily"]
            corrs, strength, interp = {}, {}, {}

            if hasattr(daily.columns, "levels"):
                assets = daily.columns.get_level_values(0).unique()
                gold_sym = (
                    self.symbols["gold"]
                    if self.symbols["gold"] in assets
                    else self.symbols["gold_etf"]
                )
                if gold_sym not in assets:
                    raise GoldAnalyzerError("Gold symbol not in daily data")

                gold_close = daily[gold_sym]["Close"].dropna()

                for name, sym in self.symbols.items():
                    if name in ["gold", "gold_etf"]:
                        continue
                    if sym not in assets:
                        continue
                    asset_close = daily[sym]["Close"].dropna()
                    idx = gold_close.index.intersection(asset_close.index)
                    if len(idx) < 30:
                        continue
                    corr = gold_close.loc[idx].corr(asset_close.loc[idx])
                    if pd.isna(corr):
                        continue
                    corr = round(corr, 3)
                    corrs[name] = corr

                    # قوة الارتباط
                    if abs(corr) > 0.7:
                        strength[name] = "قوي جداً"
                    elif abs(corr) > 0.5:
                        strength[name] = "قوي"
                    elif abs(corr) > 0.3:
                        strength[name] = "متوسط"
                    else:
                        strength[name] = "ضعيف"

                    # تفسير لبعض الأصول الأساسية
                    if name == "dxy":
                        if corr < -0.5:
                            interp[name] = "ارتباط عكسي قوي - إيجابي للذهب عند ضعف الدولار"
                        elif corr < -0.3:
                            interp[name] = "ارتباط عكسي معتدل - فرصة محتملة"
                        else:
                            interp[name] = "ارتباط ضعيف"
                    elif name == "vix":
                        interp[name] = "الذهب يستفيد من التقلبات" if corr > 0.3 else "تأثير محدود"
                    elif name == "oil":
                        interp[name] = "ارتباط قوي مع النفط - مؤشر تضخم" if abs(corr) > 0.5 else "ارتباط ضعيف"

            return {"correlations": corrs, "strength_analysis": strength, "interpretation": interp}
        except Exception as exc:
            logger.exception("Correlation analysis failed")
            return {}

    # ------------------------------------------------------------------
    # 9️⃣ تحليل حجم التداول (محسّن)
    # ------------------------------------------------------------------
    def analyze_volume_profile(self, data: pd.DataFrame) -> Dict[str, Any]:
        logger.info("📊 تحليل حجم التداول...")
        try:
            latest = data.iloc[-1]
            avg_5 = data["Volume"].tail(5).mean()
            avg_20 = data["Volume"].tail(20).mean()
            vol_ratio = latest.get("Volume_Ratio", 1)

            if vol_ratio > 2.0:
                strength, signal = "قوي جداً", "حجم استثنائي - احتمال حركة قوية"
            elif vol_ratio > 1.5:
                strength, signal = "قوي", "حجم فوق المتوسط - اهتمام متزايد"
            elif vol_ratio > 0.8:
                strength, signal = "طبيعي", "حجم طبيعي - لا إشارات خاصة"
            else:
                strength, signal = "ضعيف", "حجم ضعيف - حذر من الحركة الوهمية"

            obv_trend = "صاعد" if data["OBV"].iloc[-1] > data["OBV"].iloc[-5] else "هابط"

            return {
                "current_volume": int(latest.get("Volume", 0)),
                "avg_volume_5": int(avg_5),
                "avg_volume_20": int(avg_20),
                "volume_ratio": round(vol_ratio, 2),
                "volume_strength": strength,
                "volume_signal": signal,
                "obv_trend": obv_trend,
                "volume_price_correlation": "إيجابي"
                if (latest["Close"] > data["Close"].iloc[-2] and int(latest.get("Volume", 0)) > avg_20)
                else "سلبي",
            }
        except Exception as exc:
            logger.exception("Volume profile analysis failed")
            return {}

    # ------------------------------------------------------------------
    # 10️⃣ توليد الإشارة النهائية
    # ------------------------------------------------------------------
    def generate_professional_signals(
        self,
        tech_data: pd.DataFrame,
        correlations: Dict[str, Any],
        volume: Dict[str, Any],
        fib_levels: Dict[str, Any],
        support_resistance: Dict[str, Any],
        economic_data: Dict[str, Any],
        news_analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        logger.info("🎯 توليد الإشارات المتقدمة...")
        try:
            latest = tech_data.iloc[-1]
            prev = tech_data.iloc[-2]

            scores: Dict[str, float] = {
                "trend": 0,
                "momentum": 0,
                "volume": 0,
                "fibonacci": 0,
                "correlation": 0,
                "support_resistance": 0,
                "economic": 0,
                "news": 0,
                "ma_cross": 0,
            }

            # ---------- 1. الاتجاه ----------
            if latest["Close"] > latest["SMA_200"]:
                scores["trend"] += 2
                if latest["Close"] > latest["SMA_50"]:
                    scores["trend"] += 1
                    if latest["Close"] > latest["SMA_20"]:
                        scores["trend"] += 1
            else:
                scores["trend"] -= 2
                if latest["Close"] < latest["SMA_50"]:
                    scores["trend"] -= 1
                    if latest["Close"] < latest["SMA_20"]:
                        scores["trend"] -= 1

            # Golden / Death cross
            if latest.get("Golden_Cross", 0) == 1:
                scores["ma_cross"] = 3
            elif latest.get("Death_Cross", 0) == 1:
                scores["ma_cross"] = -3

            # ---------- 2. الزخم ----------
            if latest["MACD"] > latest["MACD_Signal"]:
                scores["momentum"] += 1
                if latest["MACD_Histogram"] > prev["MACD_Histogram"]:
                    scores["momentum"] += 1
            else:
                scores["momentum"] -= 1
                if latest["MACD_Histogram"] < prev["MACD_Histogram"]:
                    scores["momentum"] -= 1

            rsi = latest.get("RSI", 50)
            if 30 <= rsi <= 70:
                if 45 <= rsi <= 55:
                    scores["momentum"] += 0.5
                elif rsi > 55:
                    scores["momentum"] += 1
                else:
                    scores["momentum"] -= 0.5
            elif rsi < 30:
                scores["momentum"] += 2
            else:  # rsi > 70
                scores["momentum"] -= 2

            if latest.get("Stoch_K", 50) > latest.get("Stoch_D", 50):
                scores["momentum"] += 0.5

            # ---------- 3. الحجم ----------
            vol_str = volume.get("volume_strength", "")
            if vol_str == "قوي جداً":
                scores["volume"] = 3
            elif vol_str == "قوي":
                scores["volume"] = 2
            elif vol_str == "طبيعي":
                scores["volume"] = 0
            else:
                scores["volume"] = -1

            if volume.get("obv_trend") == "صاعد":
                scores["volume"] += 1

            # ---------- 4. فيبوناتشي ----------
            price = latest["Close"]
            if price > fib_levels.get("fib_38_2", 0):
                scores["fibonacci"] = 2
            elif price > fib_levels.get("fib_50_0", 0):
                scores["fibonacci"] = 1
            elif price > fib_levels.get("fib_61_8", 0):
                scores["fibonacci"] = -1
            else:
                scores["fibonacci"] = -2

            # ---------- 5. الدعم / المقاومة ----------
            if support_resistance.get("price_to_support") and support_resistance["price_to_support"] < 2:
                scores["support_resistance"] = 2
            elif support_resistance.get("price_to_resistance") and support_resistance["price_to_resistance"] < 2:
                scores["support_resistance"] = -2

            # ---------- 6. الارتباط ----------
            dxy_corr = correlations.get("correlations", {}).get("dxy", 0)
            if dxy_corr < -0.7:
                scores["correlation"] = 2
            elif dxy_corr < -0.5:
                scores["correlation"] = 1
            elif dxy_corr > 0.5:
                scores["correlation"] = -1

            # ---------- 7. البيانات الاقتصادية ----------
            econ_score = economic_data.get("score", 0)
            scores["economic"] = max(min(econ_score, 3), -3)

            # ---------- 8. الأخبار ----------
            if news_analysis.get("status") == "success":
                sentiment = news_analysis.get("summary", {}).get("overall_sentiment", "محايد")
                if sentiment == "إيجابي للذهب":
                    scores["news"] = 2
                elif sentiment == "سلبي للذهب":
                    scores["news"] = -2

                if news_analysis.get("summary", {}).get("critical_count", 0) > 0:
                    scores["news"] *= 2  # تضاعف تأثير الأخبار الحرجة

            # ---------- الوزن النهائي ----------
            weights = {
                "trend": 0.25,
                "momentum": 0.20,
                "volume": 0.15,
                "fibonacci": 0.10,
                "correlation": 0.05,
                "support_resistance": 0.10,
                "economic": 0.10,
                "news": 0.05,
                "ma_cross": 0.10,
            }

            total_score = sum(scores[k] * weights.get(k, 0) for k in scores)

            # ---------- الإشارة ----------
            if total_score >= 2.0:
                signal, confidence, action = "Strong Buy", "Very High", "شراء قوي - حجم كبير"
            elif total_score >= 1.0:
                signal, confidence, action = "Buy", "High", "شراء - حجم متوسط"
            elif total_score >= 0.3:
                signal, confidence, action = "Weak Buy", "Medium", "شراء حذر - حجم صغير"
            elif total_score <= -2.0:
                signal, confidence, action = "Strong Sell", "Very High", "بيع قوي - حجم كبير"
            elif total_score <= -1.0:
                signal, confidence, action = "Sell", "High", "بيع - حجم متوسط"
            elif total_score <= -0.3:
                signal, confidence, action = "Weak Sell", "Medium", "بيع حذر - حجم صغير"
            else:
                signal, confidence, action = "Hold", "Low", "انتظار - لا توجد إشارة واضحة"

            # ---------- إدارة المخاطر ----------
            atr = latest.get("ATR", latest["Close"] * 0.02)
            vol_percent = latest.get("ATR_Percent", 2)
            sl_mult = 1.5 if vol_percent < 1.5 else (2.0 if vol_percent < 2.5 else 2.5)

            risk = {
                "stop_loss_levels": {
                    "tight": round(latest["Close"] - (atr * sl_mult * 0.75), 2),
                    "conservative": round(latest["Close"] - (atr * sl_mult), 2),
                    "moderate": round(latest["Close"] - (atr * sl_mult * 1.5), 2),
                    "wide": round(latest["Close"] - (atr * sl_mult * 2), 2),
                },
                "profit_targets": {
                    "target_1": round(latest["Close"] + (atr * 1.5), 2),
                    "target_2": round(latest["Close"] + (atr * 3), 2),
                    "target_3": round(latest["Close"] + (atr * 5), 2),
                    "target_4": round(latest["Close"] + (atr * 8), 2),
                },
                "position_size_recommendation": self._calculate_position_size(
                    confidence, vol_percent
                ),
                "risk_reward_ratio": round(3 / sl_mult, 2),
                "max_risk_per_trade": "2%" if confidence in ["Very High", "High"] else "1%",
            }

            # ---------- استراتيجية الدخول ----------
            entry = self._generate_entry_strategy(scores, latest, support_resistance)

            # ---------- تجميع النتيجة ----------
            result = {
                "signal": signal,
                "confidence": confidence,
                "action_recommendation": action,
                "total_score": round(total_score, 2),
                "component_scores": scores,
                "current_price": round(latest["Close"], 2),
                "risk_management": risk,
                "entry_strategy": entry,
                "technical_summary": {
                    "rsi": round(latest.get("RSI", 0), 1),
                    "macd": round(latest.get("MACD", 0), 2),
                    "williams_r": round(latest.get("Williams_R", 0), 1),
                    "stoch_k": round(latest.get("Stoch_K", 0), 1),
                    "bb_position": round(latest.get("BB_Position", 0.5), 2),
                    "volume_ratio": round(latest.get("Volume_Ratio", 1), 2),
                },
                "key_levels": {
                    "sma_20": round(latest.get("SMA_20", 0), 2),
                    "sma_50": round(latest.get("SMA_50", 0), 2),
                    "sma_200": round(latest.get("SMA_200", 0), 2),
                    "bb_upper": round(latest.get("BB_Upper", 0), 2),
                    "bb_lower": round(latest.get("BB_Lower", 0), 2),
                },
            }
            return result

        except Exception as exc:
            logger.exception("Signal generation failed")
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # مساعد: حساب حجم المركز
    # ------------------------------------------------------------------
    def _calculate_position_size(self, confidence: str, volatility: float) -> str:
        if confidence == "Very High" and volatility < 2:
            return "كبير (75‑100% من رأس المال المخصص)"
        if confidence == "High" and volatility < 2.5:
            return "متوسط‑كبير (50‑75%)"
        if confidence == "High" or (confidence == "Medium" and volatility < 2):
            return "متوسط (25‑50%)"
        if confidence == "Medium":
            return "صغير (10‑25%)"
        return "صغير جداً (5‑10%) أو عدم الدخول"

    # ------------------------------------------------------------------
    # مساعد: استراتيجية الدخول
    # ------------------------------------------------------------------
    def _generate_entry_strategy(
        self, scores: Dict[str, float], latest: pd.Series, support_resistance: Dict[str, Any]
    ) -> Dict[str, Any]:
        strategy = {"entry_type": "", "entry_zones": [], "conditions": [], "warnings": []}

        if scores["trend"] > 2 and scores["momentum"] > 1:
            strategy["entry_type"] = "دخول قوي - السوق في اتجاه واضح"
            strategy["entry_zones"].append(f"دخول فوري عند {round(latest['Close'], 2)}")
        elif scores["support_resistance"] == 2:
            strategy["entry_type"] = "دخول من الدعم"
            if support_resistance.get("nearest_support"):
                strategy["entry_zones"].append(
                    f"انتظر ارتداد من {support_resistance['nearest_support']}"
                )
        elif scores["momentum"] < -1:
            strategy["warnings"].append("⚠️ ذروة شراء - انتظر تصحيح")
            strategy["entry_type"] = "انتظار تصحيح"
        else:
            strategy["entry_type"] = "دخول تدريجي"
            strategy["entry_zones"].append("قسّم الدخول على 2‑3 مراحل")

        if latest.get("RSI", 50) > 70:
            strategy["conditions"].append("انتظر RSI < 70")
        if latest.get("Volume_Ratio", 1) < 0.8:
            strategy["warnings"].append("⚠️ حجم ضعيف - تأكيد مطلوب")

        return strategy

    # ------------------------------------------------------------------
    # 11️⃣ اختبار رجعي بسيط (Back‑testing)
    # ------------------------------------------------------------------
    def _simple_backtest(self, technical_df: pd.DataFrame) -> Dict[str, Any]:
        logger.info("🔙 تشغيل اختبار رجعي مبسط...")
        try:
            df = technical_df.copy()
            # قاعدة: الدخول عندما SMA20 > SMA50 و RSI > 55 و MACD > 0
            df["position"] = 0
            long_cond = (df["SMA_20"] > df["SMA_50"]) & (df["RSI"] > 55) & (df["MACD"] > 0)
            df.loc[long_cond, "position"] = 1
            df["position"] = df["position"].ffill().fillna(0)

            df["return"] = df["Close"].pct_change().fillna(0)
            df["strategy_return"] = df["position"].shift(1).fillna(0) * df["return"]

            total_ret = (1 + df["strategy_return"]).prod() - 1
            days = (df.index[-1] - df.index[0]).days
            annual_ret = (1 + total_ret) ** (365 / days) - 1 if days > 0 else 0

            # max drawdown
            cum_ret = (1 + df["strategy_return"]).cumprod()
            rolling_max = cum_ret.cummax()
            drawdown = (cum_ret - rolling_max) / rolling_max
            max_dd = drawdown.min()

            # win rate
            wins = (df["strategy_return"] > 0).sum()
            trades = df["position"].diff().abs().sum()  # عدد مرات الدخول/الخروج
            win_rate = wins / trades if trades > 0 else 0

            # Sharpe (risk‑free = 0)
            sharpe = (
                df["strategy_return"].mean()
                / df["strategy_return"].std()
                * np.sqrt(252)
                if df["strategy_return"].std() != 0
                else 0
            )

            return {
                "total_return_%": round(total_ret * 100, 2),
                "annualized_return_%": round(annual_ret * 100, 2),
                "max_drawdown_%": round(max_dd * 100, 2),
                "win_rate_%": round(win_rate * 100, 2),
                "sharpe_ratio": round(sharpe, 2),
                "trading_days": days,
                "total_trades": int(trades),
            }
        except Exception as exc:
            logger.exception("Back‑test failed")
            return {}

    # ------------------------------------------------------------------
    # 12️⃣ تقرير بصري (مخطط السعر + مؤشرات)
    # ------------------------------------------------------------------
    def generate_visual_report(self, technical_df: pd.DataFrame, timestamp: str) -> str:
        logger.info("🖼️ إنشاء تقرير بصري...")
        try:
            plt.style.use("seaborn-darkgrid")
            fig, ax = plt.subplots(figsize=(12, 6))

            ax.plot(
                technical_df.index,
                technical_df["Close"],
                label="Close",
                color="black",
                linewidth=1,
            )
            ax.plot(
                technical_df.index,
                technical_df["SMA_20"],
                label="SMA 20",
                color="blue",
                linewidth=0.8,
            )
            ax.plot(
                technical_df.index,
                technical_df["SMA_50"],
                label="SMA 50",
                color="orange",
                linewidth=0.8,
            )
            ax.plot(
                technical_df.index,
                technical_df["EMA_9"],
                label="EMA 9",
                color="green",
                linewidth=0.8,
            )
            ax.plot(
                technical_df.index,
                technical_df["BB_Upper"],
                label="BB Upper",
                color="gray",
                linewidth=0.5,
                linestyle="--",
            )
            ax.plot(
                technical_df.index,
                technical_df["BB_Lower"],
                label="BB Lower",
                color="gray",
                linewidth=0.5,
                linestyle="--",
            )

            ax.set_title(f"Gold Price Chart – {timestamp[:10]}")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price (USD)")
            ax.legend()
            fig.autofmt_xdate()

            img_path = f"gold_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.tight_layout()
            plt.savefig(img_path, dpi=150)
            plt.close(fig)

            logger.info("🖼️ تم حفظ المخطط في %s", img_path)
            return img_path
        except Exception as exc:
            logger.exception("Visual report generation failed")
            return ""

    # ------------------------------------------------------------------
    # 13️⃣ مرحلة التحليل الكاملة (run_analysis)
    # ------------------------------------------------------------------
    def run_analysis(self) -> Dict[str, Any]:
        logger.info("🚀 بدء التحليل الاحترافي للذهب...")
        try:
            # 1️⃣ جلب البيانات
            market_data = self.fetch_multi_timeframe_data()
            if market_data is None:
                raise GoldAnalyzerError("Data download failed")

            # 2️⃣ استخراج الذهب
            gold_daily = self.extract_gold_data(market_data)
            if gold_daily is None:
                raise GoldAnalyzerError("Gold extraction failed")

            # 3️⃣ حساب المؤشرات التقنية
            tech_df = self.calculate_technical_indicators(gold_daily)

            # 4️⃣ فيبوناتشي + دعم/مقاومة
            fib_levels = self.calculate_fibonacci_levels(tech_df)
            sr_levels = self.calculate_support_resistance(tech_df)

            # 5️⃣ تحليل الحجم
            vol_analysis = self.analyze_volume_profile(tech_df)

            # 6️⃣ ارتباطات
            corr = self.analyze_correlations(market_data)

            # 7️⃣ بيانات اقتصادية
            econ = self.fetch_economic_data()

            # 8️⃣ أخبار
            news = self.fetch_news()

            # 9️⃣ إشارة نهائية
            signal = self.generate_professional_signals(
                tech_df, corr, vol_analysis, fib_levels, sr_levels, econ, news
            )

            # 🔟 اختبار رجعي
            backtest = self._simple_backtest(tech_df)

            # 📈 تقرير بصري
            chart_path = self.generate_visual_report(tech_df, datetime.now().isoformat())

            # تجميع النتيجة النهائية
            final_result = {
                "timestamp": datetime.now().isoformat(),
                "status": "success",
                "gold_analysis": signal,
                "fibonacci_levels": fib_levels,
                "support_resistance": sr_levels,
                "volume_analysis": vol_analysis,
                "market_correlations": corr,
                "economic_data": econ,
                "news_analysis": news,
                "backtest": backtest,
                "visual_report_path": chart_path,
                "market_summary": {
                    "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "data_points": len(tech_df),
                    "timeframe": "Daily",
                    "market_condition": self._determine_market_condition(signal, vol_analysis),
                },
            }

            # حفظ النتائج (JSON + تقرير نصي)
            self.save_results(final_result)

            # طباعة تقرير نصي مبسّط
            print(self.generate_report(final_result))

            logger.info("✅ التحليل مكتمل بنجاح")
            return final_result

        except Exception as exc:
            logger.exception("تحليل فشل")
            err_res = {
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "error": str(exc),
            }
            self.save_results(err_res)
            return err_res

    # ------------------------------------------------------------------
    # مساعدة: تحديد حالة السوق العامة
    # ------------------------------------------------------------------
    def _determine_market_condition(self, signals: Dict[str, Any], volume: Dict[str, Any]) -> str:
        if signals.get("signal") in ["Strong Buy", "Buy"] and volume.get(
            "volume_strength"
        ) in ["قوي", "قوي جداً"]:
            return "صاعد قوي"
        if signals.get("signal") in ["Strong Sell", "Sell"] and volume.get(
            "volume_strength"
        ) in ["قوي", "قوي جداً"]:
            return "هابط قوي"
        if signals.get("signal") == "Hold":
            return "عرضي/محايد"
        return "متقلب"

    # ------------------------------------------------------------------
    # توليد تقرير نصي شامل
    # ------------------------------------------------------------------
    def generate_report(self, analysis_result: Dict[str, Any]) -> str:
        try:
            lines = []
            lines.append("=" * 60)
            lines.append("📊 تقرير التحليل الاحترافي للذهب")
            lines.append("=" * 60)
            lines.append(f"التاريخ: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append("")

            if "gold_analysis" in analysis_result:
                ga = analysis_result["gold_analysis"]
                lines.append("🎯 الإشارة الرئيسية:")
                lines.append(f"  • الإشارة: {ga.get('signal', 'N/A')}")
                lines.append(f"  • الثقة: {ga.get('confidence', 'N/A')}")
                lines.append(f"  • التوصية: {ga.get('action_recommendation', 'N/A')}")
                lines.append(f"  • السعر الحالي: ${ga.get('current_price', 'N/A')}")
                lines.append(f"  • النقاط الإجمالية: {ga.get('total_score', 'N/A')}")
                lines.append("")

                if "component_scores" in ga:
                    lines.append("📈 تفاصيل المكوّنات:")
                    for comp, score in ga["component_scores"].items():
                        lines.append(f"  • {comp}: {score}")
                    lines.append("")

                if "risk_management" in ga:
                    rm = ga["risk_management"]
                    lines.append("⚠️ إدارة المخاطر:")
                    lines.append(
                        f"  • وقف الخسارة المحافظ: ${rm['stop_loss_levels'].get('conservative','N/A')}"
                    )
                    lines.append(f"  • هدف 1: ${rm['profit_targets'].get('target_1','N/A')}")
                    lines.append(f"  • حجم المركز: {rm.get('position_size_recommendation','N/A')}")
                    lines.append("")

                if "entry_strategy" in ga:
                    es = ga["entry_strategy"]
                    lines.append("🚪 استراتيجية الدخول:")
                    lines.append(f"  • النوع: {es.get('entry_type','')}")
                    if es.get("entry_zones"):
                        lines.append(f"  • مناطق الدخول: {', '.join(es['entry_zones'])}")
                    if es.get("conditions"):
                        lines.append("  • الشروط:")
                        for c in es["conditions"]:
                            lines.append(f"    - {c}")
                    if es.get("warnings"):
                        lines.append("  • التحذيرات:")
                        for w in es["warnings"]:
                            lines.append(f"    - {w}")
                    lines.append("")

            if "economic_data" in analysis_result:
                ed = analysis_result["economic_data"]
                lines.append("💰 البيانات الاقتصادية:")
                lines.append(f"  • التأثير الإجمالي: {ed.get('overall_impact','N/A')}")
                if "indicators" in ed:
                    for name, val in ed["indicators"].items():
                        lines.append(f"  • {name}: {val.get('value','N/A')} - {val.get('impact','')}")
                lines.append("")

            if "news_analysis" in analysis_result:
                na = analysis_result["news_analysis"]
                if na.get("status") == "success":
                    sm = na["summary"]
                    lines.append("📰 ملخص الأخبار:")
                    lines.append(f"  • المشاعر العامة: {sm.get('overall_sentiment','N/A')}")
                    lines.append(f"  • أخبار حرجة: {sm.get('critical_count',0)}")
                    lines.append(f"  • أخبار عالية التأثير: {sm.get('high_impact_count',0)}")
                    lines.append("")

            if "backtest" in analysis_result and analysis_result["backtest"]:
                bt = analysis_result["backtest"]
                lines.append("🔙 اختبار رجعي مبسط:")
                for k, v in bt.items():
                    lines.append(f"  • {k.replace('_',' ').title()}: {v}")
                lines.append("")

            if "visual_report_path" in analysis_result and analysis_result["visual_report_path"]:
                lines.append(f"📈 مخطط السعر المحفوظ في: {analysis_result['visual_report_path']}")
                lines.append("")

            lines.append("=" * 60)
            lines.append("انتهى التقرير")
            return "\n".join(lines)
        except Exception as exc:
            logger.exception("Report generation failed")
            return f"Error generating report: {exc}"

    # ------------------------------------------------------------------
    # حفظ النتائج (JSON + أرشفة + تقرير نصي)
    # ------------------------------------------------------------------
    def save_results(self, results: Dict[str, Any]) -> None:
        try:
            main_path = "gold_analysis.json"
            with open(main_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            logger.info("💾 تم حفظ النتائج في %s", main_path)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_path = f"gold_analysis_{ts}.json"
            with open(archive_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            logger.info("📁 نسخة مؤرشفة محفوظة في %s", archive_path)

            if results.get("status") == "success":
                txt_path = f"gold_report_{ts}.txt"
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(self.generate_report(results))
                logger.info("📄 تم حفظ التقرير النصي في %s", txt_path)

        except Exception as exc:
            logger.exception("Failed to save results")


# ----------------------------------------------------------------------
# Entrypoint (بدون تغيير)
# ----------------------------------------------------------------------
def main():
    analyzer = ProfessionalGoldAnalyzer()
    analyzer.run_analysis()


if __name__ == "__main__":
    main()