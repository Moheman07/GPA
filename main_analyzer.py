#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ProfessionalGoldAnalyzer - نسخة محسنة
ميزات مضافة:
- مصدر سعر الأونصة: XAUUSD=X
- تصفية مقالات بالأسماء (zero-shot) قبل تحليل المشاعر
- تحليل مشاعر دفعي (batch) عبر FinBERT إن أمكن
- تطبيع مكونات الدرجات قبل الدمج
- حفظ إشارات يومية و backtest بسيط
- مخرجات: gold_analysis.json, historical_signals.csv, backtest_report.json
"""
import os
import json
from datetime import datetime, timedelta
import requests
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta

# transformers يستخدم للنماذج (zero-shot + finbert)
from transformers import pipeline, Pipeline, AutoTokenizer, AutoModelForSequenceClassification

warnings.filterwarnings("ignore")


class ProfessionalGoldAnalyzerV2:
    def __init__(self,
                 lookback_days=365,
                 news_days=2,
                 news_api_key=None,
                 save_path=".",
                 batch_size=16):
        self.symbols = {
            'gold': 'XAUUSD=X',   # سعر الأونصة بالدولار
            'dxy': 'DX-Y.NYB',
            'vix': '^VIX',
            'treasury': '^TNX',
            'oil': 'CL=F',
            'spy': 'SPY'
        }
        self.lookback_days = lookback_days
        self.news_days = news_days
        self.news_api_key = news_api_key or os.getenv("NEWS_API_KEY")
        self.save_path = save_path
        self.batch_size = batch_size

        # النماذج: zero-shot لتحديد الصلة، و finbert لتحليل المشاعر
        self.zero_shot = None
        self.sentiment = None
        self._load_models()

    def _load_models(self):
        try:
            print("🧠 تحميل نموذج zero-shot (NLI) لتصفية الأخبار...")
            # يستخدم MNLI-based model عبر pipeline zero-shot-classification
            self.zero_shot = pipeline("zero-shot-classification",
                                      model="facebook/bart-large-mnli")
            print("✅ zero-shot جاهز.")
        except Exception as e:
            print(f"⚠️ فشل تحميل zero-shot: {e}. سيتم استخدام فلترة كلمات مفتاحية بديلة.")
            self.zero_shot = None

        try:
            print("🧠 تحميل FinBERT لتحليل المشاعر المالية (batch)...")
            # ProsusAI/finbert قد يكون متاحًا؛ كبديل نستخدم model عام إن فشل
            self.sentiment = pipeline("sentiment-analysis",
                                      model="ProsusAI/finbert")
            print("✅ FinBERT جاهز.")
        except Exception as e:
            print(f"⚠️ فشل تحميل FinBERT: {e}. محاولة تحميل نموذج عام للتحليل.")
            try:
                self.sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
                print("✅ نموذج بديل جاهز.")
            except Exception as e2:
                print(f"❌ فشل تحميل أي نموذج للمشاعر: {e2}")
                self.sentiment = None

    def fetch_market_data(self):
        print("\n📊 جلب بيانات السوق من Yahoo Finance...")
        try:
            symbols = list(self.symbols.values())
            data = yf.download(symbols, period=f"{self.lookback_days}d", interval="1d", progress=False)
            if data.empty:
                raise ValueError("البيانات الفارغة من Yahoo Finance.")
            print(f"... نجح جلب {len(data)} صفوف من البيانات.")
            return data
        except Exception as e:
            print(f"❌ خطأ في جلب بيانات السوق: {e}")
            return None

    def fetch_news(self):
        """
        جلب الأخبار من NewsAPI (إن كان متاحًا).
        ترجع قائمة مقالات (title, description, source, publishedAt).
        """
        print("\n📰 جلب الأخبار من NewsAPI...")
        if not self.news_api_key:
            print("⚠️ مفتاح NewsAPI غير موجود، سيتم تخطي جلب الأخبار.")
            return []

        query = ('gold OR XAU OR bullion OR "precious metal" OR "gold price" '
                 'OR "interest rate" OR fed OR inflation OR CPI OR NFP OR geopolitical')
        from_date = (datetime.utcnow() - timedelta(days=self.news_days)).date()
        url = ("https://newsapi.org/v2/everything"
               f"?q={requests.utils.quote(query)}&language=en&sortBy=publishedAt&pageSize=100"
               f"&from={from_date}&apiKey={self.news_api_key}")

        try:
            res = requests.get(url, timeout=20)
            res.raise_for_status()
            articles = res.json().get("articles", [])
            simple = []
            for a in articles:
                simple.append({
                    "title": a.get("title"),
                    "description": a.get("description"),
                    "content": a.get("content"),
                    "source": a.get("source", {}).get("name"),
                    "publishedAt": a.get("publishedAt")
                })
            print(f"... تم جلب {len(simple)} مقالة من NewsAPI.")
            return simple
        except Exception as e:
            print(f"❌ خطأ في جلب الأخبار: {e}")
            return []

    def filter_relevant_articles(self, articles):
        """
        فلترة المقالات باستخدام zero-shot classification عندما يكون متاحًا،
        وإلا استخدام كلمات مفتاحية بسيطة.
        """
        print("\n🔎 فلترة المقالات ذات الصلة بالذهب...")
        if not articles:
            return []

        candidates = []
        labels = ["gold", "economy", "geopolitics", "other"]
        for art in articles:
            text = ((art.get("title") or "") + " " + (art.get("description") or "")).strip()
            if not text:
                continue
            is_relevant = False
            score = 0
            if self.zero_shot:
                try:
                    out = self.zero_shot(text, candidate_labels=labels, multi_label=False)
                    # نعتبرها مرتبطة إذا كانت الفئة "gold" أعلى من عتبة
                    if out and out.get("labels"):
                        if out["labels"][0] == "gold" and out["scores"][0] >= 0.45:
                            is_relevant = True
                            score = float(out["scores"][0])
                except Exception:
                    # في حال فشل نموذج NLI نتخطى للاحتياط
                    is_relevant = False

            if not self.zero_shot:
                # فلترة كلمات مفتاحية ثانوية
                kw = ["gold", "xau", "bullion", "precious metal", "troy ounce", "spot gold"]
                lower = text.lower()
                if any(k in lower for k in kw):
                    is_relevant = True
                    score = 0.5

            if is_relevant:
                art["_relevance_score"] = score
                candidates.append(art)

        print(f"... {len(candidates)} مقالات بعد الفلترة.")
        return sorted(candidates, key=lambda x: x.get("_relevance_score", 0), reverse=True)

    def analyze_sentiment_batch(self, articles):
        """
        تحليل المشاعر دفعيًا: استلام قائمة مقالات مرشحة وإرجاع متوسط النتيجة.
        نحول نتائج النماذج إلى مقياس موحد: pos -> +score, neg -> -score, neutral -> 0
        """
        print("\n🧾 تحليل المشاعر (دفعي)...")
        if not articles or not self.sentiment:
            print("⚠️ لا توجد مقالات/أو نموذج مشاعر غير متاح — سيتم إرجاع 0.")
            return {"status": "skipped", "news_score": 0, "headlines": []}

        texts = []
        for a in articles:
            txt = (a.get("description") or a.get("title") or "")
            texts.append(txt)

        results = []
        try:
            # batch call support
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                res = self.sentiment(batch)
                results.extend(res)
        except Exception as e:
            print(f"⚠️ خطأ أثناء تحليل المشاعر دفعيًا: {e}")
            # محاولة تحليل عنصر واحد واحد
            results = []
            for t in texts:
                try:
                    out = self.sentiment(t)[0]
                    results.append(out)
                except Exception:
                    results.append({"label": "NEUTRAL", "score": 0.0})

        # تحويل الملصقات إلى مقياس موحد
        numeric = []
        headlines = []
        for art, r in zip(articles, results):
            lbl = r.get("label", "").lower()
            sc = float(r.get("score", 0.0))
            val = 0.0
            # ملاحظة: مسميات النماذج قد تكون "POSITIVE"/"NEGATIVE"/"NEUTRAL" أو "LABEL_0"...
            if "pos" in lbl or lbl == "positive":
                val = sc
            elif "neg" in lbl or lbl == "negative":
                val = -sc
            else:
                val = 0.0
            numeric.append(val)
            headlines.append({
                "title": art.get("title"),
                "source": art.get("source"),
                "relevance": art.get("_relevance_score", 0),
                "sentiment": round(val, 4)
            })

        # نحسب المتوسط المرجّح حسب relevance
        relevances = np.array([a.get("_relevance_score", 0.5) for a in articles], dtype=float)
        vals = np.array(numeric, dtype=float)
        if relevances.sum() == 0:
            news_score = float(vals.mean()) if len(vals) else 0.0
        else:
            news_score = float(np.sum(vals * relevances) / (relevances.sum()))

        news_score = round(news_score, 4)
        print(f"... news_score = {news_score} (مرجّح حسب الصلة)")
        return {"status": "success", "news_score": news_score, "headlines": headlines[:10]}

    def compute_indicators(self, gold_df):
        """
        حساب مؤشرات فنية أساسية باستخدام pandas_ta
        """
        ta_strategy = ta.Strategy(name="FullAnalysis", ta=[
            {"kind": "sma", "length": 50}, {"kind": "sma", "length": 200},
            {"kind": "rsi"}, {"kind": "macd"}, {"kind": "bbands"},
            {"kind": "atr"}, {"kind": "obv"}
        ])
        gold_df.ta.strategy(ta_strategy)
        return gold_df

    @staticmethod
    def normalize_component(val, min_val=-2, max_val=2):
        """
        تطبيع قيمة إلى نطاق [-1, 1]
        نفترض أن val يقع ضمن [min_val, max_val]
        """
        # clip to avoid extreme
        v = max(min(val, max_val), min_val)
        # scale to [-1,1]
        return 2 * (v - min_val) / (max_val - min_val) - 1

    def score_components(self, latest_row, market_data, news_score):
        """
        توليد قيم مبدئية ثم تطبيعها:
        - trend_score: -1 (strong down) .. +1 (strong up)
        - momentum_score: -1 .. +1
        - correlation_score: -1 .. +1
        - news_score: مفترض في [-1,1] بالفعل من تحليل المشاعر
        """
        # Trend: مقارنة السعر بالنسبة لـ SMA200 و SMA50
        trend_val = 0.0
        try:
            close = latest_row["Close"]
            sma200 = latest_row.get("SMA_200", np.nan)
            sma50 = latest_row.get("SMA_50", np.nan)
            if not np.isnan(sma200) and not np.isnan(sma50):
                # نسبة المسافة إلى SMA200 (مقسومة على SMA200) مضروبة بمقياس 2
                diff = (close - sma200) / sma200
                trend_val = diff * 10  # مقياس مبدئي، سيتم تطبيعه لاحقًا
        except Exception:
            trend_val = 0.0

        # Momentum: استخدام MACD histogram إن وجد
        momentum_val = 0.0
        try:
            macd_hist = latest_row.get("MACDh_12_26_9", np.nan)
            if not np.isnan(macd_hist):
                momentum_val = macd_hist / max(abs(macd_hist) + 1e-9, 1) * 2
        except Exception:
            momentum_val = 0.0

        # Correlation: ارتباط سعر الذهب مع DXY (عادة سالب)
        correlation_val = 0.0
        try:
            gold_close = market_data[('Close', self.symbols['gold'])]
            dxy_close = market_data[('Close', self.symbols['dxy'])]
            corr = gold_close.corr(dxy_close)
            # نحوّل الارتباط لمقياس حيث -1 => +1 (لأن الارتباط السلبي للذهب مع الدولار يعزز الذهب)
            correlation_val = -corr if not np.isnan(corr) else 0.0
        except Exception:
            correlation_val = 0.0

        # تطبيع كل قيمة إلى [-1,1]
        trend_norm = self.normalize_component(trend_val, min_val=-2, max_val=2)
        momentum_norm = self.normalize_component(momentum_val, min_val=-2, max_val=2)
        correlation_norm = self.normalize_component(correlation_val, min_val=-1, max_val=1)
        news_norm = max(min(news_score, 1.0), -1.0)

        components = {
            "trend_score_raw": round(trend_val, 4),
            "momentum_score_raw": round(momentum_val, 4),
            "correlation_score_raw": round(correlation_val, 4),
            "trend_score": round(trend_norm, 4),
            "momentum_score": round(momentum_norm, 4),
            "correlation_score": round(correlation_norm, 4),
            "news_score": round(news_norm, 4)
        }

        return components

    def decide_signal(self, comps, thresholds=None):
        """
        قرار نهائي: نجمع المكونات مع أوزان قصيرة قابلة للتعديل.
        الأوزان:
         - trend 40%
         - momentum 30%
         - correlation 20%
         - news 10%
        نحسب total ∈ [-1,1]. نحدد إشارة:
         - Buy إذا >= 0.4
         - Sell إذا <= -0.4
         - Else Hold
        """
        if thresholds is None:
            thresholds = {"buy": 0.4, "sell": -0.4}
        w = {"trend": 0.4, "momentum": 0.3, "correlation": 0.2, "news": 0.1}
        total = (comps["trend_score"] * w["trend"] +
                 comps["momentum_score"] * w["momentum"] +
                 comps["correlation_score"] * w["correlation"] +
                 comps["news_score"] * w["news"])
        if total >= thresholds["buy"]:
            sig = "Buy"
        elif total <= thresholds["sell"]:
            sig = "Sell"
        else:
            sig = "Hold"
        return round(total, 4), sig

    def save_json(self, filename, data):
        path = os.path.join(self.save_path, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"💾 حفظ: {path}")

    def run_full_analysis(self):
        market_data = self.fetch_market_data()
        if market_data is None:
            return {"status": "error", "error": "market fetch failed"}

        # جهز DataFrame للذهب
        gold_ticker = self.symbols["gold"]
        gold_df = pd.DataFrame({
            "Open": market_data[("Open", gold_ticker)],
            "High": market_data[("High", gold_ticker)],
            "Low": market_data[("Low", gold_ticker)],
            "Close": market_data[("Close", gold_ticker)],
            "Volume": market_data[("Volume", gold_ticker)]
        }).dropna()

        if gold_df.empty:
            return {"status": "error", "error": "no gold data"}

        # مؤشرات فنية
        gold_df = self.compute_indicators(gold_df)
        gold_df.dropna(inplace=True)

        # جلب الأخبار وفلترتها
        articles = self.fetch_news()
        relevant = self.filter_relevant_articles(articles)
        news_result = self.analyze_sentiment_batch(relevant)

        # تحضير السجل النهائي لليوم الأخير
        latest = gold_df.iloc[-1]
        comps = self.score_components(latest, market_data, news_result.get("news_score", 0))
        total_score, signal = self.decide_signal(comps)

        result = {
            "timestamp_utc": datetime.utcnow().isoformat(),
            "signal": signal,
            "total_score": total_score,
            "components": comps,
            "market_data": {
                "gold_price": float(round(latest["Close"], 4)),
                "dxy": float(round(market_data[('Close', self.symbols['dxy'])].iloc[-1], 4)),
                "vix": float(round(market_data[('Close', self.symbols['vix'])].iloc[-1], 4)) if ('Close', self.symbols['vix']) in market_data else None
            },
            "news_analysis": news_result
        }

        # حفظ النتائج اليومية والملفات اللازمة للـ backtest
        self.save_json("gold_analysis.json", result)

        # حفظ historical signals (append)
        signals_path = os.path.join(self.save_path, "historical_signals.csv")
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
        df_row = pd.DataFrame([row])
        if not os.path.exists(signals_path):
            df_row.to_csv(signals_path, index=False, encoding="utf-8")
        else:
            df_row.to_csv(signals_path, mode="a", index=False, header=False, encoding="utf-8")
        print(f"💾 تم تحديث سجل الإشارات: {signals_path}")

        return result

    def backtest_signals(self, signals_csv=None, price_series=None):
        """
        backtest بسيط:
        - إدخال: ملف historical_signals.csv أو سلسلة سعرية (pandas.Series indexed by date)
        - استراتجية: عند 'Buy' ندخل مركز طويل بكامل الرصيد (position=1)، عند 'Sell' نخرج (position=0).
          'Hold' يحتفظ بالمركز السابق.
        - حساب العوائد المئوية وتجميعها.
        مخرجات: تقرير بسيط (total_return, CAGR, max_drawdown, trades)
        """
        print("\n📈 بدء backtest بسيط...")
        try:
            if signals_csv is None:
                signals_csv = os.path.join(self.save_path, "historical_signals.csv")
            if not os.path.exists(signals_csv):
                print("⚠️ ملف historical_signals.csv غير موجود، لا يمكن تشغيل backtest.")
                return {"status": "error", "error": "no signals file"}

            sigs = pd.read_csv(signals_csv, parse_dates=["timestamp_utc"])
            sigs.sort_values("timestamp_utc", inplace=True)
            sigs.reset_index(drop=True, inplace=True)

            # للحصول على تسلسل أسعار تاريخي متوافق، نستخرج من Yahoo تاريخ يتراوح بين أول وآخر إشارة
            start = sigs["timestamp_utc"].dt.date.min()
            end = sigs["timestamp_utc"].dt.date.max() + pd.Timedelta(days=1)
            prices = yf.download(self.symbols["gold"], start=start.isoformat(), end=end.isoformat(), interval="1d", progress=False)
            if prices.empty:
                return {"status": "error", "error": "no price series fetched for backtest"}

            close = prices["Close"].ffill().dropna()
            close.index = pd.to_datetime(close.index)

            # ندمج الإشارات مع الأسعار: لكل يوم إشارة نعتبرها تُطبَّق في إغلاق اليوم نفسه (could be improved)
            sigs["date"] = pd.to_datetime(sigs["timestamp_utc"]).dt.normalize()
            # تبسيط: نأخذ آخر إشارة لكل تاريخ
            daily_sig = sigs.groupby("date").last().reindex(close.index, method="ffill").fillna(method="ffill")
            # تحويل للإشارة الموقفية: Buy -> 1, Sell -> 0, Hold -> previous
            position = []
            pos = 0
            for idx, row in daily_sig.iterrows():
                s = row["signal"]
                if s == "Buy":
                    pos = 1
                elif s == "Sell":
                    pos = 0
                # Hold leaves pos as is
                position.append(pos)
            pf = pd.DataFrame({"close": close, "position": position}, index=close.index)

            # حساب عوائد يومية
            pf["pct_change"] = pf["close"].pct_change().fillna(0)
            pf["strategy_return"] = pf["position"].shift(1).fillna(0) * pf["pct_change"]  # position 적용 على عائد اليوم التالي
            pf["cum_return"] = (1 + pf["strategy_return"]).cumprod()
            pf["buyhold_cum"] = (1 + pf["pct_change"]).cumprod()

            total_return = float(pf["cum_return"].iloc[-1] - 1)
            bh_return = float(pf["buyhold_cum"].iloc[-1] - 1)

            # CAGR تقريبي
            days = (pf.index[-1] - pf.index[0]).days or 1
            years = days / 365.25
            cagr = (pf["cum_return"].iloc[-1]) ** (1 / years) - 1 if years > 0 else 0
            bh_cagr = (pf["buyhold_cum"].iloc[-1]) ** (1 / years) - 1 if years > 0 else 0

            # max drawdown
            roll_max = pf["cum_return"].cummax()
            drawdown = pf["cum_return"] / roll_max - 1
            max_dd = float(drawdown.min())

            trades = int(((daily_sig["signal"] == "Buy") | (daily_sig["signal"] == "Sell")).sum())

            report = {
                "period_start": str(pf.index[0].date()),
                "period_end": str(pf.index[-1].date()),
                "days": int(days),
                "total_return": round(total_return, 4),
                "cagr": round(cagr, 4),
                "buyhold_return": round(bh_return, 4),
                "buyhold_cagr": round(bh_cagr, 4),
                "max_drawdown": round(max_dd, 4),
                "trades": trades
            }

            # حفظ التقرير
            self.save_json("backtest_report.json", report)
            print("✅ backtest مكتمل.")
            return {"status": "success", "report": report, "pf_head": pf.head(3).to_dict()}
        except Exception as e:
            print(f"❌ خطأ أثناء backtest: {e}")
            return {"status": "error", "error": str(e)}


if __name__ == "__main__":
    analyzer = ProfessionalGoldAnalyzerV2(
        lookback_days=365,
        news_days=2,
        news_api_key=os.getenv("NEWS_API_KEY"),
        save_path=".",
        batch_size=16
    )
    res = analyzer.run_full_analysis()
    print("\n--- النتيجة النهائية ---")
    print(json.dumps(res, indent=2, ensure_ascii=False))

    # نفذ backtest تلقائياً إن كان هناك سجل إشارات
    bt = analyzer.backtest_signals()
    print("\n--- ملخص backtest ---")
    print(json.dumps(bt, indent=2, ensure_ascii=False))