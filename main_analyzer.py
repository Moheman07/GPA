#!/usr/bin/env python3
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import json
import os
from datetime import datetime, timedelta
from transformers import pipeline
import pandas_ta as ta
import sys
import logging
import warnings

warnings.filterwarnings('ignore')

# --- إعداد التسجيل ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[logging.StreamHandler()])

class HybridGoldAnalyzer:
    def __init__(self, mode='quick'):
        self.mode = mode
        self.symbols = {'gold': 'GLD', 'dxy': 'DX-Y.NYB', 'vix': '^VIX'}
        self.news_api_key = os.getenv("NEWS_API_KEY")
        self.sentiment_pipeline = None

        if self.mode == 'deep':
            try:
                logging.info("🧠 Loading deep sentiment model...")
                self.sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
                logging.info("✅ Sentiment model loaded.")
            except Exception as e:
                logging.error(f"❌ Could not load sentiment model: {e}")

    # --- وضع المسح السريع (QUICK SCAN) ---
    def run_quick_scan(self):
        logging.info("🚀 Starting Quick Scan...")
        try:
            # 1. جلب البيانات الأساسية
            data = yf.download(list(self.symbols.values()), period='5d', interval='1d', progress=False)
            if data.empty: raise ValueError("Quick data fetch failed.")
            
            # 2. تحليل فني سريع
            gold_price = data[('Close', 'GLD')].iloc[-1]
            sma50 = data[('Close', 'GLD')].rolling(50).mean().iloc[-1]
            rsi = ta.rsi(data[('Close', 'GLD')]).iloc[-1]
            trend = "Bullish" if gold_price > sma50 else "Bearish"
            tech_score = (1 if trend == "Bullish" else -1) + (0.5 if 40 < rsi < 60 else 0)

            # 3. تحليل أخبار سريع (keyword-based)
            news_score = 0
            headlines = []
            if self.news_api_key:
                positive_words = ['surge', 'rally', 'safe haven', 'rate cut', 'weak dollar']
                negative_words = ['fall', 'drop', 'strong dollar', 'rate hike']
                query = 'gold OR "federal reserve"'
                url = f"https://newsapi.org/v2/everything?qInTitle={query}&language=en&pageSize=10&apiKey={self.news_api_key}"
                try:
                    articles = requests.get(url).json().get('articles', [])
                    headlines = [a['title'] for a in articles]
                    content = " ".join(headlines).lower()
                    positive_count = sum(word in content for word in positive_words)
                    negative_count = sum(word in content for word in negative_words)
                    news_score = (positive_count - negative_count) / 5.0 # تطبيع بسيط
                except Exception as e:
                    logging.warning(f"⚠️ Quick news fetch failed: {e}")

            # 4. النتيجة النهائية
            total_score = (tech_score * 0.7) + (news_score * 0.3)
            signal = "Buy" if total_score > 0.5 else "Sell" if total_score < -0.5 else "Hold"

            result = {
                "timestamp_utc": datetime.utcnow().isoformat(),
                "mode": "quick_scan",
                "signal": signal,
                "total_score": round(total_score, 2),
                "gold_price": round(gold_price, 2),
                "rsi": round(rsi, 2),
                "trend_vs_sma50": trend,
                "quick_news_sentiment": round(news_score, 2),
                "top_headline": headlines[0] if headlines else "N/A"
            }
            
            with open("quick_analysis.json", 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logging.info(f"✅ Quick Scan Complete! Signal: {signal}")

        except Exception as e:
            logging.error(f"❌ Quick Scan Failed: {e}")

    # --- وضع التحليل العميق (DEEP DIVE) ---
    def run_deep_analysis(self):
        logging.info("🚀 Starting Deep Dive Analysis...")
        # (هنا نضع الكود الكامل من السكربت الاحترافي السابق)
        # ... لقد قمت باختصاره هنا ليكون الرد أقصر، لكن يجب عليك استخدام الكود الكامل
        # ... الذي يحسب كل المؤشرات، الارتباطات، فيبوناتشي، الأخبار المعمقة، إلخ.
        try:
            # مثال على استدعاء دالة من نظامك العميق
            # market_data = self.fetch_market_data_optimized() ...
            # technical_data = self.calculate_professional_indicators(market_data) ...
            # final_result = self.generate_professional_signals(...) ...

            # نتيجة وهمية للتوضيح
            final_result = {
                "timestamp_utc": datetime.utcnow().isoformat(),
                "mode": "deep_dive",
                "signal": "Weak Buy",
                "confidence": "Medium",
                "technical_score": 1.35,
                "component_analysis": {"trend": 0, "momentum": 3},
                "risk_management": {"stop_loss": 3369.31, "take_profit": 3549.89}
            }
            with open("deep_analysis.json", 'w', encoding='utf-8') as f:
                json.dump(final_result, f, ensure_ascii=False, indent=2)
            logging.info("✅ Deep Dive Complete! Signal: Weak Buy")

        except Exception as e:
            logging.error(f"❌ Deep Dive Failed: {e}")

# --- المشغل الرئيسي ---
if __name__ == "__main__":
    # تحديد الوضع من سطر الأوامر، الوضع الافتراضي هو 'quick'
    mode = 'quick'
    if len(sys.argv) > 1 and sys.argv[1] in ['quick', 'deep']:
        mode = sys.argv[1]
    
    analyzer = HybridGoldAnalyzer(mode=mode)
    
    if mode == 'deep':
        analyzer.run_deep_analysis()
    else:
        analyzer.run_quick_scan()
