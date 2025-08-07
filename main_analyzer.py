#!/usr/bin/env python3
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import json
import os
from datetime import datetime, timedelta
from transformers import pipeline
import pytz
import warnings

warnings.filterwarnings('ignore')

class ProfessionalGoldAnalyzer:
    def __init__(self):
        self.symbols = {
            'gold': 'GLD', 'dxy': 'DX-Y.NYB', 'vix': '^VIX',
            'treasury': '^TNX', 'oil': 'CL=F', 'spy': 'SPY'
        }
        self.news_api_key = os.getenv("NEWS_API_KEY")
        self.sentiment_pipeline = None
        try:
            print("🧠 تحميل نموذج تحليل المشاعر المالي...")
            self.sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
            print("✅ نموذج تحليل المشاعر جاهز.")
        except Exception as e:
            print(f"⚠️ تحذير: فشل تحميل نموذج المشاعر. الخطأ: {e}")

    def fetch_data(self):
        print("\n📊 جلب بيانات السوق...")
        try:
            data = yf.download(list(self.symbols.values()), period="1y", interval="1d")
            if data.empty or len(data) < 200: raise ValueError("البيانات غير كافية للحساب.")
            data.dropna(subset=[('Close', self.symbols["gold"])], inplace=True)
            if data.empty: raise ValueError("لا توجد بيانات صالحة للذهب (GLD) بعد التنظيف.")
            print(f"... نجح! تم جلب {len(data)} يوم من البيانات.")
            return data
        except Exception as e:
            print(f"❌ خطأ في جلب البيانات: {e}")
            return None

    def analyze_news(self):
        # ... (هذه الدالة تبقى كما هي من الإصدار السابق)
        return {"status": "skipped", "news_score": 0, "headlines": []} # تبسيط مؤقت

    def run_full_analysis(self):
        market_data = self.fetch_data()
        if market_data is None: return {"status": "error", "error": "فشل جلب بيانات السوق"}

        # --- ✅ التعديل الرئيسي هنا ---
        print("\n📈 استخلاص البيانات وحساب المؤشرات الفنية...")
        # الطريقة الصحيحة للتعامل مع MultiIndex
        gold_data = pd.DataFrame()
        gold_ticker = self.symbols['gold']
        gold_data['Open'] = market_data[('Open', gold_ticker)]
        gold_data['High'] = market_data[('High', gold_ticker)]
        gold_data['Low'] = market_data[('Low', gold_ticker)]
        gold_data['Close'] = market_data[('Close', gold_ticker)]
        gold_data['Volume'] = market_data[('Volume', gold_ticker)]
        gold_data.dropna(inplace=True) # التأكد من عدم وجود قيم فارغة

        ta_strategy = ta.Strategy(name="Full Analysis", ta=[
            {"kind": "sma", "length": 50}, {"kind": "sma", "length": 200},
            {"kind": "rsi"}, {"kind": "macd"}, {"kind": "bbands"},
            {"kind": "atr"}, {"kind": "obv"}
        ])
        gold_data.ta.strategy(ta_strategy)
        gold_data.dropna(inplace=True)
        print("... تم حساب المؤشرات بنجاح.")

        news_analysis_result = self.analyze_news()
        
        latest = gold_data.iloc[-1]
        price = latest['Close']
        trend_score, momentum_score, correlation_score, news_score = 0, 0, 0, news_analysis_result['news_score']
        
        # Scoring Logic
        if price > latest['SMA_200']: trend_score = 2
        if latest['MACD_12_26_9'] > latest['MACDs_12_26_9']: momentum_score = 1
        dxy_corr = market_data[('Close', self.symbols['gold'])].corr(market_data[('Close', self.symbols['dxy'])])
        if dxy_corr < -0.5: correlation_score = 1
        
        total_score = (trend_score * 0.4) + (momentum_score * 0.3) + (correlation_score * 0.2) + (news_score * 0.1)
        
        # Determine Signal
        if total_score >= 1.0: signal = "Buy"
        elif total_score <= -1.0: signal = "Sell"
        else: signal = "Hold"
        
        final_result = {
            "timestamp_utc": datetime.utcnow().isoformat(),
            "signal": signal,
            "total_score": round(total_score, 2),
            "market_data": {
                "gold_price": round(price, 2),
                "dxy": round(market_data[('Close', self.symbols['dxy'])].iloc[-1], 2),
                "vix": round(market_data[('Close', self.symbols['vix'])].iloc[-1], 2)
            },
            "news_headlines": news_analysis_result['headlines']
        }
        
        with open("gold_analysis.json", 'w', encoding='utf-8') as f:
            json.dump(final_result, f, ensure_ascii=False, indent=2)
            
        print("\n✅ تم إتمام التحليل بنجاح وحفظ النتائج.")
        return final_result

if __name__ == "__main__":
    analyzer = ProfessionalGoldAnalyzer()
    results = analyzer.run_full_analysis()
    print("\n--- ملخص التقرير النهائي ---")
    print(json.dumps(results, indent=2, ensure_ascii=False))
