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
            print("🧠 تحميل نموذج تحليل المشاعر المالي (قد يستغرق بعض الوقت أول مرة)...")
            self.sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
            print("✅ نموذج تحليل المشاعر جاهز.")
        except Exception as e:
            print(f"⚠️ تحذير: فشل تحميل نموذج المشاعر. سيتم تخطي تحليل الأخ-بار. الخطأ: {e}")

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
        print("\n📰 بدء محرك تحليل الأخبار...")
        if not self.news_api_key or not self.sentiment_pipeline:
            return {"status": "skipped", "news_score": 0, "headlines": []}
        try:
            query = ('(gold OR XAU) OR ("interest rates" AND ("federal reserve" OR fed)) OR '
                     '(inflation AND (CPI OR "jobs report")) OR (geopolitical AND (risk OR tension))')
            url = (f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt"
                   f"&pageSize=100&from={(datetime.now() - timedelta(days=2)).date()}&apiKey={self.news_api_key}")
            
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            articles = response.json().get('articles', [])
            if not articles: raise ValueError("لم يتم العثور على مقالات.")
            
            keyword_scores = {
                'gold': 3, 'xau': 3, 'federal reserve': 2, 'fed': 2, 'interest rate': 2, 'inflation': 2,
                'cpi': 2, 'nfp': 2, 'dollar': 1, 'dxy': 1, 'geopolitical': 1, 'risk': 1
            }
            
            scored_articles = []
            for article in articles:
                content = f"{(article.get('title') or '').lower()} {(article.get('description') or '').lower()}"
                score = sum(points for keyword, points in keyword_scores.items() if keyword in content)
                if score >= 3:
                    article['relevance_score'] = score
                    scored_articles.append(article)
            
            if not scored_articles: raise ValueError("لم يتم العثور على مقالات ذات أهمية كافية.")
            print(f"🎯 تم تحديد {len(scored_articles)} مقالاً ذا أهمية عالية.")

            total_sentiment = sum(
                (res['score'] if res['label'] == 'positive' else -res['score'])
                for art in scored_articles
                if (res := self.sentiment_pipeline(art['description'] or art['title'])[0])
            )

            news_score = round(total_sentiment / len(scored_articles), 2)
            print(f"⚖️ نتيجة مشاعر الأخبار النهائية: {news_score}")
            
            sorted_articles = sorted(scored_articles, key=lambda x: x['relevance_score'], reverse=True)
            
            return {
                "status": "success", "news_score": news_score,
                "headlines": [{'title': a['title'], 'source': a.get('source', {}).get('name')} for a in sorted_articles[:5]]
            }
        except Exception as e:
            print(f"❌ خطأ في تحليل الأخبار: {e}")
            return {"status": "error", "news_score": 0, "headlines": []}

    def run_full_analysis(self):
        market_data = self.fetch_data()
        if market_data is None: return {"status": "error", "error": "فشل جلب بيانات السوق"}

        gold_data = market_data['Close'][self.symbols['gold']].to_frame('Close')
        gold_data[['Open', 'High', 'Low', 'Volume']] = market_data[['Open', 'High', 'Low', 'Volume']][self.symbols['gold']]

        print("\n📈 حساب المؤشرات الفنية...")
        ta_strategy = ta.Strategy(name="Full Analysis", ta=[
            {"kind": "sma", "length": 50}, {"kind": "sma", "length": 200},
            {"kind": "rsi"}, {"kind": "macd"}, {"kind": "bbands"},
            {"kind": "atr"}, {"kind": "obv"}
        ])
        gold_data.ta.strategy(ta_strategy)
        gold_data.dropna(inplace=True)
        print("... تم حساب المؤشرات.")

        news_analysis_result = self.analyze_news()
        
        latest = gold_data.iloc[-1]
        price = latest['Close']
        trend_score, momentum_score, correlation_score, news_score = 0, 0, 0, news_analysis_result['news_score']
        
        # Scoring Logic
        if price > latest['SMA_200']: trend_score = 2
        if latest['MACD_12_26_9'] > latest['MACDs_12_26_9']: momentum_score = 1
        dxy_corr = market_data['Close'][self.symbols['gold']].corr(market_data['Close'][self.symbols['dxy']])
        if dxy_corr < -0.5: correlation_score = 1
        
        total_score = (trend_score * 0.4) + (momentum_score * 0.3) + (correlation_score * 0.2) + (news_score * 0.1)
        
        # Determine Signal
        if total_score >= 1.0: signal = "Buy"
        elif total_score <= -1.0: signal = "Sell"
        else: signal = "Hold"
        
        # Final JSON Output
        final_result = {
            "timestamp_utc": datetime.utcnow().isoformat(),
            "signal": signal,
            "total_score": round(total_score, 2),
            "components": {
                "trend_score": trend_score, "momentum_score": momentum_score,
                "correlation_score": correlation_score, "news_score": news_score
            },
            "market_data": {
                "gold_price": round(price, 2),
                "dxy": round(market_data['Close'][self.symbols['dxy']].iloc[-1], 2),
                "vix": round(market_data['Close'][self.symbols['vix']].iloc[-1], 2)
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
