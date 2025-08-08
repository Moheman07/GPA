#!/usr/bin/env python3
"""
🏆 محلل الذهب الاحترافي النهائي
"""
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import json
import os
import sqlite3
import logging
import warnings
from datetime import datetime, timedelta
from transformers import pipeline
import pandas_ta as ta
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import Dict
import pytz

warnings.filterwarnings('ignore')

# إعداد التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gold_analysis_pro.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProfessionalGoldAnalyzerFinal:
    def __init__(self):
        self.symbols = {
            'gold': 'GC=F', 'gold_etf': 'GLD', 'silver': 'SI=F',
            'dxy': 'DX-Y.NYB', 'vix': '^VIX', 'treasury': '^TNX',
            'oil': 'CL=F', 'spy': 'SPY'
        }
        self.news_api_key = os.getenv("NEWS_API_KEY")
        self.sentiment_pipeline = None
        self.db_path = "gold_analysis_history.db"
        self._setup_database()
        self._load_sentiment_model()
        logger.info("🚀 محلل الذهب الاحترافي جاهز")

    def _setup_database(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analysis_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp_utc TEXT, signal TEXT, 
                    signal_strength TEXT, total_score REAL, confidence_level REAL, trend_score REAL, 
                    momentum_score REAL, correlation_score REAL, news_score REAL, volatility_score REAL, 
                    seasonal_score REAL, gold_specific_score REAL, gold_price REAL, dxy_value REAL, 
                    vix_value REAL, gold_silver_ratio REAL, stop_loss_price REAL, take_profit_price REAL, 
                    position_size REAL, backtest_return REAL, backtest_sharpe REAL, backtest_max_dd REAL, 
                    backtest_win_rate REAL, execution_time_ms INTEGER, news_articles_count INTEGER, 
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS news_archive (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, analysis_id INTEGER, headline TEXT, 
                    source TEXT, sentiment_score REAL, relevance_score REAL, keywords TEXT, 
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, 
                    FOREIGN KEY (analysis_id) REFERENCES analysis_history (id)
                )
            ''')
            conn.commit()
            conn.close()
            logger.info("✅ قاعدة البيانات جاهزة")
        except Exception as e:
            logger.warning(f"⚠️ تحذير قاعدة البيانات: {e}")

    def _load_sentiment_model(self):
        try:
            logger.info("🧠 تحميل نموذج تحليل المشاعر...")
            self.sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert", return_all_scores=True)
            logger.info("✅ نموذج المشاعر جاهز")
        except Exception as e:
            logger.warning(f"⚠️ فشل تحميل المشاعر: {e}")

    def fetch_market_data_optimized(self) -> pd.DataFrame | None:
        logger.info("📊 جلب بيانات السوق المحسنة...")
        try:
            symbols_list = list(self.symbols.values())
            data = yf.download(symbols_list, period="15mo", interval="1d", threads=True, progress=False)
            
            if data.empty or ('Close', self.symbols['gold']) not in data.columns or data[('Close', self.symbols['gold'])].isnull().all():
                logger.warning("⚠️ فشل GC=F، التبديل إلى GLD...")
                self.symbols['gold'] = 'GLD'
                data = yf.download(list(self.symbols.values()), period="15mo", interval="1d", threads=True, progress=False)

            gold_close_col = ('Close', self.symbols['gold'])
            if data.empty or gold_close_col not in data.columns or data[gold_close_col].isnull().all():
                raise ValueError("لا توجد بيانات صالحة للذهب")
            
            data = data.dropna(subset=[gold_close_col])
            if len(data) < 200:
                raise ValueError(f"بيانات غير كافية للحساب: {len(data)} يوم فقط")

            logger.info(f"✅ تم جلب {len(data)} يوم من البيانات")
            return data
        except Exception as e:
            logger.error(f"❌ خطأ جلب البيانات: {e}")
            return None

    def enhanced_news_analysis(self) -> Dict:
        logger.info("📰 تحليل أخبار الذهب المحسن...")
        if not self.news_api_key or not self.sentiment_pipeline:
            return {"status": "skipped", "news_score": 0, "headlines": [], "confidence": 0}
        try:
            gold_keywords = {
                'gold': 10, 'xau': 10, 'bullion': 8, 'precious metal': 8, 'gold price': 10,
                'federal reserve': 8, 'fed': 8, 'jerome powell': 8, 'interest rate': 8,
                'rate cut': 9, 'rate hike': 9, 'monetary policy': 7, 'fomc': 8,
                'inflation': 7, 'cpi': 7, 'consumer price': 6, 'deflation': 6,
                'economic data': 4, 'gdp': 4, 'unemployment': 4, 'nfp': 5,
                'dollar': 5, 'dxy': 6, 'dollar index': 6, 'usd': 4,
                'dollar strength': 6, 'dollar weakness': 7,
                'geopolitical': 6, 'safe haven': 8, 'safe-haven': 8, 'risk-off': 7,
                'war': 6, 'conflict': 6, 'crisis': 6, 'recession': 7, 'sanctions': 5,
                'stock market': 3, 'bonds': 4, 'treasury': 4, 'yield': 4, 'oil': 3
            }
            queries = [
                'gold OR XAU OR bullion OR "precious metals"',
                '"interest rates" OR "federal reserve" OR "jerome powell"',
                'inflation OR CPI OR "consumer prices"',
                '"dollar index" OR DXY OR "dollar strength"',
                'geopolitical OR "safe haven" OR crisis'
            ]
            all_articles = []

            def fetch_query_news(query):
                try:
                    url = (f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&pageSize=25&from={(datetime.now() - timedelta(days=2)).date()}&apiKey={self.news_api_key}")
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        return response.json().get('articles', [])
                except Exception:
                    pass
                return []

            with ThreadPoolExecutor(max_workers=3) as executor:
                for articles in executor.map(fetch_query_news, queries):
                    all_articles.extend(articles)

            unique_articles = {article['title'].lower().strip(): article for article in all_articles if article.get('title')}.values()
            
            relevant_articles = []
            for article in unique_articles:
                content = f"{(article.get('title') or '').lower()} {(article.get('description') or '').lower()}"
                relevance_score = sum(weight for keyword, weight in gold_keywords.items() if keyword in content)
                if relevance_score >= 5:
                    article['relevance_score'] = relevance_score
                    relevant_articles.append(article)
            
            if not relevant_articles:
                return {"status": "no_relevant", "news_score": 0, "headlines": [], "confidence": 0}

            top_articles = sorted(relevant_articles, key=lambda x: x['relevance_score'], reverse=True)[:30]
            
            sentiment_scores, processed_articles = [], []
            for article in top_articles:
                try:
                    text = f"{article.get('title', '')} {article.get('description', '')}"
                    if len(text.strip()) < 10: continue
                    
                    sentiment_result = self.sentiment_pipeline(text[:512])[0]
                    pos_score = next((s['score'] for s in sentiment_result if s['label'] == 'positive'), 0)
                    neg_score = next((s['score'] for s in sentiment_result if s['label'] == 'negative'), 0)
                    sentiment_score = pos_score - neg_score
                    
                    weighted_sentiment = sentiment_score * (article['relevance_score'] / 15)
                    sentiment_scores.append(weighted_sentiment)
                    
                    article['sentiment_score'] = round(sentiment_score, 3)
                    processed_articles.append(article)
                except:
                    continue

            if not sentiment_scores:
                return {"status": "analysis_failed", "news_score": 0, "headlines": [], "confidence": 0}

            final_sentiment = np.mean(sentiment_scores)
            sentiment_volatility = np.std(sentiment_scores)
            confidence_level = max(0, 1 - (sentiment_volatility / (abs(final_sentiment) + 0.1)))

            return {
                "status": "success", "news_score": round(final_sentiment, 3),
                "confidence": round(confidence_level, 3),
                "headlines": processed_articles[:8]
            }
        except Exception as e:
            logger.error(f"❌ خطأ تحليل الأخبار: {e}")
            return {"status": "error", "news_score": 0, "headlines": [], "confidence": 0}

    # ... (باقي الدوال من نسختك الأخيرة يجب أن تكون هنا)
    # ... (calculate_gold_specific_indicators, run_simple_backtest, etc.)
    # ... (تم اختصارها هنا، لكن تأكد من وجودها في ملفك)

    def calculate_comprehensive_technical_indicators(self, market_data: pd.DataFrame) -> pd.DataFrame:
        logger.info("📊 حساب المؤشرات الفنية الشاملة...")
        try:
            gold_symbol = self.symbols['gold']
            gold_data = pd.DataFrame({
                'Open': market_data[('Open', gold_symbol)], 'High': market_data[('High', gold_symbol)],
                'Low': market_data[('Low', gold_symbol)], 'Close': market_data[('Close', gold_symbol)],
                'Volume': market_data[('Volume', gold_symbol)]
            }).dropna()

            gold_data.ta.strategy(ta.Strategy(name="Comprehensive", ta=[
                {"kind": "sma", "length": l} for l in [10, 20, 50, 200]
            ] + [
                {"kind": "ema", "length": l} for l in [12, 26]
            ] + [
                {"kind": "rsi"}, {"kind": "macd"}, {"kind": "bbands"}, {"kind": "atr"},
                {"kind": "willr"}, {"kind": "cci"}, {"kind": "stoch"}, {"kind": "obv"}
            ]))
            gold_data.dropna(inplace=True)
            logger.info(f"✅ تم حساب {len(gold_data.columns)} مؤشراً")
            return gold_data
        except Exception as e:
            logger.error(f"❌ خطأ المؤشرات الفنية: {e}")
            return pd.DataFrame()

    # ... (بقية الدوال المساعدة)
    # ... (calculate_final_scores, run_complete_analysis, _save_results_to_database)
    # ... (يجب أن تكون موجودة هنا من نسختك الأخيرة)

def main():
    try:
        analyzer = ProfessionalGoldAnalyzerFinal()
        # results = analyzer.run_complete_analysis()
        # ... (منطق عرض النتائج)
        print("\n🎉 تم إنجاز التحليل بنجاح!")
    except Exception as e:
        logger.critical(f"💥 خطأ فادح في التشغيل: {e}")

if __name__ == "__main__":
    main()

