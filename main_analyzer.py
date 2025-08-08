#!/usr/bin/env python3
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import json
import os
import sqlite3
import logging
from datetime import datetime, timedelta
from transformers import pipeline
import pandas_ta as ta
import warnings
from concurrent.futures import ThreadPoolExecutor
import time
from functools import lru_cache

warnings.filterwarnings('ignore')

# إعداد logging محسن
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gold_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OptimizedGoldAnalyzer:
    def __init__(self):
        self.symbols = {
            'gold': 'GC=F',
            'gold_etf': 'GLD', 
            'dxy': 'DX-Y.NYB',
            'vix': '^VIX',
            'treasury': '^TNX',
            'oil': 'CL=F',
            'spy': 'SPY'
        }
        
        self.news_api_key = os.getenv("NEWS_API_KEY")
        self.sentiment_pipeline = None
        self.db_path = "gold_analysis_history.db"
        
        # إعداد سريع
        self._setup_database()
        self._load_sentiment_model()
        logger.info("🚀 محلل الذهب السريع جاهز")

    def _setup_database(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analysis_history (
                    id INTEGER PRIMARY KEY,
                    timestamp_utc TEXT,
                    signal TEXT,
                    total_score REAL,
                    gold_price REAL,
                    signal_strength TEXT,
                    news_score REAL,
                    market_data TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"تحذير قاعدة البيانات: {e}")

    def _load_sentiment_model(self):
        try:
            # تحميل النموذج مرة واحدة فقط
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis", 
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
                return_all_scores=True
            )
        except Exception as e:
            logger.warning(f"تحذير نموذج المشاعر: {e}")

    @lru_cache(maxsize=1)
    def fetch_market_data_cached(self):
        """جلب البيانات مع cache للتسريع"""
        try:
            logger.info("📊 جلب بيانات السوق (محسن)...")
            
            symbols_list = list(self.symbols.values())
            data = yf.download(
                symbols_list, 
                period="1y", 
                interval="1d",
                threads=True,
                progress=False,  # إلغاء شريط التقدم
                show_errors=False
            )
            
            if data.empty:
                raise ValueError("البيانات فارغة")
                
            # تنظيف سريع
            gold_col = ('Close', self.symbols['gold'])
            if gold_col not in data.columns:
                self.symbols['gold'] = 'GLD'
                gold_col = ('Close', 'GLD')
            
            data = data.dropna(subset=[gold_col])
            logger.info(f"✅ جلب {len(data)} يوم من البيانات")
            return data
            
        except Exception as e:
            logger.error(f"❌ خطأ جلب البيانات: {e}")
            return None

    def analyze_news_fast(self):
        """تحليل أخبار سريع ومحسن"""
        logger.info("📰 تحليل أخبار سريع...")
        
        if not self.news_api_key or not self.sentiment_pipeline:
            return {"status": "skipped", "news_score": 0, "headlines": []}

        try:
            # استعلام واحد محسن
            query = 'gold OR XAU OR "federal reserve" OR inflation OR "interest rate"'
            
            url = (
                f"https://newsapi.org/v2/everything?"
                f"q={query}&language=en&sortBy=publishedAt&pageSize=30&"
                f"from={(datetime.now() - timedelta(days=2)).date()}&"
                f"apiKey={self.news_api_key}"
            )
            
            response = requests.get(url, timeout=10)
            articles = response.json().get('articles', [])
            
            if not articles:
                return {"status": "no_articles", "news_score": 0, "headlines": []}
            
            # فلترة سريعة
            gold_keywords = ['gold', 'xau', 'bullion', 'fed', 'inflation', 'interest rate']
            relevant_articles = []
            
            for article in articles[:20]:  # معالجة 20 مقال فقط
                content = f"{article.get('title', '').lower()} {article.get('description', '').lower()}"
                if any(keyword in content for keyword in gold_keywords):
                    relevant_articles.append(article)
            
            if not relevant_articles:
                return {"status": "no_relevant", "news_score": 0, "headlines": []}
            
            # تحليل مشاعر سريع
            sentiment_scores = []
            processed_headlines = []
            
            for article in relevant_articles[:10]:  # أفضل 10 مقالات فقط
                try:
                    text = article.get('description') or article.get('title') or ""
                    if len(text) < 10:
                        continue
                        
                    result = self.sentiment_pipeline(text[:200])[0]  # نص مقصور
                    
                    pos_score = next((s['score'] for s in result if s['label'] == 'positive'), 0)
                    neg_score = next((s['score'] for s in result if s['label'] == 'negative'), 0)
                    
                    sentiment_scores.append(pos_score - neg_score)
                    processed_headlines.append({
                        'title': article['title'],
                        'source': article.get('source', {}).get('name', 'Unknown')
                    })
                    
                except Exception:
                    continue
            
            if sentiment_scores:
                avg_sentiment = np.mean(sentiment_scores)
                logger.info(f"📊 تحليل الأخبار: {avg_sentiment:.3f}")
                
                return {
                    "status": "success",
                    "news_score": round(avg_sentiment, 3),
                    "headlines": processed_headlines[:5]
                }
            else:
                return {"status": "processing_failed", "news_score": 0, "headlines": []}
                
        except Exception as e:
            logger.warning(f"⚠️ تحذير الأخبار: {e}")
            return {"status": "error", "news_score": 0, "headlines": []}

    def calculate_fast_indicators(self, gold_data):
        """حساب مؤشرات أساسية فقط للسرعة"""
        try:
            # مؤشرات أساسية فقط
            gold_data['SMA_20'] = gold_data['Close'].rolling(20).mean()
            gold_data['SMA_50'] = gold_data['Close'].rolling(50).mean()
            gold_data['SMA_200'] = gold_data['Close'].rolling(200).mean()
            
            # RSI سريع
            delta = gold_data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            gold_data['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD بسيط
            ema12 = gold_data['Close'].ewm(span=12).mean()
            ema26 = gold_data['Close'].ewm(span=26).mean()
            gold_data['MACD'] = ema12 - ema26
            gold_data['MACD_Signal'] = gold_data['MACD'].ewm(span=9).mean()
            
            # ATR للستوب لوس
            high_low = gold_data['High'] - gold_data['Low']
            high_close = np.abs(gold_data['High'] - gold_data['Close'].shift())
            low_close = np.abs(gold_data['Low'] - gold_data['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            gold_data['ATR'] = true_range.rolling(14).mean()
            
            return gold_data.dropna()
            
        except Exception as e:
            logger.error(f"❌ خطأ المؤشرات: {e}")
            return gold_data

    def run_fast_analysis(self):
        """التحليل السريع الشامل"""
        start_time = time.time()
        logger.info("🚀 بدء التحليل السريع...")
        
        # جلب البيانات
        market_data = self.fetch_market_data_cached()
        if market_data is None:
            return {"status": "error", "error": "فشل جلب البيانات"}
        
        # إعداد بيانات الذهب
        gold_ticker = self.symbols['gold']
        gold_data = pd.DataFrame({
            'Open': market_data[('Open', gold_ticker)],
            'High': market_data[('High', gold_ticker)],
            'Low': market_data[('Low', gold_ticker)], 
            'Close': market_data[('Close', gold_ticker)],
            'Volume': market_data[('Volume', gold_ticker)]
        }).dropna()
        
        # حساب المؤشرات السريع
        gold_data = self.calculate_fast_indicators(gold_data)
        
        # تحليل الأخبار بالتوازي
        news_analysis = self.analyze_news_fast()
        
        # حساب الإشارة
        latest = gold_data.iloc[-1]
        current_price = latest['Close']
        
        # نقاط سريعة
        trend_score = 2 if current_price > latest['SMA_200'] else -2
        momentum_score = 1 if latest['MACD'] > latest['MACD_Signal'] else -1
        rsi_score = 1 if 30 < latest['RSI'] < 70 else 0
        
        # DXY correlation
        try:
            dxy_current = market_data[('Close', self.symbols['dxy'])].iloc[-1]
            vix_current = market_data[('Close', self.symbols['vix'])].iloc[-1]
        except:
            dxy_current = 100
            vix_current = 20
        
        correlation_score = 1 if dxy_current < 105 else -1
        news_score = news_analysis.get('news_score', 0) * 2
        
        # النتيجة النهائية
        total_score = (trend_score * 0.4) + (momentum_score * 0.25) + (correlation_score * 0.2) + (news_score * 0.15)
        
        # الإشارة
        if total_score >= 1.5:
            signal, strength = "Buy", "Strong Buy"
        elif total_score >= 1.0:
            signal, strength = "Buy", "Buy"
        elif total_score <= -1.5:
            signal, strength = "Sell", "Strong Sell" 
        elif total_score <= -1.0:
            signal, strength = "Sell", "Sell"
        else:
            signal, strength = "Hold", "Hold"
        
        # ستوب لوس
        stop_loss = current_price - (2 * latest['ATR']) if 'buy' in signal.lower() else current_price + (2 * latest['ATR'])
        
        # النتيجة النهائية
        result = {
            "timestamp_utc": datetime.utcnow().isoformat(),
            "signal": signal,
            "signal_strength": strength,
            "total_score": round(total_score, 3),
            "components": {
                "trend_score": trend_score,
                "momentum_score": momentum_score,
                "correlation_score": correlation_score,
                "news_score": round(news_score, 2),
                "rsi_score": rsi_score
            },
            "market_data": {
                "gold_price": round(current_price, 2),
                "dxy": round(dxy_current, 2),
                "vix": round(vix_current, 2),
                "rsi": round(latest['RSI'], 2)
            },
            "stop_loss_price": round(stop_loss, 2),
            "news_analysis": news_analysis,
            "execution_time_seconds": round(time.time() - start_time, 2)
        }
        
        # حفظ النتائج
        try:
            with open("gold_analysis_enhanced.json", 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            # حفظ في قاعدة البيانات
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO analysis_history 
                (timestamp_utc, signal, total_score, gold_price, signal_strength, news_score, market_data)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                result['timestamp_utc'], result['signal'], result['total_score'],
                result['market_data']['gold_price'], result['signal_strength'],
                result['components']['news_score'], json.dumps(result['market_data'])
            ))
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.warning(f"⚠️ تحذير الحفظ: {e}")
        
        logger.info(f"✅ انتهى التحليل في {result['execution_time_seconds']} ثانية")
        return result

if __name__ == "__main__":
    try:
        analyzer = OptimizedGoldAnalyzer()
        results = analyzer.run_fast_analysis()
        
        if results.get("status") == "error":
            logger.error(f"❌ فشل: {results.get('error')}")
        else:
            print(f"\n🎯 الإشارة: {results['signal']} ({results['signal_strength']})")
            print(f"📊 النتيجة: {results['total_score']}")
            print(f"💰 سعر الذهب: ${results['market_data']['gold_price']}")
            print(f"🛑 وقف الخسارة: ${results['stop_loss_price']}")
            print(f"⏱️ وقت التنفيذ: {results['execution_time_seconds']} ثانية")
            
            headlines = results['news_analysis'].get('headlines', [])
            if headlines:
                print(f"\n📰 أهم الأخبار:")
                for i, h in enumerate(headlines[:3], 1):
                    print(f"  {i}. {h['title'][:60]}... [{h['source']}]")
                    
    except Exception as e:
        logger.error(f"❌ خطأ عام: {e}")