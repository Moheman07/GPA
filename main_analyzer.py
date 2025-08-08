#!/usr/bin/env python3
"""
🏆 محلل الذهب الاحترافي - النسخة النهائية
يعمل يدوياً وعلى GitHub Actions
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
from typing import Dict, List, Optional
import threading

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
    """محلل الذهب الاحترافي النهائي - يعمل يدوياً"""
    
    def __init__(self):
        self.symbols = {
            'gold': 'GC=F',          # Gold Futures أولاً
            'gold_etf': 'GLD',       # احتياطي
            'silver': 'SI=F',        # للنسبة
            'dxy': 'DX-Y.NYB',       
            'vix': '^VIX',           
            'treasury': '^TNX',      
            'oil': 'CL=F',           
            'spy': 'SPY'
        }
        
        self.news_api_key = os.getenv("NEWS_API_KEY")
        self.sentiment_pipeline = None
        self.db_path = "gold_analysis_history.db"
        
        # تهيئة المكونات
        self._setup_database()
        self._load_sentiment_model()
        
        logger.info("🚀 محلل الذهب الاحترافي جاهز")

    def _setup_database(self):
        """إعداد قاعدة البيانات"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analysis_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp_utc TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    signal_strength TEXT,
                    total_score REAL NOT NULL,
                    confidence_level REAL,
                    
                    -- النقاط
                    trend_score REAL,
                    momentum_score REAL,
                    correlation_score REAL,
                    news_score REAL,
                    volatility_score REAL,
                    seasonal_score REAL,
                    gold_specific_score REAL,
                    
                    -- السوق
                    gold_price REAL,
                    dxy_value REAL,
                    vix_value REAL,
                    gold_silver_ratio REAL,
                    
                    -- إدارة المخاطر
                    stop_loss_price REAL,
                    take_profit_price REAL,
                    position_size REAL,
                    
                    -- باك تيست
                    backtest_return REAL,
                    backtest_sharpe REAL,
                    backtest_max_dd REAL,
                    backtest_win_rate REAL,
                    
                    -- أداء
                    execution_time_ms INTEGER,
                    news_articles_count INTEGER,
                    
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS news_archive (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id INTEGER,
                    headline TEXT,
                    source TEXT,
                    sentiment_score REAL,
                    relevance_score REAL,
                    keywords TEXT,
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
        """تحميل نموذج المشاعر"""
        try:
            logger.info("🧠 تحميل نموذج تحليل المشاعر...")
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis", 
                model="ProsusAI/finbert",
                return_all_scores=True
            )
            logger.info("✅ نموذج المشاعر جاهز")
        except Exception as e:
            logger.warning(f"⚠️ فشل تحميل المشاعر: {e}")
            self.sentiment_pipeline = None

    def fetch_market_data_optimized(self) -> Optional[pd.DataFrame]:
        """جلب بيانات السوق محسن"""
        logger.info("📊 جلب بيانات السوق المحسنة...")
        
        try:
            symbols_list = list(self.symbols.values())
            
            # جلب البيانات مع التحسينات
            data = yf.download(
                symbols_list,
                period="15mo",  # 15 شهر (توازن)
                interval="1d",
                threads=True,
                progress=False,
                show_errors=False
            )
            
            if data.empty:
                logger.warning("⚠️ فشل GC=F، التبديل إلى GLD...")
                self.symbols['gold'] = 'GLD'
                symbols_list[0] = 'GLD'
                data = yf.download(symbols_list, period="15mo", interval="1d", threads=True, progress=False)
            
            # تنظيف
            gold_close = ('Close', self.symbols['gold'])
            if gold_close not in data.columns:
                logger.error("❌ عمود الذهب مفقود")
                return None
                
            data = data.dropna(subset=[gold_close])
            
            if len(data) < 100:
                logger.error(f"❌ بيانات غير كافية: {len(data)}")
                return None
                
            logger.info(f"✅ تم جلب {len(data)} يوم من البيانات")
            return data
            
        except Exception as e:
            logger.error(f"❌ خطأ جلب البيانات: {e}")
            return None

    def enhanced_news_analysis(self) -> Dict:
        """تحليل أخبار محسن ومتخصص"""
        logger.info("📰 تحليل أخبار الذهب المحسن...")
        
        if not self.news_api_key or not self.sentiment_pipeline:
            return {"status": "skipped", "news_score": 0, "headlines": [], "confidence": 0}

        try:
            # كلمات مفتاحية متخصصة مع أوزان
            gold_keywords = {
                # ذهب مباشر - وزن عالي
                'gold': 10, 'xau': 10, 'bullion': 8, 'precious metal': 8, 'gold price': 10,
                
                # سياسة نقدية - وزن عالي
                'federal reserve': 8, 'fed': 8, 'jerome powell': 8, 'interest rate': 8,
                'rate cut': 9, 'rate hike': 9, 'monetary policy': 7, 'fomc': 8,
                
                # تضخم واقتصاد
                'inflation': 7, 'cpi': 7, 'consumer price': 6, 'deflation': 6,
                'economic data': 4, 'gdp': 4, 'unemployment': 4, 'nfp': 5,
                
                # دولار وعملات
                'dollar': 5, 'dxy': 6, 'dollar index': 6, 'usd': 4,
                'dollar strength': 6, 'dollar weakness': 7,
                
                # جيوسياسية
                'geopolitical': 6, 'safe haven': 8, 'safe-haven': 8, 'risk-off': 7,
                'war': 6, 'conflict': 6, 'crisis': 6, 'recession': 7, 'sanctions': 5,
                
                # أسواق
                'stock market': 3, 'bonds': 4, 'treasury': 4, 'yield': 4, 'oil': 3
            }

            # استعلامات متعددة
            queries = [
                'gold OR XAU OR bullion OR "precious metals"',
                '"interest rates" OR "federal reserve" OR "jerome powell"',
                'inflation OR CPI OR "consumer prices"',
                '"dollar index" OR DXY OR "dollar strength"',
                'geopolitical OR "safe haven" OR crisis'
            ]
            
            all_articles = []
            
            # جلب الأخبار بالتوازي
            def fetch_query_news(query):
                try:
                    url = (f"https://newsapi.org/v2/everything?"
                          f"q={query}&language=en&sortBy=publishedAt&pageSize=25&"
                          f"from={(datetime.now() - timedelta(days=2)).date()}&"
                          f"apiKey={self.news_api_key}")
                    
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        return response.json().get('articles', [])
                except Exception as e:
                    logger.warning(f"⚠️ فشل استعلام: {query[:30]}... - {e}")
                return []
            
            # معالجة بالتوازي
            with ThreadPoolExecutor(max_workers=3) as executor:
                future_to_query = {executor.submit(fetch_query_news, query): query for query in queries}
                
                for future in as_completed(future_to_query):
                    articles = future.result()
                    all_articles.extend(articles)
                    logger.info(f"📥 جلب {len(articles)} مقال")

            if not all_articles:
                return {"status": "no_articles", "news_score": 0, "headlines": [], "confidence": 0}

            # إزالة المكرر
            unique_articles = []
            seen_titles = set()
            for article in all_articles:
                title = (article.get('title') or '').lower().strip()
                if title and title not in seen_titles and len(title) > 10:
                    seen_titles.add(title)
                    unique_articles.append(article)

            logger.info(f"🔍 {len(unique_articles)} مقال فريد")

            # تقييم الصلة
            relevant_articles = []
            for article in unique_articles:
                content = f"{(article.get('title') or '').lower()} {(article.get('description') or '').lower()}"
                
                relevance_score = 0
                matched_keywords = []
                
                for keyword, weight in gold_keywords.items():
                    if keyword in content:
                        relevance_score += weight
                        matched_keywords.append(keyword)
                
                # قبول المقالات المهمة
                if relevance_score >= 5:  # حد مناسب
                    article['relevance_score'] = relevance_score
                    article['matched_keywords'] = matched_keywords[:3]
                    relevant_articles.append(article)

            if not relevant_articles:
                return {"status": "no_relevant", "news_score": 0, "headlines": [], "confidence": 0}

            # ترتيب وأخذ الأفضل
            relevant_articles.sort(key=lambda x: x['relevance_score'], reverse=True)
            top_articles = relevant_articles[:30]  # أفضل 30
            
            logger.info(f"🎯 {len(top_articles)} مقال عالي الصلة")

            # تحليل المشاعر
            sentiment_scores = []
            processed_articles = []
            
            for article in top_articles:
                try:
                    text = f"{article.get('title', '')} {article.get('description', '')}"
                    if len(text.strip()) < 10:
                        continue
                        
                    # تحليل المشاعر
                    sentiment_result = self.sentiment_pipeline(text[:400])
                    
                    # استخراج النقاط
                    pos_score = next((s['score'] for s in sentiment_result[0] if s['label'] == 'positive'), 0)
                    neg_score = next((s['score'] for s in sentiment_result[0] if s['label'] == 'negative'), 0)
                    
                    sentiment_score = pos_score - neg_score
                    
                    # وزن حسب الصلة
                    weighted_sentiment = sentiment_score * (article['relevance_score'] / 15)
                    sentiment_scores.append(weighted_sentiment)
                    
                    processed_articles.append({
                        'title': article['title'],
                        'source': article.get('source', {}).get('name', 'Unknown'),
                        'sentiment_score': round(sentiment_score, 3),
                        'relevance_score': article['relevance_score'],
                        'matched_keywords': article['matched_keywords']
                    })
                    
                except Exception as e:
                    logger.warning(f"⚠️ خطأ تحليل مقال: {e}")
                    continue

            if not sentiment_scores:
                return {"status": "analysis_failed", "news_score": 0, "headlines": [], "confidence": 0}

            # النتيجة النهائية
            final_sentiment = np.mean(sentiment_scores)
            sentiment_volatility = np.std(sentiment_scores)
            confidence_level = max(0, min(1, 1 - (sentiment_volatility / (abs(final_sentiment) + 0.1))))

            # ترتيب للعرض
            processed_articles.sort(key=lambda x: x['relevance_score'], reverse=True)

            result = {
                "status": "success",
                "news_score": round(final_sentiment, 3),
                "confidence": round(confidence_level, 3),
                "headlines": processed_articles[:8],  # أفضل 8
                "analysis_details": {
                    'total_articles': len(processed_articles),
                    'average_sentiment': round(final_sentiment, 3),
                    'positive_articles': len([a for a in processed_articles if a['sentiment_score'] > 0.1]),
                    'negative_articles': len([a for a in processed_articles if a['sentiment_score'] < -0.1]),
                    'high_relevance': len([a for a in processed_articles if a['relevance_score'] > 10])
                }
            }

            logger.info(f"📊 تحليل أخبار مكتمل: {final_sentiment:.3f} (ثقة: {confidence_level:.3f})")
            return result

        except Exception as e:
            logger.error(f"❌ خطأ تحليل الأخبار: {e}")
            return {"status": "error", "news_score": 0, "headlines": [], "confidence": 0}

    def calculate_gold_specific_indicators(self, gold_data: pd.DataFrame, market_data: pd.DataFrame) -> Dict:
        """حساب المؤشرات المتخصصة بالذهب"""
        logger.info("📈 حساب المؤشرات المتخصصة بالذهب...")
        
        try:
            latest = gold_data.iloc[-1]
            gold_prices = gold_data['Close']
            
            indicators = {}
            
            # 1. نسبة الذهب/الفضة
            try:
                silver_symbol = self.symbols['silver']
                if ('Close', silver_symbol) in market_data.columns:
                    silver_prices = market_data[('Close', silver_symbol)]
                    gold_silver_ratio = gold_prices.iloc[-1] / silver_prices.iloc[-1]
                    
                    # تقييم النسبة
                    if 70 <= gold_silver_ratio <= 85:
                        gsr_score = 0  # نطاق طبيعي
                    elif gold_silver_ratio > 85:
                        gsr_score = 1  # الذهب مرتفع نسبياً
                    else:
                        gsr_score = -0.5  # الفضة مرتفعة
                        
                    indicators['gold_silver_ratio'] = gold_silver_ratio
                    indicators['gsr_score'] = gsr_score
            except:
                indicators['gold_silver_ratio'] = 75  # افتراضي
                indicators['gsr_score'] = 0

            # 2. نسبة الذهب/النفط  
            try:
                oil_symbol = self.symbols['oil']
                if ('Close', oil_symbol) in market_data.columns:
                    oil_prices = market_data[('Close', oil_symbol)]
                    gold_oil_ratio = gold_prices.iloc[-1] / oil_prices.iloc[-1]
                    
                    # تقييم (النسبة التاريخية حوالي 15-25)
                    if 15 <= gold_oil_ratio <= 25:
                        gor_score = 0
                    elif gold_oil_ratio > 25:
                        gor_score = 1  # الذهب قوي مقارنة بالنفط
                    else:
                        gor_score = -0.5
                        
                    indicators['gold_oil_ratio'] = gold_oil_ratio
                    indicators['gor_score'] = gor_score
            except:
                indicators['gold_oil_ratio'] = 20
                indicators['gor_score'] = 0

            # 3. تحليل موسمي
            current_month = datetime.now().month
            # أشهر قوة الذهب تاريخياً: يناير، فبراير، أغسطس، سبتمبر، ديسمبر
            strong_months = [1, 2, 8, 9, 12]
            seasonal_score = 1 if current_month in strong_months else -0.5
            
            indicators['current_month'] = current_month
            indicators['seasonal_score'] = seasonal_score
            indicators['is_strong_season'] = current_month in strong_months

            # 4. تحليل التقلبات
            returns = gold_prices.pct_change().dropna()
            current_volatility = returns.rolling(20).std().iloc[-1] * np.sqrt(252)  # سنوي
            
            if current_volatility > 0.25:
                volatility_regime = "high"
                volatility_score = 1  # التقلبات العالية تفيد الذهب
            elif current_volatility < 0.15:
                volatility_regime = "low"  
                volatility_score = -0.5
            else:
                volatility_regime = "normal"
                volatility_score = 0
                
            indicators['volatility'] = current_volatility
            indicators['volatility_regime'] = volatility_regime
            indicators['volatility_score'] = volatility_score

            # 5. مستويات دعم ومقاومة بسيطة
            price_current = gold_prices.iloc[-1]
            high_20 = gold_prices.rolling(20).max().iloc[-1]
            low_20 = gold_prices.rolling(20).min().iloc[-1]
            
            # موقع السعر في النطاق
            if price_current > (high_20 + low_20) / 2:
                support_resistance_score = 0.5  # النصف العلوي
            else:
                support_resistance_score = -0.5  # النصف السفلي
                
            # إذا قريب من الحدود
            if abs(price_current - high_20) / price_current < 0.02:  # قريب من المقاومة
                support_resistance_score = -1
            elif abs(price_current - low_20) / price_current < 0.02:  # قريب من الدعم
                support_resistance_score = 1
                
            indicators['resistance_level'] = high_20
            indicators['support_level'] = low_20
            indicators['support_resistance_score'] = support_resistance_score

            # 6. محاكاة مؤشر COT مبسط
            # تحليل الاتجاه طويل المدى
            sma_50 = gold_prices.rolling(50).mean().iloc[-1]
            sma_200 = gold_prices.rolling(200).mean().iloc[-1]
            
            if price_current > sma_50 > sma_200:
                cot_signal = 1  # إيجابي
            elif price_current < sma_50 < sma_200:
                cot_signal = -1  # سلبي  
            else:
                cot_signal = 0  # محايد
                
            indicators['cot_signal'] = cot_signal

            # حساب النتيجة الإجمالية للمؤشرات المتخصصة
            total_gold_score = (
                indicators.get('gsr_score', 0) * 0.2 +
                indicators.get('gor_score', 0) * 0.15 +
                indicators['seasonal_score'] * 0.2 +
                indicators['volatility_score'] * 0.2 +
                indicators['support_resistance_score'] * 0.15 +
                indicators['cot_signal'] * 0.1
            )
            
            indicators['total_gold_specific_score'] = round(total_gold_score, 3)
            
            logger.info(f"✅ مؤشرات الذهب: {total_gold_score:.3f}")
            return indicators
            
        except Exception as e:
            logger.error(f"❌ خطأ المؤشرات المتخصصة: {e}")
            return {'total_gold_specific_score': 0}

    def run_simple_backtest(self, gold_data: pd.DataFrame) -> Dict:
        """نظام باك تيست مبسط وفعال"""
        logger.info("🔬 تشغيل اختبار تاريخي مبسط...")
        
        try:
            # تحضير البيانات
            df = gold_data.copy()
            df['returns'] = df['Close'].pct_change()
            
            # حساب إشارات مبسطة للاختبار
            df['signal'] = 0
            
            # منطق الإشارات للاختبار
            for i in range(50, len(df)):
                current_data = df.iloc[i]
                
                # إشارات بسيطة
                if (current_data['Close'] > current_data['SMA_50'] and 
                    current_data['SMA_50'] > current_data['SMA_200'] and
                    current_data['RSI'] < 70):
                    df.iloc[i, df.columns.get_loc('signal')] = 1  # شراء
                elif (current_data['Close'] < current_data['SMA_50'] and 
                      current_data['SMA_50'] < current_data['SMA_200'] and
                      current_data['RSI'] > 30):
                    df.iloc[i, df.columns.get_loc('signal')] = -1  # بيع

            # حساب العوائد
            df['strategy_returns'] = df['signal'].shift(1) * df['returns']
            df['cumulative_strategy'] = (1 + df['strategy_returns']).cumprod()
            df['cumulative_market'] = (1 + df['returns']).cumprod()
            
            # إحصائيات الأداء
            total_return = (df['cumulative_strategy'].iloc[-1] - 1) * 100
            market_return = (df['cumulative_market'].iloc[-1] - 1) * 100
            
            # حساب شارب (مبسط)
            strategy_sharpe = (df['strategy_returns'].mean() / df['strategy_returns'].std() * np.sqrt(252)) if df['strategy_returns'].std() > 0 else 0
            
            # أقصى انخفاض
            rolling_max = df['cumulative_strategy'].expanding().max()
            drawdown = (df['cumulative_strategy'] - rolling_max) / rolling_max
            max_drawdown = drawdown.min() * 100
            
            # عدد الصفقات ومعدل الفوز
            trades = df[df['signal'] != 0]
            total_trades = len(trades)
            winning_trades = len(trades[trades['strategy_returns'] > 0]) if total_trades > 0 else 0
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            results = {
                'total_return_percent': round(total_return, 2),
                'market_return_percent': round(market_return, 2),
                'excess_return_percent': round(total_return - market_return, 2),
                'sharpe_ratio': round(strategy_sharpe, 2),
                'max_drawdown_percent': round(max_drawdown, 2),
                'total_trades': total_trades,
                'win_rate_percent': round(win_rate, 1),
                'test_period_days': len(df)
            }
            
            logger.info(f"📈 باك تيست: عائد {total_return:.1f}%, شارب {strategy_sharpe:.2f}")
            return results
            
        except Exception as e:
            logger.error(f"❌ خطأ الباك تيست: {e}")
            return {
                'total_return_percent': 0, 'market_return_percent': 0,
                'excess_return_percent': 0, 'sharpe_ratio': 0,
                'max_drawdown_percent': 0, 'total_trades': 0,
                'win_rate_percent': 0, 'test_period_days': 0
            }

    def calculate_comprehensive_technical_indicators(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """حساب المؤشرات الفنية الشاملة"""
        logger.info("📊 حساب المؤشرات الفنية الشاملة...")
        
        try:
            gold_symbol = self.symbols['gold']
            
            # إعداد بيانات الذهب
            gold_data = pd.DataFrame({
                'Open': market_data[('Open', gold_symbol)],
                'High': market_data[('High', gold_symbol)], 
                'Low': market_data[('Low', gold_symbol)],
                'Close': market_data[('Close', gold_symbol)],
                'Volume': market_data[('Volume', gold_symbol)]
            }).dropna()

            # المتوسطات المتحركة
            gold_data['SMA_10'] = ta.sma(gold_data['Close'], length=10)
            gold_data['SMA_20'] = ta.sma(gold_data['Close'], length=20)
            gold_data['SMA_50'] = ta.sma(gold_data['Close'], length=50)
            gold_data['SMA_200'] = ta.sma(gold_data['Close'], length=200)
            gold_data['EMA_12'] = ta.ema(gold_data['Close'], length=12)
            gold_data['EMA_26'] = ta.ema(gold_data['Close'], length=26)

            # مؤشرات الزخم
            gold_data['RSI'] = ta.rsi(gold_data['Close'], length=14)
            
            # MACD
            macd_data = ta.macd(gold_data['Close'])
            gold_data['MACD'] = macd_data['MACD_12_26_9']
            gold_data['MACD_Signal'] = macd_data['MACDs_12_26_9']
            gold_data['MACD_Hist'] = macd_data['MACDh_12_26_9']

            # Bollinger Bands
            bb_data = ta.bbands(gold_data['Close'])
            gold_data['BB_Upper'] = bb_data['BBU_5_2.0']
            gold_data['BB_Middle'] = bb_data['BBM_5_2.0'] 
            gold_data['BB_Lower'] = bb_data['BBL_5_2.0']

            # مؤشرات إضافية
            gold_data['ATR'] = ta.atr(gold_data['High'], gold_data['Low'], gold_data['Close'])
            gold_data['Williams_R'] = ta.willr(gold_data['High'], gold_data['Low'], gold_data['Close'])
            gold_data['CCI'] = ta.cci(gold_data['High'], gold_data['Low'], gold_data['Close'])
            
            # Stochastic
            stoch_data = ta.stoch(gold_data['High'], gold_data['Low'], gold_data['Close'])
            gold_data['Stoch_K'] = stoch_data['STOCHk_14_3_3']
            gold_data['Stoch_D'] = stoch_data['STOCHd_14_3_3']

            # Volume
            gold_data['OBV'] = ta.obv(gold_data['Close'], gold_data['Volume'])
            gold_data['Volume_SMA'] = ta.sma(gold_data['Volume'], length=20)

            # تنظيف البيانات
            gold_data = gold_data.dropna()
            
            logger.info(f"✅ تم حساب {len(gold_data.columns)} مؤشراً - {len(gold_data)} صف نظيف")
            return gold_data
            
        except Exception as e:
            logger.error(f"❌ خطأ المؤشرات الفنية: {e}")
            return pd.DataFrame()

    def calculate_final_scores(self, gold_data: pd.DataFrame, market_data: pd.DataFrame, gold_indicators: Dict, news_result: Dict) -> Dict:
        """حساب النقاط النهائية لجميع المكونات"""
        logger.info("🎯 حساب النقاط النهائية...")
        
        try:
            latest = gold_data.iloc[-1]
            current_price = latest['Close']
            scores = {}

            # 1. نقاط الاتجاه (30%)
            trend_score = 0
            if current_price > latest['SMA_200']: trend_score += 2
            if current_price > latest['SMA_50']: trend_score += 1.5
            if latest['SMA_50'] > latest['SMA_200']: trend_score += 1
            if current_price > latest['SMA_20']: trend_score += 0.5
            scores['trend'] = min(trend_score, 3) - 1.5  # بين -1.5 و 1.5

            # 2. نقاط الزخم (25%)
            momentum_score = 0
            rsi = latest['RSI']
            if latest['MACD'] > latest['MACD_Signal']: momentum_score += 1
            if 30 < rsi < 70: momentum_score += 1
            elif rsi < 30: momentum_score += 1.5  # تشبع بيع
            elif rsi > 70: momentum_score -= 0.5  # تشبع شراء
            if latest['Stoch_K'] > latest['Stoch_D']: momentum_score += 0.5
            scores['momentum'] = min(momentum_score, 2.5) - 1.25  # بين -1.25 و 1.25

            # 3. نقاط الارتباط (20%)
            correlation_score = 0
            try:
                dxy_current = market_data[('Close', self.symbols['dxy'])].iloc[-1]
                vix_current = market_data[('Close', self.symbols['vix'])].iloc[-1]
                
                if dxy_current < 105: correlation_score += 1  # دولار ضعيف يفيد الذهب
                if vix_current > 20: correlation_score += 1  # خوف يفيد الذهب
                if vix_current > 30: correlation_score += 0.5  # خوف شديد
            except:
                pass
            scores['correlation'] = min(correlation_score, 2) - 1  # بين -1 و 1

            # 4. نقاط التقلبات (10%)
            volatility_score = gold_indicators.get('volatility_score', 0)
            # إضافة Bollinger Bands
            try:
                bb_position = (current_price - latest['BB_Lower']) / (latest['BB_Upper'] - latest['BB_Lower'])
                if bb_position < 0.2: volatility_score += 0.5  # قرب الحد السفلي
                elif bb_position > 0.8: volatility_score -= 0.3  # قرب الحد العلوي
            except:
                pass
            scores['volatility'] = max(min(volatility_score, 1), -1)

            # 5. النقاط الموسمية (5%)
            scores['seasonal'] = gold_indicators.get('seasonal_score', 0)

            # 6. المؤشرات المتخصصة بالذهب (10%)
            scores['gold_specific'] = gold_indicators.get('total_gold_specific_score', 0)

            logger.info("✅ تم حساب جميع النقاط")
            return scores
            
        except Exception as e:
            logger.error(f"❌ خطأ حساب النقاط: {e}")
            return {'trend': 0, 'momentum': 0, 'correlation': 0, 'volatility': 0, 'seasonal': 0, 'gold_specific': 0}

    def run_complete_analysis(self) -> Dict:
        """التحليل الكامل النهائي"""
        start_time = time.time()
        logger.info("🚀 بدء التحليل الكامل النهائي...")
        
        try:
            # 1. جلب بيانات السوق
            market_data = self.fetch_market_data_optimized()
            if market_data is None:
                return {"status": "error", "error": "فشل جلب بيانات السوق"}

            # 2. حساب المؤشرات الفنية
            gold_data = self.calculate_comprehensive_technical_indicators(market_data)
            if gold_data.empty:
                return {"status": "error", "error": "فشل حساب المؤشرات الفنية"}

            # 3. تحليل الأخبار (بالتوازي)
            news_future = None
            if self.news_api_key:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    news_future = executor.submit(self.enhanced_news_analysis)

            # 4. حساب المؤشرات المتخصصة بالذهب
            gold_indicators = self.calculate_gold_specific_indicators(gold_data, market_data)

            # 5. تشغيل الباك تيست
            backtest_results = self.run_simple_backtest(gold_data)

            # 6. انتظار تحليل الأخبار
            if news_future:
                news_result = news_future.result()
            else:
                news_result = {"status": "skipped", "news_score": 0, "headlines": [], "confidence": 0}

            # 7. حساب النقاط النهائية
            scores = self.calculate_final_scores(gold_data, market_data, gold_indicators, news_result)

            # 8. حساب النتيجة الإجمالية مع الأوزان
            weights = {
                'trend': 0.30,
                'momentum': 0.25,
                'correlation': 0.20, 
                'gold_specific': 0.10,
                'volatility': 0.10,
                'seasonal': 0.05
            }
            
            technical_score = sum(scores[component] * weights[component] for component in weights)
            news_contribution = news_result.get('news_score', 0) * 0.15  # 15% للأخبار
            final_score = technical_score + news_contribution

            # 9. تحديد الإشارة
            if final_score >= 1.5:
                signal, strength = "Buy", "Very Strong Buy"
            elif final_score >= 1.0:
                signal, strength = "Buy", "Strong Buy"
            elif final_score >= 0.5:
                signal, strength = "Buy", "Buy"
            elif final_score <= -1.5:
                signal, strength = "Sell", "Very Strong Sell"
            elif final_score <= -1.0:
                signal, strength = "Sell", "Strong Sell"
            elif final_score <= -0.5:
                signal, strength = "Sell", "Sell"
            else:
                signal, strength = "Hold", "Hold"

            # 10. حساب الثقة
            confidence_factors = [
                min(abs(final_score) / 2, 1),  # ثقة من النتيجة
                news_result.get('confidence', 0),  # ثقة الأخبار
                min(backtest_results['win_rate_percent'] / 100, 1),  # ثقة الباك تيست
                min(abs(backtest_results['sharpe_ratio']) / 2, 1) if backtest_results['sharpe_ratio'] > 0 else 0
            ]
            overall_confidence = np.mean(confidence_factors)

            # 11. إدارة المخاطر
            latest = gold_data.iloc[-1]
            current_price = latest['Close']
            atr = latest['ATR']
            
            if 'buy' in signal.lower():
                stop_loss = current_price - (2.5 * atr)
                take_profit = current_price + (4 * atr)
            elif 'sell' in signal.lower():
                stop_loss = current_price + (2.5 * atr) 
                take_profit = current_price - (4 * atr)
            else:
                stop_loss = current_price
                take_profit = current_price

            # حساب حجم المركز (2% مخاطرة)
            risk_amount = abs(current_price - stop_loss) / current_price
            position_size = min(0.02 / risk_amount if risk_amount > 0 else 0.1, 0.25) * 100

            # 12. إعداد النتيجة النهائية
            execution_time = round((time.time() - start_time) * 1000)
            
            final_result = {
                "timestamp_utc": datetime.utcnow().isoformat(),
                "execution_time_ms": execution_time,
                "status": "success",

                # الإشارة الرئيسية
                "signal": signal,
                "signal_strength": strength,
                "total_score": round(final_score, 3),
                "confidence_level": round(overall_confidence, 3),

                # مكونات النقاط
                "score_components": {k: round(v, 3) for k, v in scores.items()},
                "component_weights": weights,
                "technical_score": round(technical_score, 3),
                "news_contribution": round(news_contribution, 3),

                # بيانات السوق
                "market_data": {
                    "gold_price": round(current_price, 2),
                    "dxy": round(market_data[('Close', self.symbols['dxy'])].iloc[-1], 2) if ('Close', self.symbols['dxy']) in market_data.columns else 0,
                    "vix": round(market_data[('Close', self.symbols['vix'])].iloc[-1], 2) if ('Close', self.symbols['vix']) in market_data.columns else 0,
                    "gold_silver_ratio": round(gold_indicators.get('gold_silver_ratio', 75), 2),
                    "gold_oil_ratio": round(gold_indicators.get('gold_oil_ratio', 20), 2),
                },

                # المؤشرات الفنية
                "technical_indicators": {
                    "rsi": round(latest['RSI'], 2),
                    "macd_signal": "bullish" if latest['MACD'] > latest['MACD_Signal'] else "bearish",
                    "williams_r": round(latest['Williams_R'], 2),
                    "cci": round(latest['CCI'], 2),
                    "atr": round(atr, 2)
                },

                # المؤشرات المتخصصة
                "gold_specific_analysis": {
                    "seasonal_analysis": {
                        "current_month": gold_indicators.get('current_month', 0),
                        "is_strong_season": gold_indicators.get('is_strong_season', False),
                        "seasonal_score": gold_indicators.get('seasonal_score', 0)
                    },
                    "volatility_regime": {
                        "current_volatility": round(gold_indicators.get('volatility', 0.2), 3),
                        "regime": gold_indicators.get('volatility_regime', 'normal'),
                        "score": gold_indicators.get('volatility_score', 0)
                    },
                    "support_resistance": {
                        "resistance_level": round(gold_indicators.get('resistance_level', current_price), 2),
                        "support_level": round(gold_indicators.get('support_level', current_price), 2),
                        "score": gold_indicators.get('support_resistance_score', 0)
                    }
                },

                # إدارة المخاطر
                "risk_management": {
                    "stop_loss_price": round(stop_loss, 2),
                    "take_profit_price": round(take_profit, 2),
                    "position_size_percent": round(position_size, 1),
                    "risk_reward_ratio": round(abs(take_profit - current_price) / abs(current_price - stop_loss), 2) if abs(current_price - stop_loss) > 0 else 0
                },

                # تحليل الأخبار
                "news_analysis": news_result,

                # نتائج الباك تيست
                "backtest_results": backtest_results,

                # معلومات الأداء
                "performance_info": {
                    "data_points_analyzed": len(gold_data),
                    "indicators_calculated": len(gold_data.columns),
                    "news_articles_processed": len(news_result.get('headlines', [])),
                    "backtest_period_days": backtest_results.get('test_period_days', 0)
                }
            }

            # 13. حفظ النتائج
            analysis_id = self._save_results_to_database(final_result)
            final_result["analysis_id"] = analysis_id

            # حفظ في JSON
            with open("gold_analysis_pro.json", 'w', encoding='utf-8') as f:
                json.dump(final_result, f, ensure_ascii=False, indent=2)

            logger.info(f"✅ اكتمل التحليل في {execution_time}ms")
            logger.info(f"🎯 الإشارة: {signal} ({strength}) - النتيجة: {final_score:.3f}")

            return final_result

        except Exception as e:
            logger.error(f"❌ خطأ في التحليل الكامل: {e}")
            return {
                "status": "error",
                "error": str(e), 
                "execution_time_ms": round((time.time() - start_time) * 1000)
            }

    def _save_results_to_database(self, result: Dict) -> int:
        """حفظ النتائج في قاعدة البيانات"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO analysis_history (
                    timestamp_utc, signal, signal_strength, total_score, confidence_level,
                    trend_score, momentum_score, correlation_score, news_score,
                    volatility_score, seasonal_score, gold_specific_score,
                    gold_price, dxy_value, vix_value, gold_silver_ratio,
                    stop_loss_price, take_profit_price, position_size,
                    backtest_return, backtest_sharpe, backtest_max_dd, backtest_win_rate,
                    execution_time_ms, news_articles_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result['timestamp_utc'], result['signal'], result['signal_strength'],
                result['total_score'], result['confidence_level'],
                result['score_components']['trend'], result['score_components']['momentum'],
                result['score_components']['correlation'], result['news_contribution'],
                result['score_components']['volatility'], result['score_components']['seasonal'],
                result['score_components']['gold_specific'],
                result['market_data']['gold_price'], result['market_data']['dxy'],
                result['market_data']['vix'], result['market_data']['gold_silver_ratio'],
                result['risk_management']['stop_loss_price'],
                result['risk_management']['take_profit_price'],
                result['risk_management']['position_size_percent'],
                result['backtest_results']['total_return_percent'],
                result['backtest_results']['sharpe_ratio'],
                result['backtest_results']['max_drawdown_percent'],
                result['backtest_results']['win_rate_percent'],
                result['execution_time_ms'],
                len(result['news_analysis'].get('headlines', []))
            ))
            
            analysis_id = cursor.lastrowid
            
            # حفظ الأخبار
            for headline in result['news_analysis'].get('headlines', []):
                cursor.execute('''
                    INSERT INTO news_archive 
                    (analysis_id, headline, source, sentiment_score, relevance_score, keywords)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    analysis_id, headline.get('title', ''),
                    headline.get('source', ''), headline.get('sentiment_score', 0),
                    headline.get('relevance_score', 0),
                    json.dumps(headline.get('matched_keywords', []))
                ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"💾 تم الحفظ في قاعدة البيانات - ID: {analysis_id}")
            return analysis_id
            
        except Exception as e:
            logger.warning(f"⚠️ تحذير حفظ قاعدة البيانات: {e}")
            return -1

def main():
    """الدالة الرئيسية"""
    try:
        print("🏆 محلل الذهب الاحترافي النهائي")
        print("="*50)
        
        # إنشاء المحلل
        analyzer = ProfessionalGoldAnalyzerFinal()
        
        # تشغيل التحليل
        results = analyzer.run_complete_analysis()
        
        if results.get("status") == "error":
            print(f"❌ فشل التحليل: {results.get('error')}")
            return
        
        # عرض النتائج
        print(f"\n⏱️  وقت التنفيذ: {results['execution_time_ms']}ms")
        print(f"🎯 الإشارة: {results['signal']} ({results['signal_strength']})")
        print(f"📊 النتيجة الإجمالية: {results['total_score']}")
        print(f"🔒 مستوى الثقة: {results['confidence_level']:.1%}")
        print(f"💰 سعر الذهب: ${results['market_data']['gold_price']}")
        print(f"🛑 وقف الخسارة: ${results['risk_management']['stop_loss_price']}")
        print(f"🎯 جني الأرباح: ${results['risk_management']['take_profit_price']}")
        print(f"📏 حجم المركز: {results['risk_management']['position_size_percent']:.1f}%")
        
        print(f"\n📈 مكونات التحليل:")
        for component, score in results['score_components'].items():
            print(f"  • {component.replace('_', ' ').title()}: {score:.3f}")
            
        print(f"\n🔬 باك تيست:")
        bt = results['backtest_results']
        print(f"  • العائد: {bt['total_return_percent']:.2f}%")
        print(f"  • شارب: {bt['sharpe_ratio']:.2f}")
        print(f"  • انخفاض: {bt['max_drawdown_percent']:.2f}%")
        print(f"  • فوز: {bt['win_rate_percent']:.1f}%")
        
        print(f"\n📰 الأخبار:")
        news = results['news_analysis']
        print(f"  • الحالة: {news['status']}")
        print(f"  • النتيجة: {news.get('news_score', 0):.3f}")
        print(f"  • المقالات: {len(news.get('headlines', []))}")
        
        if news.get('headlines'):
            print(f"\n📋 أهم العناوين:")
            for i, h in enumerate(news['headlines'][:5], 1):
                print(f"  {i}. {h['title'][:60]}... [{h['source']}]")
        
        print(f"\n💾 تم الحفظ في:")
        print("  • gold_analysis_pro.json")
        print("  • gold_analysis_history.db")
        print("  • gold_analysis_pro.log")
        
        print("\n🎉 تم إنجاز التحليل بنجاح!")
        
    except Exception as e:
        logger.error(f"❌ خطأ في التشغيل: {e}")
        print(f"💥 خطأ: {e}")

if __name__ == "__main__":
    main()