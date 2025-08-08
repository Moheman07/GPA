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
import pytz
import pandas_ta as ta
import warnings
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')

# إعداد نظام التسجيل المحسن
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gold_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProfessionalGoldAnalyzerV2:
    def __init__(self):
        # استخدام رموز أكثر دقة للذهب
        self.symbols = {
            'gold': 'GC=F',  # Gold Futures - أكثر دقة من GLD
            'gold_etf': 'GLD',  # كبديل احتياطي
            'dxy': 'DX-Y.NYB', 
            'vix': '^VIX',
            'treasury': '^TNX', 
            'oil': 'CL=F', 
            'spy': 'SPY',
            'silver': 'SI=F',  # إضافة الفضة للتحليل المعادن الثمينة
            'copper': 'HG=F'   # إضافة النحاس كمؤشر اقتصادي
        }
        
        self.news_api_key = os.getenv("NEWS_API_KEY")
        self.sentiment_pipeline = None
        self.db_path = "gold_analysis_history.db"
        
        # إعداد قاعدة البيانات
        self._setup_database()
        
        # تحميل نموذج تحليل المشاعر
        self._load_sentiment_model()
        
        logger.info("🚀 تم تهيئة محلل الذهب المتقدم بنجاح")

    def _setup_database(self):
        """إعداد قاعدة بيانات السجل التاريخي"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # إنشاء جدول النتائج التاريخية
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analysis_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp_utc TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    total_score REAL NOT NULL,
                    trend_score REAL,
                    momentum_score REAL,
                    correlation_score REAL,
                    news_score REAL,
                    volatility_score REAL,
                    gold_price REAL,
                    dxy_value REAL,
                    vix_value REAL,
                    signal_strength TEXT,
                    stop_loss_price REAL,
                    news_sentiment_score REAL,
                    market_volatility TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # جدول الأخبار المرتبطة بالذهب
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS news_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id INTEGER,
                    headline TEXT,
                    source TEXT,
                    sentiment_score REAL,
                    relevance_score INTEGER,
                    published_at TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (analysis_id) REFERENCES analysis_history (id)
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("✅ تم إعداد قاعدة البيانات بنجاح")
            
        except Exception as e:
            logger.error(f"❌ خطأ في إعداد قاعدة البيانات: {e}")

    def _load_sentiment_model(self):
        """تحميل نموذج تحليل المشاعر المالي"""
        try:
            logger.info("🧠 تحميل نموذج تحليل المشاعر المالي المتخصص...")
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis", 
                model="ProsusAI/finbert",
                return_all_scores=True
            )
            logger.info("✅ نموذج تحليل المشاعر جاهز")
        except Exception as e:
            logger.error(f"⚠️ فشل تحميل نموذج المشاعر: {e}")

    def fetch_market_data(self) -> Optional[pd.DataFrame]:
        """جلب بيانات السوق مع معالجة أفضل للأخطاء"""
        logger.info("📊 جلب بيانات السوق المحسنة...")
        
        try:
            # محاولة جلب بيانات العقود الآجلة للذهب أولاً
            symbols_to_fetch = list(self.symbols.values())
            data = yf.download(symbols_to_fetch, period="2y", interval="1d", threads=True)
            
            if data.empty:
                raise ValueError("فشل في جلب أي بيانات من Yahoo Finance")
            
            # التحقق من وجود بيانات الذهب
            gold_column = ('Close', self.symbols["gold"])
            if gold_column not in data.columns or data[gold_column].dropna().empty:
                logger.warning("⚠️ لا توجد بيانات للعقود الآجلة، التبديل إلى GLD...")
                # التبديل إلى GLD كبديل
                self.symbols['gold'] = 'GLD'
                symbols_to_fetch[0] = 'GLD'
                data = yf.download(symbols_to_fetch, period="2y", interval="1d", threads=True)
            
            # تنظيف البيانات
            data = data.dropna(subset=[('Close', self.symbols["gold"])])
            
            if len(data) < 100:
                raise ValueError(f"البيانات غير كافية: {len(data)} يوم فقط")
            
            logger.info(f"✅ تم جلب {len(data)} يوم من البيانات بنجاح")
            return data
            
        except Exception as e:
            logger.error(f"❌ خطأ في جلب بيانات السوق: {e}")
            return None

    def analyze_gold_news(self) -> Dict:
        """
        محرك تحليل الأخبار المتخصص والمحسن للذهب
        """
        logger.info("📰 بدء تحليل أخبار الذهب المتخصص...")
        
        if not self.news_api_key or not self.sentiment_pipeline:
            logger.warning("⚠️ مفتاح الأخبار أو نموذج المشاعر غير متاح")
            return {"status": "skipped", "news_score": 0, "headlines": [], "analysis_details": {}}

        try:
            # كلمات مفتاحية متخصصة للذهب مع أوزان محسنة
            gold_keywords = {
                # كلمات الذهب المباشرة - أعلى وزن
                'gold': 5, 'xau': 5, 'bullion': 5, 'precious metal': 5, 'precious metals': 5,
                
                # العوامل الاقتصادية المؤثرة على الذهب - وزن متوسط عالي
                'federal reserve': 4, 'fed': 4, 'interest rate': 4, 'interest rates': 4,
                'inflation': 4, 'cpi': 4, 'consumer price index': 4,
                'quantitative easing': 4, 'monetary policy': 4,
                'dollar index': 4, 'dxy': 4, 'dollar strength': 4, 'dollar weakness': 4,
                
                # العوامل الجيوسياسية - وزن متوسط
                'geopolitical': 3, 'geopolitical tension': 3, 'war': 3, 'conflict': 3,
                'sanctions': 3, 'trade war': 3, 'tariff': 3, 'tariffs': 3,
                'safe haven': 3, 'safe-haven': 3, 'risk-off': 3, 'risk off': 3,
                
                # المؤشرات الاقتصادية - وزن متوسط منخفض
                'nfp': 2, 'non-farm payroll': 2, 'unemployment': 2, 'gdp': 2,
                'retail sales': 2, 'manufacturing': 2, 'pmi': 2,
                
                # أسواق أخرى مؤثرة - وزن منخفض
                'stock market': 1, 'equity': 1, 'bonds': 1, 'treasury': 1,
                'oil prices': 1, 'commodity': 1, 'mining': 1
            }

            # استعلامات متعددة لضمان شمولية أكبر
            queries = [
                'gold OR XAU OR bullion OR "precious metals"',
                '"interest rates" OR "federal reserve" OR inflation OR CPI',
                '"dollar index" OR DXY OR "monetary policy"',
                'geopolitical OR "safe haven" OR "risk off"'
            ]

            all_articles = []
            
            for query in queries:
                try:
                    url = (
                        f"https://newsapi.org/v2/everything?"
                        f"q={query}&language=en&sortBy=publishedAt&pageSize=50&"
                        f"from={(datetime.now() - timedelta(days=3)).date()}&"
                        f"apiKey={self.news_api_key}"
                    )
                    
                    response = requests.get(url, timeout=15)
                    response.raise_for_status()
                    articles = response.json().get('articles', [])
                    all_articles.extend(articles)
                    logger.info(f"📥 جلب {len(articles)} مقال من استعلام: {query[:30]}...")
                    
                except Exception as e:
                    logger.warning(f"⚠️ فشل في استعلام: {query[:30]}... - {e}")
                    continue

            if not all_articles:
                raise ValueError("لم يتم العثور على أي مقالات")

            # إزالة المقالات المكررة
            unique_articles = []
            seen_titles = set()
            for article in all_articles:
                title = article.get('title', '').lower()
                if title and title not in seen_titles:
                    seen_titles.add(title)
                    unique_articles.append(article)

            logger.info(f"🔍 تم جلب {len(unique_articles)} مقالاً فريداً")

            # تقييم وفلترة المقالات
            relevant_articles = []
            for article in unique_articles:
                content_text = f"{(article.get('title') or '').lower()} {(article.get('description') or '').lower()}"
                
                # حساب نقاط الصلة بالذهب
                relevance_score = 0
                matched_keywords = []
                
                for keyword, weight in gold_keywords.items():
                    if keyword in content_text:
                        relevance_score += weight
                        matched_keywords.append(keyword)
                
                # قبول المقالات ذات الصلة العالية بالذهب
                if relevance_score >= 3:  # تم تقليل الحد الأدنى
                    article['relevance_score'] = relevance_score
                    article['matched_keywords'] = matched_keywords
                    relevant_articles.append(article)

            if not relevant_articles:
                raise ValueError("لا توجد مقالات ذات صلة كافية بالذهب")

            logger.info(f"🎯 تم اختيار {len(relevant_articles)} مقالاً ذا صلة عالية بالذهب")

            # تحليل المشاعر المفصل
            sentiment_scores = []
            processed_articles = []

            for article in relevant_articles:
                try:
                    text_for_analysis = article.get('description') or article.get('title') or ""
                    if not text_for_analysis:
                        continue
                        
                    # تحليل المشاعر مع النتائج الكاملة
                    sentiment_results = self.sentiment_pipeline(text_for_analysis[:512])  # قطع النص للطول المناسب
                    
                    # استخراج النتيجة النهائية
                    positive_score = next((s['score'] for s in sentiment_results[0] if s['label'] == 'positive'), 0)
                    negative_score = next((s['score'] for s in sentiment_results[0] if s['label'] == 'negative'), 0)
                    neutral_score = next((s['score'] for s in sentiment_results[0] if s['label'] == 'neutral'), 0)
                    
                    # حساب النتيجة النهائية (بين -1 و +1)
                    final_sentiment = positive_score - negative_score
                    
                    # وزن النتيجة حسب أهمية المقال
                    weighted_sentiment = final_sentiment * (article['relevance_score'] / 10)
                    
                    sentiment_scores.append(weighted_sentiment)
                    
                    processed_articles.append({
                        'title': article['title'],
                        'source': article.get('source', {}).get('name', 'Unknown'),
                        'sentiment_score': round(final_sentiment, 3),
                        'relevance_score': article['relevance_score'],
                        'matched_keywords': article['matched_keywords'][:3],  # أهم 3 كلمات
                        'published_at': article.get('publishedAt')
                    })
                    
                except Exception as e:
                    logger.warning(f"⚠️ خطأ في تحليل مشاعر مقال: {e}")
                    continue

            if not sentiment_scores:
                raise ValueError("فشل في تحليل مشاعر أي من المقالات")

            # حساب النتيجة النهائية
            average_sentiment = np.mean(sentiment_scores)
            sentiment_std = np.std(sentiment_scores)
            
            # ترتيب المقالات حسب الأهمية والصلة
            processed_articles.sort(key=lambda x: (x['relevance_score'], abs(x['sentiment_score'])), reverse=True)

            analysis_details = {
                'total_articles_analyzed': len(processed_articles),
                'average_sentiment': round(average_sentiment, 3),
                'sentiment_volatility': round(sentiment_std, 3),
                'positive_articles': len([a for a in processed_articles if a['sentiment_score'] > 0.1]),
                'negative_articles': len([a for a in processed_articles if a['sentiment_score'] < -0.1]),
                'neutral_articles': len([a for a in processed_articles if abs(a['sentiment_score']) <= 0.1])
            }

            logger.info(f"📊 تحليل الأخبار مكتمل: النتيجة النهائية {average_sentiment:.3f}")

            return {
                "status": "success",
                "news_score": round(average_sentiment, 3),
                "headlines": processed_articles[:8],  # أهم 8 مقالات
                "analysis_details": analysis_details
            }

        except Exception as e:
            logger.error(f"❌ خطأ في تحليل أخبار الذهب: {e}")
            return {
                "status": "error", 
                "news_score": 0, 
                "headlines": [],
                "analysis_details": {"error": str(e)}
            }

    def calculate_technical_indicators(self, gold_data: pd.DataFrame) -> pd.DataFrame:
        """حساب المؤشرات الفنية المحسنة"""
        logger.info("📈 حساب المؤشرات الفنية المتقدمة...")
        
        try:
            # المؤشرات الأساسية
            gold_data['SMA_20'] = ta.sma(gold_data['Close'], length=20)
            gold_data['SMA_50'] = ta.sma(gold_data['Close'], length=50)
            gold_data['SMA_200'] = ta.sma(gold_data['Close'], length=200)
            gold_data['EMA_12'] = ta.ema(gold_data['Close'], length=12)
            gold_data['EMA_26'] = ta.ema(gold_data['Close'], length=26)
            
            # مؤشرات الزخم
            gold_data['RSI'] = ta.rsi(gold_data['Close'], length=14)
            macd = ta.macd(gold_data['Close'])
            gold_data['MACD'] = macd['MACD_12_26_9']
            gold_data['MACD_Signal'] = macd['MACDs_12_26_9']
            gold_data['MACD_Hist'] = macd['MACDh_12_26_9']
            
            # Bollinger Bands
            bbands = ta.bbands(gold_data['Close'])
            gold_data['BB_Upper'] = bbands['BBU_5_2.0']
            gold_data['BB_Middle'] = bbands['BBM_5_2.0']
            gold_data['BB_Lower'] = bbands['BBL_5_2.0']
            
            # مؤشرات التقلبات
            gold_data['ATR'] = ta.atr(gold_data['High'], gold_data['Low'], gold_data['Close'], length=14)
            
            # مؤشرات الحجم
            gold_data['OBV'] = ta.obv(gold_data['Close'], gold_data['Volume'])
            
            # مؤشرات إضافية للذهب
            gold_data['Williams_R'] = ta.willr(gold_data['High'], gold_data['Low'], gold_data['Close'])
            gold_data['CCI'] = ta.cci(gold_data['High'], gold_data['Low'], gold_data['Close'])
            gold_data['Stoch_K'] = ta.stoch(gold_data['High'], gold_data['Low'], gold_data['Close'])['STOCHk_14_3_3']
            
            # إزالة الصفوف التي تحتوي على NaN
            gold_data.dropna(inplace=True)
            logger.info(f"✅ تم حساب المؤشرات الفنية - البيانات النظيفة: {len(gold_data)} صف")
            
            return gold_data
            
        except Exception as e:
            logger.error(f"❌ خطأ في حساب المؤشرات الفنية: {e}")
            return gold_data

    def calculate_adaptive_weights(self, vix_value: float, market_trend: str) -> Dict[str, float]:
        """حساب الأوزان المتكيفة حسب ظروف السوق"""
        if vix_value > 25:  # سوق عالي التقلب
            return {
                'trend': 0.25, 'momentum': 0.20, 'correlation': 0.30, 
                'news': 0.15, 'volatility': 0.10
            }
        elif vix_value < 15:  # سوق منخفض التقلب
            return {
                'trend': 0.40, 'momentum': 0.35, 'correlation': 0.15, 
                'news': 0.05, 'volatility': 0.05
            }
        else:  # سوق متوسط التقلب
            return {
                'trend': 0.35, 'momentum': 0.25, 'correlation': 0.20, 
                'news': 0.15, 'volatility': 0.05
            }

    def determine_signal_strength(self, total_score: float) -> str:
        """تحديد قوة الإشارة"""
        if total_score >= 2.0:
            return "Very Strong Buy"
        elif total_score >= 1.5:
            return "Strong Buy"
        elif total_score >= 1.0:
            return "Buy"
        elif total_score >= 0.5:
            return "Weak Buy"
        elif total_score >= -0.5:
            return "Hold"
        elif total_score >= -1.0:
            return "Weak Sell"
        elif total_score >= -1.5:
            return "Sell"
        elif total_score >= -2.0:
            return "Strong Sell"
        else:
            return "Very Strong Sell"

    def calculate_stop_loss(self, current_price: float, atr_value: float, signal: str) -> float:
        """حساب مستوى وقف الخسارة"""
        if signal.lower() in ['buy', 'strong buy', 'very strong buy']:
            return round(current_price - (2.5 * atr_value), 2)
        elif signal.lower() in ['sell', 'strong sell', 'very strong sell']:
            return round(current_price + (2.5 * atr_value), 2)
        else:
            return current_price

    def save_to_history(self, analysis_result: Dict) -> int:
        """حفظ النتائج في السجل التاريخي"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # حفظ النتيجة الرئيسية
            cursor.execute('''
                INSERT INTO analysis_history 
                (timestamp_utc, signal, total_score, trend_score, momentum_score, 
                 correlation_score, news_score, volatility_score, gold_price, dxy_value, 
                 vix_value, signal_strength, stop_loss_price, news_sentiment_score, market_volatility)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                analysis_result['timestamp_utc'],
                analysis_result['signal'],
                analysis_result['total_score'],
                analysis_result['components']['trend_score'],
                analysis_result['components']['momentum_score'],
                analysis_result['components']['correlation_score'],
                analysis_result['components']['news_score'],
                analysis_result['components'].get('volatility_score', 0),
                analysis_result['market_data']['gold_price'],
                analysis_result['market_data']['dxy'],
                analysis_result['market_data']['vix'],
                analysis_result['signal_strength'],
                analysis_result.get('stop_loss_price', 0),
                analysis_result['news_analysis'].get('news_sentiment_score', 0),
                analysis_result.get('market_volatility', 'normal')
            ))
            
            analysis_id = cursor.lastrowid
            
            # حفظ الأخبار المرتبطة
            headlines = analysis_result['news_analysis'].get('headlines', [])
            for headline in headlines:
                cursor.execute('''
                    INSERT INTO news_history 
                    (analysis_id, headline, source, sentiment_score, relevance_score, published_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    analysis_id,
                    headline.get('title', ''),
                    headline.get('source', ''),
                    headline.get('sentiment_score', 0),
                    headline.get('relevance_score', 0),
                    headline.get('published_at', '')
                ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"💾 تم حفظ النتائج في السجل التاريخي - ID: {analysis_id}")
            return analysis_id
            
        except Exception as e:
            logger.error(f"❌ خطأ في حفظ السجل التاريخي: {e}")
            return -1

    def get_historical_performance(self, days: int = 30) -> Dict:
        """استخراج أداء الإشارات التاريخية"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT signal, total_score, gold_price, timestamp_utc, signal_strength
                FROM analysis_history 
                WHERE datetime(timestamp_utc) >= datetime('now', '-{} days')
                ORDER BY timestamp_utc DESC
            '''.format(days)
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if df.empty:
                return {"status": "no_data", "message": "لا توجد بيانات تاريخية كافية"}
            
            # حساب إحصائيات الأداء
            performance_stats = {
                "total_signals": len(df),
                "buy_signals": len(df[df['signal'].str.contains('Buy', na=False)]),
                "sell_signals": len(df[df['signal'].str.contains('Sell', na=False)]),
                "hold_signals": len(df[df['signal'] == 'Hold']),
                "average_score": round(df['total_score'].mean(), 3),
                "score_volatility": round(df['total_score'].std(), 3),
                "latest_signals": df.head(10).to_dict('records')
            }
            
            return {"status": "success", "performance": performance_stats}
            
        except Exception as e:
            logger.error(f"❌ خطأ في استخراج الأداء التاريخي: {e}")
            return {"status": "error", "error": str(e)}

    def run_comprehensive_analysis(self) -> Dict:
        """تشغيل التحليل الشامل المحسن"""
        logger.info("🚀 بدء التحليل الشامل للذهب...")
        
        # جلب بيانات السوق
        market_data = self.fetch_market_data()
        if market_data is None:
            return {"status": "error", "error": "فشل جلب بيانات السوق"}
        
        # إعداد بيانات الذهب
        gold_ticker = self.symbols['gold']
        gold_data = pd.DataFrame({
            'Open': market_data[('Open', gold_ticker)],
            'High': market_data[('High', gold_ticker)],
            'Low': market_data[('Low', gold_ticker)],
            'Close': market_data[('Close', gold_ticker)],
            'Volume': market_data[('Volume', gold_ticker)]
        }).dropna()
        
        # حساب المؤشرات الفنية
        gold_data = self.calculate_technical_indicators(gold_data)
        
        if gold_data.empty:
            return {"status": "error", "error": "فشل في حساب المؤشرات الفنية"}
        
        # تحليل الأخبار المتخصص
        logger.info("📰 بدء تحليل الأخبار المتخصص...")
        news_analysis = self.analyze_gold_news()
        
        # استخراج القيم الحالية
        latest = gold_data.iloc[-1]
        current_price = latest['Close']
        current_atr = latest['ATR']
        
        # قيم السوق الأخرى
        dxy_current = market_data[('Close', self.symbols['dxy'])].iloc[-1]
        vix_current = market_data[('Close', self.symbols['vix'])].iloc[-1]
        
        # تحديد حالة السوق
        market_volatility = "high" if vix_current > 25 else "low" if vix_current < 15 else "normal"
        
        # حساب نقاط التقييم المحسنة
        scores = {}
        
        # 1. نقاط الاتجاه (Trend Score)
        if current_price > latest['SMA_200']:
            if current_price > latest['SMA_50']:
                scores['trend'] = 3.0 if current_price > latest['SMA_20'] else 2.0
            else:
                scores['trend'] = 1.0
        else:
            if current_price < latest['SMA_50']:
                scores['trend'] = -3.0 if current_price < latest['SMA_20'] else -2.0
            else:
                scores['trend'] = -1.0
        
        # 2. نقاط الزخم (Momentum Score)
        momentum_score = 0
        if latest['MACD'] > latest['MACD_Signal']:
            momentum_score += 1.5
        if latest['RSI'] > 50:
            momentum_score += 1.0
        elif latest['RSI'] > 70:
            momentum_score -= 0.5  # تشبع شراء
        if latest['Williams_R'] > -50:
            momentum_score += 0.5
        scores['momentum'] = momentum_score
        
        # 3. نقاط الارتباط (Correlation Score)
        gold_dxy_corr = market_data[('Close', gold_ticker)].tail(50).corr(
            market_data[('Close', self.symbols['dxy'])].tail(50)
        )
        if gold_dxy_corr < -0.6:
            scores['correlation'] = 2.0
        elif gold_dxy_corr < -0.3:
            scores['correlation'] = 1.0
        else:
            scores['correlation'] = -1.0
        
        # 4. نقاط الأخبار
        scores['news'] = news_analysis.get('news_score', 0) * 2  # تضخيم تأثير الأخبار
        
        # 5. نقاط التقلبات
        if vix_current > 25:
            scores['volatility'] = 1.5  # تقلبات عالية تفيد الذهب
        elif vix_current < 15:
            scores['volatility'] = -0.5
        else:
            scores['volatility'] = 0
        
        # حساب الأوزان المتكيفة
        weights = self.calculate_adaptive_weights(vix_current, "trend_up" if scores['trend'] > 0 else "trend_down")
        
        # حساب النتيجة الإجمالية
        total_score = sum(scores[key] * weights[key] for key in scores.keys() if key in weights)
        
        # تحديد الإشارة وقوتها
        signal_strength = self.determine_signal_strength(total_score)
        basic_signal = "Buy" if total_score >= 1.0 else "Sell" if total_score <= -1.0 else "Hold"
        
        # حساب مستوى وقف الخسارة
        stop_loss = self.calculate_stop_loss(current_price, current_atr, basic_signal)
        
        # إعداد النتيجة النهائية
        analysis_result = {
            "timestamp_utc": datetime.utcnow().isoformat(),
            "signal": basic_signal,
            "signal_strength": signal_strength,
            "total_score": round(total_score, 3),
            "components": {
                "trend_score": round(scores['trend'], 2),
                "momentum_score": round(scores['momentum'], 2),
                "correlation_score": round(scores['correlation'], 2),
                "news_score": round(scores['news'], 2),
                "volatility_score": round(scores['volatility'], 2)
            },
            "weights_used": weights,
            "market_data": {
                "gold_price": round(current_price, 2),
                "dxy": round(dxy_current, 2),
                "vix": round(vix_current, 2),
                "atr": round(current_atr, 2)
            },
            "market_volatility": market_volatility,
            "stop_loss_price": stop_loss,
            "technical_indicators": {
                "rsi": round(latest['RSI'], 2),
                "macd": round(latest['MACD'], 4),
                "williams_r": round(latest['Williams_R'], 2),
                "sma_50": round(latest['SMA_50'], 2),
                "sma_200": round(latest['SMA_200'], 2)
            },
            "news_analysis": news_analysis
        }
        
        # حفظ في السجل التاريخي
        history_id = self.save_to_history(analysis_result)
        analysis_result["history_id"] = history_id
        
        # حفظ في ملف JSON
        with open("gold_analysis_enhanced.json", 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, ensure_ascii=False, indent=2)
        
        logger.info("✅ تم إنجاز التحليل الشامل بنجاح")
        
        # إضافة الأداء التاريخي
        performance = self.get_historical_performance(30)
        analysis_result["historical_performance"] = performance
        
        return analysis_result

# === تشغيل التحليل ===
if __name__ == "__main__":
    try:
        analyzer = ProfessionalGoldAnalyzerV2()
        results = analyzer.run_comprehensive_analysis()
        
        if results.get("status") == "error":
            logger.error(f"❌ فشل التحليل: {results.get('error')}")
        else:
            logger.info("\n" + "="*60)
            logger.info("📋 ملخص التحليل النهائي")
            logger.info("="*60)
            logger.info(f"🎯 الإشارة: {results['signal']} ({results['signal_strength']})")
            logger.info(f"📊 النتيجة الإجمالية: {results['total_score']}")
            logger.info(f"💰 سعر الذهب: ${results['market_data']['gold_price']}")
            logger.info(f"🛑 وقف الخسارة: ${results['stop_loss_price']}")
            logger.info(f"📈 حالة السوق: {results['market_volatility']}")
            logger.info(f"📰 تحليل الأخبار: {results['news_analysis']['status']} "
                      f"(النتيجة: {results['news_analysis']['news_score']})")
            
            print("\n🎉 تم حفظ التحليل الكامل في:")
            print("  - gold_analysis_enhanced.json")
            print("  - gold_analysis_history.db") 
            print("  - gold_analysis.log")
            
            # عرض أهم العناوين
            headlines = results['news_analysis'].get('headlines', [])
            if headlines:
                print(f"\n📰 أهم {min(5, len(headlines))} أخبار متعلقة بالذهب:")
                for i, headline in enumerate(headlines[:5], 1):
                    print(f"  {i}. {headline['title'][:80]}... [{headline['source']}]")
                    
    except Exception as e:
        logger.error(f"❌ خطأ عام في التطبيق: {e}")