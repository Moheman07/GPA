#!/usr/bin/env python3
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import json
import os
import sqlite3
import joblib
from datetime import datetime, timedelta
import warnings
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
from textblob import TextBlob
import spacy
import backtrader as bt
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import aiohttp

warnings.filterwarnings('ignore')

# تحميل نموذج spaCy
try:
    nlp = spacy.load("en_core_web_sm")
except:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

class MLPredictor:
    """نظام التنبؤ بالتعلم الآلي"""
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.model_path = "gold_ml_model.pkl"
        self.scaler_path = "gold_scaler.pkl"

    def prepare_features(self, analysis_data):
        """تحضير المميزات من بيانات التحليل"""
        features = {}

        if 'gold_analysis' in analysis_data:
            scores = analysis_data['gold_analysis'].get('component_scores', {})
            features.update({f'score_{k}': v for k, v in scores.items()})
            features['total_score'] = analysis_data['gold_analysis'].get('total_score', 0)

            tech_summary = analysis_data['gold_analysis'].get('technical_summary', {})
            features.update({f'tech_{k}': v for k, v in tech_summary.items()})

        if 'volume_analysis' in analysis_data:
            vol = analysis_data['volume_analysis']
            features['volume_ratio'] = vol.get('volume_ratio', 1)
            features['volume_strength_encoded'] = self._encode_volume_strength(vol.get('volume_strength', 'طبيعي'))

        if 'market_correlations' in analysis_data:
            corr = analysis_data['market_correlations'].get('correlations', {})
            features.update({f'corr_{k}': v for k, v in corr.items()})

        if 'economic_data' in analysis_data:
            features['economic_score'] = analysis_data['economic_data'].get('score', 0)

        if 'fibonacci_levels' in analysis_data:
            fib = analysis_data['fibonacci_levels']
            features['fib_position'] = fib.get('current_position', 50)

        return features

    def _encode_volume_strength(self, strength):
        """تحويل قوة الحجم إلى رقم"""
        mapping = {'ضعيف': 0, 'طبيعي': 1, 'قوي': 2, 'قوي جداً': 3}
        return mapping.get(strength, 1)

    def train_model(self, historical_data):
        """تدريب نموذج التعلم الآلي"""
        print("🤖 بدء تدريب نموذج التعلم الآلي...")

        X, y = [], []
        for record in historical_data:
            features = self.prepare_features(record['analysis'])
            if features:
                X.append(list(features.values()))
                y.append(1 if record['price_change_5d'] > 1.0 else 0)

        if len(X) < 100:
            print("⚠️ بيانات غير كافية للتدريب")
            return False

        X, y = np.array(X), np.array(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False)
        }

        best_model, best_score = None, 0

        for name, model in models.items():
            print(f"تدريب نموذج {name}...")
            model.fit(X_train_scaled, y_train)

            y_pred = model.predict(X_test_scaled)
            f1 = f1_score(y_test, y_pred)

            print(f"  - F1 Score: {f1:.2%}")

            if f1 > best_score:
                best_score = f1
                best_model = model
                self.model = model

        print(f"\n✅ أفضل نموذج: {type(best_model).__name__} مع F1 Score: {best_score:.2%}")

        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)

        return True

    def predict_probability(self, analysis_data):
        """التنبؤ باحتمالية نجاح الإشارة"""
        try:
            if self.model is None:
                if os.path.exists(self.model_path):
                    self.model = joblib.load(self.model_path)
                    self.scaler = joblib.load(self.scaler_path)
                else:
                    return None, "النموذج غير مدرب بعد"

            features = self.prepare_features(analysis_data)
            X = np.array([list(features.values())])

            X_scaled = self.scaler.transform(X)
            probability = self.model.predict_proba(X_scaled)[0][1]

            if probability > 0.75:
                interpretation = "احتمالية عالية جداً للنجاح"
            elif probability > 0.60:
                interpretation = "احتمالية جيدة للنجاح"
            elif probability > 0.45:
                interpretation = "احتمالية متوسطة - حذر"
            else:
                interpretation = "احتمالية منخفضة - تجنب"

            return probability, interpretation

        except Exception as e:
            print(f"خطأ في التنبؤ: {e}")
            return None, str(e)

class MultiTimeframeAnalyzer:
    """محلل متعدد الأطر الزمنية"""
    def __init__(self):
        self.timeframes = {
            '1h': {'period': '5d', 'weight': 0.2},
            '4h': {'period': '1mo', 'weight': 0.3},
            '1d': {'period': '3mo', 'weight': 0.5}
        }

    def analyze_timeframe(self, symbol, interval, period):
        """تحليل إطار زمني واحد"""
        try:
            data = yf.download(symbol, period=period, interval=interval, progress=False)
            if data.empty or len(data) < 26: # التأكد من وجود بيانات كافية
                print(f"⚠️ بيانات غير كافية للإطار الزمني {interval}")
                return None

            close_prices = data['Close']
            data['SMA_20'] = close_prices.rolling(20).mean()
            data['RSI'] = self._calculate_rsi(close_prices)
            exp1 = close_prices.ewm(span=12, adjust=False).mean()
            exp2 = close_prices.ewm(span=26, adjust=False).mean()
            data['MACD'] = exp1 - exp2
            data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
            
            data.dropna(inplace=True) # إزالة الصفوف التي تحتوي على NaN بعد الحسابات
            if data.empty:
                return None

            latest = data.iloc[-1]
            score = 0
            if latest['Close'] > latest['SMA_20']: score += 1
            if 30 <= latest['RSI'] <= 70: score += 0.5 if latest['RSI'] > 50 else -0.5
            elif latest['RSI'] < 30: score += 1
            else: score -= 1
            if latest['MACD'] > latest['MACD_Signal']: score += 1
            
            return {'score': score, 'trend': 'صاعد' if score > 0 else 'هابط', 'price': latest['Close']}

        except Exception as e:
            print(f"خطأ في تحليل الإطار الزمني {interval}: {e}")
            return None

    def _calculate_rsi(self, prices, period=14):
        """حساب RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.nan) # تجنب القسمة على صفر
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def get_coherence_score(self, symbol):
        """حساب نقاط التوافق بين الأطر الزمنية"""
        print("⏰ تحليل الأطر الزمنية المتعددة...")
        results, total_weighted_score, total_weight = {}, 0, 0

        for tf_name, tf_config in self.timeframes.items():
            interval = tf_name
            analysis = self.analyze_timeframe(symbol, interval, tf_config['period'])
            if analysis:
                results[tf_name] = analysis
                total_weighted_score += analysis['score'] * tf_config['weight']
                total_weight += tf_config['weight']

        if total_weight == 0: return 0, {}

        coherence_score = total_weighted_score / total_weight
        trends = [r['trend'] for r in results.values() if r]
        
        if trends and all(t == 'صاعد' for t in trends): coherence_analysis = "توافق كامل صاعد"
        elif trends and all(t == 'هابط' for t in trends): coherence_analysis = "توافق كامل هابط"
        else: coherence_analysis = "تضارب بين الأطر الزمنية"

        return coherence_score, {'coherence_score': round(coherence_score, 2), 'analysis': coherence_analysis}

    def _get_mtf_recommendation(self, score):
        if score > 1.5: return "دخول قوي"
        elif score > 0.5: return "دخول معتدل"
        elif score < -0.5: return "تجنب الشراء"
        else: return "انتظار"

class AdvancedNewsAnalyzer:
    """محلل أخبار متقدم مع استخراج الأحداث"""
    def __init__(self, api_key):
        self.api_key = api_key
        self.event_patterns = {
            'interest_rate': {'keywords': ['rate', 'fomc', 'federal reserve'], 'multiplier': 3},
            'inflation': {'keywords': ['inflation', 'cpi'], 'multiplier': 2.5},
            'employment': {'keywords': ['jobs', 'nfp', 'unemployment'], 'multiplier': 2},
            'geopolitical': {'keywords': ['war', 'conflict', 'tension'], 'multiplier': 2.5}
        }

    def analyze_news(self, articles):
        """تحليل بسيط للمشاعر من عناوين الأخبار"""
        if not articles: return {'total_impact': 0}
        total_impact = 0
        for article in articles:
            if not article or not article.get('title'): continue
            text = f"{article.get('title', '')} {article.get('description', '')}".lower()
            sentiment = TextBlob(text).sentiment.polarity
            for event, details in self.event_patterns.items():
                if any(kw in text for kw in details['keywords']):
                    impact = sentiment * details['multiplier']
                    if event in ['interest_rate', 'inflation']: impact *= -1
                    total_impact += impact
        
        # تطبيع النتيجة
        normalized_impact = round(total_impact / len(articles) * 5, 2) if articles else 0
        return {'total_impact': normalized_impact}

    async def fetch_news_async(self):
        """جلب الأخبار بشكل غير متزامن"""
        if not self.api_key: return []
        keywords = ['"gold price"', '"federal reserve"', '"interest rates"']
        all_articles = []
        async with aiohttp.ClientSession() as session:
            tasks = [self._fetch_url(session, f"https://newsapi.org/v2/everything?q={keyword}&language=en&sortBy=publishedAt&pageSize=10&apiKey={self.api_key}") for keyword in keywords]
            results = await asyncio.gather(*tasks)
            for result in results:
                if result and 'articles' in result: all_articles.extend(result['articles'])

        unique_articles = {article['url']: article for article in all_articles}.values()
        return list(unique_articles)

    async def _fetch_url(self, session, url):
        try:
            async with session.get(url, timeout=10) as response:
                return await response.json()
        except Exception as e:
            print(f"خطأ في جلب الأخبار: {e}")
            return None

class ProfessionalBacktester:
    """نظام اختبار خلفي احترافي"""
    class GoldStrategy(bt.Strategy):
        params = (('analyzer', None), ('risk_percent', 0.02),)
        def __init__(self): self.order = None
        def next(self):
            if self.order: return
            current_data = {'close': self.data.close[0], 'open': self.data.open[0]}
            signal = self.params.analyzer.generate_signal_for_backtest(current_data)
            if not self.position and 'Buy' in signal['action']:
                size = (self.broker.getcash() * self.params.risk_percent) / self.data.close[0]
                if size > 0: self.order = self.buy(size=size)
            elif self.position and 'Sell' in signal['action']:
                self.order = self.close()
        def notify_order(self, order):
            if order.status in [order.Submitted, order.Accepted]: return
            self.order = None

    def __init__(self, analyzer): self.analyzer = analyzer
    def run_backtest(self, data, initial_cash=10000):
        print("🔄 بدء الاختبار الخلفي الاحترافي...")
        try:
            cerebro = bt.Cerebro(stdstats=False)
            data_feed = bt.feeds.PandasData(dataname=data)
            cerebro.adddata(data_feed)
            cerebro.addstrategy(self.GoldStrategy, analyzer=self.analyzer)
            cerebro.broker.setcash(initial_cash)
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
            results = cerebro.run()
            trades = results[0].analyzers.trades.get_analysis()
            final_value = cerebro.broker.getvalue()
            total_return = (final_value - initial_cash) / initial_cash * 100
            return {'total_return': round(total_return, 2), 'total_trades': trades.get('total', {}).get('total', 0)}
        except Exception as e:
            print(f"❌ خطأ في الاختبار الخلفي: {e}")
            return None
            
class DatabaseManager:
    """مدير قاعدة البيانات للتحليلات التاريخية"""
    def __init__(self, db_path="analysis_history.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """تهيئة قاعدة البيانات"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analysis_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    analysis_date DATE UNIQUE, gold_price REAL, signal TEXT, confidence TEXT, total_score REAL,
                    component_scores TEXT, technical_indicators TEXT, volume_analysis TEXT,
                    correlations TEXT, economic_score REAL, news_sentiment REAL, mtf_coherence REAL,
                    ml_probability REAL, price_change_5d REAL, signal_success BOOLEAN
                )''')

    def save_analysis(self, analysis_data):
        """حفظ التحليل في قاعدة البيانات"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                gold_analysis = analysis_data.get('gold_analysis', {})
                if 'error' in gold_analysis:
                    print("⚠️ تم تخطي الحفظ في قاعدة البيانات بسبب وجود خطأ في التحليل")
                    return

                # ✅ تصحيح: استخدام INSERT OR REPLACE لتحديث سجل اليوم الواحد
                analysis_date_str = datetime.now().date().isoformat()
                params = (
                    analysis_date_str, gold_analysis.get('current_price'), gold_analysis.get('signal'),
                    gold_analysis.get('confidence'), gold_analysis.get('total_score'),
                    json.dumps(gold_analysis.get('component_scores', {})),
                    json.dumps(gold_analysis.get('technical_summary', {})),
                    json.dumps(analysis_data.get('volume_analysis', {})),
                    json.dumps(analysis_data.get('market_correlations', {}).get('correlations', {})),
                    analysis_data.get('economic_data', {}).get('score', 0),
                    analysis_data.get('news_analysis', {}).get('total_impact'),
                    analysis_data.get('mtf_analysis', {}).get('coherence_score', 0),
                    gold_analysis.get('ml_prediction', {}).get('probability')
                )
                conn.execute('''
                    INSERT OR REPLACE INTO analysis_history (analysis_date, gold_price, signal, confidence, total_score, component_scores, technical_indicators, volume_analysis, correlations, economic_score, news_sentiment, mtf_coherence, ml_probability) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', params)
                print("✅ تم حفظ التحليل في قاعدة البيانات")
        except sqlite3.OperationalError as e:
            print(f"❌ خطأ قاعدة بيانات أثناء الحفظ (ربما قيد الاستخدام): {e}")
        except Exception as e:
            print(f"❌ خطأ في حفظ التحليل: {e}")

    def update_future_prices(self):
        """تحديث الأسعار المستقبلية للتحليلات السابقة"""
        try:
            with sqlite3.connect(self.db_path, timeout=10) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, analysis_date, gold_price FROM analysis_history WHERE price_change_5d IS NULL AND analysis_date <= date('now', '-5 days')")
                records = cursor.fetchall()
                if not records: return

                for record_id, date_str, price in records:
                    future_data = yf.download('GC=F', start=datetime.strptime(date_str, '%Y-%m-%d').date() + timedelta(days=5), end=datetime.strptime(date_str, '%Y-%m-%d').date() + timedelta(days=10), progress=False, auto_adjust=True)
                    if not future_data.empty:
                        future_price = future_data['Close'].iloc[0]
                        price_change = ((future_price - price) / price * 100) if price else 0
                        cursor.execute("UPDATE analysis_history SET price_change_5d=?, signal_success=? WHERE id=?", (price_change, price_change > 1.0, record_id))
                print(f"✅ تم تحديث {len(records)} سجل بالأسعار المستقبلية")
        except Exception as e:
            print(f"❌ خطأ في تحديث الأسعار المستقبلية: {e}")

    def get_training_data(self, min_records=100):
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query("SELECT * FROM analysis_history WHERE signal_success IS NOT NULL", conn)
            print(f"⚠️ بيانات التدريب المتاحة: {len(df)} سجل فقط")
            if len(df) < min_records: return None
            # ... (بقية الكود)
            return df.to_dict('records') # تبسيط
        except Exception as e:
            print(f"❌ خطأ في جلب بيانات التدريب: {e}")
            return None

class ProfessionalGoldAnalyzerV3:
    """الإصدار 3.0 من محلل الذهب الاحترافي"""
    def __init__(self):
        self.symbols = {
            'gold': 'GC=F', 'gold_etf': 'GLD', 'dxy': 'DX-Y.NYB',
            'vix': '^VIX', 'treasury': '^TNX', 'oil': 'CL=F',
            'spy': 'SPY', 'usdeur': 'EURUSD=X', 'silver': 'SI=F'
        }
        self.ml_predictor = MLPredictor()
        self.mtf_analyzer = MultiTimeframeAnalyzer()
        self.news_api_key = os.getenv("NEWS_API_KEY")
        self.news_analyzer = AdvancedNewsAnalyzer(self.news_api_key)
        self.db_manager = DatabaseManager()
        self.backtester = ProfessionalBacktester(self)

    def fetch_data(self):
        print("📊 جلب البيانات...")
        try:
            data = yf.download(list(self.symbols.values()), period="3y", interval="1d", group_by='ticker', progress=False, threads=False)
            if data.empty: raise ValueError("فشل جلب البيانات")
            return data
        except Exception as e:
            print(f"❌ خطأ في جلب البيانات: {e}")
            return None

    def process_data(self, raw_data):
        print("⚙️ معالجة البيانات...")
        try:
            gold_symbol = self.symbols['gold']
            if not (isinstance(raw_data.columns, pd.MultiIndex) and gold_symbol in raw_data.columns.levels[0]):
                gold_symbol = self.symbols['gold_etf']
            gold_df = raw_data[gold_symbol].copy().dropna(subset=['Close'])

            if len(gold_df) < 200: raise ValueError("بيانات غير كافية")

            for window in [20, 50, 200]: gold_df[f'SMA_{window}'] = gold_df['Close'].rolling(window).mean()
            delta = gold_df['Close'].diff()
            gain = delta.where(delta > 0, 0).ewm(com=13, adjust=False).mean()
            loss = -delta.where(delta < 0, 0).ewm(com=13, adjust=False).mean()
            gold_df['RSI'] = 100 - (100 / (1 + gain / loss.replace(0, 1e-9)))
            exp1 = gold_df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = gold_df['Close'].ewm(span=26, adjust=False).mean()
            gold_df['MACD'] = exp1 - exp2
            gold_df['MACD_Signal'] = gold_df['MACD'].ewm(span=9, adjust=False).mean()
            
            print(f"✅ بيانات نظيفة: {len(gold_df)} يوم")
            return gold_df.dropna(), raw_data
        except Exception as e:
            print(f"❌ خطأ في معالجة البيانات: {e}")
            return None, None

    def calculate_fibonacci_levels(self, data, periods=50):
        try:
            recent_data = data.tail(periods)
            high, low = recent_data['High'].max(), recent_data['Low'].min()
            diff = high - low
            if diff == 0: return {}
            current_price = data['Close'].iloc[-1]
            return {'current_position': round(((current_price - low) / diff * 100), 2)}
        except: return {}

    def analyze_volume_profile(self, data):
        try:
            avg_vol = data['Volume'].rolling(20).mean().iloc[-1]
            current_vol = data['Volume'].iloc[-1]
            volume_ratio = current_vol / avg_vol if avg_vol else 1
            if volume_ratio > 2.0: strength = 'قوي جداً'
            elif volume_ratio > 1.5: strength = 'قوي'
            else: strength = 'طبيعي'
            return {'volume_strength': strength}
        except: return {}

    def analyze_correlations(self, market_data):
        try:
            print("📊 تحليل الارتباطات...")
            close_prices = market_data.xs('Close', level=1, axis=1)
            correlations = close_prices.tail(90).corr()
            gold_symbol = self.symbols['gold']
            if gold_symbol not in correlations: gold_symbol = self.symbols['gold_etf']
            return {'correlations': {name: round(correlations[gold_symbol].get(symbol, 0), 3) for name, symbol in self.symbols.items()}}
        except Exception as e:
            print(f"❌ خطأ في تحليل الارتباطات: {e}")
            return {}

    def fetch_economic_data(self): return {'score': 3}

    def generate_signal(self, tech_data, correlations, volume, fib_levels, economic_data, news_analysis, mtf_analysis, ml_prediction):
        print("🎯 توليد الإشارة...")
        try:
            latest = tech_data.iloc[-1]
            scores = {}
            weights = {'trend': 0.25, 'momentum': 0.2, 'volume': 0.1, 'fibonacci': 0.05, 'correlation': 0.05, 'economic': 0.1, 'news': 0.1, 'mtf_coherence': 0.15}
            
            scores['trend'] = 2 if latest['Close'] > latest['SMA_50'] > latest['SMA_200'] else (-2 if latest['Close'] < latest['SMA_50'] < latest['SMA_200'] else 0)
            scores['momentum'] = (1 if latest['MACD'] > latest['MACD_Signal'] else -1) + (1.5 if latest['RSI'] > 60 else (-1.5 if latest['RSI'] < 40 else 0))
            strength_map = {'طبيعي': 0, 'قوي': 2, 'قوي جداً': 4}
            scores['volume'] = strength_map.get(volume.get('volume_strength', 'طبيعي'), 0)
            pos = fib_levels.get('current_position', 50)
            scores['fibonacci'] = 2 if pos > 70 else (-2 if pos < 30 else 0)
            scores['correlation'] = 1 if correlations.get('correlations', {}).get('dxy', 0) < -0.5 else (-1 if correlations.get('correlations', {}).get('dxy', 0) > 0.5 else 0)
            scores['economic'] = economic_data.get('score', 0)
            scores['news'] = news_analysis.get('total_impact', 0)
            scores['mtf_coherence'] = mtf_analysis.get('coherence_score', 0)
            total_score = sum(scores.get(k, 0) * v for k, v in weights.items())
            
            ml_interpretation = ""
            if ml_prediction and ml_prediction[0] is not None:
                ml_prob = ml_prediction[0]
                ml_interpretation = ml_prediction[1]
                total_score *= (1.0 + (ml_prob - 0.5) * 0.5)

            if total_score >= 1.2: signal, confidence = "Buy", "High"
            elif total_score > 0.4: signal, confidence = "Weak Buy", "Medium"
            elif total_score <= -1.2: signal, confidence = "Sell", "High"
            else: signal, confidence = "Weak Sell", "Medium"
            else: signal, confidence = "Hold", "Low"
            
            return {
                'signal': signal, 'confidence': confidence, 'total_score': round(total_score, 2),
                'component_scores': {k: round(v, 2) for k, v in scores.items()},
                'current_price': round(latest['Close'], 2),
                'ml_prediction': {'probability': ml_prediction[0] if ml_prediction and ml_prediction[0] is not None else None, 'interpretation': ml_interpretation}
            }
        except Exception as e:
            print(f"❌ خطأ في توليد الإشارة: {e}")
            return {"error": str(e)}

    def generate_signal_for_backtest(self, data):
        return {'action': 'Buy' if data['close'] > data.get('open', data['close']) else 'Sell'}

    async def run_analysis_v3(self):
        print("🚀 بدء التحليل الاحترافي...")
        final_result = {'timestamp': datetime.now().isoformat(), 'version': '3.0'}
        try:
            raw_data = self.fetch_data()
            if raw_data is None: raise ValueError("فشل جلب البيانات")
            
            technical_data, market_data = self.process_data(raw_data)
            if technical_data is None: raise ValueError("فشل معالجة البيانات")
            
            tasks = [
                self.news_analyzer.fetch_news_async(),
                asyncio.to_thread(self.mtf_analyzer.get_coherence_score, self.symbols['gold']),
                asyncio.to_thread(self.calculate_fibonacci_levels, technical_data),
                asyncio.to_thread(self.analyze_volume_profile, technical_data),
                asyncio.to_thread(self.analyze_correlations, market_data),
                asyncio.to_thread(self.fetch_economic_data),
                asyncio.to_thread(self.db_manager.update_future_prices)
            ]
            results = await asyncio.gather(*tasks)
            news_articles, (coh_score, mtf_analysis), fib_levels, vol_analysis, correlations, econ_data, _ = results
            news_analysis = self.news_analyzer.analyze_news(news_articles)
            
            training_data = self.db_manager.get_training_data()
            ml_prediction = None
            if training_data:
                # ... (ML logic) ...
                pass
            
            signals = self.generate_signal(technical_data, correlations, vol_analysis, fib_levels, econ_data, news_analysis, mtf_analysis, ml_prediction)
            
            backtest_results = self.backtester.run_backtest(technical_data)
            
            final_result.update({
                'status': 'success', 'gold_analysis': signals, 'backtest_results': backtest_results
            })
            self.db_manager.save_analysis(final_result)
        except Exception as e:
            print(f"❌ فشل التحليل الاحترافي: {e}")
            final_result.update({'status': 'error', 'error': str(e)})
        
        self.save_results_v3(final_result)
        print("\n✅ تم إتمام التحليل!")

    def save_results_v3(self, results):
        try:
            with open("gold_analysis_v3.json", 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            print(f"💾 تم حفظ التحليل في: gold_analysis_v3.json")
        except Exception as e:
            print(f"❌ خطأ في حفظ النتائج: {e}")

def main():
    analyzer = ProfessionalGoldAnalyzerV3()
    asyncio.run(analyzer.run_analysis_v3())

if __name__ == "__main__":
    main()
