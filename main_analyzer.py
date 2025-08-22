#!/usr/bin/env python3
import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
import sqlite3
import joblib
from datetime import datetime, timedelta
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import xgboost as xgb
from textblob import TextBlob
import spacy
import backtrader as bt
import asyncio
import aiohttp

warnings.filterwarnings('ignore')

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

class MLPredictor:
    def __init__(self):
        self.model, self.scaler, self.feature_columns = None, StandardScaler(), None
        self.model_path, self.scaler_path, self.columns_path = "gold_ml_model.pkl", "gold_scaler.pkl", "feature_columns.pkl"

    def prepare_features(self, analysis_data):
        features = {}
        # ... (This function remains the same as your original full script)
        if 'gold_analysis' in analysis_data:
            scores = analysis_data.get('gold_analysis', {}).get('component_scores', {})
            features.update({f'score_{k}': v for k, v in scores.items()})
            features['total_score'] = analysis_data.get('gold_analysis', {}).get('total_score', 0)
            tech_summary = analysis_data.get('gold_analysis', {}).get('technical_summary', {})
            features.update({f'tech_{k}': v for k, v in tech_summary.items()})
        if 'volume_analysis' in analysis_data:
            vol = analysis_data.get('volume_analysis', {})
            features['volume_ratio'] = vol.get('volume_ratio', 1)
            features['volume_strength_encoded'] = self._encode_volume_strength(vol.get('volume_strength', 'طبيعي'))
        if 'market_correlations' in analysis_data:
            corr = analysis_data.get('market_correlations', {}).get('correlations', {})
            features.update({f'corr_{k}': v for k, v in corr.items()})
        if 'economic_data' in analysis_data:
            features['economic_score'] = analysis_data.get('economic_data', {}).get('score', 0)
        if 'fibonacci_levels' in analysis_data:
            fib = analysis_data.get('fibonacci_levels', {})
            features['fib_position'] = fib.get('current_position', 50)
        return features

    def _encode_volume_strength(self, strength):
        return {'ضعيف': 0, 'طبيعي': 1, 'قوي': 2, 'قوي جداً': 3}.get(strength, 1)

    def train_model(self, historical_data):
        print("🤖 بدء تدريب نموذج التعلم الآلي...")
        X, y = [], []
        # ... (This function remains the same as your original full script)
        for record in historical_data:
            features = self.prepare_features(record['analysis'])
            if features:
                if self.feature_columns is None: self.feature_columns = list(features.keys())
                X.append([features.get(col, 0) for col in self.feature_columns])
                y.append(1 if record.get('price_change_5d', 0) > 1.0 else 0)
        
        if len(X) < 100:
            print(f"⚠️ بيانات غير كافية للتدريب ({len(X)} سجل)")
            return False
        if len(np.unique(y)) < 2:
            print("⚠️ لا يمكن التدريب لأن جميع البيانات تنتمي إلى فئة واحدة فقط.")
            return False

        X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size=0.2, random_state=42, stratify=y)
        X_train_scaled, X_test_scaled = self.scaler.fit_transform(X_train), self.scaler.transform(X_test)
        
        model = xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train_scaled, y_train)
        f1 = f1_score(y_test, model.predict(X_test_scaled))
        self.model = model
        print(f"\n✅ تم تدريب نموذج XGBoost مع F1 Score: {f1:.2%}")
        
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        joblib.dump(self.feature_columns, self.columns_path)
        return True
    
    def predict_probability(self, analysis_data):
        try:
            if not all(os.path.exists(p) for p in [self.model_path, self.scaler_path, self.columns_path]):
                return None, "النموذج غير مدرب"
            if self.model is None:
                self.model, self.scaler, self.feature_columns = joblib.load(self.model_path), joblib.load(self.scaler_path), joblib.load(self.columns_path)
            
            features = self.prepare_features(analysis_data)
            X = np.array([[features.get(col, 0) for col in self.feature_columns]])
            X_scaled = self.scaler.transform(X)
            probability = self.model.predict_proba(X_scaled)[0][1]
            
            if probability > 0.7: interpretation = "احتمالية عالية للنجاح"
            elif probability > 0.55: interpretation = "احتمالية جيدة للنجاح"
            else: interpretation = "احتمالية منخفضة"
            return probability, interpretation
        except Exception as e:
            print(f"خطأ في التنبؤ: {e}")
            return None, str(e)

class MultiTimeframeAnalyzer:
    """محلل متعدد الأطر الزمنية"""
    def __init__(self):
        self.timeframes = {'1h': {'period': '7d', 'weight': 0.2}, '4h': {'period': '1mo', 'weight': 0.3}, '1d': {'period': '3mo', 'weight': 0.5}}

    def analyze_timeframe(self, symbol, interval, period):
        """✅ تصحيح: تم جعل هذه الدالة أكثر قوة لتجنب الأخطاء"""
        try:
            data = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
            if data.empty or len(data) < 26: return None
            
            close = data['Close']
            sma20 = close.rolling(20).mean()
            delta = close.diff(1)
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rsi = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
            macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
            macd_signal = macd.ewm(span=9, adjust=False).mean()
            
            # التأكد من أن جميع السلاسل لها نفس الفهرس قبل المقارنة
            latest_close, latest_sma = close.iloc[-1], sma20.iloc[-1]
            latest_rsi, latest_macd, latest_signal = rsi.iloc[-1], macd.iloc[-1], macd_signal.iloc[-1]
            
            score = 0
            if latest_close > latest_sma: score += 1
            if latest_rsi > 50: score += 0.5
            if latest_macd > latest_signal: score += 1
            
            return {'score': score, 'trend': 'صاعد' if score > 1 else 'هابط'}
        except Exception as e:
            # طباعة الخطأ ولكن السماح للبرنامج بالاستمرار
            print(f"⚠️ تحذير في تحليل الإطار الزمني {interval}: {e}")
            return None
            
    def get_coherence_score(self, symbol):
        # ... (This function remains the same as your original full script)
        print("⏰ تحليل الأطر الزمنية المتعددة...")
        results, total_weighted_score, total_weight = {}, 0, 0
        for tf_name, tf_config in self.timeframes.items():
            analysis = self.analyze_timeframe(symbol, tf_name, tf_config['period'])
            if analysis:
                results[tf_name] = analysis
                total_weighted_score += analysis['score'] * tf_config['weight']
                total_weight += tf_config['weight']
        if total_weight == 0: return 0, {}
        coherence_score = total_weighted_score / total_weight
        return coherence_score, {'coherence_score': round(coherence_score, 2)}

# ... (بقية الكلاسات الأخرى تبقى كما هي)
# ... The rest of the classes (AdvancedNewsAnalyzer, ProfessionalBacktester, etc.) remain the same
# ... I will paste the full final code for clarity

class AdvancedNewsAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key

    async def fetch_news_async(self):
        if not self.api_key: return []
        keywords = ['"gold price"', '"federal reserve"', '"interest rates"']
        all_articles = []
        async with aiohttp.ClientSession() as session:
            tasks = [self._fetch_url(session, f"https://newsapi.org/v2/everything?q={keyword}&language=en&sortBy=publishedAt&pageSize=10&apiKey={self.api_key}") for keyword in keywords]
            results = await asyncio.gather(*tasks)
            for result in results:
                if result and 'articles' in result: all_articles.extend(result['articles'])
        return list({article['url']: article for article in all_articles}.values())

    async def _fetch_url(self, session, url):
        try:
            async with session.get(url, timeout=10) as response: return await response.json()
        except Exception as e:
            print(f"خطأ في جلب الأخبار: {e}")
            return None

    def analyze_news(self, articles):
        if not articles: return {'total_impact': 0}
        total_sentiment = sum(TextBlob(f"{a.get('title','')} {a.get('description','')}").sentiment.polarity for a in articles)
        return {'total_impact': round(total_sentiment / len(articles) * 5, 2) if articles else 0}

class ProfessionalBacktester:
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
            return {'total_return': round((final_value - initial_cash) / initial_cash * 100, 2), 'total_trades': trades.get('total', {}).get('total', 0)}
        except Exception as e:
            print(f"❌ خطأ في الاختبار الخلفي: {e}")
            return None

class DatabaseManager:
    def __init__(self, db_path="analysis_history.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
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
        try:
            with sqlite3.connect(self.db_path, timeout=10) as conn:
                gold_analysis = analysis_data.get('gold_analysis', {})
                if 'error' in gold_analysis: return

                analysis_date_str = datetime.now().date().isoformat()
                params = (
                    analysis_date_str, gold_analysis.get('current_price'), gold_analysis.get('signal'),
                    gold_analysis.get('confidence'), gold_analysis.get('total_score'),
                    json.dumps(gold_analysis.get('component_scores')), json.dumps(gold_analysis.get('technical_summary')),
                    json.dumps(analysis_data.get('volume_analysis')), json.dumps(analysis_data.get('market_correlations')),
                    analysis_data.get('economic_data', {}).get('score', 0), analysis_data.get('news_analysis', {}).get('total_impact'),
                    analysis_data.get('mtf_analysis', {}).get('coherence_score'), gold_analysis.get('ml_prediction', {}).get('probability')
                )
                conn.execute("INSERT OR REPLACE INTO analysis_history (analysis_date, gold_price, signal, confidence, total_score, component_scores, technical_indicators, volume_analysis, correlations, economic_score, news_sentiment, mtf_coherence, ml_probability) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", params)
                print("✅ تم حفظ التحليل في قاعدة البيانات")
        except Exception as e:
            print(f"❌ خطأ في حفظ التحليل: {e}")

    def update_future_prices(self):
        try:
            with sqlite3.connect(self.db_path, timeout=10) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, analysis_date, gold_price FROM analysis_history WHERE price_change_5d IS NULL AND analysis_date <= date('now', '-5 days')")
                records = cursor.fetchall()
                if not records: return
                for record_id, date_str, price in records:
                    date = datetime.strptime(date_str, '%Y-%m-%d').date()
                    future_data = yf.download('GC=F', start=date + timedelta(days=5), end=date + timedelta(days=10), progress=False, auto_adjust=True)
                    if not future_data.empty:
                        future_price = future_data['Close'].iloc[0]
                        price_change = ((future_price - price) / price * 100) if price else 0
                        cursor.execute("UPDATE analysis_history SET price_change_5d=?, signal_success=? WHERE id=?", (price_change, price_change > 1.0, record_id))
                if records: print(f"✅ تم تحديث {len(records)} سجل بالأسعار المستقبلية")
        except Exception as e:
            print(f"❌ خطأ في تحديث الأسعار المستقبلية: {e}")

    def get_training_data(self, min_records=100):
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query("SELECT * FROM analysis_history WHERE signal_success IS NOT NULL", conn)
            print(f"⚠️ بيانات التدريب المتاحة: {len(df)} سجل فقط")
            if len(df) < min_records: return None
            
            training_data = []
            for _, row in df.to_dict('index').items():
                try:
                    record = { 'price_change_5d': row['price_change_5d'], 'analysis': {
                            'gold_analysis': {'component_scores': json.loads(row['component_scores'] or '{}'), 'technical_summary': json.loads(row['technical_indicators'] or '{}'), 'total_score': row['total_score']},
                            'volume_analysis': json.loads(row['volume_analysis'] or '{}'), 'market_correlations': {'correlations': json.loads(row['correlations'] or '{}')},
                            'economic_data': {'score': row['economic_score']}, 'fibonacci_levels': {}
                        }}
                    training_data.append(record)
                except (json.JSONDecodeError, TypeError): continue
            return training_data
        except Exception as e:
            print(f"❌ خطأ في جلب بيانات التدريب: {e}")
            return None

class ProfessionalGoldAnalyzerV3:
    def __init__(self):
        self.symbols = {'gold': 'GC=F', 'gold_etf': 'GLD', 'dxy': 'DX-Y.NYB', 'vix': '^VIX', 'treasury': '^TNX', 'oil': 'CL=F', 'spy': 'SPY', 'silver': 'SI=F'}
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
            gold_df['Volume_SMA'] = gold_df['Volume'].rolling(20).mean()
            gold_df['Volume_Ratio'] = gold_df['Volume'] / gold_df['Volume_SMA'].replace(0, 1)

            print(f"✅ بيانات نظيفة: {len(gold_df)} يوم")
            return gold_df.dropna(), raw_data
        except Exception as e:
            print(f"❌ خطأ في معالجة البيانات: {e}")
            return None, None

    def calculate_fibonacci_levels(self, data, periods=50):
        try:
            recent = data.tail(periods)
            high, low = recent.High.max(), recent.Low.min()
            diff = high - low
            if diff == 0: return {}
            return {'current_position': round(((data['Close'].iloc[-1] - low) / diff * 100), 2)}
        except: return {}

    def analyze_volume_profile(self, data):
        try:
            volume_ratio = data['Volume_Ratio'].iloc[-1]
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

    def fetch_economic_data(self): return {'score': 3} # Simulated

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
            elif total_score < -0.4: signal, confidence = "Weak Sell", "Medium"
            else: signal, confidence = "Hold", "Low"
            
            return {
                'signal': signal, 'confidence': confidence, 'total_score': round(total_score, 2),
                'component_scores': {k: round(v, 2) for k, v in scores.items()},
                'current_price': round(latest['Close'], 2),
                'technical_summary': {'rsi': round(latest.get('RSI', 50), 1), 'macd': round(latest.get('MACD', 0), 2)},
                'ml_prediction': {'probability': ml_prediction[0] if ml_prediction and ml_prediction[0] is not None else None, 'interpretation': ml_interpretation}
            }
        except Exception as e:
            print(f"❌ خطأ في توليد الإشارة: {e}")
            return {"error": str(e)}

    def generate_signal_for_backtest(self, data):
        return {'action': 'Buy' if data['close'] > data.get('open', data['close']) else 'Sell'}

    async def run_analysis_v3(self):
        print("🚀 بدء التحليل الاحترافي المتقدم للذهب - الإصدار 3.0...")
        final_result = {'timestamp': datetime.now().isoformat(), 'version': '3.0'}
        try:
            market_data = self.fetch_data()
            if market_data is None: raise ValueError("فشل جلب البيانات")
            
            technical_data, market_data_processed = self.process_data(market_data)
            if technical_data is None: raise ValueError("فشل معالجة البيانات")

            tasks = [
                self.news_analyzer.fetch_news_async(),
                asyncio.to_thread(self.mtf_analyzer.get_coherence_score, self.symbols['gold']),
                asyncio.to_thread(self.calculate_fibonacci_levels, technical_data),
                asyncio.to_thread(self.analyze_volume_profile, technical_data),
                asyncio.to_thread(self.analyze_correlations, market_data_processed),
                asyncio.to_thread(self.fetch_economic_data),
                asyncio.to_thread(self.db_manager.update_future_prices)
            ]
            results = await asyncio.gather(*tasks)
            news_articles, (coh_score, mtf_analysis), fib_levels, vol_analysis, correlations, econ_data, _ = results
            news_analysis = self.news_analyzer.analyze_news(news_articles)
            
            training_data = self.db_manager.get_training_data()
            ml_prediction = None
            if training_data:
                if not os.path.exists(self.ml_predictor.model_path):
                    self.ml_predictor.train_model(training_data)
                
                temp_signals = self.generate_signal(technical_data, correlations, vol_analysis, fib_levels, econ_data, news_analysis, mtf_analysis, None)
                temp_analysis_for_ml = {
                    'gold_analysis': temp_signals, 'volume_analysis': vol_analysis, 
                    'market_correlations': correlations, 'economic_data': econ_data, 'fibonacci_levels': fib_levels
                }
                ml_prediction = self.ml_predictor.predict_probability(temp_analysis_for_ml)

            signals = self.generate_signal(technical_data, correlations, vol_analysis, fib_levels, econ_data, news_analysis, mtf_analysis, ml_prediction)
            
            backtest_results = self.backtester.run_backtest(technical_data)
            
            final_result.update({
                'status': 'success',
                'gold_analysis': signals,
                'backtest_results': backtest_results,
                'market_correlations': correlations, 'news_analysis': news_analysis,
                'mtf_analysis': mtf_analysis, 'volume_analysis': vol_analysis,
                'economic_data': econ_data, 'fibonacci_levels': fib_levels
            })
            self.db_manager.save_analysis(final_result)
        except Exception as e:
            print(f"❌ فشل التحليل الاحترافي: {e}")
            final_result.update({'status': 'error', 'error': str(e)})
        
        self.save_results_v3(final_result)
        report = self.generate_report_v3(final_result)
        print(report)
        print("\n✅ تم إتمام التحليل!")

    def save_results_v3(self, results):
        try:
            with open("gold_analysis_v3.json", 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            print(f"💾 تم حفظ التحليل في: gold_analysis_v3.json")
        except Exception as e:
            print(f"❌ خطأ في حفظ النتائج: {e}")

    def generate_report_v3(self, analysis_result):
        # ... (This function remains the same as your original full script)
        try:
            if analysis_result.get('status') != 'success':
                return f"فشل التحليل: {analysis_result.get('error')}"

            report = []
            report.append("=" * 80)
            report.append("📊 تقرير التحليل الاحترافي للذهب - الإصدار 3.0")
            report.append(f"التاريخ: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append("-" * 80)

            ga = analysis_result['gold_analysis']
            report.append(f"🎯 الإشارة الرئيسية: {ga.get('signal')} بثقة {ga.get('confidence')} | السعر: ${ga.get('current_price')} | النقاط: {ga.get('total_score')}")
            
            if ga.get('ml_prediction', {}).get('probability') is not None:
                ml = ga['ml_prediction']
                report.append(f"🤖 تنبؤ الآلة: {ml['probability']:.1%} نجاح ({ml['interpretation']})")

            bt_results = analysis_result.get('backtest_results')
            if bt_results:
                report.append(f"🔄 الاختبار الخلفي: عائد {bt_results.get('total_return', 0):.2f}% | صفقات {bt_results.get('total_trades', 0)}")
            
            report.append("=" * 80)
            return "\n".join(report)
        except Exception as e:
            return f"خطأ في توليد التقرير: {e}"


def main():
    analyzer = ProfessionalGoldAnalyzerV3()
    asyncio.run(analyzer.run_analysis_v3())

if __name__ == "__main__":
    main()

