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

# تحميل نموذج spaCy للغة الإنجليزية
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
        
        # استخراج النقاط من التحليل
        if 'gold_analysis' in analysis_data:
            scores = analysis_data['gold_analysis'].get('component_scores', {})
            features.update({f'score_{k}': v for k, v in scores.items()})
            features['total_score'] = analysis_data['gold_analysis'].get('total_score', 0)
            
            # المؤشرات الفنية
            tech_summary = analysis_data['gold_analysis'].get('technical_summary', {})
            features.update({f'tech_{k}': v for k, v in tech_summary.items()})
        
        # بيانات الحجم
        if 'volume_analysis' in analysis_data:
            vol = analysis_data['volume_analysis']
            features['volume_ratio'] = vol.get('volume_ratio', 1)
            features['volume_strength_encoded'] = self._encode_volume_strength(vol.get('volume_strength', 'طبيعي'))
        
        # الارتباطات
        if 'market_correlations' in analysis_data:
            corr = analysis_data['market_correlations'].get('correlations', {})
            features.update({f'corr_{k}': v for k, v in corr.items()})
        
        # البيانات الاقتصادية
        if 'economic_data' in analysis_data:
            features['economic_score'] = analysis_data['economic_data'].get('score', 0)
        
        # مستويات فيبوناتشي
        if 'fibonacci_levels' in analysis_data:
            fib = analysis_data['fibonacci_levels']
            features['fib_position'] = fib.get('current_position', 50)
        
        return features
    
    def _encode_volume_strength(self, strength):
        """تحويل قوة الحجم إلى رقم"""
        mapping = {
            'ضعيف': 0,
            'طبيعي': 1,
            'قوي': 2,
            'قوي جداً': 3
        }
        return mapping.get(strength, 1)
    
    def train_model(self, historical_data):
        """تدريب نموذج التعلم الآلي"""
        print("🤖 بدء تدريب نموذج التعلم الآلي...")
        
        # تحضير البيانات
        X = []
        y = []
        
        for record in historical_data:
            features = self.prepare_features(record['analysis'])
            if features:
                X.append(list(features.values()))
                # الهدف: هل ارتفع السعر بنسبة 1% خلال 5 أيام؟
                y.append(1 if record['price_change_5d'] > 1.0 else 0)
        
        if len(X) < 100:
            print("⚠️ بيانات غير كافية للتدريب")
            return False
        
        X = np.array(X)
        y = np.array(y)
        
        # تقسيم البيانات
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # تطبيع البيانات
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # تدريب نماذج متعددة
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False)
        }
        
        best_model = None
        best_score = 0
        
        for name, model in models.items():
            print(f"تدريب نموذج {name}...")
            model.fit(X_train_scaled, y_train)
            
            # التقييم
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            print(f"  - الدقة: {accuracy:.2%}")
            print(f"  - Precision: {precision:.2%}")
            print(f"  - Recall: {recall:.2%}")
            print(f"  - F1 Score: {f1:.2%}")
            
            # اختيار أفضل نموذج
            if f1 > best_score:
                best_score = f1
                best_model = model
                self.model = model
        
        print(f"\n✅ أفضل نموذج: {type(best_model).__name__} مع F1 Score: {best_score:.2%}")
        
        # حفظ النموذج
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        
        return True
    
    def predict_probability(self, analysis_data):
        """التنبؤ باحتمالية نجاح الإشارة"""
        try:
            # تحميل النموذج إذا لم يكن محملاً
            if self.model is None:
                if os.path.exists(self.model_path):
                    self.model = joblib.load(self.model_path)
                    self.scaler = joblib.load(self.scaler_path)
                else:
                    return None, "النموذج غير مدرب بعد"
            
            # تحضير المميزات
            features = self.prepare_features(analysis_data)
            X = np.array([list(features.values())])
            
            # التطبيع والتنبؤ
            X_scaled = self.scaler.transform(X)
            probability = self.model.predict_proba(X_scaled)[0][1]
            
            # تفسير الاحتمالية
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
            if data.empty:
                return None
            
            # حساب المؤشرات الأساسية
            data['SMA_20'] = data['Close'].rolling(20).mean()
            data['RSI'] = self._calculate_rsi(data['Close'])
            
            # حساب MACD
            exp1 = data['Close'].ewm(span=12).mean()
            exp2 = data['Close'].ewm(span=26).mean()
            data['MACD'] = exp1 - exp2
            data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
            
            latest = data.iloc[-1]
            
            # نظام نقاط مبسط
            score = 0
            
            # الاتجاه
            if latest['Close'] > latest['SMA_20']:
                score += 1
            else:
                score -= 1
            
            # RSI
            if 30 <= latest['RSI'] <= 70:
                if latest['RSI'] > 50:
                    score += 0.5
                else:
                    score -= 0.5
            elif latest['RSI'] < 30:
                score += 1  # ذروة بيع
            else:
                score -= 1  # ذروة شراء
            
            # MACD
            if latest['MACD'] > latest['MACD_Signal']:
                score += 1
            else:
                score -= 1
            
            return {
                'score': score,
                'trend': 'صاعد' if score > 0 else 'هابط',
                'strength': abs(score),
                'rsi': latest['RSI'],
                'price': latest['Close']
            }
            
        except Exception as e:
            print(f"خطأ في تحليل الإطار الزمني {interval}: {e}")
            return None
    
    def _calculate_rsi(self, prices, period=14):
        """حساب RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def get_coherence_score(self, symbol):
        """حساب نقاط التوافق بين الأطر الزمنية"""
        print("⏰ تحليل الأطر الزمنية المتعددة...")
        
        results = {}
        total_weighted_score = 0
        total_weight = 0
        
        # تحليل كل إطار زمني
        for tf_name, tf_config in self.timeframes.items():
            if tf_name == '1h':
                interval = '1h'
            elif tf_name == '4h':
                interval = '4h'
            else:
                interval = '1d'
            
            analysis = self.analyze_timeframe(symbol, interval, tf_config['period'])
            
            if analysis:
                results[tf_name] = analysis
                total_weighted_score += analysis['score'] * tf_config['weight']
                total_weight += tf_config['weight']
        
        if total_weight == 0:
            return 0, results
        
        # حساب نقاط التوافق
        coherence_score = total_weighted_score / total_weight
        
        # تحليل التوافق
        trends = [r['trend'] for r in results.values() if r]
        if all(t == 'صاعد' for t in trends):
            coherence_score += 2  # مكافأة للتوافق الكامل
            coherence_analysis = "توافق كامل صاعد - قوة استثنائية"
        elif all(t == 'هابط' for t in trends):
            coherence_score -= 2  # عقوبة للتوافق الهابط
            coherence_analysis = "توافق كامل هابط - ضعف شديد"
        elif len(set(trends)) > 1:
            coherence_analysis = "تضارب بين الأطر الزمنية - حذر"
        else:
            coherence_analysis = "توافق جزئي"
        
        return coherence_score, {
            'timeframes': results,
            'coherence_score': round(coherence_score, 2),
            'analysis': coherence_analysis,
            'recommendation': self._get_mtf_recommendation(coherence_score)
        }
    
    def _get_mtf_recommendation(self, score):
        """توصية بناءً على التوافق"""
        if score > 2:
            return "دخول قوي - جميع الأطر متوافقة"
        elif score > 1:
            return "دخول معتدل - توافق جيد"
        elif score > -1:
            return "انتظار - عدم وضوح"
        elif score > -2:
            return "تجنب الشراء - ضعف"
        else:
            return "بيع أو تجنب كامل - توافق هابط"

class AdvancedNewsAnalyzer:
    """محلل أخبار متقدم مع استخراج الأحداث"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.event_patterns = {
            'interest_rate': {
                'keywords': ['interest rate', 'rate decision', 'fomc', 'federal reserve', 'fed meeting'],
                'entities': ['Federal Reserve', 'Fed', 'FOMC', 'Jerome Powell'],
                'impact_multiplier': 3
            },
            'inflation': {
                'keywords': ['inflation', 'cpi', 'consumer price', 'pce'],
                'entities': ['Bureau of Labor Statistics', 'BLS'],
                'impact_multiplier': 2.5            },
            'employment': {
                'keywords': ['employment', 'jobs', 'nfp', 'non-farm payroll', 'unemployment'],
                'entities': ['Labor Department', 'BLS'],
                'impact_multiplier': 2
            },
            'geopolitical': {
                'keywords': ['war', 'conflict', 'sanctions', 'crisis', 'tension'],
                'entities': ['Russia', 'China', 'Middle East', 'Ukraine'],
                'impact_multiplier': 2.5
            },
            'central_bank': {
                'keywords': ['central bank', 'ecb', 'boe', 'boj', 'monetary policy'],
                'entities': ['ECB', 'Bank of England', 'Bank of Japan'],
                'impact_multiplier': 2
            },
            'dollar': {
                'keywords': ['dollar', 'dxy', 'usd', 'currency'],
                'entities': ['Dollar Index', 'DXY'],
                'impact_multiplier': 1.5
            }
        }
    
    def extract_events(self, articles):
        """استخراج الأحداث من الأخبار باستخدام NLP"""
        extracted_events = []
        
        for article in articles:
            if not article.get('title'):
                continue
            
            # دمج العنوان والوصف
            text = f"{article['title']} {article.get('description', '')}"
            
            # معالجة النص باستخدام spaCy
            doc = nlp(text)
            
            # استخراج الكيانات
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            
            # تحليل نوع الحدث
            event_type = self._classify_event(text.lower(), entities)
            
            if event_type:
                # تحليل المشاعر المتقدم
                sentiment_score = self._advanced_sentiment_analysis(text)
                
                # استخراج التفاصيل الرقمية
                numbers = self._extract_numbers(doc)
                
                event = {
                    'type': event_type,
                    'title': article['title'],
                    'source': article.get('source', {}).get('name', 'Unknown'),
                    'published': article.get('publishedAt', ''),
                    'entities': entities,
                    'numbers': numbers,
                    'sentiment_score': sentiment_score,
                    'impact_score': self._calculate_impact_score(event_type, sentiment_score),
                    'url': article.get('url', '')
                }
                
                extracted_events.append(event)
        
        return self._analyze_events_impact(extracted_events)
    
    def _classify_event(self, text, entities):
        """تصنيف نوع الحدث"""
        for event_type, patterns in self.event_patterns.items():
            # فحص الكلمات المفتاحية
            if any(keyword in text for keyword in patterns['keywords']):
                return event_type
            
            # فحص الكيانات
            entity_texts = [ent[0] for ent in entities]
            if any(entity in ' '.join(entity_texts) for entity in patterns['entities']):
                return event_type
        
        return None
    
    def _advanced_sentiment_analysis(self, text):
        """تحليل متقدم للمشاعر"""
        # استخدام TextBlob للتحليل الأساسي
        blob = TextBlob(text)
        basic_sentiment = blob.sentiment.polarity
        
        # تحليل كلمات محددة للذهب
        gold_positive = ['surge', 'rally', 'gain', 'rise', 'bullish', 'support', 'demand']
        gold_negative = ['fall', 'drop', 'decline', 'bearish', 'pressure', 'weak']
        
        positive_count = sum(1 for word in gold_positive if word in text.lower())
        negative_count = sum(1 for word in gold_negative if word in text.lower())
        
        # دمج التحليلات
        final_sentiment = basic_sentiment + (positive_count - negative_count) * 0.1
        
        return max(-1, min(1, final_sentiment))  # تقييد بين -1 و 1
    
    def _extract_numbers(self, doc):
        """استخراج الأرقام والنسب المئوية"""
        numbers = []
        
        for token in doc:
            if token.like_num or '%' in token.text:
                # محاولة استخراج السياق
                context = []
                for i in range(max(0, token.i - 3), min(len(doc), token.i + 3)):
                    context.append(doc[i].text)
                
                numbers.append({
                    'value': token.text,
                    'context': ' '.join(context)
                })
        
        return numbers
    
    def _calculate_impact_score(self, event_type, sentiment):
        """حساب نقاط التأثير"""
        base_impact = self.event_patterns.get(event_type, {}).get('impact_multiplier', 1)
        
        # تعديل بناءً على المشاعر
        if event_type in ['interest_rate', 'inflation']:
            # للذهب: ارتفاع الفائدة سلبي، انخفاضها إيجابي
            impact = base_impact * (-sentiment)
        else:
            impact = base_impact * sentiment
        
        return round(impact, 2)
    
    def _analyze_events_impact(self, events):
        """تحليل التأثير الإجمالي للأحداث"""
        if not events:
            return {
                'events': [],
                'total_impact': 0,
                'dominant_theme': None,
                'recommendation': 'لا توجد أحداث مؤثرة'
            }
        
        # تجميع حسب نوع الحدث
        event_groups = {}
        for event in events:
            event_type = event['type']
            if event_type not in event_groups:
                event_groups[event_type] = []
            event_groups[event_type].append(event)
        
        # حساب التأثير الإجمالي
        total_impact = sum(event['impact_score'] for event in events)
        
        # تحديد الموضوع المهيمن
        dominant_theme = max(event_groups.keys(), 
                           key=lambda k: sum(e['impact_score'] for e in event_groups[k]))
        
        # التوصية
        if total_impact > 5:
            recommendation = "أخبار إيجابية قوية جداً للذهب"
        elif total_impact > 2:
            recommendation = "أخبار إيجابية للذهب"
        elif total_impact < -5:
            recommendation = "أخبار سلبية قوية جداً للذهب"
        elif total_impact < -2:
            recommendation = "أخبار سلبية للذهب"
        else:
            recommendation = "تأثير محايد أو مختلط"
        
        return {
            'events': events[:10],  # أهم 10 أحداث
            'event_summary': {t: len(e) for t, e in event_groups.items()},
            'total_impact': round(total_impact, 2),
            'dominant_theme': dominant_theme,
            'recommendation': recommendation
        }
    
    async def fetch_news_async(self):
        """جلب الأخبار بشكل غير متزامن"""
        keywords = [
            '"gold price"',
            '"federal reserve"',
            '"interest rates"',
            '"inflation data"',
            '"XAU/USD"'
        ]
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for keyword in keywords:
                url = f"https://newsapi.org/v2/everything?q={keyword}&language=en&sortBy=publishedAt&pageSize=20&apiKey={self.api_key}"
                tasks.append(self._fetch_url(session, url))
            
            results = await asyncio.gather(*tasks)
            
        # دمج جميع المقالات
        all_articles = []
        for result in results:
            if result and 'articles' in result:
                all_articles.extend(result['articles'])
        
        # إزالة المكرر
        seen_urls = set()
        unique_articles = []
        for article in all_articles:
            if article.get('url') not in seen_urls:
                seen_urls.add(article.get('url'))
                unique_articles.append(article)
        
        return unique_articles
    
    async def _fetch_url(self, session, url):
        """جلب URL واحد"""
        try:
            async with session.get(url, timeout=10) as response:
                return await response.json()
        except Exception as e:
            print(f"خطأ في جلب الأخبار: {e}")
            return None

class ProfessionalBacktester:
    """نظام اختبار خلفي احترافي"""
    
    class GoldStrategy(bt.Strategy):
        params = (
            ('analyzer', None),
            ('risk_percent', 0.02),
        )
        
        def __init__(self):
            self.order = None
            self.buyprice = None
            self.buycomm = None
            self.trades = []
            
        def next(self):
            if self.order:
                return
            
            # تحليل الإشارة الحالية
            current_data = self._prepare_current_data()
            signal = self.params.analyzer.generate_signal_for_backtest(current_data)
            
            if not self.position:
                # منطق الشراء
                if signal['action'] in ['Strong Buy', 'Buy']:
                    # حساب حجم المركز
                    size = self._calculate_position_size(signal['confidence'])
                    self.order = self.buy(size=size)
                    
            else:
                # منطق البيع
                if signal['action'] in ['Strong Sell', 'Sell']:
                    self.order = self.sell()
                    
                # وقف الخسارة وجني الأرباح
                elif self.position.size > 0:
                    current_price = self.data.close[0]
                    entry_price = self.position.price
                    
                    # وقف الخسارة
                    if current_price < entry_price * 0.98:  # 2% stop loss
                        self.order = self.sell()
                        
                    # جني الأرباح
                    elif current_price > entry_price * 1.05:  # 5% take profit
                        self.order = self.sell()
        
        def _prepare_current_data(self):
            """تحضير البيانات الحالية للتحليل"""
            # هنا يتم تحضير البيانات بنفس طريقة التحليل الحي
            return {
                'close': self.data.close[0],
                'open': self.data.open[0],
                'high': self.data.high[0],
                'low': self.data.low[0],
                'volume': self.data.volume[0],
                # يمكن إضافة المزيد من البيانات
            }
        
        def _calculate_position_size(self, confidence):
            """حساب حجم المركز بناءً على الثقة"""
            base_size = self.broker.getcash() * self.params.risk_percent
            
            if confidence == 'Very High':
                return base_size * 2
            elif confidence == 'High':
                return base_size * 1.5
            else:
                return base_size
        
        def notify_order(self, order):
            if order.status in [order.Submitted, order.Accepted]:
                return
                
            if order.status in [order.Completed]:
                if order.isbuy():
                    self.buyprice = order.executed.price
                    self.buycomm = order.executed.comm
                else:
                    profit = (order.executed.price - self.buyprice) * order.executed.size
                    self.trades.append({
                        'profit': profit,
                        'return': (order.executed.price - self.buyprice) / self.buyprice
                    })
                    
            self.order = None
    
    def __init__(self, analyzer):
        self.analyzer = analyzer
        
    def run_backtest(self, data, initial_cash=10000):
        """تشغيل الاختبار الخلفي"""
        print("🔄 بدء الاختبار الخلفي الاحترافي...")
        
        # إنشاء محرك backtrader
        cerebro = bt.Cerebro()
        
        # إضافة البيانات
        data_feed = bt.feeds.PandasData(dataname=data)
        cerebro.adddata(data_feed)
        
        # إضافة الاستراتيجية
        cerebro.addstrategy(self.GoldStrategy, analyzer=self.analyzer)
        
        # إعدادات الوسيط
        cerebro.broker.setcash(initial_cash)
        cerebro.broker.setcommission(commission=0.001)  # 0.1% عمولة
        
        # إضافة المحللين
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        
        # تشغيل الاختبار
        results = cerebro.run()
        
        # استخراج النتائج
        strat = results[0]
        
        # التحليلات
        sharpe = strat.analyzers.sharpe.get_analysis()
        drawdown = strat.analyzers.drawdown.get_analysis()
        returns = strat.analyzers.returns.get_analysis()
        trades = strat.analyzers.trades.get_analysis()
        
        # حساب المقاييس الإضافية
        final_value = cerebro.broker.getvalue()
        total_return = (final_value - initial_cash) / initial_cash * 100
        
        backtest_results = {
            'initial_capital': initial_cash,
            'final_value': round(final_value, 2),
            'total_return': round(total_return, 2),
            'sharpe_ratio': round(sharpe.get('sharperatio', 0), 3),
            'max_drawdown': round(drawdown.get('max', {}).get('drawdown', 0), 2),
            'total_trades': trades.get('total', {}).get('total', 0),
            'winning_trades': trades.get('won', {}).get('total', 0),
            'losing_trades': trades.get('lost', {}).get('total', 0),
            'win_rate': round(trades.get('won', {}).get('total', 0) / max(trades.get('total', {}).get('total', 1), 1) * 100, 2),
            'avg_trade_return': round(returns.get('rtot', 0) / max(trades.get('total', {}).get('total', 1), 1), 4),
            'best_trade': round(trades.get('won', {}).get('pnl', {}).get('max', 0), 2),
            'worst_trade': round(trades.get('lost', {}).get('pnl', {}).get('max', 0), 2),
            'avg_win': round(trades.get('won', {}).get('pnl', {}).get('average', 0), 2),
            'avg_loss': round(trades.get('lost', {}).get('pnl', {}).get('average', 0), 2),
            'profit_factor': self._calculate_profit_factor(trades),
            'recovery_factor': self._calculate_recovery_factor(total_return, drawdown),
            'risk_reward_ratio': self._calculate_risk_reward(trades)
        }
        
        # إنشاء تقرير مفصل
        self._generate_backtest_report(backtest_results)
        
        return backtest_results
    
    def _calculate_profit_factor(self, trades):
        """حساب عامل الربح"""
        try:
            gross_profit = abs(trades.get('won', {}).get('pnl', {}).get('total', 0))
            gross_loss = abs(trades.get('lost', {}).get('pnl', {}).get('total', 0))
            
            if gross_loss > 0:
                return round(gross_profit / gross_loss, 2)
            return 0
        except:
            return 0
    
    def _calculate_recovery_factor(self, total_return, drawdown):
        """حساب عامل الاسترداد"""
        try:
            max_dd = abs(drawdown.get('max', {}).get('drawdown', 1))
            if max_dd > 0:
                return round(total_return / max_dd, 2)
            return 0
        except:
            return 0
    
    def _calculate_risk_reward(self, trades):
        """حساب نسبة المخاطرة إلى المكافأة"""
        try:
            avg_win = abs(trades.get('won', {}).get('pnl', {}).get('average', 0))
            avg_loss = abs(trades.get('lost', {}).get('pnl', {}).get('average', 1))
            
            if avg_loss > 0:
                return round(avg_win / avg_loss, 2)
            return 0
        except:
            return 0
    
    def _generate_backtest_report(self, results):
        """توليد تقرير الاختبار الخلفي"""
        print("\n" + "="*60)
        print("📊 تقرير الاختبار الخلفي الاحترافي")
        print("="*60)
        
        print(f"\n💰 الأداء المالي:")
        print(f"  • رأس المال الأولي: ${results['initial_capital']:,}")
        print(f"  • القيمة النهائية: ${results['final_value']:,}")
        print(f"  • العائد الإجمالي: {results['total_return']}%")
        print(f"  • نسبة شارب: {results['sharpe_ratio']}")
        print(f"  • أقصى انخفاض: {results['max_drawdown']}%")
        
        print(f"\n📈 إحصائيات التداول:")
        print(f"  • إجمالي الصفقات: {results['total_trades']}")
        print(f"  • الصفقات الرابحة: {results['winning_trades']}")
        print(f"  • الصفقات الخاسرة: {results['losing_trades']}")
        print(f"  • معدل الفوز: {results['win_rate']}%")
        
        print(f"\n💵 تحليل الأرباح والخسائر:")
        print(f"  • متوسط الربح: ${results['avg_win']}")
        print(f"  • متوسط الخسارة: ${results['avg_loss']}")
        print(f"  • أفضل صفقة: ${results['best_trade']}")
        print(f"  • أسوأ صفقة: ${results['worst_trade']}")
        print(f"  • عامل الربح: {results['profit_factor']}")
        print(f"  • نسبة المخاطرة/المكافأة: {results['risk_reward_ratio']}")
        
        # تقييم الأداء
        print(f"\n🎯 تقييم الاستراتيجية:")
        if results['sharpe_ratio'] > 2:
            print("  ✅ أداء ممتاز - نسبة شارب عالية جداً")
        elif results['sharpe_ratio'] > 1:
            print("  ✅ أداء جيد - نسبة شارب جيدة")
        elif results['sharpe_ratio'] > 0.5:
            print("  ⚠️ أداء مقبول - يحتاج تحسين")
        else:
            print("  ❌ أداء ضعيف - يحتاج مراجعة شاملة")
        
        if results['win_rate'] > 60:
            print("  ✅ معدل فوز ممتاز")
        elif results['win_rate'] > 50:
            print("  ✅ معدل فوز جيد")
        else:
            print("  ⚠️ معدل فوز يحتاج تحسين")
        
        if results['profit_factor'] > 2:
            print("  ✅ عامل ربح ممتاز")
        elif results['profit_factor'] > 1.5:
            print("  ✅ عامل ربح جيد")
        else:
            print("  ⚠️ عامل ربح ضعيف")
        
        print("="*60)

class DatabaseManager:
    """مدير قاعدة البيانات للتحليلات التاريخية"""
    
    def __init__(self, db_path="analysis_history.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """تهيئة قاعدة البيانات"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # جدول التحليلات
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                analysis_date DATE,
                gold_price REAL,
                signal TEXT,
                confidence TEXT,
                total_score REAL,
                component_scores TEXT,
                technical_indicators TEXT,
                volume_analysis TEXT,
                correlations TEXT,
                economic_score REAL,
                news_sentiment TEXT,
                mtf_coherence REAL,
                ml_probability REAL,
                price_after_1d REAL,
                price_after_5d REAL,
                price_after_10d REAL,
                price_change_1d REAL,
                price_change_5d REAL,
                price_change_10d REAL,
                signal_success BOOLEAN
            )
        ''')
        
        # جدول الأداء
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE,
                metric_name TEXT,
                metric_value REAL,
                details TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_analysis(self, analysis_data):
        """حفظ التحليل في قاعدة البيانات"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            gold_analysis = analysis_data.get('gold_analysis', {})
            
            cursor.execute('''
                INSERT INTO analysis_history (
                    analysis_date, gold_price, signal, confidence, total_score,
                    component_scores, technical_indicators, volume_analysis,
                    correlations, economic_score, news_sentiment, mtf_coherence,
                    ml_probability
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().date(),
                gold_analysis.get('current_price'),
                gold_analysis.get('signal'),
                gold_analysis.get('confidence'),
                gold_analysis.get('total_score'),
                json.dumps(gold_analysis.get('component_scores', {})),
                json.dumps(gold_analysis.get('technical_summary', {})),
                json.dumps(analysis_data.get('volume_analysis', {})),
                json.dumps(analysis_data.get('market_correlations', {}).get('correlations', {})),
                analysis_data.get('economic_data', {}).get('score', 0),
                analysis_data.get('news_analysis', {}).get('summary', {}).get('overall_sentiment'),
                analysis_data.get('mtf_analysis', {}).get('coherence_score', 0),
                analysis_data.get('ml_prediction', {}).get('probability', 0)
            ))
            
            conn.commit()
            print("✅ تم حفظ التحليل في قاعدة البيانات")
            
        except Exception as e:
            print(f"❌ خطأ في حفظ التحليل: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def update_future_prices(self):
        """تحديث الأسعار المستقبلية للتحليلات السابقة"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # جلب التحليلات التي تحتاج تحديث
            cursor.execute('''
                SELECT id, analysis_date, gold_price 
                FROM analysis_history 
                WHERE price_after_10d IS NULL 
                AND analysis_date <= date('now', '-10 days')
            ''')
            
            records = cursor.fetchall()
            
            for record_id, analysis_date, original_price in records:
                # جلب الأسعار المستقبلية
                future_prices = self._get_future_prices(analysis_date)
                
                if future_prices:
                    # حساب التغييرات
                    changes = {
                        '1d': ((future_prices.get('1d', original_price) - original_price) / original_price * 100),
                        '5d': ((future_prices.get('5d', original_price) - original_price) / original_price * 100),
                        '10d': ((future_prices.get('10d', original_price) - original_price) / original_price * 100)
                    }
                    
                    # تحديد نجاح الإشارة
                    signal_success = changes['5d'] > 1.0  # نجاح إذا ارتفع أكثر من 1%
                    
                    # تحديث السجل
                    cursor.execute('''
                        UPDATE analysis_history 
                        SET price_after_1d = ?, price_after_5d = ?, price_after_10d = ?,
                            price_change_1d = ?, price_change_5d = ?, price_change_10d = ?,
                            signal_success = ?
                        WHERE id = ?
                    ''', (
                        future_prices.get('1d'), future_prices.get('5d'), future_prices.get('10d'),
                        changes['1d'], changes['5d'], changes['10d'],
                        signal_success, record_id
                    ))
            
            conn.commit()
            print(f"✅ تم تحديث {len(records)} سجل بالأسعار المستقبلية")
            
        except Exception as e:
            print(f"❌ خطأ في تحديث الأسعار المستقبلية: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def _get_future_prices(self, analysis_date):
        """جلب الأسعار المستقبلية لتاريخ معين"""
        try:
            # تحويل التاريخ
            if isinstance(analysis_date, str):
                analysis_date = datetime.strptime(analysis_date, '%Y-%m-%d').date()
            
            # جلب بيانات الذهب
            end_date = analysis_date + timedelta(days=15)
            data = yf.download('GC=F', start=analysis_date, end=end_date, progress=False)
            
            if data.empty:
                return None
            
            prices = {}
            
            # الأسعار بعد 1، 5، 10 أيام
            for days in [1, 5, 10]:
                target_date = analysis_date + timedelta(days=days)
                
                # البحث عن أقرب تاريخ متاح
                for i in range(5):  # محاولة 5 أيام إضافية
                    check_date = target_date + timedelta(days=i)
                    if check_date in data.index:
                        prices[f'{days}d'] = data.loc[check_date, 'Close']
                        break
            
            return prices
            
        except Exception as e:
            print(f"خطأ في جلب الأسعار المستقبلية: {e}")
            return None
    
    def get_training_data(self, min_records=100):
        """جلب بيانات التدريب للنموذج"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT * FROM analysis_history 
            WHERE signal_success IS NOT NULL 
            ORDER BY analysis_date DESC 
            LIMIT ?
        '''
        
        df = pd.read_sql_query(query, conn, params=(min_records * 2,))
        conn.close()
        
        if len(df) < min_records:
            print(f"⚠️ بيانات غير كافية للتدريب: {len(df)} سجل فقط")
            return None
        
        # تحويل البيانات JSON
        training_data = []
        for _, row in df.iterrows():
            record = {
                'analysis': {
                    'gold_analysis': {
                        'current_price': row['gold_price'],
                        'signal': row['signal'],
                        'confidence': row['confidence'],
                        'total_score': row['total_score'],
                        'component_scores': json.loads(row['component_scores'] or '{}'),
                        'technical_summary': json.loads(row['technical_indicators'] or '{}')
                    },
                    'volume_analysis': json.loads(row['volume_analysis'] or '{}'),
                    'market_correlations': {
                        'correlations': json.loads(row['correlations'] or '{}')
                    },
                    'economic_data': {
                        'score': row['economic_score']
                    }
                },
                'price_change_5d': row['price_change_5d'],
                'signal_success': row['signal_success']
            }
            training_data.append(record)
        
        return training_data

class ProfessionalGoldAnalyzerV3:
    """الإصدار 3.0 من محلل الذهب الاحترافي"""
    
    def __init__(self):
        # الرموز الأساسية
        self.symbols = {
            'gold': 'GC=F', 'gold_etf': 'GLD', 'dxy': 'DX-Y.NYB',
            'vix': '^VIX', 'treasury': '^TNX', 'oil': 'CL=F',
            'spy': 'SPY', 'usdeur': 'EURUSD=X', 'silver': 'SI=F'
        }
        
        # المكونات الجديدة
        self.ml_predictor = MLPredictor()
        self.mtf_analyzer = MultiTimeframeAnalyzer()
        self.news_analyzer = AdvancedNewsAnalyzer(os.getenv("NEWS_API_KEY"))
        self.db_manager = DatabaseManager()
        self.backtester = ProfessionalBacktester(self)
        
        # APIs
        self.news_api_key = os.getenv("NEWS_API_KEY")
        self.fred_api_key = os.getenv("FRED_API_KEY")
    
    def fetch_multi_timeframe_data(self):
        """جلب بيانات متعددة الأطر الزمنية"""
        print("📊 جلب بيانات متعددة الأطر الزمنية...")
        try:
            # البيانات اليومية
            daily_data = yf.download(list(self.symbols.values()), 
                                    period="1y", interval="1d", 
                                    group_by='ticker', progress=False)
            
            # بيانات 4 ساعات
            hourly_data = yf.download(self.symbols['gold'], 
                                     period="1mo", interval="1h", 
                                     progress=False)
            
            # بيانات أسبوعية للتحليل طويل المدى
            weekly_data = yf.download(self.symbols['gold'], 
                                     period="2y", interval="1wk", 
                                     progress=False)
            
            if daily_data.empty: 
                raise ValueError("فشل جلب البيانات")
            
            return {
                'daily': daily_data, 
                'hourly': hourly_data,
                'weekly': weekly_data
            }
        except Exception as e:
            print(f"❌ خطأ في جلب البيانات: {e}")
            return None
    
    def extract_gold_data(self, market_data):
        """استخراج بيانات الذهب"""
        print("🔍 استخراج بيانات الذهب...")
        try:
            daily_data = market_data['daily']
            gold_symbol = self.symbols['gold']
            
            if not (gold_symbol in daily_data.columns.levels[0] and 
                   not daily_data[gold_symbol].dropna().empty):
                gold_symbol = self.symbols['gold_etf']
                if not (gold_symbol in daily_data.columns.levels[0] and 
                       not daily_data[gold_symbol].dropna().empty):
                    raise ValueError("لا توجد بيانات للذهب")
            
            gold_daily = daily_data[gold_symbol].copy()
            gold_daily.dropna(subset=['Close'], inplace=True)
            
            if len(gold_daily) < 200: 
                raise ValueError("بيانات غير كافية")
            
            print(f"✅ بيانات يومية نظيفة: {len(gold_daily)} يوم")
            return gold_daily
        except Exception as e:
            print(f"❌ خطأ في استخراج بيانات الذهب: {e}")
            return None
    
    def calculate_professional_indicators(self, gold_data):
        """حساب المؤشرات الاحترافية المحسّنة"""
        print("📊 حساب المؤشرات الاحترافية المحسّنة...")
        try:
            df = gold_data.copy()
            
            # المتوسطات المتحركة
            df['SMA_10'] = df['Close'].rolling(window=10).mean()
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['SMA_100'] = df['Close'].rolling(window=100).mean()
            df['SMA_200'] = df['Close'].rolling(window=200).mean()
            
            # EMA
            df['EMA_9'] = df['Close'].ewm(span=9).mean()
            df['EMA_21'] = df['Close'].ewm(span=21).mean()
            
            # التقاطعات الذهبية/الموت
            df['Golden_Cross'] = (df['SMA_50'] > df['SMA_200']).astype(int)
            df['Death_Cross'] = (df['SMA_50'] < df['SMA_200']).astype(int)
            
            # RSI محسّن
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            df['RSI'] = 100 - (100 / (1 + gain / loss))
            
            # RSI Divergence
            df['RSI_MA'] = df['RSI'].rolling(window=5).mean()
            
            # MACD محسّن
            exp1 = df['Close'].ewm(span=12).mean()
            exp2 = df['Close'].ewm(span=26).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
            df['MACD_Cross'] = np.where(df['MACD'] > df['MACD_Signal'], 1, -1)
            
            # Bollinger Bands محسّن
            std = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['SMA_20'] + (std * 2)
            df['BB_Lower'] = df['SMA_20'] - (std * 2)
            df['BB_Width'] = ((df['BB_Upper'] - df['BB_Lower']) / df['SMA_20']) * 100
            df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
            
            # ATR & Volatility
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['ATR'] = true_range.rolling(14).mean()
            df['ATR_Percent'] = (df['ATR'] / df['Close']) * 100
            
            # Volume Analysis محسّن
            df['Volume_SMA'] = df['Volume'].rolling(20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
            df['OBV'] = (df['Volume'] * (~df['Close'].diff().le(0) * 2 - 1)).cumsum()
            df['Volume_Price_Trend'] = ((df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1) * df['Volume']).cumsum()
            
            # مؤشرات إضافية
            df['ROC'] = ((df['Close'] - df['Close'].shift(14)) / df['Close'].shift(14)) * 100
            df['Williams_R'] = ((df['High'].rolling(14).max() - df['Close']) / 
                                (df['High'].rolling(14).max() - df['Low'].rolling(14).min())) * -100
            
            # Stochastic
            low_14 = df['Low'].rolling(14).min()
            high_14 = df['High'].rolling(14).max()
            df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
            df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()
            
            # Ichimoku Cloud
            high_9 = df['High'].rolling(9).max()
            low_9 = df['Low'].rolling(9).min()
            df['Tenkan_sen'] = (high_9 + low_9) / 2
            
            high_26 = df['High'].rolling(26).max()
            low_26 = df['Low'].rolling(26).min()
            df['Kijun_sen'] = (high_26 + low_26) / 2
            
            df['Senkou_Span_A'] = ((df['Tenkan_sen'] + df['Kijun_sen']) / 2).shift(26)
            
            high_52 = df['High'].rolling(52).max()
            low_52 = df['Low'].rolling(52).min()
            df['Senkou_Span_B'] = ((high_52 + low_52) / 2).shift(26)
            
            return df.dropna()
        except Exception as e:
            print(f"❌ خطأ في حساب المؤشرات: {e}")
            return gold_data
    
    def calculate_support_resistance(self, data, window=20):
        """حساب مستويات الدعم والمقاومة الديناميكية"""
        try:
            recent_data = data.tail(window * 3)
            
            # البحث عن القمم والقيعان
            highs = recent_data['High'].rolling(5, center=True).max() == recent_data['High']
            lows = recent_data['Low'].rolling(5, center=True).min() == recent_data['Low']
            
            resistance_levels = recent_data.loc[highs, 'High'].nlargest(3).tolist()
            support_levels = recent_data.loc[lows, 'Low'].nsmallest(3).tolist()
            
            current_price = data['Close'].iloc[-1]
            
            # تحديد أقرب دعم ومقاومة
            nearest_resistance = min([r for r in resistance_levels if r > current_price], default=None)
            nearest_support = max([s for s in support_levels if s < current_price], default=None)
            
            return {
                'resistance_levels': [round(r, 2) for r in resistance_levels],
                'support_levels': [round(s, 2) for s in support_levels],
                'nearest_resistance': round(nearest_resistance, 2) if nearest_resistance else None,
                'nearest_support': round(nearest_support, 2) if nearest_support else None,
                'price_to_resistance': round(((nearest_resistance - current_price) / current_price * 100), 2) if nearest_resistance else None,
                'price_to_support': round(((current_price - nearest_support) / current_price * 100), 2) if nearest_support else None
            }
        except Exception as e:
            print(f"خطأ في حساب الدعم والمقاومة: {e}")
            return {}
    
    def calculate_fibonacci_levels(self, data, periods=50):
        """حساب مستويات فيبوناتشي مع التحليل"""
        try:
            recent_data = data.tail(periods)
            high, low = recent_data['High'].max(), recent_data['Low'].min()
            diff = high - low
            current_price = data['Close'].iloc[-1]
            
            fib_levels = {
                'high': round(high, 2),
                'low': round(low, 2),
                'fib_23_6': round(high - (diff * 0.236), 2),
                'fib_38_2': round(high - (diff * 0.382), 2),
                'fib_50_0': round(high - (diff * 0.500), 2),
                'fib_61_8': round(high - (diff * 0.618), 2),
                'fib_78_6': round(high - (diff * 0.786), 2)
            }
            
            # تحليل موقع السعر
            if current_price > fib_levels['fib_23_6']:
                fib_analysis = "السعر قوي جداً فوق 23.6% - اتجاه صاعد قوي"
            elif current_price > fib_levels['fib_38_2']:
                fib_analysis = "السعر فوق 38.2% - اتجاه صاعد معتدل"
            elif current_price > fib_levels['fib_50_0']:
                fib_analysis = "السعر فوق 50% - منطقة محايدة"
            elif current_price > fib_levels['fib_61_8']:
                fib_analysis = "السعر فوق 61.8% - ضعف نسبي"
            else:
                fib_analysis = "السعر تحت 61.8% - اتجاه هابط محتمل"
            
            fib_levels['analysis'] = fib_analysis
            fib_levels['current_position'] = round(((current_price - low) / diff * 100), 2)
            
            return fib_levels
        except Exception as e:
            print(f"خطأ في حساب فيبوناتشي: {e}")
            return {}
    
    def fetch_economic_data(self):
        """جلب البيانات الاقتصادية المؤثرة على الذهب"""
        economic_data = {
            'status': 'simulated',
            'last_update': datetime.now().isoformat(),
            'indicators': {}
        }
        
        try:
            # محاكاة البيانات الاقتصادية
            economic_data['indicators'] = {
                'US_CPI': {
                    'value': 3.2,
                    'previous': 3.4,
                    'impact': 'إيجابي للذهب - تضخم منخفض',
                    'next_release': '2025-02-12'
                },
                'US_Interest_Rate': {
                    'value': 4.5,
                    'previous': 4.75,
                                        'impact': 'إيجابي للذهب - خفض الفائدة',
                    'next_release': '2025-01-29 FOMC'
                },
                'US_NFP': {
                    'value': 256000,
                    'previous': 227000,
                    'impact': 'سلبي للذهب - سوق عمل قوي',
                    'next_release': '2025-02-07'
                },
                'DXY_Index': {
                    'value': 108.5,
                    'trend': 'هابط',
                    'impact': 'إيجابي للذهب - ضعف الدولار'
                },
                'Geopolitical_Risk': {
                    'level': 'متوسط',
                    'events': ['توترات تجارية', 'قلق من التضخم'],
                    'impact': 'محايد إلى إيجابي للذهب'
                }
            }
            
            # حساب التأثير الإجمالي
            positive_factors = sum(1 for ind in economic_data['indicators'].values() 
                                 if 'إيجابي' in str(ind.get('impact', '')))
            negative_factors = sum(1 for ind in economic_data['indicators'].values() 
                                 if 'سلبي' in str(ind.get('impact', '')))
            
            if positive_factors > negative_factors:
                economic_data['overall_impact'] = 'إيجابي للذهب'
                economic_data['score'] = positive_factors - negative_factors
            elif negative_factors > positive_factors:
                economic_data['overall_impact'] = 'سلبي للذهب'
                economic_data['score'] = positive_factors - negative_factors
            else:
                economic_data['overall_impact'] = 'محايد'
                economic_data['score'] = 0
                
        except Exception as e:
            print(f"خطأ في جلب البيانات الاقتصادية: {e}")
            economic_data['error'] = str(e)
            
        return economic_data
    
    def analyze_volume_profile(self, data):
        """تحليل محسّن لحجم التداول"""
        try:
            latest = data.iloc[-1]
            prev_5 = data.tail(5)
            prev_20 = data.tail(20)
            
            current_volume = int(latest.get('Volume', 0))
            avg_volume_5 = int(prev_5['Volume'].mean())
            avg_volume_20 = int(prev_20['Volume'].mean())
            volume_ratio = latest.get('Volume_Ratio', 1)
            
            # تحليل قوة الحجم
            if volume_ratio > 2.0:
                volume_strength = 'قوي جداً'
                volume_signal = 'حجم استثنائي - احتمال حركة قوية'
            elif volume_ratio > 1.5:
                volume_strength = 'قوي'
                volume_signal = 'حجم فوق المتوسط - اهتمام متزايد'
            elif volume_ratio > 0.8:
                volume_strength = 'طبيعي'
                volume_signal = 'حجم طبيعي - لا إشارات خاصة'
            else:
                volume_strength = 'ضعيف'
                volume_signal = 'حجم ضعيف - حذر من الحركة الوهمية'
            
            # تحليل OBV
            obv_trend = 'صاعد' if data['OBV'].iloc[-1] > data['OBV'].iloc[-5] else 'هابط'
            
            return {
                'current_volume': current_volume,
                'avg_volume_5': avg_volume_5,
                'avg_volume_20': avg_volume_20,
                'volume_ratio': round(volume_ratio, 2),
                'volume_strength': volume_strength,
                'volume_signal': volume_signal,
                'obv_trend': obv_trend,
                'volume_price_correlation': 'إيجابي' if (latest['Close'] > data['Close'].iloc[-2] and current_volume > avg_volume_20) else 'سلبي'
            }
        except Exception as e:
            print(f"خطأ في تحليل الحجم: {e}")
            return {}
    
    def analyze_correlations(self, market_data):
        """تحليل الارتباطات مع تفسير محسّن"""
        try:
            print("📊 تحليل الارتباطات المتقدم...")
            daily_data = market_data['daily']
            correlations = {}
            strength = {}
            interpretation = {}
            
            if hasattr(daily_data.columns, 'levels'):
                available_symbols = daily_data.columns.get_level_values(0).unique()
                gold_symbol = self.symbols['gold'] if self.symbols['gold'] in available_symbols else self.symbols['gold_etf']
                
                if gold_symbol in available_symbols:
                    gold_prices = daily_data[gold_symbol]['Close'].dropna()
                    
                    for name, symbol in self.symbols.items():
                        if name not in ['gold', 'gold_etf'] and symbol in available_symbols:
                            if not daily_data[symbol].empty:
                                asset_prices = daily_data[symbol]['Close'].dropna()
                                common_index = gold_prices.index.intersection(asset_prices.index)
                                
                                if len(common_index) > 30:
                                    corr = gold_prices.loc[common_index].corr(asset_prices.loc[common_index])
                                    
                                    if pd.notna(corr):
                                        correlations[name] = round(corr, 3)
                                        
                                        # تحديد القوة
                                        if abs(corr) > 0.7:
                                            strength[name] = 'قوي جداً'
                                        elif abs(corr) > 0.5:
                                            strength[name] = 'قوي'
                                        elif abs(corr) > 0.3:
                                            strength[name] = 'متوسط'
                                        else:
                                            strength[name] = 'ضعيف'
                                        
                                        # التفسير
                                        if name == 'dxy':
                                            if corr < -0.5:
                                                interpretation[name] = 'ارتباط عكسي قوي - إيجابي للذهب عند ضعف الدولار'
                                            elif corr < -0.3:
                                                interpretation[name] = 'ارتباط عكسي معتدل - فرصة محتملة'
                                            else:
                                                interpretation[name] = 'ارتباط ضعيف - تأثير محدود'
                                        
                                        elif name == 'vix':
                                            if corr > 0.3:
                                                interpretation[name] = 'الذهب يستفيد من زيادة التقلبات'
                                            else:
                                                interpretation[name] = 'تأثير محدود من التقلبات'
                                        
                                        elif name == 'oil':
                                            if abs(corr) > 0.5:
                                                interpretation[name] = 'ارتباط قوي - مؤشر على التضخم'
                                            else:
                                                interpretation[name] = 'ارتباط ضعيف'
            
            return {
                'correlations': correlations,
                'strength_analysis': strength,
                'interpretation': interpretation
            }
        except Exception as e:
            print(f"❌ خطأ في تحليل الارتباطات: {e}")
            return {}
    
    async def fetch_news_enhanced(self):
        """جلب وتحليل الأخبار بشكل متقدم"""
        print("📰 جلب وتحليل أخبار الذهب المتقدم...")
        
        if not self.news_api_key:
            return {"status": "no_api_key", "message": "يتطلب مفتاح API للأخبار"}
        
        try:
            # جلب الأخبار بشكل غير متزامن
            articles = await self.news_analyzer.fetch_news_async()
            
            # استخراج وتحليل الأحداث
            events_analysis = self.news_analyzer.extract_events(articles)
            
            return {
                "status": "success",
                "events_analysis": events_analysis,
                "articles_count": len(articles)
            }
            
        except Exception as e:
            print(f"❌ خطأ في جلب الأخبار: {e}")
            return {"status": "error", "message": str(e)}
    
    def generate_professional_signals_v3(self, tech_data, correlations, volume, fib_levels, 
                                       support_resistance, economic_data, news_analysis, 
                                       mtf_analysis, ml_prediction):
        """توليد إشارات احترافية محسّنة V3 مع ML والتحليل متعدد الأطر"""
        print("🎯 توليد إشارات احترافية متقدمة V3...")
        
        try:
            latest = tech_data.iloc[-1]
            prev = tech_data.iloc[-2]
            
            # نظام النقاط المحسّن V3
            scores = {
                'trend': 0,
                'momentum': 0,
                'volume': 0,
                'fibonacci': 0,
                'correlation': 0,
                'support_resistance': 0,
                'economic': 0,
                'news': 0,
                'ma_cross': 0,
                'mtf_coherence': 0  # جديد
            }
            
            # 1. تحليل الاتجاه (25%)
            if latest['Close'] > latest['SMA_200']:
                scores['trend'] += 2
                if latest['Close'] > latest['SMA_50']:
                    scores['trend'] += 1
                    if latest['Close'] > latest['SMA_20']:
                        scores['trend'] += 1
            else:
                scores['trend'] -= 2
                if latest['Close'] < latest['SMA_50']:
                    scores['trend'] -= 1
                    if latest['Close'] < latest['SMA_20']:
                        scores['trend'] -= 1
            
            # التقاطعات الذهبية
            if latest.get('Golden_Cross', 0) == 1:
                scores['ma_cross'] = 3
            elif latest.get('Death_Cross', 0) == 1:
                scores['ma_cross'] = -3
            
            # 2. تحليل الزخم (20%)
            # MACD
            if latest['MACD'] > latest['MACD_Signal']:
                scores['momentum'] += 1
                if latest['MACD_Histogram'] > prev['MACD_Histogram']:
                    scores['momentum'] += 1
            else:
                scores['momentum'] -= 1
                if latest['MACD_Histogram'] < prev['MACD_Histogram']:
                    scores['momentum'] -= 1
            
            # RSI
            if 30 <= latest['RSI'] <= 70:
                if 45 <= latest['RSI'] <= 55:
                    scores['momentum'] += 0.5
                elif latest['RSI'] > 55:
                    scores['momentum'] += 1
                else:
                    scores['momentum'] -= 0.5
            elif latest['RSI'] < 30:
                scores['momentum'] += 2
            elif latest['RSI'] > 70:
                scores['momentum'] -= 2
            
            # Stochastic
            if latest.get('Stoch_K', 50) > latest.get('Stoch_D', 50):
                scores['momentum'] += 0.5
            
            # 3. تحليل الحجم (10%)
            if volume.get('volume_strength') == 'قوي جداً':
                scores['volume'] = 3
            elif volume.get('volume_strength') == 'قوي':
                scores['volume'] = 2
            elif volume.get('volume_strength') == 'طبيعي':
                scores['volume'] = 0
            else:
                scores['volume'] = -1
            
            # OBV
            if volume.get('obv_trend') == 'صاعد':
                scores['volume'] += 1
            
            # 4. تحليل فيبوناتشي (8%)
            if fib_levels:
                current_price = latest['Close']
                if current_price > fib_levels.get('fib_38_2', 0):
                    scores['fibonacci'] = 2
                elif current_price > fib_levels.get('fib_50_0', 0):
                    scores['fibonacci'] = 1
                elif current_price > fib_levels.get('fib_61_8', 0):
                    scores['fibonacci'] = -1
                else:
                    scores['fibonacci'] = -2
            
            # 5. تحليل الدعم والمقاومة (8%)
            if support_resistance:
                if support_resistance.get('price_to_support') and support_resistance['price_to_support'] < 2:
                    scores['support_resistance'] = 2
                elif support_resistance.get('price_to_resistance') and support_resistance['price_to_resistance'] < 2:
                    scores['support_resistance'] = -2
            
            # 6. تحليل الارتباطات (5%)
            dxy_corr = correlations.get('correlations', {}).get('dxy', 0)
            if dxy_corr < -0.7:
                scores['correlation'] = 2
            elif dxy_corr < -0.5:
                scores['correlation'] = 1
            elif dxy_corr > 0.5:
                scores['correlation'] = -1
            
            # 7. البيانات الاقتصادية (8%)
            if economic_data:
                econ_score = economic_data.get('score', 0)
                scores['economic'] = min(max(econ_score, -3), 3)
            
            # 8. تحليل الأخبار المتقدم (6%)
            if news_analysis and news_analysis.get('status') == 'success':
                events = news_analysis.get('events_analysis', {})
                total_impact = events.get('total_impact', 0)
                
                if total_impact > 5:
                    scores['news'] = 3
                elif total_impact > 2:
                    scores['news'] = 2
                elif total_impact < -5:
                    scores['news'] = -3
                elif total_impact < -2:
                    scores['news'] = -2
                else:
                    scores['news'] = 0
            
            # 9. توافق الأطر الزمنية (10%) - جديد
            if mtf_analysis:
                coherence_score = mtf_analysis.get('coherence_score', 0)
                scores['mtf_coherence'] = coherence_score
            
            # حساب النتيجة النهائية مع الأوزان المحدثة
            weights = {
                'trend': 0.20,
                'momentum': 0.15,
                'volume': 0.10,
                'fibonacci': 0.08,
                'correlation': 0.05,
                'support_resistance': 0.08,
                'economic': 0.08,
                'news': 0.06,
                'ma_cross': 0.10,
                'mtf_coherence': 0.10
            }
            
            total_score = sum(scores[key] * weights.get(key, 0) for key in scores)
            
            # دمج التنبؤ بالتعلم الآلي
            confidence_boost = 1.0
            ml_interpretation = ""
            
            if ml_prediction and ml_prediction[0] is not None:
                ml_probability = ml_prediction[0]
                ml_interpretation = ml_prediction[1]
                
                # تعديل النتيجة بناءً على ML
                if ml_probability > 0.75:
                    confidence_boost = 1.3
                elif ml_probability > 0.60:
                    confidence_boost = 1.15
                elif ml_probability < 0.40:
                    confidence_boost = 0.7
                elif ml_probability < 0.25:
                    confidence_boost = 0.5
                
                total_score *= confidence_boost
            
            # تحديد الإشارة والثقة
            if total_score >= 2.5:
                signal = "Strong Buy"
                confidence = "Very High"
                action = "شراء قوي - حجم كبير"
            elif total_score >= 1.5:
                signal = "Buy"
                confidence = "High"
                action = "شراء - حجم متوسط"
            elif total_score >= 0.5:
                signal = "Weak Buy"
                confidence = "Medium"
                action = "شراء حذر - حجم صغير"
            elif total_score <= -2.5:
                signal = "Strong Sell"
                confidence = "Very High"
                action = "بيع قوي - حجم كبير"
            elif total_score <= -1.5:
                signal = "Sell"
                confidence = "High"
                action = "بيع - حجم متوسط"
            elif total_score <= -0.5:
                signal = "Weak Sell"
                confidence = "Medium"
                action = "بيع حذر - حجم صغير"
            else:
                signal = "Hold"
                confidence = "Low"
                action = "انتظار - لا توجد إشارة واضحة"
            
            # إدارة المخاطر المحسّنة V3
            atr = latest.get('ATR', latest['Close'] * 0.02)
            price = latest['Close']
            volatility = latest.get('ATR_Percent', 2)
            
            # تعديل مستويات وقف الخسارة حسب التقلبات وML
            sl_multiplier = 1.5 if volatility < 1.5 else (2.0 if volatility < 2.5 else 2.5)
            
            # تعديل إضافي بناءً على ML
            if ml_prediction and ml_prediction[0] is not None:
                if ml_prediction[0] < 0.4:
                    sl_multiplier *= 0.8  # وقف خسارة أضيق للإشارات الضعيفة
            
            risk_management = {
                'stop_loss_levels': {
                    'tight': round(price - (atr * sl_multiplier * 0.75), 2),
                    'conservative': round(price - (atr * sl_multiplier), 2),
                    'moderate': round(price - (atr * sl_multiplier * 1.5), 2),
                    'wide': round(price - (atr * sl_multiplier * 2), 2)
                },
                'profit_targets': {
                    'target_1': round(price + (atr * 1.5), 2),
                    'target_2': round(price + (atr * 3), 2),
                    'target_3': round(price + (atr * 5), 2),
                    'target_4': round(price + (atr * 8), 2)
                },
                'position_size_recommendation': self._calculate_position_size_v3(confidence, volatility, ml_prediction),
                'risk_reward_ratio': round(3 / sl_multiplier, 2),
                'max_risk_per_trade': '2%' if confidence in ['Very High', 'High'] else '1%',
                'volatility_adjusted': True,
                'ml_adjusted': ml_prediction[0] is not None
            }
            
            # توصيات إضافية محسّنة
            entry_strategy = self._generate_entry_strategy_v3(scores, latest, support_resistance, mtf_analysis)
            
            # تحليل الأحداث القادمة
            upcoming_events = self._analyze_upcoming_events(economic_data, news_analysis)
            
            return {
                'signal': signal,
                'confidence': confidence,
                'action_recommendation': action,
                'total_score': round(total_score, 2),
                'component_scores': scores,
                'current_price': round(price, 2),
                'risk_management': risk_management,
                'entry_strategy': entry_strategy,
                'ml_prediction': {
                    'probability': round(ml_prediction[0], 3) if ml_prediction and ml_prediction[0] else None,
                    'interpretation': ml_interpretation
                },
                'mtf_summary': mtf_analysis.get('analysis', '') if mtf_analysis else '',
                'upcoming_events': upcoming_events,
                'technical_summary': {
                    'rsi': round(latest.get('RSI', 0), 1),
                    'macd': round(latest.get('MACD', 0), 2),
                    'williams_r': round(latest.get('Williams_R', 0), 1),
                    'stoch_k': round(latest.get('Stoch_K', 0), 1),
                    'bb_position': round(latest.get('BB_Position', 0.5), 2),
                    'volume_ratio': round(latest.get('Volume_Ratio', 1), 2)
                },
                'key_levels': {
                    'sma_20': round(latest.get('SMA_20', 0), 2),
                    'sma_50': round(latest.get('SMA_50', 0), 2),
                    'sma_200': round(latest.get('SMA_200', 0), 2),
                    'bb_upper': round(latest.get('BB_Upper', 0), 2),
                    'bb_lower': round(latest.get('BB_Lower', 0), 2)
                }
            }
            
        except Exception as e:
            print(f"❌ خطأ في توليد الإشارات: {e}")
            return {"error": str(e)}
    
    def _calculate_position_size_v3(self, confidence, volatility, ml_prediction):
        """حساب حجم المركز المحسّن مع ML"""
        base_recommendation = ""
        
        # التوصية الأساسية
        if confidence == "Very High" and volatility < 2:
            base_recommendation = "كبير (75-100% من رأس المال المخصص)"
        elif confidence == "High" and volatility < 2.5:
            base_recommendation = "متوسط-كبير (50-75%)"
        elif confidence == "High" or (confidence == "Medium" and volatility < 2):
            base_recommendation = "متوسط (25-50%)"
        elif confidence == "Medium":
            base_recommendation = "صغير (10-25%)"
        else:
            base_recommendation = "صغير جداً (5-10%) أو عدم الدخول"
        
        # تعديل بناءً على ML
        if ml_prediction and ml_prediction[0] is not None:
            if ml_prediction[0] > 0.75:
                base_recommendation += " (ML يؤكد بقوة)"
            elif ml_prediction[0] < 0.4:
                base_recommendation += " (ML يحذر - قلل الحجم)"
        
        return base_recommendation
    
    def _generate_entry_strategy_v3(self, scores, latest_data, support_resistance, mtf_analysis):
        """توليد استراتيجية دخول محسّنة V3"""
        strategy = {
            'entry_type': '',
            'entry_zones': [],
            'conditions': [],
            'warnings': [],
            'mtf_confirmation': ''
        }
        
        # تحليل توافق الأطر الزمنية
        if mtf_analysis:
            if mtf_analysis.get('coherence_score', 0) > 2:
                strategy['mtf_confirmation'] = '✅ توافق قوي عبر جميع الأطر الزمنية'
                strategy['entry_type'] = 'دخول مؤكد - توافق متعدد الأطر'
            elif mtf_analysis.get('coherence_score', 0) < -1:
                strategy['warnings'].append('⚠️ تضارب بين الأطر الزمنية')
                strategy['entry_type'] = 'انتظار توضيح الاتجاه'
            else:
                strategy['mtf_confirmation'] = 'توافق جزئي - حذر مطلوب'
        
        # تحديد نوع الدخول
        if scores['trend'] > 2 and scores['momentum'] > 1 and scores['mtf_coherence'] > 1:
            strategy['entry_type'] = 'دخول قوي - اتجاه واضح مع توافق'
            strategy['entry_zones'].append(f"دخول فوري عند {round(latest_data['Close'], 2)}")
        elif scores['support_resistance'] == 2:
            strategy['entry_type'] = 'دخول من الدعم'
            if support_resistance.get('nearest_support'):
                strategy['entry_zones'].append(f"انتظر ارتداد من {support_resistance['nearest_support']}")
        elif scores['momentum'] < -1:
            strategy['warnings'].append('⚠️ ذروة شراء - انتظر تصحيح')
            strategy['entry_type'] = 'انتظار تصحيح'
        else:
            strategy['entry_type'] = 'دخول تدريجي'
            strategy['entry_zones'].append('قسّم الدخول على 2-3 مراحل')
        
        # الشروط المطلوبة
        if latest_data.get('RSI', 50) > 70:
            strategy['conditions'].append('انتظر RSI < 70')
        if latest_data.get('Volume_Ratio', 1) < 0.8:
            strategy['warnings'].append('⚠️ حجم ضعيف - تأكيد مطلوب')
        
        # إضافة مناطق دخول بناءً على المستويات الفنية
        if latest_data.get('BB_Position', 0.5) < 0.2:
            strategy['entry_zones'].append(f"قرب الحد السفلي لبولينجر - فرصة شراء عند {round(latest_data.get('BB_Lower', 0), 2)}")
        elif latest_data.get('BB_Position', 0.5) > 0.8:
            strategy['warnings'].append('⚠️ قرب الحد العلوي لبولينجر - احتمال تصحيح')
        
        return strategy
    
    def _analyze_upcoming_events(self, economic_data, news_analysis):
        """تحليل الأحداث القادمة المؤثرة"""
        events = []
        
        # الأحداث الاقتصادية
        if economic_data and 'indicators' in economic_data:
            for indicator, data in economic_data['indicators'].items():
                if 'next_release' in data:
                    events.append({
                        'type': 'economic',
                        'name': indicator,
                        'date': data['next_release'],
                        'expected_impact': data.get('impact', 'غير محدد')
                    })
        
        # الأحداث من الأخبار
        if news_analysis and news_analysis.get('status') == 'success':
            events_data = news_analysis.get('events_analysis', {})
            if 'dominant_theme' in events_data:
                events.append({
                    'type': 'news_theme',
                    'name': events_data['dominant_theme'],
                    'impact': 'مستمر',
                    'recommendation': events_data.get('recommendation', '')
                })
        
        return events
    
    def generate_signal_for_backtest(self, data):
        """توليد إشارة مبسطة للاختبار الخلفي"""
        # نسخة مبسطة من generate_professional_signals_v3 للاختبار الخلفي
        score = 0
        
        # تحليل بسيط بناءً على البيانات المتاحة
        if 'close' in data and 'open' in data:
            if data['close'] > data['open']:
                score += 1
            else:
                score -= 1
        
        if score > 0:
            return {'action': 'Buy', 'confidence': 'Medium'}
        elif score < 0:
            return {'action': 'Sell', 'confidence': 'Medium'}
        else:
            return {'action': 'Hold', 'confidence': 'Low'}
    
    def generate_report_v3(self, analysis_result):
        """توليد تقرير نصي شامل V3"""
        try:
            report = []
            report.append("=" * 80)
            report.append("📊 تقرير التحليل الاحترافي للذهب - الإصدار 3.0")
            report.append("=" * 80)
            report.append(f"التاريخ: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append("")
            
            # الإشارة الرئيسية
            if 'gold_analysis' in analysis_result:
                ga = analysis_result['gold_analysis']
                report.append("🎯 الإشارة الرئيسية:")
                report.append(f"  • الإشارة: {ga.get('signal', 'N/A')}")
                report.append(f"  • الثقة: {ga.get('confidence', 'N/A')}")
                report.append(f"  • التوصية: {ga.get('action_recommendation', 'N/A')}")
                report.append(f"  • السعر الحالي: ${ga.get('current_price', 'N/A')}")
                report.append(f"  • النقاط الإجمالية: {ga.get('total_score', 'N/A')}")
                report.append("")
                
                # التنبؤ بالتعلم الآلي
                if 'ml_prediction' in ga and ga['ml_prediction'].get('probability') is not None:
                    report.append("🤖 تنبؤ التعلم الآلي:")
                    report.append(f"  • احتمالية النجاح: {ga['ml_prediction']['probability']:.1%}")
                    report.append(f"  • التفسير: {ga['ml_prediction']['interpretation']}")
                    report.append("")
                
                # توافق الأطر الزمنية
                if 'mtf_summary' in ga and ga['mtf_summary']:
                    report.append("⏰ تحليل الأطر الزمنية المتعددة:")
                    report.append(f"  • {ga['mtf_summary']}")
                    report.append("")
                
                # تفاصيل النقاط
                if 'component_scores' in ga:
                    report.append("📈 تحليل المكونات:")
                    for component, score in ga['component_scores'].items():
                        report.append(f"  • {component}: {score}")
                    report.append("")
                
                # إدارة المخاطر
                if 'risk_management' in ga:
                    rm = ga['risk_management']
                    report.append("⚠️ إدارة المخاطر:")
                    report.append(f"  • وقف الخسارة المحافظ: ${rm['stop_loss_levels'].get('conservative', 'N/A')}")
                    report.append(f"  • الهدف الأول: ${rm['profit_targets'].get('target_1', 'N/A')}")
                    report.append(f"  • الهدف الثاني: ${rm['profit_targets'].get('target_2', 'N/A')}")
                    report.append(f"  • حجم المركز: {rm.get('position_size_recommendation', 'N/A')}")
                    if rm.get('ml_adjusted'):
                        report.append("  • ✅ تم تعديل المخاطر بناءً على التعلم الآلي")
                    report.append("")
                
                # استراتيجية الدخول
                if 'entry_strategy' in ga:
                    es = ga['entry_strategy']
                    report.append("📍 استراتيجية الدخول:")
                    report.append(f"  • النوع: {es.get('entry_type', 'N/A')}")
                    if es.get('mtf_confirmation'):
                        report.append(f"  • {es['mtf_confirmation']}")
                    for zone in es.get('entry_zones', []):
                        report.append(f"  • {zone}")
                    for warning in es.get('warnings', []):
                        report.append(f"  • {warning}")
                    report.append("")
                
                # الأحداث القادمة
                if 'upcoming_events' in ga and ga['upcoming_events']:
                    report.append("📅 الأحداث القادمة المؤثرة:")
                    for event in ga['upcoming_events'][:5]:
                        report.append(f"  • {event.get('name', 'N/A')}: {event.get('date', 'N/A')} - {event.get('expected_impact', 'N/A')}")
                    report.append("")
            
            # تحليل الأطر الزمنية المتعددة المفصل
            if 'mtf_analysis' in analysis_result and analysis_result['mtf_analysis']:
                mtf = analysis_result['mtf_analysis']
                report.append("⏱️ تفاصيل الأطر الزمنية:")
                if 'timeframes' in mtf:
                    for tf_name, tf_data in mtf['timeframes'].items():
                        if tf_data:
                            report.append(f"  • {tf_name}: {tf_data.get('trend', 'N/A')} (نقاط: {tf_data.get('score', 0):.1f})")
                report.append(f"  • نقاط التوافق الإجمالية: {mtf.get('coherence_score', 0)}")
                report.append(f"  • التوصية: {mtf.get('recommendation', 'N/A')}")
                report.append("")
            
            # البيانات الاقتصادية
            if 'economic_data' in analysis_result:
                ed = analysis_result['economic_data']
                if ed.get('status') != 'error':
                    report.append("💰 البيانات الاقتصادية:")
                    report.append(f"  • التأثير الإجمالي: {ed.get('overall_impact', 'N/A')}")
                    if 'indicators' in ed:
                        for ind_name, ind_data in ed['indicators'].items():
                            if isinstance(ind_data, dict):
                                report.append(f"  • {ind_name}: {ind_data.get('value', 'N/A')} - {ind_data.get('impact', '')}")
                    report.append("")
            
            # تحليل الأخبار المتقدم
            if 'news_analysis' in analysis_result:
                na = analysis_result['news_analysis']
                if na.get('status') == 'success' and 'events_analysis' in na:
                    ea = na['events_analysis']
                    report.append("📰 تحليل الأحداث الإخبارية:")
                    report.append(f"  • التأثير الإجمالي: {ea.get('total_impact', 0):.1f}")
                    report.append(f"  • الموضوع المهيمن: {ea.get('dominant_theme', 'N/A')}")
                    report.append(f"  • التوصية: {ea.get('recommendation', 'N/A')}")
                    
                    if 'event_summary' in ea:
                        report.append("  • ملخص الأحداث:")
                        for event_type, count in ea['event_summary'].items():
                            report.append(f"    - {event_type}: {count} حدث")
                    report.append("")
            
            # الارتباطات
            if 'market_correlations' in analysis_result:
                mc = analysis_result['market_correlations']
                if 'correlations' in mc:
                    report.append("🔗 الارتباطات الرئيسية:")
                    for asset, corr in mc['correlations'].items():
                        interpretation = mc.get('interpretation', {}).get(asset, '')
                        report.append(f"  • {asset.upper()}: {corr} - {interpretation}")
                    report.append("")
            
            # نتائج الاختبار الخلفي (إن وجدت)
            if 'backtest_results' in analysis_result:
                bt_results = analysis_result['backtest_results']
                report.append("🔄 نتائج الاختبار الخلفي:")
                report.append(f"  • العائد الإجمالي: {bt_results.get('total_return', 0):.2f}%")
                report.append(f"  • نسبة شارب: {bt_results.get('sharpe_ratio', 0):.3f}")
                report.append(f"  • معدل الفوز: {bt_results.get('win_rate', 0):.1f}%")
                report.append(f"  • عامل الربح: {bt_results.get('profit_factor', 0):.2f}")
                report.append("")
            
            # ملخص السوق
            if 'market_summary' in analysis_result:
                ms = analysis_result['market_summary']
                report.append("📊 ملخص حالة السوق:")
                report.append(f"  • الحالة العامة: {ms.get('market_condition', 'N/A')}")
                report.append(f"  • آخر تحديث: {ms.get('last_update', 'N/A')}")
                report.append(f"  • نقاط البيانات: {ms.get('data_points', 0)}")
                report.append("")
            
            report.append("=" * 80)
            report.append("انتهى التقرير - الإصدار 3.0")
            report.append("تم دمج: التعلم الآلي | تحليل متعدد الأطر | تحليل أخبار متقدم | اختبار خلفي احترافي")
            
            return "\n".join(report)
            
        except Exception as e:
            return f"خطأ في توليد التقرير: {e}"
    
    async def run_analysis_v3(self):
        """تشغيل التحليل الاحترافي الشامل V3"""
        print("🚀 بدء التحليل الاحترافي المتقدم للذهب - الإصدار 3.0...")
        print("=" * 80)
        
        try:
            # 1. جلب البيانات
            market_data = self.fetch_multi_timeframe_data()
            if market_data is None:
                raise ValueError("فشل في جلب بيانات السوق")
            
            # 2. استخراج بيانات الذهب
            gold_data = self.extract_gold_data(market_data)
            if gold_data is None:
                raise ValueError("فشل في استخراج بيانات الذهب")
            
            # 3. حساب المؤشرات الفنية
            technical_data = self.calculate_professional_indicators(gold_data)
            
            # 4. تحليل متعدد الأطر الزمنية
            print("\n⏰ تحليل الأطر الزمنية المتعددة...")
            coherence_score, mtf_analysis = self.mtf_analyzer.get_coherence_score(self.symbols['gold'])
            
            # 5. حساب مستويات فيبوناتشي
            fibonacci_levels = self.calculate_fibonacci_levels(technical_data)
            
            # 6. حساب الدعم والمقاومة
            support_resistance = self.calculate_support_resistance(technical_data)
            
            # 7. تحليل الحجم
            volume_analysis = self.analyze_volume_profile(technical_data)
            
            # 8. تحليل الارتباطات
            correlations = self.analyze_correlations(market_data)
            
            # 9. جلب البيانات الاقتصادية
            economic_data = self.fetch_economic_data()
            
            # 10. جلب وتحليل الأخبار المتقدم
            news_data = await self.fetch_news_enhanced()
            
            # 11. التنبؤ بالتعلم الآلي
            print("\n🤖 التنبؤ بالتعلم الآلي...")
            # تحديث الأسعار المستقبلية للبيانات التاريخية
            self.db_manager.update_future_prices()
            
            # جلب بيانات التدريب
            training_data = self.db_manager.get_training_data()
            
            ml_prediction = None
            if training_data and len(training_data) >= 100:
                # تدريب النموذج إذا لزم الأمر
                if not os.path.exists(self.ml_predictor.model_path):
                    print("  • تدريب نموذج جديد...")
                    self.ml_predictor.train_model(training_data)
                
                # التنبؤ
                current_analysis = {
                    'gold_analysis': {
                        'component_scores': {},  # سيتم ملؤها لاحقاً
                        'technical_summary': {}
                    },
                    'volume_analysis': volume_analysis,
                    'market_correlations': correlations,
                    'economic_data': economic_data
                }
                ml_prediction = self.ml_predictor.predict_probability(current_analysis)
            else:
                print("  • بيانات غير كافية للتعلم الآلي")
            
            # 12. توليد الإشارات النهائية V3
            signals = self.generate_professional_signals_v3(
                technical_data, correlations, volume_analysis, 
                fibonacci_levels, support_resistance, 
                economic_data, news_data, mtf_analysis, ml_prediction
            )
            
            # 13. اختبار خلفي (اختياري)
            backtest_results = None
            if len(technical_data) > 100:
                print("\n🔄 تشغيل الاختبار الخلفي...")
                backtest_results = self.backtester.run_backtest(technical_data)
            
            # تجميع النتائج النهائية
            final_result = {
                'timestamp': datetime.now().isoformat(),
                'status': 'success',
                'version': '3.0',
                'gold_analysis': signals,
                'mtf_analysis': mtf_analysis,
                'fibonacci_levels': fibonacci_levels,
                'support_resistance': support_resistance,
                'volume_analysis': volume_analysis,
                'market_correlations': correlations,
                'economic_data': economic_data,
                'news_analysis': news_data,
                'backtest_results': backtest_results,
                'market_summary': {
                    'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'data_points': len(technical_data),
                    'timeframe': 'Multi-timeframe',
                    'market_condition': self._determine_market_condition_v3(signals, volume_analysis, mtf_analysis)
                }
            }
            
            # حفظ في قاعدة البيانات
            self.db_manager.save_analysis(final_result)
            
            # حفظ النتائج
            self.save_results_v3(final_result)
            
            # توليد وطباعة التقرير
            report = self.generate_report_v3(final_result)
            print(report)
            
            print("\n✅ تم إتمام التحليل الاحترافي V3.0 بنجاح!")
            return final_result
            
        except Exception as e:
            error_message = f"❌ فشل التحليل الاحترافي: {e}"
            print(error_message)
            error_result = {
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'version': '3.0',
                'error': str(e)
            }
            self.save_results_v3(error_result)
            return error_result
    
    def _determine_market_condition_v3(self, signals, volume, mtf_analysis):
        """تحديد حالة السوق العامة V3"""
        conditions = []
        
        # تحليل الإشارة الأساسية
        if signals.get('signal') in ['Strong Buy', 'Buy']:
            if volume.get('volume_strength') in ['قوي', 'قوي جداً']:
                conditions.append('صاعد قوي')
            else:
                conditions.append('صاعد')
        elif signals.get('signal') in ['Strong Sell', 'Sell']:
            if volume.get('volume_strength') in ['قوي', 'قوي جداً']:
                conditions.append('هابط قوي')
            else:
                conditions.append('هابط')
        else:
            conditions.append('عرضي')
        
        # تحليل توافق الأطر الزمنية
        if mtf_analysis and mtf_analysis.get('coherence_score', 0) > 2:
            conditions.append('توافق قوي')
        elif mtf_analysis and mtf_analysis.get('coherence_score', 0) < -2:
            conditions.append('تضارب شديد')
        
        # تحليل ML
        if signals.get('ml_prediction', {}).get('probability'):
            ml_prob = signals['ml_prediction']['probability']
            if ml_prob > 0.7:
                conditions.append('ML إيجابي')
            elif ml_prob < 0.3:
                conditions.append('ML سلبي')
        
        # دمج الحالات
        if 'صاعد قوي' in conditions and 'توافق قوي' in conditions:
            return 'صاعد قوي جداً - فرصة ممتازة'
        elif 'هابط قوي' in conditions and 'توافق قوي' in conditions:
            return 'هابط قوي جداً - تجنب الشراء'
        elif 'تضارب شديد' in conditions:
            return 'متقلب - عدم وضوح'
        elif 'عرضي' in conditions:
            return 'عرضي/محايد - انتظار'
        else:
            return ' | '.join(conditions[:2])
    
    def save_results_v3(self, results):
        """حفظ النتائج في ملفات متعددة V3"""
        try:
            # حفظ JSON الرئيسي
            filename = "gold_analysis_v3.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            print(f"💾 تم حفظ التحليل في: {filename}")
            
            # حفظ نسخة مؤرخة
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            archive_filename = f"gold_analysis_v3_{timestamp}.json"
            with open(archive_filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            print(f"📁 تم حفظ نسخة مؤرخة: {archive_filename}")
            
            # حفظ التقرير النصي
            if results.get('status') == 'success':
                report = self.generate_report_v3(results)
                report_filename = f"gold_report_v3_{timestamp}.txt"
                with open(report_filename, 'w', encoding='utf-8') as f:
                    f.write(report)
                print(f"📄 تم حفظ التقرير: {report_filename}")
            
            # حفظ ملخص للمراجعة السريعة
            summary_filename = "gold_analysis_summary.json"
            summary = {
                'last_update': results.get('timestamp'),
                'version': results.get('version'),
                'signal': results.get('gold_analysis', {}).get('signal'),
                'confidence': results.get('gold_analysis', {}).get('confidence'),
                'price': results.get('gold_analysis', {}).get('current_price'),
                'ml_probability': results.get('gold_analysis', {}).get('ml_prediction', {}).get('probability'),
                'market_condition': results.get('market_summary', {}).get('market_condition')
            }
            with open(summary_filename, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"📋 تم حفظ الملخص: {summary_filename}")
            
        except Exception as e:
            print(f"❌ خطأ في حفظ النتائج: {e}")

def main():
    """الدالة الرئيسية لتشغيل المحلل"""
    analyzer = ProfessionalGoldAnalyzerV3()
    
    # تشغيل التحليل بشكل غير متزامن
    asyncio.run(analyzer.run_analysis_v3())

if __name__ == "__main__":
    main()
