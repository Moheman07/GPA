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
                    size = self._calculate_position_size(signal['confidence'], self.data.close[0])
                    if size > 0:
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
        
        def _calculate_position_size(self, confidence, price):
            """✅ تصحيح: حساب حجم المركز (عدد الوحدات) بناءً على الثقة والسعر"""
            if price <= 0:
                return 0 # تجنب القسمة على صفر
                
            cash_to_risk = self.broker.getcash() * self.params.risk_percent
            base_size = cash_to_risk / price # حساب عدد الوحدات
            
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
            # التأكد من أن gold_analysis ليس خطأً
            if not gold_analysis or 'error' in gold_analysis:
                print("⚠️ تم تخطي الحفظ في قاعدة البيانات بسبب وجود خطأ في التحليل")
                return

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
                gold_analysis.get('ml_prediction', {}).get('probability', 0)
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
            if len(records) > 0:
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
            # ✅ تصحيح: زيادة الفترة الزمنية لضمان وجود بيانات كافية للاختبار الخلفي
            daily_data = yf.download(list(self.symbols.values()), 
                                    period="3y", interval="1d", 
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
        """حساب المؤشرات الاحترافية المحسّنة"
