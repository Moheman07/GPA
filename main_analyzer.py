#!/usr/bin/env python3
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import json
import os
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

class ProfessionalGoldAnalyzer:
    def __init__(self):
        self.symbols = {
            'gold': 'GC=F', 'gold_etf': 'GLD', 'dxy': 'DX-Y.NYB',
            'vix': '^VIX', 'treasury': '^TNX', 'oil': 'CL=F',
            'spy': 'SPY', 'usdeur': 'EURUSD=X', 'silver': 'SI=F'
        }
        self.news_api_key = os.getenv("NEWS_API_KEY")
        self.fred_api_key = os.getenv("FRED_API_KEY")  # للبيانات الاقتصادية

    def fetch_multi_timeframe_data(self):
        print("📊 جلب بيانات متعددة الأطر الزمنية...")
        try:
            daily_data = yf.download(list(self.symbols.values()), 
                                    period="1y", interval="1d", 
                                    group_by='ticker', progress=False)

            # جلب بيانات 4 ساعات للتحليل قصير المدى
            hourly_data = yf.download(self.symbols['gold'], 
                                     period="1mo", interval="1h", 
                                     progress=False)

            if daily_data.empty: 
                raise ValueError("فشل جلب البيانات")

            return {'daily': daily_data, 'hourly': hourly_data}
        except Exception as e:
            print(f"❌ خطأ في جلب البيانات: {e}")
            return None

    def extract_gold_data(self, market_data):
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
            # محاكاة البيانات الاقتصادية (يمكن استبدالها بـ API حقيقي)
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

    def fetch_news(self):
        """جلب وتحليل الأخبار المؤثرة على الذهب"""
        print("📰 جلب وتحليل أخبار الذهب...")

        if not self.news_api_key:
            return {"status": "no_api_key", "message": "يتطلب مفتاح API للأخبار"}

        try:
            # كلمات مفتاحية محسّنة ومركزة
            keywords = '"gold price" OR "XAU/USD" OR "federal reserve interest" OR "US inflation" OR "FOMC meeting"'

            url = f"https://newsapi.org/v2/everything?q={keywords}&language=en&sortBy=publishedAt&pageSize=30&apiKey={self.news_api_key}"

            response = requests.get(url, timeout=15)
            articles = response.json().get('articles', [])

            # تصنيف الأخبار حسب الأهمية والتأثير
            high_impact_keywords = ['federal reserve', 'fed', 'interest rate', 'fomc', 'inflation', 'cpi', 'employment', 'nfp']
            medium_impact_keywords = ['dollar', 'dxy', 'treasury', 'geopolitical', 'crisis', 'war']
            gold_specific_keywords = ['gold', 'xau', 'precious metal', 'bullion']

            categorized_news = {
                'critical': [],
                'high_impact': [],
                'medium_impact': [],
                'gold_specific': []
            }

            for article in articles:
                if not article.get('title'):
                    continue

                title_lower = article['title'].lower()
                content_lower = (article.get('description') or '').lower()
                full_text = f"{title_lower} {content_lower}"

                news_item = {
                    'title': article['title'][:150],
                    'source': article.get('source', {}).get('name', 'Unknown'),
                    'published': article.get('publishedAt', ''),
                    'url': article.get('url', ''),
                    'impact': None,
                    'sentiment': None
                }

                # تحديد التأثير
                if any(kw in full_text for kw in ['rate decision', 'fomc decision', 'emergency']):
                    news_item['impact'] = 'حرج - تأثير فوري'
                    news_item['sentiment'] = self._analyze_sentiment(full_text)
                    categorized_news['critical'].append(news_item)
                elif any(kw in full_text for kw in high_impact_keywords):
                    news_item['impact'] = 'عالي'
                    news_item['sentiment'] = self._analyze_sentiment(full_text)
                    categorized_news['high_impact'].append(news_item)
                elif any(kw in full_text for kw in medium_impact_keywords):
                    news_item['impact'] = 'متوسط'
                    categorized_news['medium_impact'].append(news_item)
                elif any(kw in full_text for kw in gold_specific_keywords):
                    news_item['impact'] = 'مباشر للذهب'
                    categorized_news['gold_specific'].append(news_item)

            # تحليل إجمالي للأخبار
            total_news = sum(len(v) for v in categorized_news.values())

            news_summary = {
                'total_relevant_news': total_news,
                'critical_count': len(categorized_news['critical']),
                'high_impact_count': len(categorized_news['high_impact']),
                'overall_sentiment': self._calculate_overall_sentiment(categorized_news)
            }

            return {
                "status": "success",
                "summary": news_summary,
                "categorized_news": {
                    k: v[:3] for k, v in categorized_news.items() if v  # أول 3 أخبار من كل فئة
                }
            }

        except Exception as e:
            print(f"❌ خطأ في جلب الأخبار: {e}")
            return {"status": "error", "message": str(e)}

    def _analyze_sentiment(self, text):
        """تحليل بسيط للمشاعر في النص"""
        positive_words = ['rise', 'gain', 'up', 'high', 'boost', 'surge', 'rally', 'bullish']
        negative_words = ['fall', 'drop', 'down', 'low', 'decline', 'plunge', 'bearish', 'crisis']

        text_lower = text.lower()
        positive_score = sum(1 for word in positive_words if word in text_lower)
        negative_score = sum(1 for word in negative_words if word in text_lower)

        if positive_score > negative_score:
            return 'إيجابي'
        elif negative_score > positive_score:
            return 'سلبي'
        else:
            return 'محايد'

    def _calculate_overall_sentiment(self, categorized_news):
        """حساب المشاعر الإجمالية من الأخبار"""
        sentiments = []
        for category, news_list in categorized_news.items():
            for news in news_list:
                if news.get('sentiment'):
                    sentiments.append(news['sentiment'])

        if not sentiments:
            return 'محايد'

        positive = sentiments.count('إيجابي')
        negative = sentiments.count('سلبي')

        if positive > negative:
            return 'إيجابي للذهب'
        elif negative > positive:
            return 'سلبي للذهب'
        else:
            return 'محايد'

    def generate_professional_signals(self, tech_data, correlations, volume, fib_levels, support_resistance, economic_data, news_analysis):
        """توليد إشارات احترافية محسّنة بناءً على جميع المدخلات"""
        print("🎯 توليد إشارات احترافية متقدمة...")

        try:
            latest = tech_data.iloc[-1]
            prev = tech_data.iloc[-2]

            # نظام النقاط المحسّن
            scores = {
                'trend': 0,
                'momentum': 0,
                'volume': 0,
                'fibonacci': 0,
                'correlation': 0,
                'support_resistance': 0,
                'economic': 0,
                'news': 0,
                'ma_cross': 0
            }

            # 1. تحليل الاتجاه (30%)
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

            # 2. تحليل الزخم (25%)
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
                    scores['momentum'] += 0.5  # منطقة محايدة قوية
                elif latest['RSI'] > 55:
                    scores['momentum'] += 1  # زخم صاعد
                else:
                    scores['momentum'] -= 0.5  # زخم هابط
            elif latest['RSI'] < 30:
                scores['momentum'] += 2  # ذروة بيع
            elif latest['RSI'] > 70:
                scores['momentum'] -= 2  # ذروة شراء

            # Stochastic
            if latest.get('Stoch_K', 50) > latest.get('Stoch_D', 50):
                scores['momentum'] += 0.5

            # 3. تحليل الحجم (15%)
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

            # 4. تحليل فيبوناتشي (10%)
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

            # 5. تحليل الدعم والمقاومة (10%)
            if support_resistance:
                if support_resistance.get('price_to_support') and support_resistance['price_to_support'] < 2:
                    scores['support_resistance'] = 2  # قريب من دعم قوي
                elif support_resistance.get('price_to_resistance') and support_resistance['price_to_resistance'] < 2:
                    scores['support_resistance'] = -2  # قريب من مقاومة قوية

            # 6. تحليل الارتباطات (5%)
            dxy_corr = correlations.get('correlations', {}).get('dxy', 0)
            if dxy_corr < -0.7:
                scores['correlation'] = 2
            elif dxy_corr < -0.5:
                scores['correlation'] = 1
            elif dxy_corr > 0.5:
                scores['correlation'] = -1

            # 7. البيانات الاقتصادية (10%)
            if economic_data:
                econ_score = economic_data.get('score', 0)
                scores['economic'] = min(max(econ_score, -3), 3)  # حد أقصى ±3

            # 8. تحليل الأخبار (5%)
            if news_analysis and news_analysis.get('status') == 'success':
                sentiment = news_analysis.get('summary', {}).get('overall_sentiment', 'محايد')
                if sentiment == 'إيجابي للذهب':
                    scores['news'] = 2
                elif sentiment == 'سلبي للذهب':
                    scores['news'] = -2

                # أخبار حرجة
                if news_analysis.get('summary', {}).get('critical_count', 0) > 0:
                    scores['news'] *= 2  # مضاعفة التأثير

            # حساب النتيجة النهائية
            weights = {
                'trend': 0.25,
                'momentum': 0.20,
                'volume': 0.15,
                'fibonacci': 0.10,
                'correlation': 0.05,
                'support_resistance': 0.10,
                'economic': 0.10,
                'news': 0.05,
                'ma_cross': 0.10
            }

            total_score = sum(scores[key] * weights.get(key, 0) for key in scores)

            # تحديد الإشارة والثقة
            if total_score >= 2.0:
                signal = "Strong Buy"
                confidence = "Very High"
                action = "شراء قوي - حجم كبير"
            elif total_score >= 1.0:
                signal = "Buy"
                confidence = "High"
                action = "شراء - حجم متوسط"
            elif total_score >= 0.3:
                signal = "Weak Buy"
                confidence = "Medium"
                action = "شراء حذر - حجم صغير"
            elif total_score <= -2.0:
                signal = "Strong Sell"
                confidence = "Very High"
                action = "بيع قوي - حجم كبير"
            elif total_score <= -1.0:
                signal = "Sell"
                confidence = "High"
                action = "بيع - حجم متوسط"
            elif total_score <= -0.3:
                signal = "Weak Sell"
                confidence = "Medium"
                action = "بيع حذر - حجم صغير"
            else:
                signal = "Hold"
                confidence = "Low"
                action = "انتظار - لا توجد إشارة واضحة"

            # إدارة المخاطر المحسّنة
            atr = latest.get('ATR', latest['Close'] * 0.02)
            price = latest['Close']
            volatility = latest.get('ATR_Percent', 2)

            # تعديل مستويات وقف الخسارة حسب التقلبات
            sl_multiplier = 1.5 if volatility < 1.5 else (2.0 if volatility < 2.5 else 2.5)

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
                'position_size_recommendation': self._calculate_position_size(confidence, volatility),
                'risk_reward_ratio': round(3 / sl_multiplier, 2),
                'max_risk_per_trade': '2%' if confidence in ['Very High', 'High'] else '1%'
            }

            # توصيات إضافية
            entry_strategy = self._generate_entry_strategy(scores, latest, support_resistance)

            return {
                'signal': signal,
                'confidence': confidence,
                'action_recommendation': action,
                'total_score': round(total_score, 2),
                'component_scores': scores,
                'current_price': round(price, 2),
                'risk_management': risk_management,
                'entry_strategy': entry_strategy,
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

    def _calculate_position_size(self, confidence, volatility):
        """حساب حجم المركز بناءً على الثقة والتقلبات"""
        if confidence == "Very High" and volatility < 2:
            return "كبير (75-100% من رأس المال المخصص)"
        elif confidence == "High" and volatility < 2.5:
            return "متوسط-كبير (50-75%)"
        elif confidence == "High" or (confidence == "Medium" and volatility < 2):
            return "متوسط (25-50%)"
        elif confidence == "Medium":
            return "صغير (10-25%)"
        else:
            return "صغير جداً (5-10%) أو عدم الدخول"

    def _generate_entry_strategy(self, scores, latest_data, support_resistance):
        """توليد استراتيجية دخول مفصلة"""
        strategy = {
            'entry_type': '',
            'entry_zones': [],
            'conditions': [],
            'warnings': []
        }

        # تحديد نوع الدخول
        if scores['trend'] > 2 and scores['momentum'] > 1:
            strategy['entry_type'] = 'دخول قوي - السوق في اتجاه واضح'
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

        return strategy

    def generate_report(self, analysis_result):
        """توليد تقرير نصي شامل"""
        try:
            report = []
            report.append("=" * 60)
            report.append("📊 تقرير التحليل الاحترافي للذهب")
            report.append("=" * 60)
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

            # الأخبار
            if 'news_analysis' in analysis_result:
                na = analysis_result['news_analysis']
                if na.get('status') == 'success' and 'summary' in na:
                    report.append("📰 ملخص الأخبار:")
                    summary = na['summary']
                    report.append(f"  • المشاعر العامة: {summary.get('overall_sentiment', 'N/A')}")
                    report.append(f"  • أخبار حرجة: {summary.get('critical_count', 0)}")
                    report.append(f"  • أخبار عالية التأثير: {summary.get('high_impact_count', 0)}")
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

            report.append("=" * 60)
            report.append("انتهى التقرير")

            return "\n".join(report)

        except Exception as e:
            return f"خطأ في توليد التقرير: {e}"

    def run_analysis(self):
        """تشغيل التحليل الاحترافي الشامل المحسّن"""
        print("🚀 بدء التحليل الاحترافي المتقدم للذهب...")
        print("=" * 60)

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

            # 4. حساب مستويات فيبوناتشي
            fibonacci_levels = self.calculate_fibonacci_levels(technical_data)

            # 5. حساب الدعم والمقاومة
            support_resistance = self.calculate_support_resistance(technical_data)

            # 6. تحليل الحجم
            volume_analysis = self.analyze_volume_profile(technical_data)

            # 7. تحليل الارتباطات
            correlations = self.analyze_correlations(market_data)

            # 8. جلب البيانات الاقتصادية
            economic_data = self.fetch_economic_data()

            # 9. جلب وتحليل الأخبار
            news_data = self.fetch_news()

            # 10. توليد الإشارات النهائية
            signals = self.generate_professional_signals(
                technical_data, correlations, volume_analysis, 
                fibonacci_levels, support_resistance, 
                economic_data, news_data
            )

            # تجميع النتائج النهائية
            final_result = {
                'timestamp': datetime.now().isoformat(),
                'status': 'success',
                'gold_analysis': signals,
                'fibonacci_levels': fibonacci_levels,
                'support_resistance': support_resistance,
                'volume_analysis': volume_analysis,
                'market_correlations': correlations,
                'economic_data': economic_data,
                'news_analysis': news_data,
                'market_summary': {
                    'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'data_points': len(technical_data),
                    'timeframe': 'Daily',
                    'market_condition': self._determine_market_condition(signals, volume_analysis)
                }
            }

            # حفظ النتائج
            self.save_results(final_result)

            # توليد وطباعة التقرير
            report = self.generate_report(final_result)
            print(report)

            print("\n✅ تم إتمام التحليل الاحترافي بنجاح!")
            return final_result

        except Exception as e:
            error_message = f"❌ فشل التحليل الاحترافي: {e}"
            print(error_message)
            error_result = {
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'error': str(e)
            }
            self.save_results(error_result)
            return error_result

    def _determine_market_condition(self, signals, volume):
        """تحديد حالة السوق العامة"""
        if signals.get('signal') in ['Strong Buy', 'Buy'] and volume.get('volume_strength') in ['قوي', 'قوي جداً']:
            return 'صاعد قوي'
        elif signals.get('signal') in ['Strong Sell', 'Sell'] and volume.get('volume_strength') in ['قوي', 'قوي جداً']:
            return 'هابط قوي'
        elif signals.get('signal') == 'Hold':
            return 'عرضي/محايد'
        else:
            return 'متقلب'

    def save_results(self, results):
        """حفظ النتائج في ملفات متعددة"""
        try:
            # حفظ JSON الرئيسي
            filename = "gold_analysis.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            print(f"💾 تم حفظ التحليل في: {filename}")

            # حفظ نسخة مؤرخة
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            archive_filename = f"gold_analysis_{timestamp}.json"
            with open(archive_filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            print(f"📁 تم حفظ نسخة مؤرخة: {archive_filename}")

            # حفظ التقرير النصي
            if results.get('status') == 'success':
                report = self.generate_report(results)
                report_filename = f"gold_report_{timestamp}.txt"
                with open(report_filename, 'w', encoding='utf-8') as f:
                    f.write(report)
                print(f"📄 تم حفظ التقرير: {report_filename}")

        except Exception as e:
            print(f"❌ خطأ في حفظ النتائج: {e}")

def main():
    analyzer = ProfessionalGoldAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main()