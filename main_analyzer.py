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
            'gold': 'GC=F',
            'gold_etf': 'GLD',
            'dxy': 'DX-Y.NYB',
            'vix': '^VIX',
            'treasury': '^TNX',
            'oil': 'CL=F',
            'spy': 'SPY',
            'usdeur': 'EURUSD=X'
        }
        self.news_api_key = os.getenv("NEWS_API_KEY")
        
    def fetch_multi_timeframe_data(self):
        """جلب بيانات متعددة الأطر الزمنية"""
        print("📊 جلب بيانات متعددة الأطر الزمنية...")
        
        try:
            symbols_list = list(self.symbols.values())
            
            # جلب بيانات يومية (6 شهور)
            daily_data = yf.download(symbols_list, period="6mo", interval="1d", group_by='ticker')
            
            # جلب بيانات ساعية (30 يوم للذهب فقط)
            hourly_data = yf.download(self.symbols['gold'], period="30d", interval="1h")
            
            print("✅ تم جلب البيانات متعددة الأطر")
            return {
                'daily': daily_data,
                'hourly': hourly_data
            }
            
        except Exception as e:
            print(f"❌ خطأ في جلب البيانات: {e}")
            return None

    def extract_gold_data(self, market_data):
        """استخراج بيانات الذهب مع تحسينات"""
        try:
            daily_data = market_data['daily']
            hourly_data = market_data['hourly']
            
            # استخراج البيانات اليومية
            if hasattr(daily_data.columns, 'levels') and len(daily_data.columns.levels) > 1:
                available_symbols = daily_data.columns.levels[0].tolist()
                
                if self.symbols['gold'] in available_symbols:
                    gold_daily = daily_data[self.symbols['gold']].copy()
                elif self.symbols['gold_etf'] in available_symbols:
                    gold_daily = daily_data[self.symbols['gold_etf']].copy()
                else:
                    raise ValueError("لا يمكن العثور على بيانات الذهب")
            else:
                gold_daily = daily_data.copy()
            
            # تنظيف البيانات
            gold_daily = gold_daily.dropna(subset=['Close'])
            
            if not hourly_data.empty:
                hourly_data = hourly_data.dropna(subset=['Close'])
            
            print(f"✅ بيانات يومية: {len(gold_daily)} يوم")
            print(f"✅ بيانات ساعية: {len(hourly_data)} ساعة")
            
            return {
                'daily': gold_daily,
                'hourly': hourly_data
            }
            
        except Exception as e:
            print(f"❌ خطأ في استخراج بيانات الذهب: {e}")
            return None

    def calculate_professional_indicators(self, gold_data):
        """حساب مؤشرات احترافية متقدمة"""
        try:
            print("📊 حساب المؤشرات الاحترافية...")
            
            daily_df = gold_data['daily'].copy()
            
            # المؤشرات الأساسية
            daily_df['SMA_20'] = daily_df['Close'].rolling(window=20).mean()
            daily_df['SMA_50'] = daily_df['Close'].rolling(window=50).mean()
            daily_df['SMA_200'] = daily_df['Close'].rolling(window=200).mean()
            
            # RSI
            delta = daily_df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            daily_df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = daily_df['Close'].ewm(span=12).mean()
            exp2 = daily_df['Close'].ewm(span=26).mean()
            daily_df['MACD'] = exp1 - exp2
            daily_df['MACD_Signal'] = daily_df['MACD'].ewm(span=9).mean()
            daily_df['MACD_Histogram'] = daily_df['MACD'] - daily_df['MACD_Signal']
            
            # Bollinger Bands
            std = daily_df['Close'].rolling(window=20).std()
            daily_df['BB_Upper'] = daily_df['SMA_20'] + (std * 2)
            daily_df['BB_Lower'] = daily_df['SMA_20'] - (std * 2)
            daily_df['BB_Width'] = ((daily_df['BB_Upper'] - daily_df['BB_Lower']) / daily_df['SMA_20']) * 100
            
            # ATR
            high_low = daily_df['High'] - daily_df['Low']
            high_close = np.abs(daily_df['High'] - daily_df['Close'].shift())
            low_close = np.abs(daily_df['Low'] - daily_df['Close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            daily_df['ATR'] = true_range.rolling(14).mean()
            daily_df['ATR_Percent'] = (daily_df['ATR'] / daily_df['Close']) * 100
            
            # حجم التداول المتقدم
            daily_df['Volume_SMA'] = daily_df['Volume'].rolling(20).mean()
            daily_df['Volume_Ratio'] = daily_df['Volume'] / daily_df['Volume_SMA']
            daily_df['OBV'] = (daily_df['Volume'] * ((daily_df['Close'] - daily_df['Close'].shift()) / daily_df['Close'].shift().abs())).cumsum()
            
            # مؤشرات الزخم المتقدمة
            daily_df['ROC'] = ((daily_df['Close'] - daily_df['Close'].shift(14)) / daily_df['Close'].shift(14)) * 100
            daily_df['Williams_R'] = ((daily_df['High'].rolling(14).max() - daily_df['Close']) / 
                                     (daily_df['High'].rolling(14).max() - daily_df['Low'].rolling(14).min())) * -100
            
            # مستويات الدعم والمقاومة
            daily_df['Resistance_20'] = daily_df['High'].rolling(window=20).max()
            daily_df['Support_20'] = daily_df['Low'].rolling(window=20).min()
            daily_df['Resistance_50'] = daily_df['High'].rolling(window=50).max()
            daily_df['Support_50'] = daily_df['Low'].rolling(window=50).min()
            
            # مؤشر القوة النسبية المتقدم
            daily_df['Strength_Index'] = (
                (daily_df['RSI'] - 50) * 0.3 +
                (daily_df['ROC']) * 0.4 +
                ((daily_df['Close'] - daily_df['SMA_50']) / daily_df['SMA_50'] * 100) * 0.3
            )
            
            print("✅ تم حساب المؤشرات الاحترافية")
            return daily_df
            
        except Exception as e:
            print(f"❌ خطأ في حساب المؤشرات: {e}")
            return gold_data['daily']

    def calculate_fibonacci_levels(self, data, periods=50):
        """حساب مستويات فيبوناتشي"""
        try:
            recent_data = data.tail(periods)
            high = recent_data['High'].max()
            low = recent_data['Low'].min()
            
            diff = high - low
            
            fib_levels = {
                'high': round(high, 2),
                'low': round(low, 2),
                'fib_23.6': round(high - (diff * 0.236), 2),
                'fib_38.2': round(high - (diff * 0.382), 2),
                'fib_50.0': round(high - (diff * 0.500), 2),
                'fib_61.8': round(high - (diff * 0.618), 2),
                'fib_78.6': round(high - (diff * 0.786), 2)
            }
            
            return fib_levels
            
        except Exception as e:
            print(f"❌ خطأ في حساب فيبوناتشي: {e}")
            return {}

    def analyze_volume_profile(self, data):
        """تحليل حجم التداول المتقدم"""
        try:
            latest = data.iloc[-1]
            recent_data = data.tail(20)
            
            volume_analysis = {
                'current_volume': int(latest['Volume']),
                'avg_volume_20': int(recent_data['Volume'].mean()),
                'volume_ratio': round(latest['Volume_Ratio'], 2),
                'obv_trend': 'صاعد' if data['OBV'].iloc[-1] > data['OBV'].iloc[-10] else 'هابط',
                'volume_strength': 'قوي' if latest['Volume_Ratio'] > 1.5 else ('ضعيف' if latest['Volume_Ratio'] < 0.7 else 'طبيعي')
            }
            
            return volume_analysis
            
        except Exception as e:
            print(f"❌ خطأ في تحليل الحجم: {e}")
            return {}

    def analyze_correlations(self, market_data):
        """تحليل الارتباطات المتقدم"""
        try:
            print("📊 تحليل الارتباطات المتقدم...")
            
            daily_data = market_data['daily']
            correlations = {}
            correlation_strength = {}
            
            if hasattr(daily_data.columns, 'levels') and len(daily_data.columns.levels) > 1:
                available_symbols = daily_data.columns.levels[0].tolist()
                
                gold_symbol = None
                if self.symbols['gold'] in available_symbols:
                    gold_symbol = self.symbols['gold']
                elif self.symbols['gold_etf'] in available_symbols:
                    gold_symbol = self.symbols['gold_etf']
                
                if gold_symbol:
                    gold_prices = daily_data[gold_symbol]['Close'].dropna()
                    
                    for name, symbol in self.symbols.items():
                        if name not in ['gold', 'gold_etf'] and symbol in available_symbols:
                            try:
                                asset_prices = daily_data[symbol]['Close'].dropna()
                                common_index = gold_prices.index.intersection(asset_prices.index)
                                
                                if len(common_index) > 30:
                                    corr = gold_prices.loc[common_index].corr(asset_prices.loc[common_index])
                                    if not pd.isna(corr):
                                        correlations[name] = round(corr, 3)
                                        
                                        # تصنيف قوة الارتباط
                                        if abs(corr) > 0.7:
                                            correlation_strength[name] = 'قوي جداً'
                                        elif abs(corr) > 0.5:
                                            correlation_strength[name] = 'قوي'
                                        elif abs(corr) > 0.3:
                                            correlation_strength[name] = 'متوسط'
                                        else:
                                            correlation_strength[name] = 'ضعيف'
                                            
                            except Exception as e:
                                continue
            
            return {
                'correlations': correlations,
                'strength_analysis': correlation_strength
            }
            
        except Exception as e:
            print(f"❌ خطأ في تحليل الارتباطات: {e}")
            return {'correlations': {}, 'strength_analysis': {}}

    def fetch_news(self):
        """جلب الأخبار المتطورة"""
        print("📰 جلب أخبار الذهب المتخصصة...")
        
        if not self.news_api_key:
            return {"status": "no_api_key", "articles": []}
        
        try:
            # كلمات مفتاحية أكثر تخصصاً
            keywords = (
                "gold OR XAU OR \"gold price\" OR \"precious metals\" OR \"federal reserve\" OR "
                "\"interest rate\" OR inflation OR \"dollar index\" OR \"safe haven\" OR "
                "\"central bank\" OR \"monetary policy\" OR \"gold futures\" OR \"bullion\""
            )
            
            url = (
                f"https://newsapi.org/v2/everything?"
                f"q={keywords}&"
                f"language=en&"
                f"sortBy=publishedAt&"
                f"pageSize=20&"
                f"from={(datetime.now() - timedelta(days=2)).date()}&"
                f"apiKey={self.news_api_key}"
            )
            
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            
            articles = response.json().get('articles', [])
            
            # تصفية وتصنيف الأخبار
            high_impact = []
            medium_impact = []
            
            high_impact_keywords = ['federal reserve', 'fed', 'interest rate', 'inflation', 'monetary policy']
            medium_impact_keywords = ['gold', 'xau', 'precious metals', 'dollar', 'bullion']
            
            for article in articles:
                title = (article.get('title', '') or '').lower()
                desc = (article.get('description', '') or '').lower()
                content = f"{title} {desc}"
                
                if any(keyword in content for keyword in high_impact_keywords):
                    high_impact.append({
                        'title': article.get('title', ''),
                        'source': article.get('source', {}).get('name', ''),
                        'publishedAt': article.get('publishedAt', ''),
                        'impact': 'عالي'
                    })
                elif any(keyword in content for keyword in medium_impact_keywords):
                    medium_impact.append({
                        'title': article.get('title', ''),
                        'source': article.get('source', {}).get('name', ''),
                        'publishedAt': article.get('publishedAt', ''),
                        'impact': 'متوسط'
                    })
            
            return {
                "status": "success",
                "high_impact_news": high_impact[:3],
                "medium_impact_news": medium_impact[:3],
                "total_analyzed": len(articles)
            }
            
        except Exception as e:
            print(f"❌ خطأ في جلب الأخبار: {e}")
            return {"status": "error", "error": str(e)}

    def generate_professional_signals(self, technical_data, correlations, volume_analysis, fibonacci_levels):
        """توليد إشارات احترافية متقدمة"""
        try:
            print("🎯 توليد إشارات احترافية...")
            
            latest = technical_data.iloc[-1]
            prev = technical_data.iloc[-2]
            
            # نظام نقاط متقدم
            signals = {}
            score = 0
            
            # 1. تحليل الاتجاه المتقدم (وزن 30%)
            trend_score = 0
            if pd.notna(latest['SMA_200']) and pd.notna(latest['SMA_50']):
                if latest['Close'] > latest['SMA_200']:
                    signals['long_term_trend'] = "صاعد قوي" if latest['Close'] > latest['SMA_50'] else "صاعد"
                    trend_score += 3 if latest['Close'] > latest['SMA_50'] else 2
                else:
                    signals['long_term_trend'] = "هابط قوي" if latest['Close'] < latest['SMA_50'] else "هابط"
                    trend_score -= 3 if latest['Close'] < latest['SMA_50'] else -2
            
            # 2. تحليل الزخم المتعدد (وزن 25%)
            momentum_score = 0
            
            # MACD
            if pd.notna(latest['MACD']) and pd.notna(latest['MACD_Signal']):
                if latest['MACD'] > latest['MACD_Signal']:
                    if latest['MACD_Histogram'] > prev['MACD_Histogram']:
                        signals['macd'] = "إيجابي متزايد"
                        momentum_score += 2
                    else:
                        signals['macd'] = "إيجابي"
                        momentum_score += 1
                else:
                    signals['macd'] = "سلبي"
                    momentum_score -= 1
            
            # RSI متقدم
            if pd.notna(latest['RSI']):
                rsi = latest['RSI']
                if 40 <= rsi <= 60:
                    signals['rsi_status'] = "منطقة متوازنة"
                    momentum_score += 1
                elif rsi > 70:
                    signals['rsi_status'] = "ذروة شراء - حذر"
                    momentum_score -= 1
                elif rsi < 30:
                    signals['rsi_status'] = "ذروة بيع - فرصة"
                    momentum_score += 2
                else:
                    signals['rsi_status'] = f"طبيعي ({rsi:.1f})"
            
            # ROC (معدل التغير)
            if pd.notna(latest['ROC']):
                if latest['ROC'] > 2:
                    signals['roc'] = "زخم صاعد قوي"
                    momentum_score += 1
                elif latest['ROC'] < -2:
                    signals['roc'] = "زخم هابط قوي"
                    momentum_score -= 1
                else:
                    signals['roc'] = "زخم معتدل"
            
            # 3. تحليل الحجم (وزن 15%)
            volume_score = 0
            if volume_analysis.get('volume_strength') == 'قوي':
                signals['volume_confirmation'] = "حجم مؤكد للاتجاه"
                volume_score += 1
            elif volume_analysis.get('volume_strength') == 'ضعيف':
                signals['volume_confirmation'] = "حجم ضعيف - حذر"
                volume_score -= 0.5
            else:
                signals['volume_confirmation'] = "حجم طبيعي"
            
            # 4. تحليل فيبوناتشي (وزن 15%)
            fib_score = 0
            current_price = latest['Close']
            if fibonacci_levels:
                # تحديد موقع السعر من مستويات فيبوناتشي
                if current_price > fibonacci_levels.get('fib_61.8', 0):
                    signals['fibonacci_position'] = "فوق 61.8% - قوة"
                    fib_score += 1
                elif current_price < fibonacci_levels.get('fib_38.2', 0):
                    signals['fibonacci_position'] = "تحت 38.2% - ضعف"
                    fib_score -= 1
                else:
                    signals['fibonacci_position'] = "داخل النطاق الطبيعي"
            
            # 5. تحليل الارتباطات (وزن 15%)
            correlation_score = 0
            dxy_corr = correlations.get('correlations', {}).get('dxy', 0)
            if dxy_corr < -0.7:
                signals['dollar_relationship'] = "ارتباط سلبي قوي جداً - مفيد للذهب"
                correlation_score += 2
            elif dxy_corr < -0.5:
                signals['dollar_relationship'] = "ارتباط سلبي قوي - مفيد للذهب"
                correlation_score += 1
            elif dxy_corr > 0.3:
                signals['dollar_relationship'] = "ارتباط إيجابي غير طبيعي"
                correlation_score -= 1
            else:
                signals['dollar_relationship'] = f"ارتباط معتدل ({dxy_corr})"
            
            # حساب النتيجة النهائية المرجحة
            total_score = (
                trend_score * 0.30 +
                momentum_score * 0.25 +
                volume_score * 0.15 +
                fib_score * 0.15 +
                correlation_score * 0.15
            )
            
            # تحديد الإشارة النهائية مع مستويات دقة
            if total_score >= 3:
                final_signal = "Strong Buy"
                confidence = "Very High"
                action = "افتح صفقة شراء بحجم كامل"
            elif total_score >= 1.5:
                final_signal = "Buy"
                confidence = "High"
                action = "افتح صفقة شراء بحجم متوسط"
            elif total_score >= 0.5:
                final_signal = "Weak Buy"
                confidence = "Medium"
                action = "شراء حذر بحجم صغير"
            elif total_score <= -3:
                final_signal = "Strong Sell"
                confidence = "Very High"
                action = "افتح صفقة بيع بحجم كامل"
            elif total_score <= -1.5:
                final_signal = "Sell"
                confidence = "High"
                action = "افتح صفقة بيع بحجم متوسط"
            elif total_score <= -0.5:
                final_signal = "Weak Sell"
                confidence = "Medium"
                action = "بيع حذر بحجم صغير"
            else:
                final_signal = "Hold"
                confidence = "Low"
                action = "ابق خارج السوق وانتظر"
            
            # إدارة المخاطر المتقدمة
            current_price = latest['Close']
            atr = latest.get('ATR', current_price * 0.02)
            atr_percent = latest.get('ATR_Percent', 2.0)
            
            # حساب مستويات متعددة لوقف الخسارة
            conservative_sl = current_price - (atr * 1.5)  # محافظ
            moderate_sl = current_price - (atr * 2.0)      # متوسط
            aggressive_sl = current_price - (atr * 2.5)    # عدواني
            
            # أهداف متعددة
            target_1 = current_price + (atr * 2)    # هدف قريب
            target_2 = current_price + (atr * 3.5)  # هدف متوسط
            target_3 = current_price + (atr * 5)    # هدف بعيد
            
            result = {
                'signal': final_signal,
                'confidence': confidence,
                'action_recommendation': action,
                'total_score': round(total_score, 2),
                'component_scores': {
                    'trend': round(trend_score, 1),
                    'momentum': round(momentum_score, 1),
                    'volume': round(volume_score, 1),
                    'fibonacci': round(fib_score, 1),
                    'correlation': round(correlation_score, 1)
                },
                'current_price': round(current_price, 2),
                'risk_management': {
                    'stop_loss_levels': {
                        'conservative': round(conservative_sl, 2),
                        'moderate': round(moderate_sl, 2),
                        'aggressive': round(aggressive_sl, 2)
                    },
                    'profit_targets': {
                        'target_1': round(target_1, 2),
                        'target_2': round(target_2, 2),
                        'target_3': round(target_3, 2)
                    },
                    'position_size_recommendation': self.calculate_position_size(atr_percent, confidence),
                    'risk_reward_ratios': {
                        'conservative': round((target_1 - current_price) / (current_price - conservative_sl), 2),
                        'moderate': round((target_2 - current_price) / (current_price - moderate_sl), 2),
                        'aggressive': round((target_3 - current_price) / (current_price - aggressive_sl), 2)
                    }
                },
                'technical_details': signals,
                'advanced_indicators': {
                    'rsi': round(latest.get('RSI', 0), 1),
                    'williams_r': round(latest.get('Williams_R', 0), 1),
                    'roc': round(latest.get('ROC', 0), 2),
                    'bb_width': round(latest.get('BB_Width', 0), 2),
                    'atr_percent': round(atr_percent, 2),
                    'strength_index': round(latest.get('Strength_Index', 0), 2)
                }
            }
            
            print(f"✅ الإشارة الاحترافية: {final_signal} ({confidence})")
            return result
            
        except Exception as e:
            print(f"❌ خطأ في توليد الإشارات: {e}")
            return {"error": str(e)}

    def calculate_position_size(self, volatility, confidence):
        """حساب حجم الصفقة المناسب"""
        if confidence == "Very High":
            if volatility > 3:
                return "متوسط (تقلبات عالية رغم الثقة)"
            else:
                return "كبير (ثقة عالية + تقلبات معقولة)"
        elif confidence == "High":
            if volatility > 2.5:
                return "صغير (تقلبات عالية)"
            else:
                return "متوسط"
        else:
            return "صغير جداً (ثقة منخفضة)"

    def get_market_status(self):
        """حالة السوق المتقدمة"""
        try:
            import pytz
            
            # أوقات متعددة
            utc_time = datetime.now(pytz.UTC)
            ny_time = utc_time.astimezone(pytz.timezone('America/New_York'))
            london_time = utc_time.astimezone(pytz.timezone('Europe/London'))
            
            # حالة الأسواق
            ny_trading = ny_time.weekday() < 5 and 9 <= ny_time.hour < 16
            london_trading = london_time.weekday() < 5 and 8 <= london_time.hour < 17
            
            return {
                'current_time_utc': utc_time.strftime('%Y-%m-%d %H:%M:%S UTC'),
                'ny_time': ny_time.strftime('%Y-%m-%d %H:%M:%S EST'),
                'london_time': london_time.strftime('%Y-%m-%d %H:%M:%S GMT'),
                'ny_market_status': 'Open' if ny_trading else 'Closed',
                'london_market_status': 'Open' if london_trading else 'Closed',
                'is_major_trading_session': ny_trading or london_trading,
                'market_overlap': ny_trading and london_trading
            }
            
        except:
            return {
                'current_time_utc': datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'),
                'status': 'Unknown'
            }

    def run_analysis(self):
        """تشغيل التحليل الاحترافي الشامل"""
        print("🚀 بدء التحليل الاحترافي للذهب...")
        
        try:
            # 1. جلب البيانات متعددة الأطر
            market_data = self.fetch_multi_timeframe_data()
            if market_data is None:
                raise ValueError("فشل في جلب بيانات السوق")
            
            # 2. استخراج بيان
