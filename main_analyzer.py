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
            
            print("✅ تم جلب البيانات متعددة الأطر")
            return {
                'daily': daily_data
            }
            
        except Exception as e:
            print(f"❌ خطأ في جلب البيانات: {e}")
            return None

    def extract_gold_data(self, market_data):
        """استخراج بيانات الذهب مع تحسينات"""
        try:
            daily_data = market_data['daily']
            
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
            
            print(f"✅ بيانات يومية: {len(gold_daily)} يوم")
            
            return gold_daily
            
        except Exception as e:
            print(f"❌ خطأ في استخراج بيانات الذهب: {e}")
            return None

    def calculate_professional_indicators(self, gold_data):
        """حساب مؤشرات احترافية متقدمة"""
        try:
            print("📊 حساب المؤشرات الاحترافية...")
            
            daily_df = gold_data.copy()
            
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
            
            # مؤشرات الزخم المتقدمة
            daily_df['ROC'] = ((daily_df['Close'] - daily_df['Close'].shift(14)) / daily_df['Close'].shift(14)) * 100
            daily_df['Williams_R'] = ((daily_df['High'].rolling(14).max() - daily_df['Close']) / 
                                     (daily_df['High'].rolling(14).max() - daily_df['Low'].rolling(14).min())) * -100
            
            # مستويات الدعم والمقاومة
            daily_df['Resistance_20'] = daily_df['High'].rolling(window=20).max()
            daily_df['Support_20'] = daily_df['Low'].rolling(window=20).min()
            
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
            return gold_data

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
                'fib_23_6': round(high - (diff * 0.236), 2),
                'fib_38_2': round(high - (diff * 0.382), 2),
                'fib_50_0': round(high - (diff * 0.500), 2),
                'fib_61_8': round(high - (diff * 0.618), 2),
                'fib_78_6': round(high - (diff * 0.786), 2)
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
                'current_volume': int(latest.get('Volume', 0)),
                'avg_volume_20': int(recent_data['Volume'].mean()),
                'volume_ratio': round(latest.get('Volume_Ratio', 1), 2),
                'volume_strength': 'قوي' if latest.get('Volume_Ratio', 1) > 1.5 else ('ضعيف' if latest.get('Volume_Ratio', 1) < 0.7 else 'طبيعي')
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
            return {"status": "no_api_key", "high_impact_news": [], "medium_impact_news": []}
        
        try:
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
            return {"status": "error", "error": str(e), "high_impact_news": [], "medium_impact_news": []}

    def generate_professional_signals(self, technical_data, correlations, volume_analysis, fibonacci_levels):
        """توليد إشارات احترافية متقدمة"""
        try:
            print("🎯 توليد إشارات احترافية...")
            
            latest = technical_data.iloc[-1]
            prev = technical_data.iloc[-2]
            
            # نظام نقاط متقدم
            signals = {}
            score = 0
            
            # 1. تحليل الاتجاه المتقدم
            trend_score = 0
            if pd.notna(latest.get('SMA_200')) and pd.notna(latest.get('SMA_50')):
                if latest['Close'] > latest['SMA_200']:
                    signals['long_term_trend'] = "صاعد قوي" if latest['Close'] > latest['SMA_50'] else "صاعد"
                    trend_score += 3 if latest['Close'] > latest['SMA_50'] else 2
                else:
                    signals['long_term_trend'] = "هابط قوي" if latest['Close'] < latest['SMA_50'] else "هابط"
                    trend_score -= 3 if latest['Close'] < latest['SMA_50'] else -2
            
            # 2. تحليل الزخم المتعدد
            momentum_score = 0
            
            # MACD
            if pd.notna(latest.get('MACD')) and pd.notna(latest.get('MACD_Signal')):
                if latest['MACD'] > latest['MACD_Signal']:
                    if latest.get('MACD_Histogram', 0) > prev.get('MACD_Histogram', 0):
                        signals['macd'] = "إيجابي متزايد"
                        momentum_score += 2
                    else:
                        signals['macd'] = "إيجابي"
                        momentum_score += 1
                else:
                    signals['macd'] = "سلبي"
                    momentum_score -= 1
            
            # RSI متقدم
            if pd.notna(latest.get('RSI')):
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
            if pd.notna(latest.get('ROC')):
                if latest['ROC'] > 2:
                    signals['roc'] = "زخم صاعد قوي"
                    momentum_score += 1
                elif latest['ROC'] < -2:
                    signals['roc'] = "زخم هابط قوي"
                    momentum_score -= 1
                else:
                    signals['roc'] = "زخم معتدل"
            
            # 3. تحليل الحجم
            volume_score = 0
            if volume_analysis.get('volume_strength') == 'قوي':
                signals['volume_confirmation'] = "حجم مؤكد للاتجاه"
                volume_score += 1
            elif volume_analysis.get('volume_strength') == 'ضعيف':
                signals['volume_confirmation'] = "حجم ضعيف - حذر"
                volume_score -= 0.5
            else:
                signals['volume_confirmation'] = "حجم طبيعي"
            
            # 4. تحليل فيبوناتشي
            fib_score = 0
            current_price = latest['Close']
            if fibonacci_levels:
                if current_price > fibonacci_levels.get('fib_61_8', 0):
                    signals['fibonacci_position'] = "فوق 61.8% - قوة"
                    fib_score += 1
                elif current_price < fibonacci_levels.get('fib_38_2', 0):
                    signals['fibonacci_position'] = "تحت 38.2% - ضعف"
                    fib_score -= 1
                else:
                    signals['fibonacci_position'] = "داخل النطاق الطبيعي"
            
            # 5. تحليل الارتباطات
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
            
            # تحديد الإشارة النهائية
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
            conservative_sl = current_price - (atr * 1.5)
            moderate_sl = current_price - (atr * 2.0)
            aggressive_sl = current_price - (atr * 2.5)
            
            # أهداف متعددة
            target_1 = current_price + (atr * 2)
            target_2 = current_price + (atr * 3.5)
            target_3 = current_price + (atr * 5)
            
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
                        'conservative': round((target_1 - current_price) / (current_price - conservative_sl), 2) if (current_price - conservative_sl) > 0 else 0,
                        'moderate': round((target_2 - current_price) / (current_price - moderate_sl), 2) if (current_price - moderate_sl) > 0 else 0,
                        'aggressive': round((target_3 - current_price) / (current_price - aggressive_sl), 2) if (current_price - aggressive_sl) > 0 else 0
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
            
            utc_time = datetime.now(pytz.UTC)
            ny_time = utc_time.astimezone(pytz.timezone('America/New_York'))
            london_time = utc_time.astimezone(pytz.timezone('Europe/London'))
            
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
            
            # 2. استخراج بيانات الذهب
            gold_data = self.extract_gold_data(market_data)
            if gold_data is None:
                raise ValueError("فشل في استخراج بيانات الذهب")
            
            # 3. حساب المؤشرات الاحترافية
            technical_data = self.calculate_professional_indicators(gold_data)
            
            # 4. حساب مستويات فيبوناتشي
            fibonacci_levels = self.calculate_fibonacci_levels(technical_data)
            
            # 5. تحليل الحجم
            volume_analysis = self.analyze_volume_profile(technical_data)
            
            # 6. تحليل الارتباطات
            correlations = self.analyze_correlations(market_data)
            
            # 7. جلب الأخبار
            news_data = self.fetch_news()
            
            # 8. توليد الإشارات الاحترافية
            signals = self.generate_professional_signals(
                technical_data, correlations, volume_analysis, fibonacci_levels
            )
            
            # 9. تجميع النتائج النهائية
            results = {
                'timestamp': datetime.now().isoformat(),
                'last_update': datetime.now().strftime('%Y-%m-%d %H:%M UTC'),
                'market_status': self.get_market_status(),
                'gold_analysis': {
                    'price_usd': signals.get('current_price'),
                    'signal': signals.get('signal'),
                    'confidence': signals.get('confidence'),
                    'action_recommendation': signals.get('action_recommendation'),
                    'technical_score': signals.get('total_score'),
                    'component_analysis': signals.get('component_scores', {}),
                    'technical_details': signals.get('technical_details', {}),
                    'advanced_indicators': signals.get('advanced_indicators', {}),
                    'risk_management': signals.get('risk_management', {})
                },
                'fibonacci_levels': fibonacci_levels,
                'volume_analysis': volume_analysis,
                'market_correlations': correlations,
                'news_analysis': news_data,
                 'summary': {
                    'signal': signals.get('signal', 'N/A'),
                    'price': signals.get('current_price', 'N/A'),
                    'confidence': signals.get('confidence', 'N/A'),
                    'action': signals.get('action_recommendation', 'N/A'),
                    'rsi': signals.get('advanced_indicators', {}).get('rsi', 'N/A'),
                    'trend': signals.get('technical_details', {}).get('long_term_trend', 'N/A')
                }
            }
            
            # 10. حفظ النتيجة
            self.save_single_result(results)
            
            print("✅ تم إتمام التحليل الاحترافي بنجاح!")
            return results
            
        except Exception as e:
            print(f"❌ فشل التحليل الاحترافي: {e}")
            
            error_result = {
                'timestamp': datetime.now().isoformat(),
                'last_update': datetime.now().strftime('%Y-%m-%d %H:%M UTC'),
                'status': 'error',
                'error': str(e),
                'market_status': self.get_market_status()
            }
            
            self.save_single_result(error_result)
            return error_result

    def save_single_result(self, results):
        """حفظ النتيجة في ملف واحد فقط"""
        try:
            filename = "gold_analysis.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            
            print(f"💾 تم تحديث الملف: {filename}")
            
            if os.path.exists(filename):
                file_size = os.path.getsize(filename)
                print(f"📁 حجم الملف: {file_size} بايت")
            else:
                print("❌ لم يتم إنشاء الملف!")
            
        except Exception as e:
            print(f"❌ خطأ في حفظ النتائج: {e}")

def main():
    """الدالة الرئيسية"""
    print("=" * 60)
    print("🏆 محلل الذهب الاحترافي المتطور")
    print("=" * 60)
    
    analyzer = ProfessionalGoldAnalyzer()
    results = analyzer.run_analysis()
    
    # طباعة ملخص احترافي
    print("\n" + "=" * 60)
    print("📋 ملخص التحليل الاحترافي:")
    print("=" * 60)
    
    if results.get('status') != 'error' and 'gold_analysis' in results:
        gold = results['gold_analysis']
        
        # المعلومات الأساسية
        print(f"💰 السعر الحالي: ${gold.get('price_usd', 'N/A')}")
        print(f"🎯 الإشارة: {gold.get('signal', 'N/A')}")
        print(f"🔍 مستوى الثقة: {gold.get('confidence', 'N/A')}")
        print(f"📊 النقاط الإجمالية: {gold.get('technical_score', 'N/A')}")
        print(f"💡 التوصية: {gold.get('action_recommendation', 'N/A')}")
        
        # تحليل المكونات
        components = gold.get('component_analysis', {})
        if components:
            print(f"\n📊 تحليل المكونات:")
            print(f"   • الاتجاه: {components.get('trend', 'N/A')}")
            print(f"   • الزخم: {components.get('momentum', 'N/A')}")
            print(f"   • الحجم: {components.get('volume', 'N/A')}")
            print(f"   • فيبوناتشي: {components.get('fibonacci', 'N/A')}")
            print(f"   • الارتباط: {components.get('correlation', 'N/A')}")
        
        # المؤشرات المتقدمة
        indicators = gold.get('advanced_indicators', {})
        if indicators:
            print(f"\n📈 المؤشرات المتقدمة:")
            print(f"   • RSI: {indicators.get('rsi', 'N/A')}")
            print(f"   • Williams %R: {indicators.get('williams_r', 'N/A')}")
            print(f"   • معدل التغير (ROC): {indicators.get('roc', 'N/A')}%")
            print(f"   • عرض البولينجر: {indicators.get('bb_width', 'N/A')}%")
            print(f"   • ATR النسبي: {indicators.get('atr_percent', 'N/A')}%")
        
        # إدارة المخاطر
        risk = gold.get('risk_management', {})
        if risk:
            print(f"\n🛡️ إدارة المخاطر:")
            
            # مستويات وقف الخسارة
            stop_levels = risk.get('stop_loss_levels', {})
            print(f"   🛑 وقف الخسارة:")
            print(f"      • محافظ: ${stop_levels.get('conservative', 'N/A')}")
            print(f"      • متوسط: ${stop_levels.get('moderate', 'N/A')}")
            print(f"      • عدواني: ${stop_levels.get('aggressive', 'N/A')}")
            
            # الأهداف
            targets = risk.get('profit_targets', {})
            print(f"   🎯 أهداف الربح:")
            print(f"      • الهدف الأول: ${targets.get('target_1', 'N/A')}")
            print(f"      • الهدف الثاني: ${targets.get('target_2', 'N/A')}")
            print(f"      • الهدف الثالث: ${targets.get('target_3', 'N/A')}")
            
            # نسب المخاطرة للربح
            ratios = risk.get('risk_reward_ratios', {})
            print(f"   ⚖️ نسب المخاطرة للربح:")
            print(f"      • محافظ: 1:{ratios.get('conservative', 'N/A')}")
            print(f"      • متوسط: 1:{ratios.get('moderate', 'N/A')}")
            print(f"      • عدواني: 1:{ratios.get('aggressive', 'N/A')}")
            
            # حجم الصفقة المقترح
            position_size = risk.get('position_size_recommendation', 'N/A')
            print(f"   📏 حجم الصفقة المقترح: {position_size}")
        
        # مستويات فيبوناتشي
        fibonacci = results.get('fibonacci_levels', {})
        if fibonacci:
            print(f"\n🌟 مستويات فيبوناتشي:")
            print(f"   • أعلى نقطة: ${fibonacci.get('high', 'N/A')}")
            print(f"   • 78.6%: ${fibonacci.get('fib_78_6', 'N/A')}")
            print(f"   • 61.8%: ${fibonacci.get('fib_61_8', 'N/A')}")
            print(f"   • 50.0%: ${fibonacci.get('fib_50_0', 'N/A')}")
            print(f"   • 38.2%: ${fibonacci.get('fib_38_2', 'N/A')}")
            print(f"   • 23.6%: ${fibonacci.get('fib_23_6', 'N/A')}")
            print(f"   • أدنى نقطة: ${fibonacci.get('low', 'N/A')}")
        
        # تحليل الحجم
        volume = results.get('volume_analysis', {})
        if volume:
            print(f"\n📊 تحليل الحجم:")
            print(f"   • الحجم الحالي: {volume.get('current_volume', 'N/A'):,}")
            print(f"   • متوسط الحجم (20): {volume.get('avg_volume_20', 'N/A'):,}")
            print(f"   • نسبة الحجم: {volume.get('volume_ratio', 'N/A')}")
            print(f"   • قوة الحجم: {volume.get('volume_strength', 'N/A')}")
        
        # الارتباطات
        correlations = results.get('market_correlations', {}).get('correlations', {})
        strength = results.get('market_correlations', {}).get('strength_analysis', {})
        if correlations:
            print(f"\n🔗 ارتباطات السوق:")
            for asset, corr in correlations.items():
                strength_level = strength.get(asset, 'غير محدد')
                print(f"   • {asset.upper()}: {corr} ({strength_level})")
        
        # الأخبار
        news = results.get('news_analysis', {})
        if news.get('status') == 'success':
            high_impact = news.get('high_impact_news', [])
            if high_impact:
                print(f"\n📰 أخبار عالية التأثير:")
                for i, article in enumerate(high_impact, 1):
                    print(f"   {i}. {article.get('title', 'بدون عنوان')}")
        
        # حالة السوق
        market_status = results.get('market_status', {})
        if market_status:
            print(f"\n🌍 حالة الأسواق:")
            print(f"   • نيويورك: {market_status.get('ny_market_status', 'N/A')}")
            print(f"   • لندن: {market_status.get('london_market_status', 'N/A')}")
            print(f"   • جلسة رئيسية: {'نعم' if market_status.get('is_major_trading_session', False) else 'لا'}")
        
    else:
        print(f"❌ حالة التحليل: {results.get('status', 'غير معروف')}")
        if 'error' in results:
            print(f"الخطأ: {results['error']}")
    
    print("=" * 60)
    print("🔔 انتهى التحليل الاحترافي")
    print("=" * 60)

if __name__ == "__main__":
    main()
