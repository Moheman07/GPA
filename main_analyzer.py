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
            'gold': 'GC=F', 'gold_etf': 'GLD',
            'dxy': 'DX-Y.NYB', 'vix': '^VIX',
            'treasury': '^TNX', 'oil': 'CL=F',
            'spy': 'SPY', 'usdeur': 'EURUSD=X'
        }
        self.news_api_key = os.getenv("NEWS_API_KEY")
        
    def fetch_multi_timeframe_data(self):
        """جلب بيانات متعددة الأطر الزمنية"""
        print("📊 جلب بيانات متعددة الأطر الزمنية...")
        try:
            symbols_list = list(self.symbols.values())
            daily_data = yf.download(symbols_list, period="1y", interval="1d", group_by='ticker')
            if daily_data.empty: raise ValueError("لا توجد بيانات")
            print("✅ تم جلب البيانات متعددة الأطر")
            return {'daily': daily_data}
        except Exception as e:
            print(f"❌ خطأ في جلب البيانات: {e}")
            return None

    def extract_gold_data(self, market_data):
        """استخراج بيانات الذهب مع تحسينات"""
        try:
            daily_data = market_data['daily']
            
            if hasattr(daily_data.columns, 'levels') and len(daily_data.columns.levels) > 1:
                available_symbols = daily_data.columns.levels[0].tolist()
                gold_symbol_to_use = None
                if self.symbols['gold'] in available_symbols and not daily_data[self.symbols['gold']].dropna().empty:
                    gold_symbol_to_use = self.symbols['gold']
                elif self.symbols['gold_etf'] in available_symbols and not daily_data[self.symbols['gold_etf']].dropna().empty:
                    gold_symbol_to_use = self.symbols['gold_etf']
                
                if gold_symbol_to_use:
                    gold_daily = daily_data[gold_symbol_to_use].copy()
                    print(f"✅ تم استخدام بيانات {gold_symbol_to_use}")
                else:
                    raise ValueError("لا يمكن العثور على بيانات الذهب (GC=F or GLD)")
            else:
                gold_daily = daily_data.copy()

            gold_daily = gold_daily.dropna(subset=['Close'])
            print(f"✅ بيانات يومية نظيفة: {len(gold_daily)} يوم")
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
            
            # مؤشر القوة المخصص
            daily_df['Strength_Index'] = (
                (daily_df['RSI'] - 50) * 0.3 +
                (daily_df['ROC']) * 0.4 +
                ((daily_df['Close'] - daily_df['SMA_50']) / daily_df['SMA_50'] * 100) * 0.3
            )
            
            print("✅ تم حساب المؤشرات الاحترافية")
            return daily_df.dropna() # إزالة أي صفوف قد تحتوي على NaN بعد الحسابات
            
        except Exception as e:
            print(f"❌ خطأ في حساب المؤشرات: {e}")
            return gold_data

    # ... (باقي الدوال: calculate_fibonacci, analyze_volume_profile, etc. تبقى كما هي)
    def calculate_fibonacci_levels(self, data, periods=50):
        try:
            recent_data = data.tail(periods)
            high = recent_data['High'].max()
            low = recent_data['Low'].min()
            diff = high - low
            return {
                'high': round(high, 2), 'low': round(low, 2),
                'fib_23_6': round(high - (diff * 0.236), 2),
                'fib_38_2': round(high - (diff * 0.382), 2),
                'fib_50_0': round(high - (diff * 0.500), 2),
                'fib_61_8': round(high - (diff * 0.618), 2),
                'fib_78_6': round(high - (diff * 0.786), 2)
            }
        except Exception: return {}
    
    def analyze_volume_profile(self, data):
        try:
            latest = data.iloc[-1]
            return {
                'current_volume': int(latest.get('Volume', 0)),
                'avg_volume_20': int(data.tail(20)['Volume'].mean()),
                'volume_ratio': round(latest.get('Volume_Ratio', 1), 2),
                'volume_strength': 'قوي' if latest.get('Volume_Ratio', 1) > 1.5 else ('ضعيف' if latest.get('Volume_Ratio', 1) < 0.7 else 'طبيعي')
            }
        except Exception: return {}

    def analyze_correlations(self, market_data):
        try:
            print("📊 تحليل الارتباطات المتقدم...")
            daily_data = market_data['daily']
            correlations, strength_analysis = {}, {}
            if hasattr(daily_data.columns, 'levels'):
                available_symbols = daily_data.columns.levels[0].tolist()
                gold_symbol = self.symbols['gold'] if self.symbols['gold'] in available_symbols else self.symbols['gold_etf']
                if gold_symbol in available_symbols:
                    gold_prices = daily_data[gold_symbol]['Close'].dropna()
                    for name, symbol in self.symbols.items():
                        if name not in ['gold', 'gold_etf'] and symbol in available_symbols:
                            asset_prices = daily_data[symbol]['Close'].dropna()
                            common_index = gold_prices.index.intersection(asset_prices.index)
                            if len(common_index) > 30:
                                corr = gold_prices.loc[common_index].corr(asset_prices.loc[common_index])
                                if pd.notna(corr):
                                    correlations[name] = round(corr, 3)
                                    if abs(corr) > 0.7: strength_analysis[name] = 'قوي جداً'
                                    elif abs(corr) > 0.5: strength_analysis[name] = 'قوي'
                                    elif abs(corr) > 0.3: strength_analysis[name] = 'متوسط'
                                    else: strength_analysis[name] = 'ضعيف'
            return {'correlations': correlations, 'strength_analysis': strength_analysis}
        except Exception: return {'correlations': {}, 'strength_analysis': {}}

    def fetch_news(self):
        print("📰 جلب أخبار الذهب المتخصصة...")
        if not self.news_api_key: return {"status": "no_api_key", "high_impact_news": []}
        try:
            keywords = "\"gold price\" OR \"federal reserve\" OR \"interest rate\" OR inflation OR \"dollar index\""
            url = f"https://newsapi.org/v2/everything?q={keywords}&language=en&sortBy=publishedAt&pageSize=20&from={(datetime.now() - timedelta(days=2)).date()}&apiKey={self.news_api_key}"
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            articles = response.json().get('articles', [])
            high_impact, medium_impact = [], []
            high_kw = ['federal reserve', 'fed', 'interest rate', 'inflation', 'monetary policy']
            for article in articles:
                content = f"{(article.get('title') or '').lower()} {(article.get('description') or '').lower()}"
                news_item = {'title': article.get('title'), 'source': article.get('source', {}).get('name'), 'publishedAt': article.get('publishedAt')}
                if any(kw in content for kw in high_kw):
                    high_impact.append({**news_item, 'impact': 'عالي'})
                else:
                    medium_impact.append({**news_item, 'impact': 'متوسط'})
            return {"status": "success", "high_impact_news": high_impact[:3], "medium_impact_news": medium_impact[:3]}
        except Exception: return {"status": "error", "high_impact_news": []}
        
    def generate_professional_signals(self, technical_data, correlations, volume_analysis, fibonacci_levels):
        try:
            print("🎯 توليد إشارات احترافية...")
            latest = technical_data.iloc[-1]
            prev = technical_data.iloc[-2]
            signals = {}
            trend_score, momentum_score, volume_score, fib_score, correlation_score = 0, 0, 0, 0, 0
            
            # Trend Analysis
            if pd.notna(latest.get('SMA_200')) and pd.notna(latest.get('SMA_50')):
                if latest['Close'] > latest['SMA_200']:
                    signals['long_term_trend'] = "صاعد قوي" if latest['Close'] > latest['SMA_50'] else "صاعد"
                    trend_score = 3 if latest['Close'] > latest['SMA_50'] else 2
                else:
                    signals['long_term_trend'] = "هابط قوي" if latest['Close'] < latest['SMA_50'] else "هابط"
                    trend_score = -3 if latest['Close'] < latest['SMA_50'] else -2
            
            # Momentum Analysis
            if pd.notna(latest.get('MACD')) and pd.notna(latest.get('MACD_Signal')):
                if latest['MACD'] > latest['MACD_Signal']:
                    signals['macd'] = "إيجابي متزايد" if latest['MACD_Histogram'] > prev['MACD_Histogram'] else "إيجابي"
                    momentum_score = 2 if latest['MACD_Histogram'] > prev['MACD_Histogram'] else 1
                else:
                    signals['macd'] = "سلبي"; momentum_score = -1
            
            if pd.notna(latest.get('RSI')):
                rsi = latest['RSI']
                if 40 <= rsi <= 60: signals['rsi_status'] = "متوازن"; momentum_score += 1
                elif rsi > 70: signals['rsi_status'] = "ذروة شراء"; momentum_score -= 1
                elif rsi < 30: signals['rsi_status'] = "ذروة بيع"; momentum_score += 2
            
            if pd.notna(latest.get('ROC')):
                if latest['ROC'] > 2: signals['roc'] = "زخم صاعد قوي"; momentum_score += 1
                elif latest['ROC'] < -2: signals['roc'] = "زخم هابط قوي"; momentum_score -= 1

            # Volume & Fibonacci & Correlation Scores
            if volume_analysis.get('volume_strength') == 'قوي': volume_score = 1
            if fibonacci_levels and latest['Close'] > fibonacci_levels.get('fib_61_8', 0): fib_score = 1
            dxy_corr = correlations.get('correlations', {}).get('dxy', 0)
            if dxy_corr < -0.7: correlation_score = 2
            elif dxy_corr < -0.5: correlation_score = 1
            
            # Weighted Score
            total_score = (trend_score * 0.30 + momentum_score * 0.25 + volume_score * 0.15 + fib_score * 0.15 + correlation_score * 0.15)
            
            # Final Signal
            if total_score >= 1.5: final_signal, confidence, action = "Buy", "High", "شراء بحجم متوسط"
            elif total_score >= 0.5: final_signal, confidence, action = "Weak Buy", "Medium", "شراء حذر بحجم صغير"
            elif total_score <= -1.5: final_signal, confidence, action = "Sell", "High", "بيع بحجم متوسط"
            elif total_score <= -0.5: final_signal, confidence, action = "Weak Sell", "Medium", "بيع حذر بحجم صغير"
            else: final_signal, confidence, action = "Hold", "Low", "انتظر"
            
            # Risk Management
            current_price = latest['Close']
            atr = latest.get('ATR', current_price * 0.02)
            atr_percent = latest.get('ATR_Percent', 2.0)
            
            result = {'signal': final_signal, 'confidence': confidence, 'action_recommendation': action, 'total_score': round(total_score, 2),
                      'component_scores': {'trend': trend_score, 'momentum': momentum_score, 'volume': volume_score, 'fibonacci': fib_score, 'correlation': correlation_score},
                      'current_price': round(current_price, 2),
                      'risk_management': {
                          'stop_loss_levels': {'conservative': round(current_price - (atr * 1.5), 2), 'moderate': round(current_price - (atr * 2.0), 2)},
                          'profit_targets': {'target_1': round(current_price + (atr * 2), 2), 'target_2': round(current_price + (atr * 3.5), 2)},
                          'position_size_recommendation': self.calculate_position_size(atr_percent, confidence)
                      },
                      'technical_details': signals,
                      'advanced_indicators': {'rsi': round(latest.get('RSI', 0), 1), 'williams_r': round(latest.get('Williams_R', 0), 1), 'roc': round(latest.get('ROC', 0), 2)}
            }
            print(f"✅ الإشارة الاحترافية: {final_signal} ({confidence})")
            return result
        except Exception as e:
            print(f"❌ خطأ في توليد الإشارات: {e}")
            return {"error": str(e)}

    def calculate_position_size(self, volatility, confidence):
        if confidence in ["Very High", "High"]: return "متوسط"
        else: return "صغير"

    def get_market_status(self):
        try:
            import pytz
            utc_time = datetime.now(pytz.UTC)
            ny_time = utc_time.astimezone(pytz.timezone('America/New_York'))
            london_time = utc_time.astimezone(pytz.timezone('Europe/London'))
            ny_trading = ny_time.weekday() < 5 and 9 <= ny_time.hour < 16
            london_trading = london_time.weekday() < 5 and 8 <= london_time.hour < 17
            return {
                'ny_market_status': 'Open' if ny_trading else 'Closed',
                'london_market_status': 'Open' if london_trading else 'Closed',
                'is_major_trading_session': ny_trading or london_trading
            }
        except Exception: return {'status': 'Unknown'}

    def run_analysis(self):
        """تشغيل التحليل الاحترافي الشامل"""
        print("🚀 بدء التحليل الاحترافي للذهب...")
        try:
            market_data = self.fetch_multi_timeframe_data()
            if market_data is None: raise ValueError("فشل جلب بيانات السوق")
            
            gold_data = self.extract_gold_data(market_data)
            if gold_data is None: raise ValueError("فشل استخراج بيانات الذهب")
            
            technical_data = self.calculate_professional_indicators(gold_data)
            fibonacci_levels = self.calculate_fibonacci_levels(technical_data)
            volume_analysis = self.analyze_volume_profile(technical_data)
            correlations = self.analyze_correlations(market_data)
            news_data = self.fetch_news()
            
            signals = self.generate_professional_signals(technical_data, correlations, volume_analysis, fibonacci_levels)
            
            # --- ✅ التعديل هنا ---
            # بناء قاموس gold_analysis أولاً
            gold_analysis_data = {
                'price_usd': signals.get('current_price'),
                'signal': signals.get('signal'),
                'confidence': signals.get('confidence'),
                'action_recommendation': signals.get('action_recommendation'),
                'technical_score': signals.get('total_score'),
                'component_analysis': signals.get('component_scores', {}),
                'technical_details': signals.get('technical_details', {}),
                'advanced_indicators': signals.get('advanced_indicators', {}),
                'risk_management': signals.get('risk_management', {})
            }
            
            # بناء قاموس الملخص من القاموس الذي تم إنشاؤه للتو
            summary_data = {
                'signal': gold_analysis_data.get('signal', 'N/A'),
                'price': gold_analysis_data.get('price_usd', 'N/A'),
                'confidence': gold_analysis_data.get('confidence', 'N/A'),
                'action': gold_analysis_data.get('action_recommendation', 'N/A'),
                'rsi': gold_analysis_data.get('advanced_indicators', {}).get('rsi', 'N/A'),
                'trend': gold_analysis_data.get('technical_details', {}).get('long_term_trend', 'N/A')
            }
            
            # تجميع النتائج النهائية
            results = {
                'timestamp': datetime.now().isoformat(),
                'market_status': self.get_market_status(),
                'gold_analysis': gold_analysis_data,
                'fibonacci_levels': fibonacci_levels,
                'volume_analysis': volume_analysis,
                'market_correlations': correlations,
                'news_analysis': news_data,
                'summary': summary_data
            }
            
            self.save_single_result(results)
            print("✅ تم إتمام التحليل الاحترافي بنجاح!")
            return results
        except Exception as e:
            print(f"❌ فشل التحليل الاحترافي: {e}")
            error_result = {
                'timestamp': datetime.now().isoformat(), 'status': 'error',
                'error': str(e), 'market_status': self.get_market_status()
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
    if 'summary' in results:
        summary = results['summary']
        print(f"🎯 الإشارة: {summary.get('signal', 'N/A')} ({summary.get('confidence', 'N/A')})")
        print(f"💡 التوصية: {summary.get('action', 'N/A')}")
        print(f"💰 السعر الحالي: ${summary.get('price', 'N/A')}")
        print(f"📈 الاتجاه العام: {summary.get('trend', 'N/A')}")
        print(f"📊 مؤشر القوة النسبية: {summary.get('rsi', 'N/A')}")
    else:
        print("❌ لم يتم إنشاء الملخص بسبب خطأ.")
    print("=" * 60)

if __name__ == "__main__":
    main()
