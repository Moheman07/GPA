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

class SimpleGoldAnalyzer:
    def __init__(self):
        self.symbols = {
            'gold': 'GC=F',
            'gold_etf': 'GLD',
            'dxy': 'DX-Y.NYB',
            'vix': '^VIX',
            'treasury': '^TNX',
            'oil': 'CL=F',
            'spy': 'SPY'
        }
        self.news_api_key = os.getenv("NEWS_API_KEY")
        
    def fetch_data(self):
        """جلب البيانات"""
        print("📊 جلب بيانات السوق...")
        
        try:
            symbols_list = list(self.symbols.values())
            print(f"جلب البيانات لـ: {symbols_list}")
            
            data = yf.download(symbols_list, period="6mo", interval="1d", group_by='ticker')
            
            if data.empty:
                raise ValueError("لا توجد بيانات")
                
            print(f"✅ تم جلب البيانات - الشكل: {data.shape}")
            return data
            
        except Exception as e:
            print(f"❌ خطأ في جلب البيانات: {e}")
            return None

    def extract_gold_data(self, market_data):
        """استخراج بيانات الذهب"""
        try:
            print("🔍 استخراج بيانات الذهب...")
            
            # التحقق من هيكل البيانات
            if hasattr(market_data.columns, 'levels') and len(market_data.columns.levels) > 1:
                # Multi-level columns
                gold_symbol = self.symbols['gold']
                
                # التحقق من وجود رمز الذهب
                available_symbols = market_data.columns.levels[0].tolist()
                print(f"الرموز المتاحة: {available_symbols}")
                
                if gold_symbol in available_symbols:
                    gold_data = market_data[gold_symbol].copy()
                    print(f"✅ تم استخراج بيانات الذهب: {gold_data.shape}")
                elif self.symbols['gold_etf'] in available_symbols:
                    gold_data = market_data[self.symbols['gold_etf']].copy()
                    print(f"✅ تم استخراج بيانات GLD بدلاً من GC=F: {gold_data.shape}")
                else:
                    raise ValueError(f"لا يمكن العثور على بيانات الذهب")
            else:
                gold_data = market_data.copy()
                print(f"✅ بيانات مستوى واحد: {gold_data.shape}")
            
            # تنظيف البيانات
            gold_data = gold_data.dropna(subset=['Close'])
            
            print(f"✅ بيانات الذهب نظيفة: {len(gold_data)} يوم")
            print(f"آخر سعر: ${gold_data['Close'].iloc[-1]:.2f}")
            
            return gold_data
            
        except Exception as e:
            print(f"❌ خطأ في استخراج بيانات الذهب: {e}")
            return None

    def calculate_simple_indicators(self, prices):
        """حساب مؤشرات بسيطة"""
        try:
            print("📊 حساب المؤشرات الفنية...")
            df = prices.copy()
            
            if 'Close' not in df.columns:
                print(f"❌ لا يوجد عمود Close. الأعمدة المتاحة: {df.columns.tolist()}")
                return df
            
            # المتوسطات المتحركة
            df['SMA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
            df['SMA_50'] = df['Close'].rolling(window=50, min_periods=1).mean()
            df['SMA_200'] = df['Close'].rolling(window=200, min_periods=1).mean()
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['Close'].ewm(span=12).mean()
            exp2 = df['Close'].ewm(span=26).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            
            # Bollinger Bands
            std = df['Close'].rolling(window=20, min_periods=1).std()
            df['BB_Upper'] = df['SMA_20'] + (std * 2)
            df['BB_Lower'] = df['SMA_20'] - (std * 2)
            
            # ATR
            if all(col in df.columns for col in ['High', 'Low']):
                high_low = df['High'] - df['Low']
                high_close = np.abs(df['High'] - df['Close'].shift())
                low_close = np.abs(df['Low'] - df['Close'].shift())
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                df['ATR'] = true_range.rolling(14, min_periods=1).mean()
            else:
                df['ATR'] = df['Close'] * 0.02
            
            print("✅ تم حساب المؤشرات الفنية")
            return df
            
        except Exception as e:
            print(f"❌ خطأ في حساب المؤشرات: {e}")
            return prices

    def analyze_correlations(self, market_data):
        """تحليل الارتباطات"""
        try:
            print("📊 تحليل الارتباطات...")
            correlations = {}
            
            if hasattr(market_data.columns, 'levels') and len(market_data.columns.levels) > 1:
                available_symbols = market_data.columns.levels[0].tolist()
                
                # اختيار رمز الذهب المتاح
                gold_symbol = None
                if self.symbols['gold'] in available_symbols:
                    gold_symbol = self.symbols['gold']
                elif self.symbols['gold_etf'] in available_symbols:
                    gold_symbol = self.symbols['gold_etf']
                
                if gold_symbol:
                    gold_prices = market_data[gold_symbol]['Close'].dropna()
                    
                    for name, symbol in self.symbols.items():
                        if name not in ['gold', 'gold_etf'] and symbol in available_symbols:
                            try:
                                asset_prices = market_data[symbol]['Close'].dropna()
                                common_index = gold_prices.index.intersection(asset_prices.index)
                                if len(common_index) > 30:
                                    corr = gold_prices.loc[common_index].corr(asset_prices.loc[common_index])
                                    if not pd.isna(corr):
                                        correlations[name] = round(corr, 3)
                            except Exception as e:
                                continue
            
            return correlations
            
        except Exception as e:
            print(f"❌ خطأ في تحليل الارتباطات: {e}")
            return {}

    def fetch_news(self):
        """جلب الأخبار"""
        print("📰 جلب أخبار الذهب...")
        
        if not self.news_api_key:
            print("⚠️ مفتاح الأخبار غير متوفر")
            return {"status": "no_api_key", "articles": []}
        
        try:
            keywords = "gold OR XAU OR \"federal reserve\" OR inflation"
            url = (
                f"https://newsapi.org/v2/everything?"
                f"q={keywords}&"
                f"language=en&"
                f"sortBy=publishedAt&"
                f"pageSize=15&"
                f"from={(datetime.now() - timedelta(days=1)).date()}&"
                f"apiKey={self.news_api_key}"
            )
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            articles = response.json().get('articles', [])
            
            relevant = []
            for article in articles:
                title = (article.get('title', '') or '').lower()
                desc = (article.get('description', '') or '').lower()
                
                if any(word in f"{title} {desc}" for word in ['gold', 'xau', 'fed', 'inflation']):
                    relevant.append({
                        'title': article.get('title', ''),
                        'source': article.get('source', {}).get('name', ''),
                        'publishedAt': article.get('publishedAt', '')
                    })
            
            print(f"✅ تم جلب {len(relevant)} خبر مهم")
            return {"status": "success", "articles": relevant[:5]}
            
        except Exception as e:
            print(f"❌ خطأ في جلب الأخبار: {e}")
            return {"status": "error", "error": str(e), "articles": []}

    def generate_signals(self, technical_data, correlations):
        """توليد الإشارات"""
        try:
            print("🎯 توليد إشارات التداول...")
            
            if technical_data is None or technical_data.empty:
                raise ValueError("لا توجد بيانات فنية")
            
            latest = technical_data.iloc[-1]
            
            # حساب النقاط
            score = 0
            signals = {}
            
            # الاتجاه
            if pd.notna(latest.get('SMA_200')):
                if latest['Close'] > latest['SMA_200']:
                    signals['trend'] = "صاعد"
                    score += 2
                else:
                    signals['trend'] = "هابط"
                    score -= 2
            else:
                signals['trend'] = "غير محدد"
            
            # الزخم
            if pd.notna(latest.get('MACD')) and pd.notna(latest.get('MACD_Signal')):
                if latest['MACD'] > latest['MACD_Signal']:
                    signals['momentum'] = "إيجابي"
                    score += 1
                else:
                    signals['momentum'] = "سلبي"
                    score -= 1
            else:
                signals['momentum'] = "غير محدد"
            
            # RSI
            if pd.notna(latest.get('RSI')):
                rsi = latest['RSI']
                if rsi > 70:
                    signals['rsi'] = "ذروة شراء"
                    score -= 1
                elif rsi < 30:
                    signals['rsi'] = "ذروة بيع"
                    score += 1
                else:
                    signals['rsi'] = f"عادي ({rsi:.1f})"
            else:
                signals['rsi'] = "غير محدد"
            
            # إشارة الارتباط مع الدولار
            dxy_corr = correlations.get('dxy', 0)
            if dxy_corr < -0.5:
                signals['dxy_relationship'] = "سلبي قوي - مفيد للذهب"
                score += 0.5
            elif dxy_corr > 0.3:
                signals['dxy_relationship'] = "إيجابي - غير طبيعي"
                score -= 0.5
            else:
                signals['dxy_relationship'] = f"معتدل ({dxy_corr})"
            
            # الإشارة النهائية
            if score >= 2:
                final_signal = "Buy"
                confidence = "High" if score >= 3 else "Medium"
            elif score <= -2:
                final_signal = "Sell"
                confidence = "High" if score <= -3 else "Medium"
            else:
                final_signal = "Hold"
                confidence = "Low"
            
            # إدارة المخاطر
            current_price = latest['Close']
            atr = latest.get('ATR', current_price * 0.02)
            
            result = {
                'signal': final_signal,
                'confidence': confidence,
                'score': round(score, 1),
                'current_price': round(current_price, 2),
                'stop_loss': round(current_price - (atr * 2), 2),
                'take_profit': round(current_price + (atr * 3), 2),
                'technical_details': signals,
                'indicators': {
                    'rsi': round(latest.get('RSI', 0), 1),
                    'sma_20': round(latest.get('SMA_20', 0), 2),
                    'sma_50': round(latest.get('SMA_50', 0), 2),
                    'sma_200': round(latest.get('SMA_200', 0), 2),
                    'macd': round(latest.get('MACD', 0), 3),
                    'macd_signal': round(latest.get('MACD_Signal', 0), 3)
                }
            }
            
            print(f"✅ الإشارة النهائية: {final_signal} ({confidence})")
            return result
            
        except Exception as e:
            print(f"❌ خطأ في توليد الإشارات: {e}")
            return {"error": str(e)}

    def get_market_status(self):
        """حالة السوق"""
        try:
            import pytz
            ny_tz = pytz.timezone('America/New_York')
            ny_time = datetime.now(ny_tz)
            
            is_weekday = ny_time.weekday() < 5
            is_trading_hours = 9 <= ny_time.hour < 16
            
            return {
                'current_time_est': ny_time.strftime('%Y-%m-%d %H:%M:%S EST'),
                'is_trading_hours': is_weekday and is_trading_hours,
                'status': 'Open' if (is_weekday and is_trading_hours) else 'Closed'
            }
        except:
            return {
                'current_time_est': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'is_trading_hours': False,
                'status': 'Unknown'
            }

    def run_analysis(self):
        """تشغيل التحليل الكامل"""
        print("🚀 بدء تحليل الذهب...")
        
        try:
            # 1. جلب البيانات
            market_data = self.fetch_data()
            if market_data is None:
                raise ValueError("فشل في جلب بيانات السوق")
            
            # 2. استخراج بيانات الذهب
            gold_data = self.extract_gold_data(market_data)
            if gold_data is None:
                raise ValueError("فشل في استخراج بيانات الذهب")
            
            # 3. حساب المؤشرات
            technical_data = self.calculate_simple_indicators(gold_data)
            
            # 4. تحليل الارتباطات
            correlations = self.analyze_correlations(market_data)
            
            # 5. جلب الأخبار
            news_data = self.fetch_news()
            
            # 6. توليد الإشارات
            signals = self.generate_signals(technical_data, correlations)
            
            # 7. تجميع النتائج النهائية
            results = {
                'timestamp': datetime.now().isoformat(),
                'last_update': datetime.now().strftime('%Y-%m-%d %H:%M UTC'),
                'market_status': self.get_market_status(),
                'gold_analysis': {
                    'price_usd': signals.get('current_price'),
                    'signal': signals.get('signal'),
                    'confidence': signals.get('confidence'),
                    'technical_score': signals.get('score'),
                    'technical_details': signals.get('technical_details', {}),
                    'indicators': signals.get('indicators', {}),
                    'risk_management': {
                        'stop_loss': signals.get('stop_loss'),
                        'take_profit': signals.get('take_profit')
                    }
                },
                'market_correlations': correlations,
                'news_analysis': {
                    'status': news_data.get('status'),
                    'articles_count': len(news_data.get('articles', [])),
                    'headlines': [article.get('title') for article in news_data.get('articles', [])]
                },
                'summary': {
                    'signal': signals.get('signal', 'N/A'),
                    'price': signals.get('current_price', 'N/A'),
                    'confidence': signals.get('confidence', 'N/A'),
                    'rsi': signals.get('indicators', {}).get('rsi', 'N/A'),
                    'trend': signals.get('technical_details', {}).get('trend', 'N/A')
                }
            }
            
            # 8. حفظ النتيجة في ملف واحد فقط
            self.save_single_result(results)
            
            print("✅ تم إتمام التحليل بنجاح!")
            return results
            
        except Exception as e:
            print(f"❌ فشل التحليل: {e}")
            
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
            # اسم الملف الثابت الذي يتم تحديثه في كل مرة
            filename = "gold_analysis.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            
            print(f"💾 تم تحديث الملف: {filename}")
            
            # التحقق من إنشاء الملف
            if os.path.exists(filename):
                file_size = os.path.getsize(filename)
                print(f"📁 حجم الملف: {file_size} بايت")
            else:
                print("❌ لم يتم إنشاء الملف!")
            
        except Exception as e:
            print(f"❌ خطأ في حفظ النتائج: {e}")

def main():
    """الدالة الرئيسية"""
    print("=" * 50)
    print("🏆 محلل الذهب المتقدم")
    print("=" * 50)
    
    analyzer = SimpleGoldAnalyzer()
    results = analyzer.run_analysis()
    
    # طباعة ملخص سريع
    print("\n" + "=" * 50)
    print("📋 ملخص النتائج:")
    print("=" * 50)
    
    if results.get('status') != 'error' and 'gold_analysis' in results:
        gold = results['gold_analysis']
        print(f"💰 السعر: ${gold.get('price_usd', 'N/A')}")
        print(f"🎯 الإشارة: {gold.get('signal', 'N/A')}")
        print(f"🔍 الثقة: {gold.get('confidence', 'N/A')}")
        print(f"📊 النقاط: {gold.get('technical_score', 'N/A')}")
        
        indicators = gold.get('indicators', {})
        print(f"📈 RSI: {indicators.get('rsi', 'N/A')}")
        print(f"📊 SMA 200: ${indicators.get('sma_200', 'N/A')}")
        
        risk = gold.get('risk_management', {})
        print(f"🛑 وقف الخسارة: ${risk.get('stop_loss', 'N/A')}")
        print(f"🎯 جني الأرباح: ${risk.get('take_profit', 'N/A')}")
        
    else:
        print(f"❌ حالة: {results.get('status', 'غير معروف')}")
        if 'error' in results:
            print(f"الخطأ: {results['error']}")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
