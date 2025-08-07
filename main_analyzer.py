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
            data = yf.download(symbols_list, period="6mo", interval="1d")
            
            if data.empty:
                raise ValueError("لا توجد بيانات")
                
            print("✅ تم جلب البيانات")
            return data
            
        except Exception as e:
            print(f"❌ خطأ في جلب البيانات: {e}")
            return None

    def calculate_simple_indicators(self, prices):
        """حساب مؤشرات بسيطة بدون pandas_ta"""
        try:
            df = prices.copy()
            
            # المتوسطات المتحركة
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['SMA_200'] = df['Close'].rolling(window=200).mean()
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['Close'].ewm(span=12).mean()
            exp2 = df['Close'].ewm(span=26).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            
            # Bollinger Bands
            std = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['SMA_20'] + (std * 2)
            df['BB_Lower'] = df['SMA_20'] - (std * 2)
            
            # ATR (Average True Range)
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            true_range = pd.DataFrame([high_low, high_close, low_close]).max()
            df['ATR'] = true_range.rolling(14).mean()
            
            return df
            
        except Exception as e:
            print(f"❌ خطأ في حساب المؤشرات: {e}")
            return prices

    def analyze_correlations(self, data):
        """تحليل الارتباطات"""
        try:
            correlations = {}
            
            if len(data.columns.levels) > 1:
                # Multi-level columns
                gold_prices = data[self.symbols['gold']]['Close']
                
                for name, symbol in self.symbols.items():
                    if name != 'gold' and symbol in data.columns.levels[0]:
                        try:
                            asset_prices = data[symbol]['Close']
                            corr = gold_prices.corr(asset_prices)
                            if not pd.isna(corr):
                                correlations[name] = round(corr, 3)
                        except:
                            continue
            
            return correlations
            
        except Exception as e:
            print(f"❌ خطأ في تحليل الارتباطات: {e}")
            return {}

    def fetch_news(self):
        """جلب الأخبار"""
        if not self.news_api_key:
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
            
            # فلترة الأخبار المهمة
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
            
            return {"status": "success", "articles": relevant[:5]}
            
        except Exception as e:
            print(f"❌ خطأ في جلب الأخبار: {e}")
            return {"status": "error", "error": str(e), "articles": []}

    def generate_signals(self, technical_data, correlations):
        """توليد الإشارات"""
        try:
            latest = technical_data.iloc[-1]
            
            # حساب النقاط
            score = 0
            signals = {}
            
            # الاتجاه
            if pd.notna(latest['SMA_200']):
                if latest['Close'] > latest['SMA_200']:
                    signals['trend'] = "صاعد"
                    score += 2
                else:
                    signals['trend'] = "هابط"
                    score -= 2
            
            # الزخم
            if pd.notna(latest['MACD']) and pd.notna(latest['MACD_Signal']):
                if latest['MACD'] > latest['MACD_Signal']:
                    signals['momentum'] = "إيجابي"
                    score += 1
                else:
                    signals['momentum'] = "سلبي"
                    score -= 1
            
            # RSI
            if pd.notna(latest['RSI']):
                rsi = latest['RSI']
                if rsi > 70:
                    signals['rsi'] = "ذروة شراء"
                    score -= 1
                elif rsi < 30:
                    signals['rsi'] = "ذروة بيع"
                    score += 1
                else:
                    signals['rsi'] = f"عادي ({rsi:.1f})"
            
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
            
            return {
                'signal': final_signal,
                'confidence': confidence,
                'score': score,
                'current_price': round(current_price, 2),
                'stop_loss': round(current_price - (atr * 2), 2),
                'take_profit': round(current_price + (atr * 3), 2),
                'technical_details': signals
            }
            
        except Exception as e:
            print(f"❌ خطأ في توليد الإشارات: {e}")
            return {"error": str(e)}

    def run_analysis(self):
        """تشغيل التحليل الكامل"""
        print("🚀 بدء تحليل الذهب...")
        
        try:
            # 1. جلب البيانات
            market_data = self.fetch_data()
            if market_data is None:
                raise ValueError("فشل في جلب البيانات")
            
            # 2. استخراج بيانات الذهب
            if len(market_data.columns.levels) > 1:
                gold_data = market_data[self.symbols['gold']]
            else:
                gold_data = market_data
            
            # 3. حساب المؤشرات
            technical_data = self.calculate_simple_indicators(gold_data)
            
            # 4. تحليل الارتباطات
            correlations = self.analyze_correlations(market_data)
            
            # 5. جلب الأخبار
            news_data = self.fetch_news()
            
            # 6. توليد الإشارات
            signals = self.generate_signals(technical_data, correlations)
            
            # 7. تجميع النتائج
            results = {
                'timestamp': datetime.now().isoformat(),
                'market_status': self.get_market_status(),
                'gold_analysis': {
                    'price_usd': signals.get('current_price'),
                    'signal': signals.get('signal'),
                    'confidence': signals.get('confidence'),
                    'technical_score': signals.get('score'),
                    'technical_details': signals.get('technical_details', {}),
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
                }
            }
            
            # 8. حفظ النتائج
            self.save_results(results)
            
            print("✅ تم إتمام التحليل بنجاح!")
            return results
            
        except Exception as e:
            error_result = {
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'error': str(e)
            }
            self.save_results(error_result)
            print(f"❌ فشل التحليل: {e}")
            return error_result

    def get_market_status(self):
        """حالة السوق"""
        from datetime import datetime
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

    def save_results(self, results):
        """حفظ النتائج في JSON"""
        try:
            # ملف بالطابع الزمني
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            timestamped_file = f"gold_analysis_{timestamp}.json"
            
            # ملف النتيجة الأخيرة
            latest_file = "gold_analysis_latest.json"
            
            # حفظ الملفين
            for filename in [timestamped_file, latest_file]:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            
            print(f"💾 تم حفظ النتائج في: {timestamped_file}")
            
        except Exception as e:
            print(f"❌ خطأ في حفظ النتائج: {e}")

def main():
    """الدالة الرئيسية"""
    analyzer = SimpleGoldAnalyzer()
    results = analyzer.run_analysis()
    
    # طباعة ملخص سريع
    if 'gold_analysis' in results:
        gold = results['gold_analysis']
        print(f"\n📊 الملخص:")
        print(f"   السعر: ${gold.get('price_usd', 'N/A')}")
        print(f"   الإشارة: {gold.get('signal', 'N/A')}")
        print(f"   الثقة: {gold.get('confidence', 'N/A')}")

if __name__ == "__main__":
    main()
